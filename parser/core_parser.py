import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
import tree_sitter_java as tsjava
import tree_sitter_cpp as tscpp
import tree_sitter_c as tsc
import tree_sitter_go as tsgo
import tree_sitter_rust as tsrust

from tree_sitter import Language, Parser, Query, QueryCursor
from parser.models import ParsedFile, CodeEntity, CodeRelationship

LANGUAGE_REGISTRY = {
    "python": Language(tspython.language()),
    "javascript": Language(tsjavascript.language()),
    "typescript": Language(tstypescript.language_typescript()),
    "java": Language(tsjava.language()),
    "cpp": Language(tscpp.language()),
    "c": Language(tsc.language()),
    "go": Language(tsgo.language()),
    "rust": Language(tsrust.language()),
}

QUERY_REGISTRY = {
    "python": """
        (class_definition name: (identifier) @class.name)
        (function_definition name: (identifier) @function.name)
        (import_statement name: (dotted_name) @import.name)
        (import_statement (dotted_name) @import.name)
        (aliased_import name: (dotted_name) @import.name)
        (import_from_statement module_name: (dotted_name) @import.module)
        (import_from_statement module_name: (relative_import) @import.module)
        (call function: (identifier) @call.target)
        (call function: (attribute attribute: (identifier) @call.target))
    """,
    "javascript": """
        (class_declaration name: (identifier) @class.name)
        (function_declaration name: (identifier) @function.name)
        (call_expression function: (identifier) @call.target)
        (call_expression function: (member_expression property: (property_identifier) @call.target))
    """,
    "typescript": """
        (class_declaration name: (type_identifier) @class.name)
        (function_declaration name: (identifier) @function.name)
        (call_expression function: (identifier) @call.target)
    """,
    "java": """
        (class_declaration name: (identifier) @class.name)
        (method_declaration name: (identifier) @function.name)
        (method_invocation name: (identifier) @call.target)
    """,
    "cpp": """
        (class_specifier name: (type_identifier) @class.name)
        (function_definition declarator: (function_declarator declarator: (identifier) @function.name))
        (call_expression function: (identifier) @call.target)
    """,
    "c": """
        (function_definition declarator: (function_declarator declarator: (identifier) @function.name))
        (call_expression function: (identifier) @call.target)
    """,
    "go": """
        (type_spec name: (type_identifier) @class.name type: (struct_type))
        (function_declaration name: (identifier) @function.name)
        (call_expression function: (identifier) @call.target)
    """,
    "rust": """
        (struct_item name: (type_identifier) @class.name)
        (function_item name: (identifier) @function.name)
        (call_expression function: (identifier) @call.target)
    """,
}

# Parent node type -> field holding the receiver in obj.method().
RECEIVER_FIELDS = {
    "attribute": "object",          # Python
    "member_expression": "object",  # JavaScript
    "method_invocation": "object",  # Java
}
MAX_RECEIVER_LENGTH = 80
UNKNOWN_RECEIVER = "<expr>"


def _node_text(node) -> str:
    text = node.text
    if isinstance(text, bytes):
        return text.decode("utf8")
    return text or ""


def _call_receiver(target_node) -> str:
    """Return the receiver of a method call (``self`` in ``self.run()``), or ''."""
    parent = target_node.parent
    field_name = RECEIVER_FIELDS.get(parent.type) if parent is not None else None
    if field_name is None:
        return ""

    receiver = parent.child_by_field_name(field_name)
    if receiver is None:
        return ""

    text = " ".join(_node_text(receiver).split())
    if not text or len(text) > MAX_RECEIVER_LENGTH:
        return UNKNOWN_RECEIVER
    return text


def _innermost_function(entities: list[CodeEntity], line: int) -> CodeEntity | None:
    """Return the smallest function whose line range contains ``line``."""
    containing = [
        entity
        for entity in entities
        if entity.type == "function"
        and entity.start_line <= line <= entity.end_line
    ]
    return min(
        containing,
        key=lambda entity: entity.end_line - entity.start_line,
        default=None,
    )


def parse_source_code(file_path: str, content: str, language: str) -> ParsedFile:
    if language not in LANGUAGE_REGISTRY or language not in QUERY_REGISTRY:
        return ParsedFile(file_path=file_path, language=language, entities=[], relationships=[])

    ts_lang = LANGUAGE_REGISTRY[language]
    parser = Parser(ts_lang)
    query = Query(ts_lang, QUERY_REGISTRY[language])

    tree = parser.parse(bytes(content, "utf8"))
    captures = QueryCursor(query).captures(tree.root_node)

    entities: list[CodeEntity] = []
    relationships: list[CodeRelationship] = []
    entity_ids: set[str] = set()



    # 1. Collect entities (classes and functions).
    # Sort captures by source position so duplicate names receive stable IDs.
    entity_captures = []
    for capture_name, nodes in captures.items():
        if capture_name not in ("class.name", "function.name"):
            continue

        entity_type = (
            "class" if capture_name == "class.name" else "function"
        )
        for node in nodes:
            parent_node = node.parent if node.parent else node
            entity_captures.append(
                (parent_node.start_point[0], node.start_point[1],
                 entity_type, node, parent_node)
            )

    entity_captures.sort(key=lambda item: (item[0], item[1]))

    for _, _, entity_type, node, parent_node in entity_captures:
        name = _node_text(node).strip()
        start_line = parent_node.start_point[0]

        entity_id = f"{file_path}:{entity_type}:{name}"
        if entity_id in entity_ids:
            entity_id = f"{entity_id}@{start_line}"
        entity_ids.add(entity_id)

        entities.append(CodeEntity(
            id=entity_id,
            type=entity_type,
            name=name,
            file_path=file_path,
            start_line=start_line,
            end_line=parent_node.end_point[0],
        ))

        # CONTAINS edge: File -> Entity
        relationships.append(CodeRelationship(
            source_id=file_path,
            target_id=entity_id,
            type="CONTAINS",
        ))



    # 2. Extract IMPORTS
    import_nodes = captures.get("import.name", []) + captures.get("import.module", [])
    for node in import_nodes:
        relationships.append(CodeRelationship(
            source_id=file_path,
            target_id=f"module:{_node_text(node).strip()}",
            type="IMPORTS",
        ))

    # 3. Extract CALLS, attributed to the innermost enclosing function.
    #    The receiver is kept (symbol:self.search vs symbol:pipeline.search)
    #    so GraphBuilder can resolve the call to the right definition.
    for call_node in captures.get("call.target", []):
        callee_name = _node_text(call_node)
        receiver = _call_receiver(call_node)
        target = f"{receiver}.{callee_name}" if receiver else callee_name

        caller = _innermost_function(entities, call_node.start_point[0])
        relationships.append(CodeRelationship(
            source_id=caller.id if caller else file_path,
            target_id=f"symbol:{target}",
            type="CALLS",
        ))

    return ParsedFile(
        file_path=file_path,
        language=language,
        entities=entities,
        relationships=relationships,
        source_code=content,
    )
