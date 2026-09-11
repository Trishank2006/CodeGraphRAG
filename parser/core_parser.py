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
    """
}

def parse_source_code(file_path: str, content: str, language: str) -> ParsedFile:
    if language not in LANGUAGE_REGISTRY or language not in QUERY_REGISTRY:
        return ParsedFile(file_path=file_path, language=language, entities=[], relationships=[])

    ts_lang = LANGUAGE_REGISTRY[language]
    query_string = QUERY_REGISTRY[language]
    
    parser = Parser(ts_lang)
    query = Query(ts_lang, query_string)

    tree = parser.parse(bytes(content, "utf8"))
    query_cursor = QueryCursor(query)
    captures = query_cursor.captures(tree.root_node)

    entities = []
    relationships = []

    # 1. Collect entities (Classes and Functions)
    for capture_name, nodes in captures.items():
        if capture_name in ("class.name", "function.name"):
            for node in nodes:
                entity_type = "class" if "class" in capture_name else "function"
                name_text = node.text
                name = name_text.decode('utf8') if isinstance(name_text, bytes) else name_text
                parent_node = node.parent if node.parent else node
                
                entity_id = f"{file_path}:{entity_type}:{name}"
                entities.append(CodeEntity(
                    id=entity_id,
                    type=entity_type,
                    name=name,
                    file_path=file_path,
                    start_line=parent_node.start_point[0],
                    end_line=parent_node.end_point[0]
                ))
                # Generate CONTAINS edge: File -> Entity
                relationships.append(CodeRelationship(
                    source_id=file_path,
                    target_id=entity_id,
                    type="CONTAINS"
                ))

    # 2. Extract IMPORTS
    import_captures = captures.get("import.name", []) + captures.get("import.module", [])
    for node in import_captures:
        raw_name = node.text
        import_name = raw_name.decode('utf8') if isinstance(raw_name, bytes) else raw_name
        relationships.append(CodeRelationship(
            source_id=file_path,
            target_id=f"module:{import_name.strip()}",
            type="IMPORTS"
        ))

    # 3. Extract CALLS (Attributing invocation to the enclosing function)
    call_captures = captures.get("call.target", [])
    for call_node in call_captures:
        raw_name = call_node.text
        callee_name = raw_name.decode('utf8') if isinstance(raw_name, bytes) else raw_name
        call_line = call_node.start_point[0]

        # Find enclosing function
        caller_entity = None
        for entity in entities:
            if entity.type == "function" and entity.start_line <= call_line <= entity.end_line:
                caller_entity = entity
                break

        caller_id = caller_entity.id if caller_entity else file_path
        relationships.append(CodeRelationship(
            source_id=caller_id,
            target_id=f"symbol:{callee_name}",
            type="CALLS"
        ))

    return ParsedFile(
        file_path=file_path, 
        language=language, 
        entities=entities, 
        relationships=relationships
    )