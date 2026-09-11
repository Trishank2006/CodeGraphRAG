import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
import tree_sitter_java as tsjava
import tree_sitter_cpp as tscpp
import tree_sitter_c as tsc
import tree_sitter_go as tsgo
import tree_sitter_rust as tsrust

from tree_sitter import Language, Parser, Query, QueryCursor
from parser.models import ParsedFile, CodeEntity,CodeRelationship

# 1. The Language Registry: Map Person 1's string to the Tree-sitter Language
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

# 2. The Query Registry: Every language has a slightly different syntax tree!
QUERY_REGISTRY = {
    "python": """
        (class_definition name: (identifier) @class.name)
        (function_definition name: (identifier) @function.name)
    """,
    "javascript": """
        (class_declaration name: (identifier) @class.name)
        (function_declaration name: (identifier) @function.name)
    """,
    "typescript": """
        (class_declaration name: (type_identifier) @class.name)
        (function_declaration name: (identifier) @function.name)
    """,
    "java": """
        (class_declaration name: (identifier) @class.name)
        (method_declaration name: (identifier) @function.name)
    """,
    "cpp": """
        (class_specifier name: (type_identifier) @class.name)
        (function_definition declarator: (function_declarator declarator: (identifier) @function.name))
    """,
    "c": """
        (function_definition declarator: (function_declarator declarator: (identifier) @function.name))
    """,
    "go": """
        (type_spec name: (type_identifier) @class.name type: (struct_type))
        (function_declaration name: (identifier) @function.name)
    """,
    "rust": """
        (struct_item name: (type_identifier) @class.name)
        (function_item name: (identifier) @function.name)
    """
}

def parse_source_code(file_path: str, content: str, language: str) -> ParsedFile:
    # If it's a language we don't support yet, just return an empty ParsedFile
    if language not in LANGUAGE_REGISTRY or language not in QUERY_REGISTRY:
        return ParsedFile(file_path=file_path, language=language, entities=[], relationships=[])

    # Dynamically select the right grammar and query
    ts_lang = LANGUAGE_REGISTRY[language]
    query_string = QUERY_REGISTRY[language]
    
    parser = Parser(ts_lang)
    query = Query(ts_lang, query_string)

    tree = parser.parse(bytes(content, "utf8"))
    query_cursor = QueryCursor(query)
    captures = query_cursor.captures(tree.root_node)
    
    entities = []
    
    for capture_name, nodes in captures.items():
        for node in nodes:
            entity_type = "class" if "class" in capture_name else "function"
            name_text = node.text
            name = name_text.decode('utf8') if isinstance(name_text, bytes) else name_text
            
            # Get the parent block for accurate line numbers
            parent_node = node.parent if node.parent else node
            
            entities.append(CodeEntity(
                id=f"{file_path}:{entity_type}:{name}",
                type=entity_type,
                name=name,
                file_path=file_path,
                start_line=parent_node.start_point[0],
                end_line=parent_node.end_point[0]
            ))
        # --- BUILD THE CONTAINS RELATIONSHIPS HERE ---
    relationships = []
    for entity in entities:
        relationships.append(CodeRelationship(
            source_id=file_path,
            target_id=entity.id,
            type="CONTAINS"
        ))
        
    return ParsedFile(
        file_path=file_path, 
        language=language, 
        entities=entities, 
        relationships=relationships
    )