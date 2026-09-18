from enum import Enum


class NodeType(str, Enum):
    REPOSITORY = "Repository"
    FILE = "File"
    CLASS = "Class"
    FUNCTION = "Function"
    METHOD = "Method"
    VARIABLE = "Variable"
    MODULE = "Module"
    SYMBOL = "Symbol"


class RelationType(str, Enum):
    CONTAINS = "CONTAINS"
    IMPORTS = "IMPORTS"
    CALLS = "CALLS"
    INHERITS = "INHERITS"
    USES = "USES"
    DEFINES = "DEFINES"
