from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import logging

from graph.models import GraphEdge, GraphNode
from graph.schema import NodeType, RelationType
from parser.models import CodeEntity, ParsedFile

logger = logging.getLogger(__name__)

SELF_RECEIVERS = frozenset({"self", "cls", "this"})

# Languages where a bare call inside a class can target one of its methods.
IMPLICIT_THIS_LANGUAGES = frozenset({"java", "cpp"})
SAME_PACKAGE_LANGUAGES = frozenset({"go", "java"})


class GraphBuilder:
    """Build a repository-scoped graph from AST-parsed files."""

    def __init__(self, repository_name: str = "default_repo") -> None:
        if not repository_name or not repository_name.strip():
            raise ValueError("Repository name must not be empty")

        self.repository_name = repository_name

    def _scoped_id(self, node_id: str) -> str:
        """Make node IDs unique across repositories."""
        return f"repo:{self.repository_name}:{node_id}"

    def build_from_parsed_files(
        self, parsed_files: list[ParsedFile]
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        seen_node_ids: set[str] = set()
        seen_edges: set[tuple[str, str, str]] = set()
        definitions = _DefinitionIndex.build(parsed_files)

        def add_node(node: GraphNode) -> None:
            if node.id not in seen_node_ids:
                nodes.append(node)
                seen_node_ids.add(node.id)

        def add_reference_node(reference_id: str, language: str) -> None:
            properties: dict[str, object] = {
                "repository": self.repository_name,
                "reference": True,
            }

            if reference_id.startswith("module:"):
                label = NodeType.MODULE
                name = reference_id.removeprefix("module:")
            else:
                label = NodeType.SYMBOL
                receiver, _, name = (
                    reference_id.removeprefix("symbol:").rpartition(".")
                )
                if receiver:
                    properties["receiver"] = receiver

            # Placeholders aren't defined in any repository file.
            add_node(GraphNode(
                id=self._scoped_id(reference_id),
                label=label,
                name=name,
                file_path="",
                language=language,
                properties=properties,
            ))

        # Pass 1: file and entity nodes.
        for pf in parsed_files:
            add_node(GraphNode(
                id=self._scoped_id(f"file:{pf.file_path}"),
                label=NodeType.FILE,
                name=pf.file_path.split("/")[-1],
                file_path=pf.file_path,
                language=pf.language,
                properties={"repository": self.repository_name},
            ))

            for entity in pf.entities:
                add_node(GraphNode(
                    id=self._scoped_id(entity.id),
                    label=(
                        NodeType.CLASS
                        if entity.type == "class"
                        else NodeType.FUNCTION
                    ),
                    name=entity.name,
                    file_path=entity.file_path,
                    language=pf.language,
                    start_line=entity.start_line,
                    end_line=entity.end_line,
                    properties={"repository": self.repository_name},
                ))

        # Pass 2: relationships, with imports normalized and calls resolved.
        for pf in parsed_files:
            for rel in pf.relationships:
                try:
                    relation_type = RelationType(rel.type)
                except ValueError:
                    logger.warning(
                        "Skipping unknown relationship type %r in %s",
                        rel.type,
                        pf.file_path,
                    )
                    continue

                source_id = (
                    f"file:{rel.source_id}"
                    if rel.source_id == pf.file_path
                    else rel.source_id
                )
                target_id = rel.target_id

                if rel.type == "IMPORTS" and target_id.startswith("module:"):
                    absolute = _absolute_module(
                        target_id.removeprefix("module:"), pf.file_path
                    )
                    if absolute:
                        target_id = f"module:{absolute}"

                if rel.type == "CALLS" and target_id.startswith("symbol:"):
                    resolved = definitions.resolve_call(
                        pf,
                        caller_id=rel.source_id,
                        target=target_id.removeprefix("symbol:"),
                    )
                    if resolved is not None:
                        target_id = resolved

                scoped_source = self._scoped_id(source_id)
                if scoped_source not in seen_node_ids:
                    logger.debug("Skipping edge with a missing source: %s", source_id)
                    continue

                if target_id.startswith(("module:", "symbol:")):
                    add_reference_node(target_id, pf.language)

                scoped_target = self._scoped_id(target_id)
                if scoped_target not in seen_node_ids:
                    logger.debug("Skipping edge with a missing target: %s", target_id)
                    continue

                edge_key = (scoped_source, scoped_target, relation_type.value)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                edges.append(GraphEdge(
                    source_id=scoped_source,
                    target_id=scoped_target,
                    type=relation_type,
                ))

        return nodes, edges


@dataclass
class _DefinitionIndex:
    """Lookup tables used to resolve call targets to real definitions."""

    entities_by_id: dict[str, CodeEntity] = field(default_factory=dict)
    by_name: dict[str, list[CodeEntity]] = field(
        default_factory=lambda: defaultdict(list)
    )
    by_file_and_name: dict[tuple[str, str], list[CodeEntity]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # Function/method entity ID -> innermost enclosing class.
    owner_class: dict[str, CodeEntity] = field(default_factory=dict)
    # File path -> repository files it imports.
    imported_files: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, parsed_files: list[ParsedFile]) -> _DefinitionIndex:
        index = cls()
        module_lookup = _module_lookup(pf.file_path for pf in parsed_files)

        for pf in parsed_files:
            classes = [e for e in pf.entities if e.type == "class"]

            for entity in pf.entities:
                index.entities_by_id[entity.id] = entity
                index.by_name[entity.name].append(entity)
                index.by_file_and_name[(pf.file_path, entity.name)].append(entity)

                if entity.type != "class":
                    owner = _innermost_container(classes, entity)
                    if owner is not None:
                        index.owner_class[entity.id] = owner

            imported: set[str] = set()
            for rel in pf.relationships:
                if rel.type == "IMPORTS" and rel.target_id.startswith("module:"):
                    module = _absolute_module(
                        rel.target_id.removeprefix("module:"), pf.file_path
                    )
                    imported |= module_lookup.get(module, set())
            index.imported_files[pf.file_path] = imported

        return index

    def resolve_call(
        self,
        parsed_file: ParsedFile,
        caller_id: str,
        target: str,
    ) -> str | None:
        """Return the entity ID a call refers to, or None if it can't be
        determined without guessing."""
        receiver, _, name = target.rpartition(".")
        caller = self.entities_by_id.get(caller_id)
        owner = self.owner_class.get(caller_id)

        if receiver in SELF_RECEIVERS:
            return _unique(self._methods_of(owner, name)) if owner else None

        if receiver:
            # Class-level calls such as ApplicationSettings.from_environment().
            receiver_class = _closest([
                [e for e in tier if e.type == "class"]
                for tier in self._visible_tiers(parsed_file, receiver)
            ])
            if receiver_class is None:
                return None
            return _unique(self._methods_of(receiver_class, name))

        tiers: list[list[CodeEntity]] = []
        if caller is not None:
            # Helpers defined inside the caller.
            tiers.append([
                e
                for e in self.by_file_and_name.get((parsed_file.file_path, name), [])
                if e is not caller and _contains(caller, e)
            ])
        if owner is not None and parsed_file.language in IMPLICIT_THIS_LANGUAGES:
            tiers.append(self._methods_of(owner, name))
        tiers.extend(self._visible_tiers(parsed_file, name))

        match = _closest(tiers)
        return match.id if match else None

    def _visible_tiers(
        self, parsed_file: ParsedFile, name: str
    ) -> list[list[CodeEntity]]:
        """Top-level definitions of ``name`` visible from a file, closest first.

        There is deliberately no repository-wide tier: a name that isn't
        defined locally, imported, or in the same package is treated as
        external (a builtin or library function) rather than guessed.
        """
        file_path = parsed_file.file_path
        imported = self.imported_files.get(file_path, set())
        top_level = [
            e for e in self.by_name.get(name, []) if e.id not in self.owner_class
        ]

        tiers = [
            [e for e in top_level if e.file_path == file_path],
            [e for e in top_level if e.file_path in imported],
        ]
        if parsed_file.language in SAME_PACKAGE_LANGUAGES:
            tiers.append([
                e for e in top_level if _same_directory(e.file_path, file_path)
            ])
        return tiers

    def _methods_of(self, owner: CodeEntity, name: str) -> list[CodeEntity]:
        return [
            e
            for e in self.by_file_and_name.get((owner.file_path, name), [])
            if self.owner_class.get(e.id) is owner
        ]


def _contains(outer: CodeEntity, inner: CodeEntity) -> bool:
    return outer.start_line <= inner.start_line and inner.end_line <= outer.end_line


def _innermost_container(
    classes: list[CodeEntity], entity: CodeEntity
) -> CodeEntity | None:
    containers = [c for c in classes if c is not entity and _contains(c, entity)]
    return min(
        containers,
        key=lambda c: c.end_line - c.start_line,
        default=None,
    )


def _unique(candidates: list[CodeEntity]) -> str | None:
    return candidates[0].id if len(candidates) == 1 else None

def _closest(tiers: list[list[CodeEntity]]) -> CodeEntity | None:
    """Return the only candidate in the first non-empty tier, or None if that
    tier is ambiguous or every tier is empty."""
    for candidates in tiers:
        if candidates:
            return candidates[0] if len(candidates) == 1 else None
    return None


def _same_directory(a: str, b: str) -> bool:
    return a.rpartition("/")[0] == b.rpartition("/")[0]


def _module_lookup(file_paths) -> dict[str, set[str]]:
    """Map dotted module names to Python files in the repository.

    A module's full path from the repository root (a.b.c for a/b/c.py) always
    wins. Shorter suffixes (b.c, c) count only when exactly one file has them,
    which supports src/ layouts without guessing between same-named modules.
    """
    exact: dict[str, set[str]] = defaultdict(set)
    suffixes: dict[str, set[str]] = defaultdict(set)

    for path in file_paths:
        if not path.endswith(".py"):
            continue

        stem = path.removesuffix(".py")
        if stem == "__init__":
            continue
        stem = stem.removesuffix("/__init__")

        parts = stem.split("/")
        exact[".".join(parts)].add(path)
        for start in range(1, len(parts)):
            suffixes[".".join(parts[start:])].add(path)

    lookup = {
        module: paths for module, paths in suffixes.items() if len(paths) == 1
    }
    lookup.update(exact)
    return lookup


def _absolute_module(module: str, file_path: str) -> str:
    """Turn a relative import (.models, ..graph.schema) into a dotted path
    from the repository root. Returns '' if it climbs above the root."""
    if not module.startswith("."):
        return module

    level = len(module) - len(module.lstrip("."))
    package = file_path.split("/")[:-1]
    if level - 1 > len(package):
        return ""

    package = package[: len(package) - (level - 1)]
    remainder = module.lstrip(".")
    return ".".join(package + (remainder.split(".") if remainder else []))