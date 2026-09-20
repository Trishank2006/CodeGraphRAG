from typing import Dict, Any, List, Optional
from graph.evidence import GraphEvidence


class GraphCitationValidator:
    """
    Validates 3-way alignment for graph-derived citations:
    Source Code Content <---> AST Entity Coordinates <---> Neo4j Entity
    """

    @staticmethod
    def validate_3way_alignment(
        evidence: GraphEvidence,
        file_content: Optional[str] = None,
        ast_start_line: Optional[int] = None,
        ast_end_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Checks whether the line numbers from graph evidence match AST nodes and source text."""
        ast_aligned = True
        if ast_start_line is not None:
            ast_aligned = ast_aligned and (evidence.source_start_line == ast_start_line)
        if ast_end_line is not None:
            ast_aligned = ast_aligned and (evidence.source_end_line == ast_end_line)

        source_aligned = True
        if file_content is not None:
            lines = file_content.splitlines()
            start_idx = max(0, evidence.source_start_line - 1)
            end_idx = min(len(lines), evidence.source_end_line)
            source_slice = "\n".join(lines[start_idx:end_idx])
            source_aligned = evidence.source_entity in source_slice

        return {
            "entity": evidence.source_entity,
            "ast_aligned": ast_aligned,
            "source_aligned": source_aligned,
            "is_valid": ast_aligned and source_aligned,
        }

    @staticmethod
    def batch_validate_citations(
        evidences: List[GraphEvidence],
        file_contents: Dict[str, str],
    ) -> float:
        """Computes citation validity rate across multiple graph evidence objects."""
        if not evidences:
            return 1.0

        valid_count = 0
        for ev in evidences:
            content = file_contents.get(ev.source_file)
            res = GraphCitationValidator.validate_3way_alignment(ev, file_content=content)
            if res["is_valid"]:
                valid_count += 1

        return float(valid_count / len(evidences))