from graph.evidence import EntityMetadata, GraphEvidence, PathStep, GraphPath


def test_entity_metadata_creation():
    meta = EntityMetadata(
        id="fn:upload",
        name="upload_meeting",
        label="Function",
        file_path="routers/meetings.py",
        start_line=53,
        end_line=80,
    )
    assert meta.id == "fn:upload"
    assert meta.start_line == 53
    assert meta.end_line == 80


def test_graph_evidence_contract():
    evidence = GraphEvidence(
        source_entity="upload_meeting",
        relationship="CALLS",
        target_entity="process_meeting_pipeline",
        distance=1,
        source_file="routers/meetings.py",
        target_file="services/pipeline.py",
        source_start_line=53,
        source_end_line=80,
        target_start_line=52,
        target_end_line=149,
    )
    assert evidence.relationship == "CALLS"
    assert evidence.source_file == "routers/meetings.py"
    assert evidence.target_file == "services/pipeline.py"


def test_graph_path_formatting():
    s1 = EntityMetadata(id="1", name="upload_meeting")
    s2 = EntityMetadata(id="2", name="process_meeting_pipeline")
    s3 = EntityMetadata(id="3", name="transcribe")

    path = GraphPath(
        source="upload_meeting",
        target="transcribe",
        steps=[
            PathStep(source=s1, relationship="CALLS", target=s2),
            PathStep(source=s2, relationship="CALLS", target=s3),
        ],
        path_length=2,
    )
    formatted = path.format_path()
    assert formatted == "upload_meeting --[CALLS]--> process_meeting_pipeline --[CALLS]--> transcribe"