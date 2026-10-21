$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

# Activate Python environment
.\.venv\Scripts\Activate.ps1

# Start Neo4j if it is stopped
$container = docker inspect -f '{{.State.Running}}' codegraphrag-bench-neo4j
if ($container -ne "true") {
    docker start codegraphrag-bench-neo4j
}

# Configure Neo4j
$env:NEO4J_URI = "bolt://localhost:17687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_PASSWORD = "benchpassword123"
$env:CODEGRAPHRAG_NEO4J_URI = "bolt://localhost:17687"
$env:CODEGRAPHRAG_NEO4J_PASSWORD = "benchpassword123"

Write-Host "CodeGraphRAG environment is ready."
Write-Host "Run tests with: python -m pytest -q -rs"
Write-Host "Enable live tests with: `$env:CODEGRAPHRAG_RUN_LIVE_TESTS = '1'"
