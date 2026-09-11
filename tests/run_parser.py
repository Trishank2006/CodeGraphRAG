import sys
import os

# This forces Python to look in your CodeGraphRAG folder FIRST
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.core_parser import parse_source_code

# Read your dummy file
file_path = os.path.join("tests", "fixtures", "sample.py")
with open(file_path, "r", encoding="utf-8") as f:
    code_content = f.read()

# Run it through your parser
result = parse_source_code(file_path, code_content, "python")

print(f"Output ready for File: {result.file_path}")
print(f"Found {len(result.entities)} entities:")

for entity in result.entities:
    print(f" -> [{entity.type}] {entity.name} (Lines {entity.start_line}-{entity.end_line})")