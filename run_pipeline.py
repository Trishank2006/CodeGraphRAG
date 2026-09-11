import os
import sys
import shutil
import stat

# Ensure Python finds our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.main import ingest_repository
from parser.core_parser import parse_source_code

def remove_readonly(func, path, exc_info):
    """Clear the readonly bit and reattempt the removal."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def main():
    # Let's test the pipeline by having it parse YOUR OWN repository!
    repo_url = "https://github.com/Akki-24/meeting_summarizer.git"
    destination = "./temp_test_repo"
    
    # Clean up the temp folder if it already exists from a previous run
    if os.path.exists(destination):
        shutil.rmtree(destination, onexc=remove_readonly)

    print(f"📥 Step 1: Person 1's Code - Cloning repository...")
    
    # 1. Run Person 1's Ingestion Pipeline
    files_data = ingest_repository(repo_url, destination)
    print(f"✅ Ingestion complete. Discovered {len(files_data)} supported files.")

    print("\n🧠 Step 2: Person 2's Code - Passing files to AST Parser...\n")
    
    total_entities = 0
    
    # 2. Run Person 2's Parser on the output
    for file_record in files_data:
        # We only set up the Tree-sitter grammar for Python so far
        
            
            # Pass Person 1's text output directly into your parser
            parsed = parse_source_code(
                file_path=file_record["file_path"],
                content=file_record["content"],
                language=file_record["language"]
            )
            
            # If your parser found classes or functions, print them out!
            if parsed.entities:
                print(f"📄 {parsed.file_path}")
                for entity in parsed.entities:
                    print(f"   -> [{entity.type}] {entity.name} (Lines {entity.start_line}-{entity.end_line})")
                # --- ADD THIS TO PRINT YOUR NEW RELATIONSHIPS ---
                for rel in parsed.relationships:
                    print(f"   🔗 Edge: ({rel.source_id}) -[:{rel.type}]-> ({rel.target_id})") 
                total_entities += len(parsed.entities)
                print("-" * 50)

    print(f"\n🎉 Integration complete! Extracted a total of {total_entities} Python entities.")

if __name__ == "__main__":
    main()