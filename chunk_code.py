import os
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# 1. Initialize the Tree-sitter Python parser
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

def extract_code_blocks(filepath):
    """Parses a Python file and extracts functions/classes as discrete chunks."""
    with open(filepath, "r", encoding="utf-8") as f:
        code_bytes = f.read().encode("utf-8")
    
    tree = parser.parse(code_bytes)
    root_node = tree.root_node
    
    chunks = []
    
    # 2. Traverse the Abstract Syntax Tree (AST)
    def traverse(node):
        if node.type in ['function_definition', 'class_definition']:
            chunk_bytes = code_bytes[node.start_byte:node.end_byte]
            
            name_node = node.child_by_field_name('name')
            name = name_node.text.decode('utf-8') if name_node else "unknown"
            
            chunks.append({
                "type": node.type,
                "name": name,
                "code": chunk_bytes.decode('utf-8'),
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1
            })
            
        for child in node.children:
            traverse(child)

    traverse(root_node)
    return chunks

# 3. Test on a file from our cloned repository
def get_test_file(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                return os.path.join(root, file)
    return None

print("Looking for a Python file in ./repo_data to parse...")
test_file = get_test_file("./repo_data")

if test_file:
    print(f"Found: {test_file}")
    blocks = extract_code_blocks(test_file)
    
    print(f"\nSuccessfully extracted {len(blocks)} logical blocks.")
    if blocks:
        print("\n--- Example Block Extracted ---")
        print(f"Type: {blocks[0]['type']}")
        print(f"Name: {blocks[0]['name']}")
        print(f"Lines: {blocks[0]['start_line']} to {blocks[0]['end_line']}")
        print("Code Snippet:")
        
        snippet_lines = blocks[0]['code'].split('\n')[:5]
        print('\n'.join(snippet_lines))
        if len(snippet_lines) >= 5:
            print("    ...")
else:
    print("Could not find a Python file to test.")