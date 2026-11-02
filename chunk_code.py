import os
import sys
import re
from typing import List, Dict, Any
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript

def get_parser_for_language(lang: Language) -> Parser:
    try:
        return Parser(lang)
    except TypeError:
        p = Parser()
        p.language = lang
        return p

PY_LANG = Language(tspython.language())
JS_LANG = Language(tsjavascript.language())
TS_LANG = Language(tstypescript.language_typescript())

py_parser = get_parser_for_language(PY_LANG)
js_parser = get_parser_for_language(JS_LANG)
ts_parser = get_parser_for_language(TS_LANG)

def chunk_markdown(content: str, filepath: str, repo_name: str) -> List[Dict[str, Any]]:
    lines = content.splitlines()
    sections = []
    current_title = "Overview"
    current_lines = []
    start_line = 1

    for idx, line in enumerate(lines, start=1):
        if re.match(r"^#{1,4}\s+", line):
            if current_lines and "\n".join(current_lines).strip():
                sec_text = "\n".join(current_lines).strip()
                sections.append({
                    "type": "markdown",
                    "name": current_title,
                    "code": f"# Repo: {repo_name} | File: {filepath} | Section: {current_title}\n{sec_text[:3000]}",
                    "start_line": start_line,
                    "end_line": idx - 1,
                    "file": filepath,
                    "repo": repo_name
                })
            current_title = line.lstrip("#").strip()
            current_lines = [line]
            start_line = idx
        else:
            current_lines.append(line)

    if current_lines and "\n".join(current_lines).strip():
        sec_text = "\n".join(current_lines).strip()
        sections.append({
            "type": "markdown",
            "name": current_title,
            "code": f"# Repo: {repo_name} | File: {filepath} | Section: {current_title}\n{sec_text[:3000]}",
            "start_line": start_line,
            "end_line": len(lines),
            "file": filepath,
            "repo": repo_name
        })

    return sections

def extract_python_blocks(content: str, code_bytes: bytes, filepath: str, repo_name: str) -> List[Dict[str, Any]]:
    tree = py_parser.parse(code_bytes)
    chunks = []

    def traverse(node, current_class=None):
        node_type = node.type
        target_node = node
        if node.parent and node.parent.type == 'decorated_definition':
            target_node = node.parent

        if node_type == 'class_definition':
            name_node = node.child_by_field_name('name')
            class_name = name_node.text.decode('utf-8', errors="ignore") if name_node else "unknown"
            chunk_bytes = code_bytes[target_node.start_byte:target_node.end_byte]
            code_text = chunk_bytes.decode('utf-8', errors="ignore")
            chunks.append({
                "type": "class_definition",
                "name": class_name,
                "code": f"# Repo: {repo_name} | File: {filepath} | Class: {class_name}\n" + code_text[:3500],
                "start_line": target_node.start_point[0] + 1,
                "end_line": target_node.end_point[0] + 1,
                "file": filepath,
                "repo": repo_name
            })
            for child in node.children:
                traverse(child, current_class=class_name)
        elif node_type in ('function_definition', 'async_function_definition'):
            name_node = node.child_by_field_name('name')
            fn_name = name_node.text.decode('utf-8', errors="ignore") if name_node else "unknown"
            chunk_bytes = code_bytes[target_node.start_byte:target_node.end_byte]
            code_text = chunk_bytes.decode('utf-8', errors="ignore")
            qual_name = f"{current_class}.{fn_name}" if current_class else fn_name
            chunks.append({
                "type": node_type,
                "name": qual_name,
                "code": f"# Repo: {repo_name} | File: {filepath} | {qual_name}\n" + code_text[:2500],
                "start_line": target_node.start_point[0] + 1,
                "end_line": target_node.end_point[0] + 1,
                "file": filepath,
                "repo": repo_name
            })
            for child in node.children:
                traverse(child, current_class=current_class)
        else:
            for child in node.children:
                traverse(child, current_class=current_class)

    traverse(tree.root_node)
    return chunks

def extract_js_ts_blocks(content: str, code_bytes: bytes, filepath: str, repo_name: str, is_ts: bool = False) -> List[Dict[str, Any]]:
    parser = ts_parser if is_ts else js_parser
    tree = parser.parse(code_bytes)
    chunks = []

    def traverse(node, current_class=None):
        node_type = node.type
        target_node = node
        if node.parent and node.parent.type in ('export_statement', 'export_default_declaration'):
            target_node = node.parent

        if node_type == 'class_declaration':
            name_node = node.child_by_field_name('name')
            class_name = name_node.text.decode('utf-8', errors="ignore") if name_node else "AnonymousClass"
            chunk_bytes = code_bytes[target_node.start_byte:target_node.end_byte]
            code_text = chunk_bytes.decode('utf-8', errors="ignore")
            chunks.append({
                "type": "class_declaration",
                "name": class_name,
                "code": f"// Repo: {repo_name} | File: {filepath} | Class: {class_name}\n" + code_text[:3500],
                "start_line": target_node.start_point[0] + 1,
                "end_line": target_node.end_point[0] + 1,
                "file": filepath,
                "repo": repo_name
            })
            for child in node.children:
                traverse(child, current_class=class_name)
        elif node_type in ('method_definition', 'public_field_definition'):
            name_node = node.child_by_field_name('name')
            fn_name = name_node.text.decode('utf-8', errors="ignore") if name_node else "unknown"
            chunk_bytes = code_bytes[node.start_byte:node.end_byte]
            code_text = chunk_bytes.decode('utf-8', errors="ignore")
            qual_name = f"{current_class}.{fn_name}" if current_class else fn_name
            chunks.append({
                "type": "method_definition",
                "name": qual_name,
                "code": f"// Repo: {repo_name} | File: {filepath} | {qual_name}\n" + code_text[:2500],
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "file": filepath,
                "repo": repo_name
            })
            for child in node.children:
                traverse(child, current_class=current_class)
        elif node_type in ('function_declaration', 'generator_function_declaration'):
            name_node = node.child_by_field_name('name')
            fn_name = name_node.text.decode('utf-8', errors="ignore") if name_node else "unknown"
            chunk_bytes = code_bytes[target_node.start_byte:target_node.end_byte]
            code_text = chunk_bytes.decode('utf-8', errors="ignore")
            qual_name = f"{current_class}.{fn_name}" if current_class else fn_name
            chunks.append({
                "type": node_type,
                "name": qual_name,
                "code": f"// Repo: {repo_name} | File: {filepath} | {qual_name}\n" + code_text[:2500],
                "start_line": target_node.start_point[0] + 1,
                "end_line": target_node.end_point[0] + 1,
                "file": filepath,
                "repo": repo_name
            })
            for child in node.children:
                traverse(child, current_class=current_class)
        elif node_type in ('interface_declaration', 'type_alias_declaration'):
            name_node = node.child_by_field_name('name')
            decl_name = name_node.text.decode('utf-8', errors="ignore") if name_node else "type"
            chunk_bytes = code_bytes[target_node.start_byte:target_node.end_byte]
            code_text = chunk_bytes.decode('utf-8', errors="ignore")
            chunks.append({
                "type": node_type,
                "name": decl_name,
                "code": f"// Repo: {repo_name} | File: {filepath} | {decl_name}\n" + code_text[:2500],
                "start_line": target_node.start_point[0] + 1,
                "end_line": target_node.end_point[0] + 1,
                "file": filepath,
                "repo": repo_name
            })
        else:
            for child in node.children:
                traverse(child, current_class=current_class)

    traverse(tree.root_node)
    return chunks

def extract_code_blocks(filepath: str, repo_name: str = "codebase") -> List[Dict[str, Any]]:
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

    if not content.strip():
        return []

    ext = os.path.splitext(filepath)[1].lower()
    code_bytes = content.encode("utf-8")

    if ext == ".md":
        return chunk_markdown(content, filepath, repo_name)
    elif ext == ".py":
        blocks = extract_python_blocks(content, code_bytes, filepath, repo_name)
    elif ext in (".js", ".jsx", ".mjs", ".cjs"):
        blocks = extract_js_ts_blocks(content, code_bytes, filepath, repo_name, is_ts=False)
    elif ext in (".ts", ".tsx"):
        blocks = extract_js_ts_blocks(content, code_bytes, filepath, repo_name, is_ts=True)
    else:
        blocks = []

    # Fallback for procedural scripts, HTML/CSS/C/C++/Java, or files without top-level classes/functions
    if not blocks and content.strip():
        lines = content.splitlines()
        fname = os.path.basename(filepath)
        window = 80
        for start in range(0, len(lines), window):
            segment = "\n".join(lines[start:start+window])
            if segment.strip():
                part_num = (start // window) + 1
                block_name = fname if len(lines) <= window else f"{fname} (part {part_num})"
                blocks.append({
                    "type": "file_content",
                    "name": block_name,
                    "code": f"# Repo: {repo_name} | File: {filepath} | {block_name}\n" + segment[:2800],
                    "start_line": start + 1,
                    "end_line": min(start + window, len(lines)),
                    "file": filepath,
                    "repo": repo_name
                })

    return blocks

extract_ast_blocks = extract_code_blocks