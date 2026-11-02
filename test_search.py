import sys
from db_config import client, embed_fn, code_col, history_col

def search_code(query: str, repo: str = None, n_results: int = 3):
    print(f"\n--- Searching Codebase for: '{query}' [Repo: {repo or 'All'}] ---")
    where_filter = {"repo": repo} if repo else None
    try:
        results = code_col.query(query_texts=[query], n_results=n_results, where=where_filter)
    except Exception as e:
        print(f"Filtered query notice: {e}, falling back to general query")
        results = code_col.query(query_texts=[query], n_results=n_results)

    if not results["ids"] or not results["ids"][0]:
        print("No matching code chunks found.")
        return

    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        print(f"\nMatch {i+1} (Distance: {dist:.4f})")
        print(f"Repo: {meta.get('repo', 'unknown')}")
        print(f"File: {meta.get('file', 'unknown')}")
        print(f"Symbol: {meta.get('name', 'unknown')} (Lines {meta.get('start_line')}-{meta.get('end_line')})")
        snippet = results["documents"][0][i]
        print(f"Preview: {snippet[:150].strip()}...")

def search_history(query: str, repo: str = None, n_results: int = 3):
    print(f"\n--- Searching GitHub History for: '{query}' [Repo: {repo or 'All'}] ---")
    where_filter = {"repo": repo} if repo else None
    try:
        results = history_col.query(query_texts=[query], n_results=n_results, where=where_filter)
    except Exception as e:
        results = history_col.query(query_texts=[query], n_results=n_results)

    if not results["ids"] or not results["ids"][0]:
        print("No matching issues or PRs found.")
        return

    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        doc = results["documents"][0][i]
        print(f"\nMatch {i+1} (Distance: {dist:.4f})")
        print(f"Repo: {meta.get('repo', 'unknown')}")
        print(f"Type: {'PR' if meta.get('is_pr') else 'Issue'} #{meta.get('number')} - {meta.get('title')}")
        print(f"Preview: {doc.split('Body: ')[-1][:150].strip()}...")

if __name__ == "__main__":
    test_query = sys.argv[1] if len(sys.argv) > 1 else "How does palindrome check work?"
    test_repo = sys.argv[2] if len(sys.argv) > 2 else "superswan/beginner"
    search_code(test_query, repo=test_repo)
    search_history(test_query, repo=test_repo)
