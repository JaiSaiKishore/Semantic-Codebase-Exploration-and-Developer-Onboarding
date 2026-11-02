import os
import stat
import shutil
import sqlite3
import urllib.request
from db_config import client, embed_fn, code_col, history_col

def remove_readonly(func, path, excinfo):
    """Clears the readonly bit on Windows .git packfiles and reattempts removal."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

def wipe_collection(col, name: str):
    """Deletes all records from a ChromaDB collection safely."""
    try:
        data = col.get()
        ids = data.get("ids", [])
        if ids:
            batch_size = 500
            for i in range(0, len(ids), batch_size):
                col.delete(ids=ids[i:i+batch_size])
        print(f"  [OK] Wiped collection '{name}' ({len(ids)} chunks removed, now {col.count()}).")
    except Exception as e:
        try:
            client.delete_collection(name)
            new_col = client.get_or_create_collection(name=name, embedding_function=embed_fn)
            print(f"  [OK] Recreated collection '{name}' (count: {new_col.count()}).")
        except Exception as inner_e:
            print(f"  [-] Notice resetting '{name}': {e} / {inner_e}")

def reset_all_data():
    print("==================================================")
    print("     FULL SYSTEM RESET: SEMANTIC CODEBASE RAG     ")
    print("==================================================")

    # 1. If backend server is running on port 8000, trigger live in-memory & DB reset
    try:
        req = urllib.request.Request("http://localhost:8000/api/reset", method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                print("  [OK] Notified running FastAPI server to clear active repository & status cache.")
    except Exception:
        pass

    # 2. Wipe ChromaDB vector collections
    wipe_collection(code_col, "codebase_chunks")
    wipe_collection(history_col, "github_history")

    # 3. Delete all cloned repositories inside ./repos and legacy ./chroma_dbs
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repos_dir = os.path.join(base_dir, "repos")
    if os.path.exists(repos_dir):
        shutil.rmtree(repos_dir, onerror=remove_readonly)
    os.makedirs(repos_dir, exist_ok=True)
    print("  [OK] Deleted all cloned repositories in ./repos/")

    legacy_dbs = os.path.join(base_dir, "chroma_dbs")
    if os.path.exists(legacy_dbs):
        shutil.rmtree(legacy_dbs, onerror=remove_readonly)
        print("  [OK] Removed legacy ./chroma_dbs/ directory.")

    # 4. Clear SQLite conversation checkpoints
    ckpt_path = os.path.join(base_dir, "checkpoints.db")
    removed_files = False
    for suffix in ["", "-wal", "-shm"]:
        fpath = ckpt_path + suffix
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
                removed_files = True
            except PermissionError:
                # Server has checkpoints.db open; truncate tables via SQLite directly
                if suffix == "":
                    try:
                        conn = sqlite3.connect(fpath)
                        cur = conn.cursor()
                        cur.execute("SELECT name FROM sqlite_schema WHERE type='table'")
                        for (tname,) in cur.fetchall():
                            cur.execute(f"DELETE FROM {tname}")
                        conn.commit()
                        conn.close()
                        print("  [OK] Cleared all chat history tables in active checkpoints.db")
                    except Exception as sqle:
                        print(f"  [-] Notice clearing checkpoints.db tables: {sqle}")
    if removed_files:
        print("  [OK] Deleted checkpoints.db conversation history files.")

    print("\n==================================================")
    print("DONE! All repository vectors, clones, and chat")
    print("history have been completely wiped.")
    print("==================================================")

if __name__ == "__main__":
    reset_all_data()