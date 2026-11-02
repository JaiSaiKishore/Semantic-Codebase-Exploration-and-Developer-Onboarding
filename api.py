"""
Backward-compatible API entrypoint that re-exports the FastAPI app from server.py.
Allows running either `python server.py` or `uvicorn api:app --reload`.
"""
import uvicorn
from server import app

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)