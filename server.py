import uvicorn
from backend.routes import app

# Expose both app and api for uvicorn compatibility
api = app

if __name__ == '__main__':
    uvicorn.run('server:app', host='0.0.0.0', port=8000, reload=True)
