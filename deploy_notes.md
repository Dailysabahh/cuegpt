# CueGPT — AWS Deployment Notes

## Files in this folder
- `App.py` — Flask application
- `build_index_from_texts.py` — rebuilds ChromaDB from teaching texts
- `requirements.txt` — Python dependencies (pinned versions)
- `gunicorn.conf.py` — production server config
- `cuegpt.service` — systemd service file (copy to server)
- `nginx.conf` — Nginx reverse proxy config (copy to server)
- `.gitignore` — excludes .env, venv, shots/
- `templates/index.html` — ChatGPT-style UI
- `Extracted teachings/` — all .txt source files
- `chroma_db/` — pre-built vector database (163 chunks)

## To start locally
```bash
PYTHONPATH=/Users/pro/Downloads/human_chatbot/venv311 python3.11 App.py
```

## To rebuild the brain (after adding new .txt files)
```bash
PYTHONPATH=/Users/pro/Downloads/human_chatbot/venv311 python3.11 build_index_from_texts.py
```
