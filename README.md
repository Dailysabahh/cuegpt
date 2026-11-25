# CueGPT

Chatbot for Cue Banks' students at Wall Street Academy.

A question is embedded, matched against a Chroma index of his teaching sessions, and answered in that method. A chart image can be sent with the question. The bot does not place trades.

## What is not in this repository

Lesson audio, transcripts, and the vector index stay off GitHub. They are source material, not the application.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python build_index.py   # requires the local teaching texts
python App.py
```

Open http://localhost:5001.
