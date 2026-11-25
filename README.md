# CueGPT

Chatbot for Cue Banks' students. It answers from his class sessions, in the way he teaches. You can send a chart with the question.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python build_index.py
python App.py
```

Open http://localhost:5001.
