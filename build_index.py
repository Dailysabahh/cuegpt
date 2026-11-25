# build_index.py

from dotenv import load_dotenv
import os
import json
import chromadb
from openai import OpenAI

# ------------------ LOAD ENV ------------------ #

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------ LOAD DATA ------------------ #

with open("master_trading_brain.json") as f:
    data = json.load(f)

# ------------------ PREPARE TEXT ------------------ #

def build_text(item):
    return f"""
Concept: {item.get('concept', '')}
Rule: {item.get('rule', '')}
Strategy: {item.get('strategy', '')}
Explanation: {item.get('explanation', '')}
Type: {item.get('knowledge_type', '')}
"""

texts = [build_text(item) for item in data]

# ------------------ CREATE CHROMA CLIENT (PERSISTENT) ------------------ #

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 🔥 IMPORTANT: delete old collection if exists (clean rebuild)
try:
    chroma_client.delete_collection(name="cue_trading_brain")
    print("🗑 Old collection deleted")
except:
    pass

collection = chroma_client.get_or_create_collection(
    name="cue_trading_brain"
)

# ------------------ CREATE EMBEDDINGS ------------------ #

print("🔄 Creating embeddings...")

embeddings = []
for t in texts:
    emb = client.embeddings.create(
        input=t,
        model="text-embedding-3-small"
    ).data[0].embedding

    embeddings.append(emb)

# ------------------ IDS + METADATA ------------------ #

ids = [f"cue_{i}" for i in range(len(data))]

metas = [
    {
        "concept": item.get("concept", ""),
        "knowledge_type": item.get("knowledge_type", ""),
        "pair": item.get("pair", "UNKNOWN"),
        "video_type": item.get("video_type", ""),
        "source": item.get("source_file", "")
    }
    for item in data
]

# ------------------ STORE ------------------ #

collection.upsert(
    ids=ids,
    documents=texts,
    embeddings=embeddings,
    metadatas=metas
)


print("🔥 CueGPT trading brain indexed and saved successfully!")
