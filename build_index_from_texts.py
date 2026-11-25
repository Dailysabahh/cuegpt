# build_index_from_texts.py
# Reads all .txt teaching files from "Extracted teachings/" and indexes them into ChromaDB

from dotenv import load_dotenv
import os
import glob
import chromadb
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── collect all .txt files ────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEACHINGS_DIR = os.path.join(BASE_DIR, "Extracted teachings")

txt_files = glob.glob(os.path.join(TEACHINGS_DIR, "**", "*.txt"), recursive=True)
print(f"Found {len(txt_files)} teaching files")

# ── chunk each file into ~600-word segments ───────────────────────────────────

def chunk_text(text, chunk_size=600):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

all_docs   = []
all_metas  = []
all_ids    = []

chunk_id = 0
for fpath in txt_files:
    filename = os.path.basename(fpath)
    subfolder = os.path.basename(os.path.dirname(fpath))
    source_label = f"{subfolder}/{filename}" if subfolder != "Extracted teachings" else filename

    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
    except Exception as e:
        print(f"  ⚠ Could not read {fpath}: {e}")
        continue

    if not text:
        continue

    chunks = chunk_text(text, chunk_size=600)
    for idx, chunk in enumerate(chunks):
        all_docs.append(chunk)
        all_metas.append({
            "source_file": source_label,
            "chunk_index": idx,
            "knowledge_type": "cue_teaching",
        })
        all_ids.append(f"cue_{chunk_id}")
        chunk_id += 1

print(f"Total chunks to embed: {len(all_docs)}")

# ── connect to Chroma ─────────────────────────────────────────────────────────

chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    chroma_client.delete_collection(name="cue_trading_brain")
    print("Old collection deleted")
except Exception:
    pass

collection = chroma_client.get_or_create_collection(name="cue_trading_brain")

# ── embed and upsert in batches of 100 ───────────────────────────────────────

BATCH = 100
for start in range(0, len(all_docs), BATCH):
    end = min(start + BATCH, len(all_docs))
    batch_docs  = all_docs[start:end]
    batch_metas = all_metas[start:end]
    batch_ids   = all_ids[start:end]

    print(f"  Embedding chunks {start+1}–{end} of {len(all_docs)} ...")

    response = client.embeddings.create(
        input=batch_docs,
        model="text-embedding-3-small"
    )
    batch_embeddings = [item.embedding for item in response.data]

    collection.upsert(
        ids=batch_ids,
        documents=batch_docs,
        embeddings=batch_embeddings,
        metadatas=batch_metas,
    )

print(f"\nCueGPT brain indexed successfully! {len(all_docs)} chunks stored in ChromaDB.")
