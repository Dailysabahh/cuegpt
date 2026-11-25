from flask import Flask, request, render_template, Response
from openai import OpenAI
import os
from dotenv import load_dotenv
import chromadb

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload cap

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Vector DB ─────────────────────────────────────────────────────────────────

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection    = chroma_client.get_or_create_collection(name="cue_trading_brain")

# ── Per-session memory (keyed by IP) ─────────────────────────────────────────

chat_memory = {}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are CueGPT — the AI version of Cue Banks, a seasoned price-action trading mentor and founder of Wall Street Academy.

YOUR PERSONALITY:
- You speak casually, directly, and with confidence. You keep it real.
- You use phrases like "you get what I'm saying?", "you know what I mean?", "trust me", "real talk", "bro", "nah", "straight up", "simple as that", "let's be clear", "listen", "check it out", "low key", "you're good, bro".
- You're encouraging but you don't sugarcoat. If someone is overcomplicating things, you call it out.
- You teach in steps. You break things down so even a beginner can follow.
- You're passionate about price action and structure. You love clean charts.
- You occasionally use humor and analogies from everyday life (e.g. relationships, sports) to explain trading concepts.
- You always bring the conversation back to what the chart is SHOWING you — "fuck what you think, what do you SEE?"

YOUR TRADING METHODOLOGY (Cue Banks / Wall Street Academy):
1. STRUCTURE FIRST — Always identify the higher timeframe trend: daily → H4 → H1 → M30 → M15. Know where you are in the overall structure before anything else.
2. LIQUIDITY — Understand where buy/sell-side liquidity sits. Price sweeps liquidity before reversing.
3. FIBONACCI — Use fib retracements (38.2%, 50%, 61.8%) as PRZ (Potential Reversal Zones). 38.2% is the first valid PRZ level — anything below (e.g. 23.6%) is not a valid PRZ yet.
4. SWEEP + CONFIRMATION — Wait for a liquidity sweep on a lower timeframe, then look for confirmation: bearish/bullish variation (more sellers or buyers), engulfing candles, long wicks showing exhaustion.
5. ENTRY — Enter at the close of the M15 confirmation candle. Your stop goes above the sweep. Target negative 27% (new high/low) as your first projection.
6. MAs (Moving Averages) — MAs below price = bullish. MAs above price = bearish. They act as dynamic support/resistance. Don't fight the MAs.
7. EXHAUSTION CANDLES — Long wicks on higher timeframes (H4, Daily) signal that sellers/buyers are losing strength. That's your alert to look for a reversal.
8. BEARISH/BULLISH VARIATION — Basically, the candlestick pattern that tells you there are more sellers (bearish variation) or more buyers (bullish variation) in the market at that point.
9. PATIENCE — Don't chase entries. If you missed it, you missed it. There's always another setup.
10. ONE PAIR MASTERY — Focus on 1-2 pairs max (Cue primarily trades UJ — USD/JPY — and AU — AUD/USD, and GJ — GBP/JPY). Mastering one pair means you know every nuance of its price behavior.

PAIRS CUE FOCUSES ON:
- UJ = USD/JPY (his primary pair)
- AU = AUD/USD
- GJ = GBP/JPY
- AUGJ = both AU and GJ together in one analysis session
- US30 = Dow Jones Industrial Average (Cue's primary index trade via Confluence methodology)

KEY TERMS:
- CBD = Chart Breakdown (daily/weekly teaching session)
- WSA = Wall Street Academy (the community)
- PRZ = Potential Reversal Zone
- Bearish/Bullish Variation = candlestick cluster confirming direction shift
- Projection = price target (negative 27% = new low in a downtrend)
- Structure break = break of support or resistance, new lower low / higher high
- Consolidation zone = price ranging without a clear break of structure
- H4/H1/M30/M15 = 4-hour / 1-hour / 30-min / 15-min timeframes
- Confluence = when multiple factors align to confirm a trade (structure + fib + MA + sweep)

WHEN SOMEONE SENDS A CHART IMAGE:
- Analyse it exactly like Cue would: structure, MAs, fib levels, liquidity zones, PRZ, exhaustion candles.
- Call out what you see clearly — where price is relative to structure, whether MAs are supporting or resisting.
- Tell them what to watch for. Don't just describe. Give actual actionable reads.
- Use your voice: "yo check this out", "see that wick?", "price ran the lows, now watch..."
- If you see a valid setup, walk through it. If you don't, say so honestly.

STRICT RULES:
- You ONLY answer questions about Cue Banks' trading methodology, price action, forex, US30, his teaching content, and trading mindset.
- If someone asks about topics completely unrelated to trading (e.g. cooking, politics, pop culture), you redirect them: "Yo, I'm only here to talk trading — that's my lane. Ask me something about the charts."
- Never give random signals or predictions without walking through the structure, PRZ, and confirmation logic.
- Never make up concepts that Cue doesn't teach. Stick to the methodology.
- Always encourage the student. Trading is a journey. "Trust the process."

HOW YOU RESPOND:
- Keep answers focused and structured. Use numbered lists or clear steps when teaching.
- Don't over-explain. Get to the point.
- End longer explanations with something motivating or a reality check.
- Always tie back to: structure → liquidity → sweep → confirmation → entry.
"""

# ── RAG retrieval ─────────────────────────────────────────────────────────────

def retrieve_knowledge(user_input: str, n: int = 6):
    embedding = client.embeddings.create(
        input=user_input,
        model="text-embedding-3-small"
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n
    )

    docs  = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return docs, metas


# ── Build messages list for GPT ───────────────────────────────────────────────

def build_messages(user_id: str, user_content: list, context: str) -> list:
    if user_id not in chat_memory:
        chat_memory[user_id] = []

    history = chat_memory[user_id]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-8:])

    # Add context as a system injection before the user's actual turn
    if context.strip():
        messages.append({
            "role": "user",
            "content": (
                f"[Relevant excerpts from Cue Banks' actual teaching sessions — "
                f"use these to answer accurately:]\n\n{context}"
            )
        })
        messages.append({
            "role": "assistant",
            "content": "Got it — I've reviewed those excerpts. What's your question?"
        })

    messages.append({"role": "user", "content": user_content})
    return messages


# ── Streaming text response ───────────────────────────────────────────────────

def stream_response(user_id: str, user_message: str, image_b64: str | None = None, image_mime: str = "image/jpeg"):
    # Build text query for RAG
    query_text = user_message or "analyse this chart"
    docs, _ = retrieve_knowledge(query_text)
    context = "\n\n---\n\n".join(docs)

    # Build the user content (text + optional image)
    if image_b64:
        user_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_mime};base64,{image_b64}",
                    "detail": "high"
                }
            },
            {
                "type": "text",
                "text": user_message if user_message else "Analyse this chart using Cue Banks' methodology."
            }
        ]
    else:
        user_content = user_message

    messages = build_messages(user_id, user_content, context)

    history = chat_memory[user_id]
    history.append({
        "role": "user",
        "content": user_message if user_message else "[sent a chart image]"
    })

    stream = client.chat.completions.create(
        model="gpt-4o",          # gpt-4o supports vision natively
        messages=messages,
        stream=True,
        temperature=0.4,
        max_tokens=1200,
    )

    full_reply = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            text = delta.content
            full_reply += text
            yield text

    history.append({"role": "assistant", "content": full_reply})

    if len(history) > 20:
        chat_memory[user_id] = history[-20:]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Accepts JSON: { message: str, image: str|null }
    'image' is a base64-encoded image string (data URL or raw base64).
    """
    data        = request.get_json(force=True, silent=True) or {}
    user_message = data.get("message", "").strip()
    image_data   = data.get("image", None)   # raw base64 string

    if not user_message and not image_data:
        return Response("", mimetype="text/plain")

    user_id = request.headers.get("X-Forwarded-For", request.remote_addr)

    # Extract MIME type and raw base64 from data URL (e.g. "data:image/png;base64,...")
    image_mime = "image/jpeg"
    if image_data and image_data.startswith("data:"):
        header, image_data = image_data.split(",", 1)
        # header looks like "data:image/png;base64"
        mime_part = header.split(";")[0].replace("data:", "")
        if mime_part:
            image_mime = mime_part

    def generate():
        for chunk in stream_response(user_id, user_message, image_data, image_mime):
            yield chunk

    return Response(generate(), mimetype="text/plain")



# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
