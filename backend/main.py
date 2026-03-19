"""
DeepTutor Backend - Render.com deployment version
"""
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx, json

app = FastAPI(title="DeepTutor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY  = os.getenv("LLM_API_KEY", "minimax-agent")
API_HOST = os.getenv("LLM_HOST", "http://127.0.0.1:8766")
MODEL    = os.getenv("LLM_MODEL", "MiniMax-M2.5")

print(f"DeepTutor Backend starting")
print(f"  API: {API_HOST}/v1/messages")
print(f"  Model: {MODEL}")

def oai_to_anthropic(messages, body):
    system, anth_messages = "", []
    for msg in messages:
        role, content = msg.get("role","user"), msg.get("content","")
        if isinstance(content, list):
            content = " ".join(c.get("text","") for c in content if c.get("type")=="text")
        if role == "system":
            system = content
        elif role in ("user","assistant"):
            anth_messages.append({"role": role, "content": content})
    payload = {"model": MODEL, "messages": anth_messages}
    if system:
        payload["system"] = system
    payload["max_tokens"] = body.get("max_tokens") or 8192
    if body.get("temperature"):
        payload["temperature"] = body["temperature"]
    return payload

def anth_to_oai(ar):
    usage = ar.get("usage", {})
    text = ""
    for b in (ar.get("content") or []):
        if b.get("type") == "text":
            text = b.get("text",""); break
        if b.get("type") == "thinking" and not text:
            text = f"[思考中...]"
    return {
        "id": ar.get("id","chatcmpl"),
        "object": "chat.completion",
        "created": 1700000000,
        "model": MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": sum(usage.values()) if isinstance(usage, dict) else 0,
        }
    }

@app.api_route("/api/v1/chat", methods=["GET","POST"])
@app.api_route("/api/v1/chat/completions", methods=["GET","POST"])
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    payload = oai_to_anthropic(messages, body)
    hdrs = {"Authorization": f"Bearer {API_KEY}", "x-api-key": API_KEY, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            resp = await client.post(f"{API_HOST}/v1/messages", json=payload, headers=hdrs)
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": {"message": str(e)}})

        if not resp.is_success:
            return JSONResponse(status_code=resp.status_code,
                content={"error": {"message": f"API error {resp.status_code}: {resp.text[:300]"}})

        try:
            anthropic_resp = resp.json()
        except Exception as e:
            return JSONResponse(status_code=502,
                content={"error": {"message": f"Parse error: {e}", "raw": resp.text[:300]}})

        if stream:
            async def gen():
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        d = line[6:]
                        try:
                            dj = json.loads(d)
                            if dj.get("type") == "content_block_delta":
                                delta = dj.get("delta",{})
                                if delta.get("type") == "text_delta":
                                    txt = delta.get("text","")
                                    yield f"data: {json.dumps({'id':'chatcmpl','object':'chat.completion.chunk','created':1700000000,'model':MODEL,'choices':[{'index':0,'delta':{'content':txt},'finish_reason':None}]})}\n\n"
                            elif dj.get("type") == "message_stop":
                                yield "data: [DONE]\n\n"
                        except: pass
            return StreamingResponse(gen(), media_type="text/event-stream")
        else:
            return JSONResponse(content=anth_to_oai(anthropic_resp))

@app.get("/v1/models")
@app.get("/api/v1/models")
async def list_models():
    return {"data": [{"id": MODEL, "object": "model", "created": 1700000000, "owned_by": "minimax"}]}

@app.get("/health")
@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "model": MODEL, "backend": "deepTutor-adapter-render"}

@app.get("/api/v1/knowledge/list")
async def knowledge_list():
    return {"data": [{"id": "default", "name": "Default Knowledge Base"}]}

@app.get("/api/v1/history/sessions")
async def history_sessions():
    return {"data": []}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
