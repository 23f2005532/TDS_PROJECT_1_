# eval_receiver.py
from fastapi import FastAPI, Request
import uvicorn, json

app = FastAPI()
received = []

@app.post("/evaluation-")
async def receive(req: Request):
    payload = await req.json()
    print("=== /evaluation-receiver got payload ===")
    print(json.dumps(payload, indent=2))
    received.append(payload)
    return {"status":"ok"}

@app.get("/received")
def list_received():
    return {"count": len(received), "items": received}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
