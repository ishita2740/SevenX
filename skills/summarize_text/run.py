import json, sys, requests
data = json.loads(sys.stdin.read())
params = data["params"]
text = params["text"]
max_words = params.get("max_words", 50)
try:
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen2.5:3b",
            "messages": [
                {"role": "system", "content": f"Summarize the following text in {max_words} words or fewer. Be concise."},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=90,
    )
    resp.raise_for_status()
    summary = resp.json()["message"]["content"].strip()
    print(json.dumps({"success": True, "data": {"summary": summary, "original_length": len(text), "summary_length": len(summary)}}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
