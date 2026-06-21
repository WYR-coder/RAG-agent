import json as _json
import urllib.request

from ..config import settings


def _call_llm(messages: list[dict], temperature: float = 0.3,
              max_tokens: int = 2000, stream: bool = False):
    body = _json.dumps({
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{settings.llm_base_url}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        },
    )
    return urllib.request.urlopen(req, timeout=120)


def chat_stream(messages: list[dict], temperature: float = 0.3):
    resp = _call_llm(messages, temperature=temperature, stream=True)
    buffer = b""
    for chunk in iter(lambda: resp.read(1024), b""):
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if not line or line == b"data: [DONE]":
                continue
            if line.startswith(b"data: "):
                try:
                    event = _json.loads(line[6:])
                    delta = event.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token
                except Exception:
                    pass


def chat(messages: list[dict], temperature: float = 0.3) -> str:
    resp = _call_llm(messages, temperature=temperature, stream=False)
    result = _json.loads(resp.read())
    return result["choices"][0]["message"]["content"]
