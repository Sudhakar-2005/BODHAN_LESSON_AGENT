"""Probe script field + error envelope on the working chat translate transport."""

from __future__ import annotations

import httpx

from bodhan.config import load_settings

settings = load_settings()
headers = {"Authorization": f"Bearer {settings.bodhan_translate_key}"}
URL = "https://api.bodhan.ai/v1/chat/completions"


def show(label: str, r: httpx.Response) -> None:
    body = r.text[:400].encode("ascii", "backslashreplace").decode("ascii")
    print(f"{label} -> HTTP {r.status_code}: {body}")


# script field accepted?
try:
    r = httpx.post(
        URL,
        headers=headers,
        timeout=45,
        json={
            "model": "indic-translate",
            "messages": [{"role": "user", "content": "Good morning"}],
            "target_language": "hi",
            "script": "roman",
        },
    )
    show("script=roman", r)
except Exception as exc:
    print(f"script=roman: EXC {type(exc).__name__}")

# unsupported language -> documented envelope?
try:
    r = httpx.post(
        URL,
        headers=headers,
        timeout=45,
        json={
            "model": "indic-translate",
            "messages": [{"role": "user", "content": "hello"}],
            "target_language": "xx",
        },
    )
    show("target_language=xx", r)
except Exception as exc:
    print(f"target_language=xx: EXC {type(exc).__name__}")

# two user turns -> documented invalid_prompt?
try:
    r = httpx.post(
        URL,
        headers=headers,
        timeout=45,
        json={
            "model": "indic-translate",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "user", "content": "world"},
            ],
            "target_language": "hi",
        },
    )
    show("two user turns", r)
except Exception as exc:
    print(f"two user turns: EXC {type(exc).__name__}")
