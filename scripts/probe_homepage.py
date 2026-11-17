"""Fetch the Bodhan gateway homepage and list candidate API paths."""

from __future__ import annotations

import re

import httpx

html = httpx.get("https://api.bodhan.ai/", timeout=30, follow_redirects=True).text

paths = sorted(
    set(
        re.findall(
            r"""["'`](/[a-zA-Z0-9_\-/]*(?:translate|speech|transcri|ocr|chat|audio|v1)[a-zA-Z0-9_\-/]*)["'`]""",
            html,
            re.I,
        )
    )
)
print("candidate paths:", paths)

urls = sorted(set(re.findall(r"""https?://[a-zA-Z0-9_\-./]+""", html)))
print("urls:")
for url in urls:
    print(" ", url)

for link in sorted(set(re.findall(r"""href=["']([^"']+)["']""", html))):
    print(" href:", link)
