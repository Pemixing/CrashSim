import gzip
import json
import re
import urllib.error
import urllib.request
import zlib
from typing import Dict, List, Optional, Union


def _decode_response_body(data: bytes, content_encoding: Optional[str] = None) -> str:
    """Decode HTTP body bytes to UTF-8 text, handling gzip/deflate payloads."""
    encoding = (content_encoding or "").strip().lower()
    if encoding == "gzip":
        data = gzip.decompress(data)
    elif encoding in ("deflate", "zlib"):
        try:
            data = zlib.decompress(data)
        except zlib.error:
            data = zlib.decompress(data, -zlib.MAX_WBITS)
    elif len(data) >= 2 and data[:2] == b"\x1f\x8b":
        # Some gateways return gzip without a Content-Encoding header.
        data = gzip.decompress(data)

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise RuntimeError(
            "Failed to decode response body as UTF-8 "
            f"(len={len(data)}, content_encoding={content_encoding!r}, "
            f"prefix={data[:16]!r})"
        ) from e


def call_openai_chat(
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: Union[str, List[Dict]],
    temperature: float = 0.0,
    *,
    timeout: float = 120.0,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept-Encoding": "identity",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = _decode_response_body(
                resp.read(),
                resp.headers.get("Content-Encoding"),
            )
    except urllib.error.HTTPError as e:
        body = _decode_response_body(
            e.read(),
            e.headers.get("Content-Encoding"),
        )
        raise RuntimeError(f"OpenAI HTTPError {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenAI URLError: {e}") from e

    obj = json.loads(raw)
    choices = obj.get("choices", [])
    if not choices:
        raise RuntimeError(f"Empty choices in model response: {raw[:500]}")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"Empty message content in model response: {raw[:500]}")
    return content


def extract_json(text: str) -> Dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, flags=re.IGNORECASE)
    if fence:
        return json.loads(fence.group(1))

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and first < last:
        return json.loads(text[first : last + 1])

    raise ValueError("Cannot parse JSON from model output.")
