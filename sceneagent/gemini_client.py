"""Gemini calls for the section 3.1 loop and the section 3.4 material vote.

The paper names Gemini 2.5 Flash. This client tries that id first. A 404 is
stored, then a Flash id the key's model list actually contains is used. The
substitute is not labeled 2.5.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

PAPER_MODEL_ID = "gemini-2.5-flash"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"


def redact(text, secret):
    if not text or not secret:
        return text or ""
    return str(text).replace(secret, "[redacted]")


def _request(url, key, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read().decode("utf-8")
            return response.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return exc.code, raw


def _candidate_text(raw):
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts).strip()


class GeminiClient:
    """HTTP client. The key is not included in returned bodies."""

    def __init__(self, api_key):
        self.api_key = api_key

    def generate(self, prompt, model_id, image_bytes=None, mime_type="image/jpeg"):
        parts = [{"text": prompt}]
        if image_bytes is not None:
            import base64
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                }
            })
        status, raw = _request(
            f"{API_ROOT}/models/{model_id}:generateContent",
            self.api_key,
            {"contents": [{"parts": parts}]},
        )
        raw = redact(raw, self.api_key)
        return {
            "requested_model_id": model_id,
            "model_id": model_id,
            "http_status": status,
            "text": _candidate_text(raw) if status == 200 else "",
            "error_body": "" if status == 200 else raw,
        }

    def list_model_ids(self):
        names = []
        page = ""
        for _ in range(5):
            url = f"{API_ROOT}/models"
            if page:
                url = f"{url}?pageToken={urllib.parse.quote(page, safe='')}"
            status, raw = _request(url, self.api_key)
            raw = redact(raw, self.api_key)
            if status != 200:
                break
            payload = json.loads(raw)
            names.extend(item.get("name", "") for item in payload.get("models") or [])
            page = payload.get("nextPageToken") or ""
            if not page:
                break
        return names

    def generate_flash(self, prompt, image_bytes=None):
        """Try the paper id, then one listed Flash id if that call is not 200."""
        first = self.generate(prompt, PAPER_MODEL_ID, image_bytes=image_bytes)
        if first["http_status"] == 200 and first["text"]:
            first["paper_model_id"] = PAPER_MODEL_ID
            first["used_paper_model"] = True
            return first
        listed = self.list_model_ids()
        fallback = choose_flash_model(
            first.get("error_body") or "",
            listed,
            exclude=(PAPER_MODEL_ID,),
        )
        if not fallback:
            first["paper_model_id"] = PAPER_MODEL_ID
            first["used_paper_model"] = False
            first["fallback_model_id"] = None
            return first
        fallback_id = fallback.split("/", 1)[-1]
        second = self.generate(prompt, fallback_id, image_bytes=image_bytes)
        second["paper_model_id"] = PAPER_MODEL_ID
        second["paper_http_status"] = first["http_status"]
        second["paper_error_body"] = first.get("error_body") or ""
        second["fallback_model_id"] = fallback_id
        second["used_paper_model"] = False
        second["model_id"] = fallback_id
        return second


def choose_flash_model(error_body, model_names, exclude=(PAPER_MODEL_ID,)):
    """Pick a Flash id the API lists, other than the id that just failed.

    The 404 text names the unavailable model first, then the replacement.
    The replacement is used. It is not labeled as the paper model.
    """
    available = set(model_names)
    excluded = set()
    for item in exclude:
        excluded.add(item)
        excluded.add(item.split("/")[-1])
        if not str(item).startswith("models/"):
            excluded.add("models/" + str(item))
    named = []
    token = "models/gemini-"
    cursor = 0
    while True:
        start = error_body.find(token, cursor)
        if start < 0:
            break
        end = start + len(token)
        while end < len(error_body) and (error_body[end].isalnum() or error_body[end] in ".-"):
            end += 1
        named.append(error_body[start:end])
        cursor = end

    def usable(name):
        short = name.split("/")[-1]
        if name in excluded or short in excluded:
            return False
        if not _is_text_flash(name):
            return False
        return name in available or ("models/" + short) in available

    for name in named:
        if usable(name):
            return name if name in available else "models/" + name.split("/")[-1]
    preferred = (
        "models/gemini-3.8-flash",
        "models/gemini-flash-latest",
        "models/gemini-2.5-flash-lite",
    )
    for name in preferred:
        if usable(name):
            return name
    for name in model_names:
        if usable(name):
            return name if name.startswith("models/") else "models/" + name
    return None


def _is_text_flash(name):
    lowered = name.lower()
    if "flash" not in lowered:
        return False
    blocked = ("tts", "image", "computer-use", "omni")
    return not any(piece in lowered for piece in blocked)
