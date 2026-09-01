from __future__ import annotations

import re

# Gỡ bỏ ký hiệu biểu cảm phổ biến và các dấu gạch đầu dòng dạng sticker.
_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF\uFE0F]"
)


def compact_message(message: str) -> str:
    """Giữ text ngắn gọn, bỏ emoji và khoảng trắng thừa."""
    text = str(message or "")
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_response_language(text: str) -> str:
    """Trả về 'Vietnamese' nếu query chủ yếu là tiếng Việt, ngược lại là 'English'."""
    sample = (text or "").strip().lower()
    if not sample:
        return "English"

    vietnamese_markers = [
        "tôi", "mình", "này", "bài báo", "tìm", "cho tôi", "hãy", "viết", "báo cáo",
        "về", "trong", "các", "điều", "nghiên cứu", "mô hình", "học", "các bài",
        "điều tra", "ngôn ngữ", "toán", "đại học", "tiếng việt"
    ]
    if any(marker in sample for marker in vietnamese_markers):
        return "Vietnamese"
    return "English"
