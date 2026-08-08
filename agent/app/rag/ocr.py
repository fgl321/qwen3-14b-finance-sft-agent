from __future__ import annotations

import io
import threading
from typing import Any


_engine: Any | None = None
_engine_lock = threading.Lock()


def _get_engine() -> Any:
    """懒加载 rapidocr（PP-OCRv6，模型随 wheel 打包，离线可用）。"""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from rapidocr import RapidOCR

                _engine = RapidOCR()
    return _engine


def _join_texts(output: Any) -> str:
    texts = getattr(output, "txts", None) or []
    return "\n".join(
        text.strip()
        for text in texts
        if text and text.strip()
    )


def ocr_text_from_bytes(data: bytes) -> str:
    """对图片字节做 OCR，失败返回空串。"""
    try:
        import numpy as np
        from PIL import Image

        image = Image.open(io.BytesIO(data)).convert("RGB")
        output = _get_engine()(np.array(image))
        return _join_texts(output)
    except Exception:
        return ""


def ocr_text_from_pixmap(pixmap: Any) -> str:
    """对 PyMuPDF 渲染的页面位图做 OCR，失败返回空串。"""
    try:
        import numpy as np

        width = int(pixmap.width)
        height = int(pixmap.height)
        channels = int(pixmap.n)
        samples = bytes(pixmap.samples)
        image = np.frombuffer(samples, dtype=np.uint8).reshape(
            height,
            width,
            channels,
        )
        if channels >= 4:
            image = image[:, :, :3]
        output = _get_engine()(image)
        return _join_texts(output)
    except Exception:
        return ""


def ocr_available() -> bool:
    try:
        _get_engine()
        return True
    except Exception:
        return False
