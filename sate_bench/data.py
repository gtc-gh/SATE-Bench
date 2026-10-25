"""Annotation loading and image helpers."""

import base64
import io
import json
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from .config import IMAGE_RESOLUTION


def load_samples(annotation_path, data_root=None, limit: Optional[int] = None) -> List[dict]:
    """Loads a SATE-Bench annotation file and resolves image paths to absolute paths.

    `image_paths` in the annotation files are relative to the folder that contains the
    annotation file (e.g. `data/benchmark/`), unless `data_root` is given.
    """
    annotation_path = Path(annotation_path)
    data_root = Path(data_root) if data_root else annotation_path.parent
    with open(annotation_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    for s in samples:
        if "image_paths" in s:
            s["image_paths"] = [str(data_root / p) for p in s["image_paths"]]
        if "image_path" in s:
            s["image_path"] = str(data_root / s["image_path"])
    return samples[:limit] if limit else samples


def check_images(samples: List[dict]) -> List[str]:
    """Returns the list of image files referenced by `samples` that are missing on disk."""
    missing = set()
    for s in samples:
        for p in s.get("image_paths", [s.get("image_path")]):
            if p and not Path(p).exists():
                missing.add(p)
    return sorted(missing)


def load_image(path, resolution: Tuple[int, int] = IMAGE_RESOLUTION) -> Image.Image:
    """Loads an RGB image and resizes it to `resolution` (720p by default)."""
    img = Image.open(path).convert("RGB")
    if img.size != tuple(resolution):
        img = img.resize(resolution, Image.Resampling.LANCZOS)
    return img


def image_to_jpeg_bytes(path, resolution: Tuple[int, int] = IMAGE_RESOLUTION) -> bytes:
    """Loads, resizes and JPEG-encodes an image (format used for API models)."""
    with Image.open(path) as img:
        if img.size != tuple(resolution):
            img = img.resize(resolution, Image.Resampling.LANCZOS)
        with io.BytesIO() as output:
            img.save(output, format="JPEG")
            return output.getvalue()


def image_to_base64(path, resolution: Tuple[int, int] = IMAGE_RESOLUTION) -> str:
    return base64.b64encode(image_to_jpeg_bytes(path, resolution)).decode("utf-8")
