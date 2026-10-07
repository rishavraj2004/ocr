"""Image I/O and conversion utilities for OpenCV and PIL."""

from __future__ import annotations

from pathlib import Path
from typing import Union
import cv2
import numpy as np
from PIL import Image


def load_image_as_cv2(image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """Load an image from filepath, PIL Image, or numpy array into an OpenCV BGR/Grayscale array.

    Uses numpy fromfile + imdecode on filepaths to safely support Windows non-ASCII / Unicode paths.
    """
    if isinstance(image_input, np.ndarray):
        return image_input.copy()

    if isinstance(image_input, Image.Image):
        # Convert PIL to OpenCV array
        if image_input.mode == "RGBA":
            rgb = image_input.convert("RGB")
            arr = np.array(rgb)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif image_input.mode == "RGB":
            arr = np.array(image_input)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif image_input.mode == "L":
            return np.array(image_input)
        else:
            rgb = image_input.convert("RGB")
            return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

    path = Path(image_input)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found at: {path}")

    # Read binary stream to handle any UTF-8 path on Windows
    with open(path, "rb") as f:
        file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"OpenCV failed to decode image at: {path}")

    return img


def save_cv2_image(image: np.ndarray, output_path: Union[str, Path]) -> Path:
    """Save OpenCV array to disk safely using imencode and file writing."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    ext = p.suffix.lower() or ".png"
    success, encoded = cv2.imencode(ext, image)
    if not success:
        raise ValueError(f"Failed to encode image to format '{ext}'")

    with open(p, "wb") as f:
        f.write(encoded.tobytes())

    return p


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV numpy array to a PIL Image."""
    if len(image.shape) == 2:
        return Image.fromarray(image)
    elif len(image.shape) == 3:
        if image.shape[2] == 4:
            rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(rgba)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    raise ValueError(f"Unsupported image shape for PIL conversion: {image.shape}")
