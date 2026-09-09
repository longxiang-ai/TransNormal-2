"""Image IO and resize helpers for TransNormal-2 inference."""

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_image(path: str) -> torch.Tensor:
    """Load an image file as a (1, 3, H, W) float tensor in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    ts = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return ts / 127.5 - 1.0


def resize_to_multiple_of_16(image_tensor: torch.Tensor) -> torch.Tensor:
    """Rescale (B, C, H, W) so both sides are multiples of 16 (aspect kept)."""
    h, w = image_tensor.shape[2], image_tensor.shape[3]
    min_side = min(h, w)
    if min_side < 16:
        raise ValueError("Both image dimensions must be at least 16 pixels after resizing.")
    scale = (min_side // 16) * 16 / min_side

    new_h = (int(h * scale) // 16) * 16
    new_w = (int(w * scale) // 16) * 16

    if (new_h, new_w) == (h, w):
        return image_tensor
    return F.interpolate(
        image_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
    )


def resize_image_first(image_tensor: torch.Tensor, process_res: Optional[int] = None) -> torch.Tensor:
    """Optionally cap the max edge at ``process_res``, then snap to /16."""
    if process_res is not None and process_res < 16:
        raise ValueError("process_res must be at least 16 pixels.")
    if process_res:
        max_edge = max(image_tensor.shape[2], image_tensor.shape[3])
        if max_edge > process_res:
            scale = process_res / max_edge
            new_height = int(image_tensor.shape[2] * scale)
            new_width = int(image_tensor.shape[3] * scale)
            image_tensor = F.interpolate(
                image_tensor, size=(new_height, new_width), mode="bilinear", align_corners=False
            )
    return resize_to_multiple_of_16(image_tensor)


def tensor_to_output(
    normal_01: torch.Tensor, output_type: str = "pt"
) -> Union[torch.Tensor, np.ndarray, List[Image.Image]]:
    """Convert a (B, 3, H, W) [0, 1] tensor to the requested output format."""
    if output_type == "pt":
        return normal_01
    arr = normal_01.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    if output_type == "np":
        return arr
    if output_type == "pil":
        return [Image.fromarray((a * 255).round().astype(np.uint8)) for a in arr]
    raise ValueError(f"Unsupported output_type: {output_type} (use 'pt', 'np' or 'pil')")


def save_normal_map(
    normal: Union[torch.Tensor, np.ndarray],
    path: str,
    save_npy: Optional[str] = None,
) -> None:
    """Save a normal map prediction as a PNG (and optionally raw .npy).

    Accepts (3, H, W) / (1, 3, H, W) tensors or (H, W, 3) arrays in [0, 1]
    using the ``(n + 1) / 2`` encoding in RGB channel order.
    """
    if isinstance(normal, torch.Tensor):
        t = normal.detach().float().cpu()
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 3 and t.shape[0] == 3:
            t = t.permute(1, 2, 0)
        arr = t.numpy()
    else:
        arr = np.asarray(normal, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[0]

    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255).round().astype(np.uint8)).save(path)
    if save_npy:
        np.save(save_npy, arr)
