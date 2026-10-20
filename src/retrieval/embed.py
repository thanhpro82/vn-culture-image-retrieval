"""SigLIP 2 (ảnh + text) và bge-m3 (dense + sparse). Nạp model lười, chạy tuần tự để vừa 4 GB VRAM."""
import gc

import numpy as np
from PIL import Image

SIGLIP_ID = "google/siglip2-base-patch16-256"
BGE_ID = "BAAI/bge-m3"


def _feat(out):
    """transformers 5 trả đối tượng có pooler_output; bản cũ trả thẳng tensor."""
    return out if hasattr(out, "float") else out.pooler_output


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)


class Embedder:
    def __init__(self, device: str | None = None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._siglip = None
        self._bge = None

    def _load_siglip(self):
        if self._siglip is None:
            from transformers import AutoModel, AutoProcessor
            model = AutoModel.from_pretrained(SIGLIP_ID, dtype=self.dtype).to(self.device).eval()
            self._siglip = (model, AutoProcessor.from_pretrained(SIGLIP_ID))
        return self._siglip

    def _load_bge(self):
        if self._bge is None:
            from FlagEmbedding import BGEM3FlagModel
            self._bge = BGEM3FlagModel(BGE_ID, use_fp16=self.device == "cuda", device=self.device)
        return self._bge

    def unload(self, which: str) -> None:
        setattr(self, f"_{which}", None)
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    def image(self, batch: list[Image.Image]) -> np.ndarray:
        model, proc = self._load_siglip()
        inputs = proc(images=[im.convert("RGB") for im in batch], return_tensors="pt")
        with self.torch.no_grad():
            out = model.get_image_features(pixel_values=inputs["pixel_values"].to(self.device, self.dtype))
        return _norm(_feat(out).float().cpu().numpy())

    def text_siglip(self, texts: list[str]) -> np.ndarray:
        model, proc = self._load_siglip()
        inputs = proc(text=[t.lower() for t in texts], padding="max_length", max_length=64,
                      truncation=True, return_tensors="pt")
        with self.torch.no_grad():
            out = model.get_text_features(input_ids=inputs["input_ids"].to(self.device))
        return _norm(_feat(out).float().cpu().numpy())

    def text_bge(self, texts: list[str]) -> tuple[np.ndarray, list[dict[int, float]]]:
        res = self._load_bge().encode(texts, batch_size=32, max_length=256,
                                      return_dense=True, return_sparse=True, return_colbert_vecs=False)
        sparse = [{int(k): float(v) for k, v in w.items()} for w in res["lexical_weights"]]
        return _norm(np.asarray(res["dense_vecs"], dtype="float32")), sparse
