"""Test cần model thật (tải vài GB lần đầu) nên đánh dấu slow."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from retrieval.embed import Embedder

pytestmark = pytest.mark.slow
IMG_DIR, MANIFEST = Path("Data/images"), Path("Data/manifest.parquet")


@pytest.fixture(scope="module")
def emb():
    return Embedder()


def test_image_embedding_shape_and_norm(emb):
    imgs = [Image.new("RGB", (300, 200), c) for c in ("red", "blue")]
    v = emb.image(imgs)
    assert v.shape == (2, 768)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-3)


def test_text_embeddings_shapes(emb):
    v = emb.text_siglip(["áo bà ba", "bãi biển Nha Trang"])
    assert v.shape == (2, 768)
    dense, sparse = emb.text_bge(["áo bà ba", "bãi biển Nha Trang"])
    assert dense.shape == (2, 1024) and len(sparse) == 2 and all(isinstance(s, dict) for s in sparse)


def test_vietnamese_text_finds_its_image(emb):
    if not MANIFEST.exists() or len(list(IMG_DIR.glob("*.webp"))) < 60:
        pytest.skip("cần >= 60 ảnh đã tải trong Data/images")
    df = pd.read_parquet(MANIFEST)
    df = df[df["image_id"].map(lambda i: (IMG_DIR / f"{i}.webp").exists())]
    df = df[df["caption"].notna()].head(30)
    imgs = [Image.open(IMG_DIR / f"{i}.webp").convert("RGB") for i in df["image_id"]]
    iv = emb.image(imgs)
    tv = emb.text_siglip(df["caption"].tolist())
    sims = tv @ iv.T
    hit10 = np.mean([i in np.argsort(-sims[i])[:10] for i in range(len(df))])
    assert hit10 >= 0.5
