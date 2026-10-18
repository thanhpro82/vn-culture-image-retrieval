"""Tải sẵn các model vào HF_HOME (idempotent). Chạy một lần trước khi api khởi động."""
from huggingface_hub import snapshot_download

from retrieval.embed import BGE_ID, SIGLIP_ID
from retrieval.rerank import RERANK_ID

IGNORE = ["onnx/*", "*.onnx", "*.h5", "*.msgpack", "*.ot", "imgs/*", "*.tflite"]


def main() -> None:
    for repo in (SIGLIP_ID, BGE_ID, RERANK_ID):
        print("fetch", repo, flush=True)
        snapshot_download(repo, ignore_patterns=IGNORE)
    print("models ready")


if __name__ == "__main__":
    main()
