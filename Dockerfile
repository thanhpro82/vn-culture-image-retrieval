# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder
ENV UV_LINK_MODE=copy PIP_NO_CACHE_DIR=1
RUN pip install uv
WORKDIR /app
COPY pyproject.toml ./
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
# torch bản CPU (phục vụ không cần GPU); các thư viện còn lại tách lớp để cache khi sửa mã
RUN uv pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install -r pyproject.toml --extra ml

FROM python:3.12-slim
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONPATH=/app/src HF_HOME=/models WEB_DIR=/app/web
RUN useradd -m -u 1000 app && mkdir /models && chown app /models
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY src ./src
COPY web ./web
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"
CMD ["uvicorn", "retrieval.main:create", "--factory", "--host", "0.0.0.0", "--port", "8000"]
