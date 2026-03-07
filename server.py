#!/usr/bin/env python3
"""
OpenAI-Compatible MLX Chat Server
==================================
A production-quality local inference server for MLX models that speaks
the OpenAI Chat Completions protocol, including Azure OpenAI endpoints.

Usage:
    python server.py                                          # default model
    python server.py --model mlx-community/MiniMax-M2.5-8bit  # pick a model
    python server.py --port 8888 --host 0.0.0.0               # custom bind

Then hit it from any OpenAI client:

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
    r = client.chat.completions.create(
        model="minimax",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    )

Or from an Azure OpenAI client:

    from openai import AzureOpenAI
    client = AzureOpenAI(
        azure_endpoint="http://localhost:8080",
        api_key="local",
        api_version="2025-03-01-preview",
    )
    r = client.responses.create(
        model="my-deployment",
        input="Hello!",
    )
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*device_info is deprecated.*")

import argparse
import logging

import uvicorn

from mlx_server.app import create_app
from mlx_server.model import holder

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mlx-server")

# Create the FastAPI app (importable for uvicorn direct usage)
app = create_app()


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible MLX chat server")
    p.add_argument(
        "--model",
        default="mlx-community/Qwen3.5-9B-bf16",
        help="HuggingFace model ID or local path (default: mlx-community/Qwen3.5-9B-bf16)",
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    p.add_argument("--workers", type=int, default=1, help="Uvicorn workers (default: 1)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load model before starting the server so it's immediately available
    holder.load(args.model)

    log.info("Starting server on %s:%d", args.host, args.port)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
