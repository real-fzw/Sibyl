"""Run Sibyl inference on independent ALFWorld states from JSONL input."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .runtime import Actor, OpenAICompatibleCloud, decide


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path,
                        help="Merged Hugging Face actor model directory")
    parser.add_argument("--input", type=Path,
                        help="JSONL states; defaults to standard input")
    parser.add_argument("--cloud-base-url", help="OpenAI-compatible API base URL")
    parser.add_argument("--cloud-model", help="Cloud provider model identifier")
    parser.add_argument("--max-response-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048,
                        help="Actor input token limit; defaults to 2048 for ALFWorld")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    if args.max_response_tokens < 1:
        parser.error("--max-response-tokens must be positive")
    if args.max_prompt_tokens is not None and args.max_prompt_tokens < 1:
        parser.error("--max-prompt-tokens must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if args.cloud_base_url:
        if not args.cloud_model:
            parser.error("--cloud-model is required with --cloud-base-url")
        if not os.environ.get("SIBYL_CLOUD_API_KEY"):
            parser.error("Set SIBYL_CLOUD_API_KEY for cloud delegation")
    elif args.cloud_model:
        parser.error("--cloud-model requires --cloud-base-url")

    cloud = (OpenAICompatibleCloud(base_url=args.cloud_base_url, model=args.cloud_model)
             if args.cloud_base_url else None)
    actor = Actor(
        args.model, max_response_tokens=args.max_response_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    source = args.input.expanduser().open(encoding="utf-8") if args.input else sys.stdin
    try:
        for line in source:
            if not line.strip():
                continue
            decision = decide(actor, json.loads(line), cloud)
            # Prompts remain available to callers of decide(); the CLI emits
            # only the response and action information needed by the caller.
            output = {key: value for key, value in decision.items()
                      if key not in ("actor_prompt", "cloud_prompt")}
            print(json.dumps(output, ensure_ascii=False), flush=True)
    finally:
        if args.input:
            source.close()


if __name__ == "__main__":
    main()
