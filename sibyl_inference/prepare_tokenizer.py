"""Prepare an already merged Qwen3 tokenizer for Sibyl inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MARKER = "sibyl_qwen3_fixed_protocol_template"


def remove_qwen_prefill(template: str) -> tuple[str, bool]:
    lines = template.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines)
                  if "if enable_thinking is defined" in " ".join(line.split())
                  and "enable_thinking is false" in " ".join(line.split())), None)
    if start is None:
        return template, False
    end = next((i for i in range(start + 1, len(lines)) if "endif" in lines[i]), None)
    if end is None or "<think>" not in "".join(lines[start:end + 1]):
        raise ValueError("Cannot safely remove Qwen3 non-thinking prefill")
    return "".join(lines[:start] + lines[end + 1:]), True


def fix_tokenizer(model_dir: Path) -> None:
    from transformers import PreTrainedTokenizerBase

    config_path = model_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    templates = {}
    if config.get("chat_template") is not None:
        if not isinstance(config["chat_template"], str):
            raise ValueError("Expected a single Qwen3 chat-template string")
        templates["config"] = config["chat_template"]
    jinja_path = model_dir / "chat_template.jinja"
    if jinja_path.is_file():
        templates["jinja"] = jinja_path.read_text(encoding="utf-8")
    if not templates:
        raise ValueError("No Qwen3 chat template found in merged model")

    # Validate every template before writing either file. Transformers may
    # load the separate Jinja file in preference to tokenizer_config.json.
    prepared = {}
    for name, template in templates.items():
        fixed, _ = remove_qwen_prefill(template)
        tokenizer = PreTrainedTokenizerBase(chat_template=fixed)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Sibyl tokenizer check"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        if not rendered.rstrip().endswith("<|im_start|>assistant"):
            raise ValueError("Chat template must end with a plain Qwen3 assistant prefix")
        prepared[name] = fixed

    if "config" in prepared:
        config["chat_template"] = prepared["config"]
    if "jinja" in prepared:
        jinja_path.write_text(prepared["jinja"], encoding="utf-8")
    config[MARKER] = True
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path,
                        help="Merged Hugging Face model directory")
    args = parser.parse_args()
    fix_tokenizer(args.model.expanduser().resolve())
    print("Tokenizer is ready for Sibyl inference.")


if __name__ == "__main__":
    main()
