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
    config_path = model_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seen_template = False
    changed = False
    if isinstance(config.get("chat_template"), str):
        seen_template = True
        config["chat_template"], edited = remove_qwen_prefill(config["chat_template"])
        changed |= edited
    jinja_path = model_dir / "chat_template.jinja"
    if jinja_path.is_file():
        seen_template = True
        template, edited = remove_qwen_prefill(jinja_path.read_text(encoding="utf-8"))
        if edited:
            jinja_path.write_text(template, encoding="utf-8")
        changed |= edited
    if not seen_template:
        raise ValueError("No Qwen3 chat template found in merged model")
    already_verified = (
        config.get(MARKER) is True
        or config.get("cada_qwen3_fixed_protocol_template") is True
    )
    if not changed and not already_verified:
        templates = [config.get("chat_template", "")]
        if jinja_path.is_file():
            templates.append(jinja_path.read_text(encoding="utf-8"))
        if not any("add_generation_prompt" in value and "<|im_start|>assistant" in value
                   and "<think>" not in value for value in templates):
            raise ValueError("Unverified chat template; refusing to mark as fixed protocol")
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
