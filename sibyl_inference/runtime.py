"""Local actor inference and optional one-step cloud execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping
from urllib import error, request

from .protocol import (
    CALL_CLOUD, build_actor_prompt, build_cloud_prompt, normalize_action,
    project_actor_response, project_cloud_response,
)


class Actor:
    def __init__(
        self, model_dir: Path, *, max_response_tokens: int = 512,
        max_prompt_tokens: int | None = None, gpu_memory_utilization: float = 0.9,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        model_dir = model_dir.expanduser().resolve()
        config = json.loads((model_dir / "tokenizer_config.json").read_text())
        if config.get("sibyl_qwen3_fixed_protocol_template") is not True:
            raise ValueError(
                "Model must use the Sibyl fixed-protocol tokenizer; "
                "prepare it with python -m sibyl_inference.prepare_tokenizer"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.llm = LLM(
            model=str(model_dir), tensor_parallel_size=1, dtype="bfloat16",
            trust_remote_code=True, gpu_memory_utilization=gpu_memory_utilization,
        )
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_response_tokens)
        self.max_prompt_tokens = max_prompt_tokens

    def generate(self, prompt: str) -> str:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=False, enable_thinking=False,
        )
        if self.max_prompt_tokens is not None:
            tokens = self.tokenizer.encode(rendered, add_special_tokens=False)
            if len(tokens) > self.max_prompt_tokens:
                rendered = self.tokenizer.decode(
                    tokens[-self.max_prompt_tokens:], skip_special_tokens=False
                )
        output = self.llm.generate([rendered], self.sampling, use_tqdm=False)
        return str(output[0].outputs[0].text or "")


class OpenAICompatibleCloud:
    def __init__(self, *, base_url: str, model: str, api_key: str | None = None,
                 timeout: float = 180.0) -> None:
        if not base_url:
            raise ValueError("Cloud base URL is required")
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self.model = model
        self.api_key = api_key or os.environ.get("SIBYL_CLOUD_API_KEY")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("Set SIBYL_CLOUD_API_KEY for cloud delegation")
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }).encode("utf-8")
        req = request.Request(
            self.url, data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                payload = json.load(response)
        except error.HTTPError as exc:
            raise RuntimeError(f"Cloud HTTP {exc.code}") from exc
        return str(payload["choices"][0]["message"]["content"] or "")


def decide(actor: Any, state: Mapping[str, Any], cloud: Any = None) -> dict[str, Any]:
    """Choose one ALFWorld action; unresolved calls have executed_action=None."""
    observation = str(state["observation"])
    task = str(state["task"])
    admissible = [normalize_action(a) for a in state["admissible_actions"]]
    history = list(state.get("history", []))
    prompt = build_actor_prompt(
        observation=observation, task=task, admissible_actions=admissible,
        history=history, history_length=int(state.get("history_length", 5)),
    )
    response = actor.generate(prompt)
    action, valid_protocol = project_actor_response(response)
    result: dict[str, Any] = {
        "actor_prompt": prompt, "actor_response": response, "actor_action": action,
        "actor_protocol_valid": valid_protocol,
        "actor_action_admissible": action == CALL_CLOUD or action in admissible,
        "cloud_called": action == CALL_CLOUD,
        "executed_action": action,
    }
    if action != CALL_CLOUD:
        return result
    if cloud is None:
        result["executed_action"] = None
        result["cloud_error"] = "cloud_not_configured"
        return result
    cloud_prompt = build_cloud_prompt(
        task=task, observation=observation, admissible_actions=admissible,
        history=state.get("cloud_history", history), inventory=state.get("inventory", []),
    )
    result["cloud_prompt"] = cloud_prompt
    try:
        cloud_response = cloud.generate(cloud_prompt)
        cloud_action, valid_cloud = project_cloud_response(cloud_response, admissible)
        result["cloud_response"] = cloud_response
        result["cloud_action"] = cloud_action
        result["cloud_action_valid"] = valid_cloud
        result["executed_action"] = cloud_action if valid_cloud else CALL_CLOUD
    except Exception as exc:
        result["cloud_action_valid"] = False
        result["cloud_error"] = f"{type(exc).__name__}: {exc}"
        result["executed_action"] = CALL_CLOUD
    return result
