"""ALFWorld prompts and action parsing for Sibyl inference."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

CALL_CLOUD = "call_cloud"
CALL_DESCRIPTION = (
    "The special action 'call_cloud' delegates only the current step to a "
    "cloud model, which will choose and execute one of the original "
    "admissible ALFWorld actions. Use it only when delegation is likely to "
    "improve task success."
)
ADMISSIBLE_RE = re.compile(
    r"(Your admissible actions of the current situation are:\s*)"
    r"\[(?P<body>.*?)\]\.",
    flags=re.IGNORECASE | re.DOTALL,
)

NO_HISTORY = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {observation}
Your admissible actions of the current situation are: [{actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.\x20
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WITH_HISTORY = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {history}
You are now at step {current_step} and your current observation is: {observation}
Your admissible actions of the current situation are: [{actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

RESPONSE = re.compile(
    r"\A\s*<think>(?P<think>[^<>]+)</think>\s*<action>(?P<action>[^<>]+)</action>\s*\Z",
    re.DOTALL | re.IGNORECASE,
)


def normalize_action(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def extract_task(initial_observation: str) -> str:
    marker = "Your task is to: "
    if marker not in initial_observation:
        raise ValueError("Initial observation has no task description")
    return initial_observation.split(marker, 1)[1].strip()


def build_actor_prompt(
    *, observation: str, task: str, admissible_actions: Sequence[str],
    history: Sequence[Mapping[str, Any]] = (), history_length: int = 5,
) -> str:
    if history_length < 0:
        raise ValueError("history_length must be nonnegative")
    actions = [normalize_action(a) for a in admissible_actions if normalize_action(a) != "help"]
    rendered_actions = "\n ".join(f"'{action}'" for action in actions)
    if history and history_length:
        recent = history[-history_length:]
        start = len(history) - len(recent)
        memory = "\n".join(
            f"[Observation {start + i + 1}: '{row['observation']}', "
            f"Action {start + i + 1}: '{row['action']}']"
            for i, row in enumerate(recent)
        )
        prompt = WITH_HISTORY.format(
            task=task, step_count=len(history), history_length=len(recent),
            history=memory, current_step=len(history) + 1,
            observation=observation, actions=rendered_actions,
        )
    else:
        # Native initial observations already include the goal. Independent
        # states may supply it separately; preserve the native prompt when
        # present and include the goal when it would otherwise be omitted.
        if task.strip() and task.strip() not in observation:
            observation = f"{observation.rstrip()}\nYour task is to: {task.strip()}"
        prompt = NO_HISTORY.format(observation=observation, actions=rendered_actions)
    match = ADMISSIBLE_RE.search(prompt)
    if not match:
        raise ValueError("ALFWorld admissible-action list is missing")
    body = match.group("body").rstrip()
    if not re.search(r"['\"]call_cloud['\"]", body, flags=re.IGNORECASE):
        body = f"{body},\n 'call_cloud'" if body else "'call_cloud'"
    replacement = f"{match.group(1)}[{body}].\n{CALL_DESCRIPTION}"
    return prompt[:match.start()] + replacement + prompt[match.end():]


def project_actor_response(response: str) -> tuple[str, bool]:
    """Return the requested action and whether the fixed protocol is valid."""
    text = str(response or "")
    lowered = text.lower()
    start, end = lowered.find("<action>"), lowered.find("</action>")
    if start < 0 or end < 0 or end <= start:
        return normalize_action(lowered[-30:]), False
    action = normalize_action(lowered[start + 8:end])
    match = RESPONSE.fullmatch(text)
    valid = bool(match and match.group("think").strip() and match.group("action").strip())
    return action, valid


def build_cloud_prompt(
    *, task: str, observation: str, admissible_actions: Sequence[str],
    history: Sequence[Mapping[str, Any]] = (), inventory: Sequence[str] = (),
) -> str:
    """One-step cloud request with the complete action/observation trajectory."""
    lines = [
        "You are the one-step cloud controller for an ALFWorld episode.",
        "Choose exactly one action for the current state.",
        "Return only <action>ONE EXACT ADMISSIBLE ACTION</action>.",
        "Do not return a plan, explanation, multiple actions, or call_cloud.",
        "", f"Task: {task.strip()}",
        f"Inventory: {', '.join(inventory) if inventory else 'empty'}",
        "", "Complete action-observation-result trajectory:",
    ]
    if not history:
        lines.append("(no previous actions)")
    for i, row in enumerate(history, 1):
        lines.extend([
            f"Step {i} observation: {str(row['observation']).strip()}",
            f"Step {i} action: {normalize_action(row['action'])}",
            f"Step {i} result: {str(row.get('result', '')).strip()}",
        ])
    lines.extend([
        "", f"Current observation: {observation.strip()}",
        "Current admissible actions:",
        *[f"- {normalize_action(a)}" for a in admissible_actions],
    ])
    return "\n".join(lines).strip() + "\n"


def project_cloud_response(response: str, admissible_actions: Sequence[str]) -> tuple[str, bool]:
    match = re.fullmatch(r"\s*<action>\s*([^<>]+?)\s*</action>\s*", str(response or ""), re.I | re.S)
    action = normalize_action(match.group(1)) if match else ""
    valid_actions = {normalize_action(a) for a in admissible_actions}
    return action, bool(action and action != CALL_CLOUD and action in valid_actions)
