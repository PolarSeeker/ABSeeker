"""
Online LLM judge for ABSeeker anchor-clue per-turn scoring.

Configuration is read lazily via os.environ on every call (same pattern as
llm_judge.py). Set variables in RL/.env and source it from train_abseeker.sh.

Env vars:
  ABSEEKER_ANCHOR_API_BASE       OpenAI-compatible base URL (e.g. http://localhost:30016/v1)
  ABSEEKER_ANCHOR_API_KEY        Bearer token (leave empty for local / no-auth endpoints)
  ABSEEKER_ANCHOR_MODEL_NAME     Model id on the endpoint
  ABSEEKER_ANCHOR_ENABLE_THINKING  "true" / "false" (default: false)
  ABSEEKER_ANCHOR_TIMEOUT        Request timeout seconds (default: 120)
  ABSEEKER_ANCHOR_TEMPERATURE    Sampling temperature (default: 0.0)
"""

from __future__ import annotations

import json
import os
import re

import requests


def create_scoring_prompt(step, core_clues, query: str = "", correct_answer: str = ""):
    """Create prompt for LLM to score a step."""
    system_prompt = (
        "You are a trajectory scorer. Your task is to evaluate each step in an AI "
        "agent's problem-solving trajectory based on the scoring rubric.\n\n"
        "Focus on INTERMEDIATE STEPS and KEY ENTITIES. Base your scoring ONLY on the "
        "step's content (Context, Reasoning, and Tool Call)."
    )

    core_clues_text = "\n".join([
        f"  {i + 1}. {clue.get('name', 'unknown')} ({clue.get('type', 'unknown')}): "
        f"{clue.get('description', '')}"
        for i, clue in enumerate(core_clues)
    ])

    rubric = """
## Scoring Rubric:

All "entity/clue" in criteria 1-6 refer ONLY to the entities listed in "Core Clues (Key Entities - Reference)" below. Evaluate this step based on them, starting from the standard score of 1.0:

1. **Making Progress (baseline 1.0)**: 
   - If the step makes reasonable progress without major errors, it receives the standard score of 1.0.
   - Example: General searches, exploring possibilities without excluding correct entities.

2. **Excluding Wrong Entity/Clue (+0.4)**: 
   - If the step correctly excludes, dismisses, or identifies as wrong an entity/clue that is INCORRECT or IRRELEVANT, add points.
   - This shows good reasoning and helps narrow down the search.

3. **Finding/Exploring Correct Entity/Clue (+0.8)**: 
   - If the step discovers, mentions, or explores a key entity/clue that matches the core clues, add points.
   - Example: Finding or exploring entities that match core clues.

4. **Treating Wrong Entity as Final Answer (-2.0)**: 
   - ONLY when the step concludes or affirms a wrong entity as the final answer (the answer to the question), deduct significant points.
   - Exploring or considering wrong candidates during the process does NOT deduct points; only concluding with a wrong final answer does.

5. **Excluding Correct Entity/Clue (-0.8)**: 
   - If the step excludes a key entity/clue that is CORRECT according to core clues, deduct significant points.
   - This is a critical error that leads away from the solution.
   - Exploring incorrect candidates does NOT deduct points; exploration is normal and expected.

6. **Context Has Clue But Not Explored (-0.2)**: 
   - ONLY when the User/Context explicitly contains/mentions a key entity/clue from core clues, but the step does not explore or mention it, deduct a small amount. If the context does not contain the clue, this criterion does not apply.

7. **Correct Final Answer (+1.0)**:
   - ONLY when the step explicitly concludes or outputs the final answer to the question (e.g. inside `<answer>` tags or an unambiguous final conclusion), and that answer matches the Correct Answer below, add points.
   - Treat minor formatting differences as equivalent (e.g. with/without book-title brackets, simplified vs traditional Chinese).
   - If the step states a wrong final answer, do NOT apply criterion 7; apply criterion 4 instead.

## Question:
{query}

## Correct Answer:
{correct_answer}

## Core Clues (Key Entities - Reference):
These are the key entities that should be identified/explored:

{core_clues_text}

## Step to Score:
Step Number: {step_num}

Context: {user_content}

Reasoning: {reasoning}

Tool Call: {action}

## Task:
Score this step based on the rubric above.

Return ONLY a JSON object with this exact structure:
{{
    "score": <float between 0.0 and 2, with standard score being 1.0>,
    "criteria": <array of int 1-7, the rubric criterion numbers that apply, e.g. [1] or [1, 3, 7]>,
    "explanation": "<brief explanation. When applying criteria, MUST name the specific clue(s)/entity, e.g. 'clue X', 'entity Y'>"
}}

Important: Return ONLY valid JSON, no other text.
""".format(
        core_clues_text=core_clues_text,
        query=query,
        correct_answer=correct_answer,
        step_num=step.get("step", "N/A"),
        user_content=step.get("user_content", ""),
        reasoning=step.get("standard_reasoning", ""),
        action=str(step.get("standard_tool_call", "N/A")),
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rubric},
    ]


def _llm_gateway_proxies() -> dict | None:
    p = os.environ.get("ABSEEKER_ANCHOR_HTTP_PROXY", "").strip()
    return {"http": p, "https": p} if p else None


def _get_anchor_judge_config():
    base = os.environ.get("ABSEEKER_ANCHOR_API_BASE", "")
    key = os.environ.get("ABSEEKER_ANCHOR_API_KEY", "")
    model = os.environ.get("ABSEEKER_ANCHOR_MODEL_NAME", "")
    enable_thinking = os.environ.get("ABSEEKER_ANCHOR_ENABLE_THINKING", "false").lower() == "true"
    timeout = float(os.environ.get("ABSEEKER_ANCHOR_TIMEOUT", "120"))
    temperature = float(os.environ.get("ABSEEKER_ANCHOR_TEMPERATURE", "0.0"))
    return base, key, model, enable_thinking, timeout, temperature


def call_anchor_judge_api(messages: list[dict]) -> str:
    base, api_key, model_name, enable_thinking, timeout, temperature = _get_anchor_judge_config()
    if not base:
        raise RuntimeError(
            "[ABSeeker-Anchor-Judge] ABSEEKER_ANCHOR_API_BASE is not set. "
            "Copy RL/.env.example to RL/.env and fill in the anchor judge endpoint."
        )
    if not model_name:
        raise RuntimeError("[ABSeeker-Anchor-Judge] ABSEEKER_ANCHOR_MODEL_NAME is not set.")

    base = base.rstrip("/")
    api_url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "n": 1,
    }
    if enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    # Block until the proxy tunnel recovers instead of bailing out fast and
    # letting score_turn inject a fake 1.0. On failure we wait
    # ABSEEKER_ANCHOR_RETRY_WAIT_SECONDS (default 60s) before retrying — long enough
    # for the watchdog in train_abseeker.sh (checks every 30s, reconnects
    # immediately) to detect the outage and rebuild the tunnels. We keep
    # retrying until ABSEEKER_ANCHOR_MAX_RETRY_SECONDS (default 300s) so a proxy
    # outage during rollout causes zero reward contamination.
    import time as _time
    retry_wait = float(os.environ.get("ABSEEKER_ANCHOR_RETRY_WAIT_SECONDS", "60"))
    max_retry_seconds = float(os.environ.get("ABSEEKER_ANCHOR_MAX_RETRY_SECONDS", "300"))
    _start = _time.monotonic()
    _attempt = 0
    while True:
        _attempt += 1
        try:
            response = requests.post(
                api_url, json=payload, headers=headers, timeout=timeout,
                proxies=_llm_gateway_proxies(),
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return (content or "").strip()
        except Exception as _e:
            _elapsed = _time.monotonic() - _start
            if _elapsed >= max_retry_seconds:
                print(
                    f"[ABSeeker-Anchor-Judge] API unreachable after {_attempt} attempts "
                    f"({_elapsed:.0f}s >= {max_retry_seconds:.0f}s), giving up: {_e}",
                    flush=True,
                )
                raise
            print(
                f"[ABSeeker-Anchor-Judge] API call failed (attempt {_attempt}, "
                f"elapsed {_elapsed:.0f}s): {_e}. Waiting {retry_wait:.0f}s for "
                f"proxy to recover before retry...",
                flush=True,
            )
            _time.sleep(retry_wait)


def parse_anchor_result(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "score" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    m = re.search(r'"score"\s*:\s*([-+]?\d+(?:\.\d+)?)', raw)
    if m:
        return {"score": float(m.group(1)), "criteria": [], "explanation": ""}
    return None


def score_turn(
    step: dict,
    core_clues: list[dict],
    query: str = "",
    correct_answer: str = "",
) -> float:
    """Score one assistant turn. Returns rubric float (default 1.0 on parse failure)."""
    try:
        result = score_turn_detail(step, core_clues, query, correct_answer)
        return result["score"]
    except Exception as _e:
        print(
            f"[ABSeeker-Anchor-Judge] score_turn failed (step={step.get('step', 'N/A')}), "
            f"DEFAULTING TO 1.0 — reward may be contaminated: {_e}",
            flush=True,
        )
        return 1.0


def score_turn_detail(
    step: dict,
    core_clues: list[dict],
    query: str = "",
    correct_answer: str = "",
) -> dict:
    """Score one assistant turn and return full JSON result."""
    messages = create_scoring_prompt(step, core_clues, query, correct_answer)
    raw = call_anchor_judge_api(messages)
    parsed = parse_anchor_result(raw)
    if parsed is None:
        print(f"[ABSeeker-Anchor-Judge] failed to parse JSON, defaulting to 1.0. raw={raw[:200]!r}")
        return {"score": 1.0, "criteria": [1], "explanation": "parse failed"}
    return {
        "score": float(parsed["score"]),
        "criteria": parsed.get("criteria", []),
        "explanation": parsed.get("explanation", ""),
    }


def judge_answer_correct(query: str, ground_truth: str, answer: str) -> float:
    """Judge whether `answer` is correct for `query` given `ground_truth`.

    Returns 1.0 (correct) or 0.0 (wrong). Uses the same proxy endpoint /
    config and retry policy as score_turn (call_anchor_judge_api: 60s wait,
    300s total window).
    """
    if not answer or not answer.strip():
        return 0.0
    messages = [
        {"role": "system", "content": "You are a strict grader deciding whether a model's answer is correct."},
        {"role": "user", "content": (
            f"## Question:\n{query}\n\n"
            f"## Correct Answer:\n{ground_truth}\n\n"
            f"## Model Answer:\n{answer}\n\n"
            "Does the Model Answer match the Correct Answer? Treat minor formatting, "
            "synonym, or simplified/traditional Chinese differences as equivalent. "
            "Reply with ONLY 'yes' or 'no'."
        )},
    ]
    try:
        raw = call_anchor_judge_api(messages)
    except Exception as _e:
        print(
            f"[ABSeeker-Anchor-Judge] judge_answer_correct failed, "
            f"DEFAULTING TO 0.0 — outcome_acc may be wrong: {_e}",
            flush=True,
        )
        return 0.0
    m = re.search(r"\b(yes|no)\b", raw, re.IGNORECASE)
    if not m:
        print(
            f"[ABSeeker-Anchor-Judge] judge_answer_correct could not parse yes/no, "
            f"DEFAULTING TO 0.0. raw={raw[:200]!r}",
            flush=True,
        )
        return 0.0
    return 1.0 if m.group(1).lower() == "yes" else 0.0
