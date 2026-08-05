"""
Turn-level format penalty for DeepResearcher agent outputs.

Expected formats (Qwen3.5 / Qwen3 share the same think + answer tags):
  Non-answer turn: <think>...</think>(<tool_call>...</tool_call>)+
  Answer turn:     <think>...</think><answer>...</answer>

Tool-call body accepts either:
  - Qwen3.5 XML: <function=name><parameter=key>value</parameter></function>
  - Qwen3 JSON:  {"name": "...", "arguments": {...}}
  - PythonInterpreter with <code>...</code>

Returns a binary result: True (correct) or False (any error detected).
"""

import re

try:
    import json5 as _json_lib
except ImportError:
    import json as _json_lib

_THINK_OPEN = re.compile(r"<think>")
_THINK_CLOSE = re.compile(r"</think>")
_TC_OPEN = re.compile(r"<tool_call>")
_TC_CLOSE = re.compile(r"</tool_call>")
_ANS_OPEN = re.compile(r"<answer>")
_ANS_CLOSE = re.compile(r"</answer>")
_QWEN35_FUNCTION_RE = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL)
_QWEN35_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _check_json_tool_body(tc_body: str) -> bool:
    try:
        parsed = _json_lib.loads(tc_body)
        if not isinstance(parsed, dict):
            return False
        if "name" not in parsed or "arguments" not in parsed:
            return False
        if not isinstance(parsed["arguments"], dict):
            return False
        return bool(str(parsed["name"]).strip())
    except Exception:
        return False


def _check_qwen35_tool_body(tc_body: str) -> bool:
    fm = _QWEN35_FUNCTION_RE.search(tc_body)
    if not fm:
        return False
    name = fm.group(1).strip()
    if not name:
        return False
    params_text = fm.group(2)
    if "pythoninterpreter" in name.lower():
        return "<code>" in params_text and "</code>" in params_text
    params = list(_QWEN35_PARAM_RE.finditer(params_text))
    if not params:
        return False
    for m in params:
        if not m.group(1).strip() or not m.group(2).strip():
            return False
    return True


def _check_tool_call_body(tc_body: str) -> bool:
    if not tc_body:
        return False
    if ("pythoninterpreter" in tc_body.lower()
            and "<code>" in tc_body and "</code>" in tc_body):
        return True
    if "<function=" in tc_body:
        return _check_qwen35_tool_body(tc_body)
    return _check_json_tool_body(tc_body)


def _assistant_content(messages, turn_boundaries, turn_idx: int) -> str | None:
    if (messages is None or (hasattr(messages, '__len__') and len(messages) == 0)) or \
       (turn_boundaries is None or (hasattr(turn_boundaries, '__len__') and len(turn_boundaries) == 0)) or \
       turn_idx >= len(turn_boundaries):
        return None
    msg_idx = turn_boundaries[turn_idx].get("msg_count")
    if msg_idx is None:
        msg_idx = 2 + 2 * turn_idx
    if msg_idx >= len(messages):
        return None
    msg = messages[msg_idx]
    if msg.get("role") != "assistant":
        return None
    return msg.get("content", "")


def apply_format_deduction_to_turn_scores(
    turn_scores,
    messages,
    turn_boundaries,
    deduction: float = 0.2,
):
    """Overwrite each malformed turn score with a fixed penalty (-*deduction*).

    Uses a fixed negative value instead of subtracting from the anchor score so
    the penalty does not get diluted by a high per-turn anchor score.
    """
    if turn_scores is None or (hasattr(turn_scores, '__len__') and len(turn_scores) == 0):
        return turn_scores, 0

    adjusted = [float(s) if s is not None else 0.0 for s in turn_scores]
    if (messages is None or (hasattr(messages, '__len__') and len(messages) == 0)) or \
       (turn_boundaries is None or (hasattr(turn_boundaries, '__len__') and len(turn_boundaries) == 0)):
        return adjusted, 0

    num_turns = len(turn_boundaries)
    penalty_count = 0
    for turn_idx in range(num_turns):
        if turn_idx >= len(adjusted):
            break
        content = _assistant_content(messages, turn_boundaries, turn_idx)
        if content is None:
            continue
        is_final = turn_idx == num_turns - 1
        if check_turn_format(content, is_final):
            continue
        adjusted[turn_idx] = -deduction
        penalty_count += 1
    return adjusted, penalty_count


def deduct_outcome_for_answer_format(
    outcome_score: float,
    messages,
    turn_boundaries,
    deduction: float = 0.2,
) -> tuple[float, bool]:
    """Deduct from outcome when the final answer turn has bad format."""
    if not messages or not turn_boundaries:
        return float(outcome_score), False
    content = _assistant_content(messages, turn_boundaries, len(turn_boundaries) - 1)
    if content is None or check_turn_format(content, True):
        return float(outcome_score), False
    return float(outcome_score) - deduction, True


def count_bad_format_turns(messages, turn_boundaries) -> int:
    """Count assistant turns that fail check_turn_format."""
    if (messages is None or (hasattr(messages, "__len__") and len(messages) == 0)
            or turn_boundaries is None
            or (hasattr(turn_boundaries, "__len__") and len(turn_boundaries) == 0)):
        return 0
    num_turns = len(turn_boundaries)
    n_bad = 0
    for turn_idx in range(num_turns):
        content = _assistant_content(messages, turn_boundaries, turn_idx)
        if content is None:
            continue
        if not check_turn_format(content, turn_idx == num_turns - 1):
            n_bad += 1
    return n_bad


def check_turn_format(content: str, is_final_turn: bool) -> bool:
    """Check whether a single assistant turn has correct format.

    Args:
        content: The raw assistant message content for this turn.
        is_final_turn: True if this is the last turn (should contain <answer>).

    Returns:
        True if format is correct, False if any error detected.
    """
    if not content or not content.strip():
        return False

    think_opens = [m.start() for m in _THINK_OPEN.finditer(content)]
    think_closes = [m.start() for m in _THINK_CLOSE.finditer(content)]
    tc_opens = [m.start() for m in _TC_OPEN.finditer(content)]
    tc_closes = [m.start() for m in _TC_CLOSE.finditer(content)]
    ans_opens = [m.start() for m in _ANS_OPEN.finditer(content)]
    ans_closes = [m.start() for m in _ANS_CLOSE.finditer(content)]

    # --- Think tags: must exist, properly paired, strictly sequential ---
    if not think_opens or len(think_opens) != len(think_closes):
        return False
    # No stray text before the first <think>
    if content[:think_opens[0]].strip():
        return False
    for i in range(len(think_opens)):
        if think_opens[i] >= think_closes[i]:
            return False
        if i > 0 and think_closes[i - 1] >= think_opens[i]:
            return False

    last_think_close = think_closes[-1]

    last_think_close_end = last_think_close + len("</think>")

    if is_final_turn:
        # --- Answer turn: <think>...</think><answer>...</answer> ---
        if tc_opens or tc_closes:
            return False
        if len(ans_opens) != 1 or len(ans_closes) != 1:
            return False
        if ans_opens[0] < last_think_close or ans_closes[0] < ans_opens[0]:
            return False

        # No stray text between </think> and <answer>
        if content[last_think_close_end:ans_opens[0]].strip():
            return False
        # No trailing text after </answer>
        if content[ans_closes[0] + len("</answer>"):].strip():
            return False
        # Answer body must not be empty
        if not content[ans_opens[0] + len("<answer>"):ans_closes[0]].strip():
            return False
    else:
        # --- Tool-call turn: <think>...</think>(<tool_call>...</tool_call>)+ ---
        if ans_opens or ans_closes:
            return False
        if len(tc_opens) != len(tc_closes) or len(tc_opens) < 1:
            return False
        # First tool call must start after the last </think>
        if tc_opens[0] < last_think_close:
            return False
        # No stray text between </think> and the first <tool_call>
        if content[last_think_close_end:tc_opens[0]].strip():
            return False
        # No trailing text after the last </tool_call>
        if content[tc_closes[-1] + len("</tool_call>"):].strip():
            return False

        for i in range(len(tc_opens)):
            if tc_opens[i] >= tc_closes[i]:
                return False
            if i > 0 and tc_closes[i - 1] >= tc_opens[i]:
                return False
            # Only whitespace allowed between consecutive tool calls
            if i > 0 and content[tc_closes[i - 1] + len("</tool_call>"):tc_opens[i]].strip():
                return False
            tc_body = content[tc_opens[i] + len("<tool_call>"):tc_closes[i]].strip()
            if not _check_tool_call_body(tc_body):
                return False

    return True
