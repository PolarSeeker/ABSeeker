"""ABSeeker agent inference loop — tools from tools/."""
from __future__ import annotations

import argparse
import asyncio
import datetime
import importlib.util
import json
import os
import random
import re
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

_T = TypeVar("_T")
_thread_log = threading.local()


def _plog(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    f = getattr(_thread_log, "file", None)
    print(*args, file=f, **kwargs)


def run_with_log_file(log_path: str, fn: Callable[..., _T], *args, **kwargs) -> _T:
    with open(log_path, "w", encoding="utf-8") as lf:
        _thread_log.file = lf
        try:
            return fn(*args, **kwargs)
        finally:
            _thread_log.file = None

import jinja2
import requests
from requests.adapters import HTTPAdapter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from prompts.sft_react import SYSTEM_PROMPT

TOOLS_DIR = os.path.join(ROOT, "tools")
TEMPLATE_PATH = os.path.join(ROOT, "configs", "abseeker_chat_template.jinja")


def _load_tool_fn(module_name: str, fn_name: str):
    path = os.path.join(TOOLS_DIR, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)


_search_web = _load_tool_fn("search_web", "search_web")
_visit_web = _load_tool_fn("visit_web", "visit_web")


def print_colored(text: str, color: int) -> None:
    _plog(f"\033[{color}m{text}\033[0m", end="", flush=True)


def _tool_color(tool_name: str) -> int:
    if tool_name == "search":
        return 34
    if tool_name == "visit":
        return 33
    return 35


def _print_tool_call(tool_name: str, tool_args: Any, tool_response: str) -> None:
    c = _tool_color(tool_name)
    try:
        args_str = json.dumps(tool_args, ensure_ascii=False, sort_keys=True)
    except Exception:
        args_str = str(tool_args)
    print_colored(f"\n[{tool_name}] args={args_str}\n", c)
    print_colored(f"[{tool_name}] response={'' if tool_response is None else str(tool_response)}\n", c)


def strftime_now_function(fmt: str) -> str:
    return datetime.datetime.now().strftime(fmt)


def _tojson(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _raise_exception(msg: str) -> str:
    raise jinja2.TemplateError(msg)


with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    _template_string = f.read()

_env = jinja2.Environment()
_env.filters["tojson"] = _tojson
_env.globals["strftime_now"] = strftime_now_function
_env.globals["raise_exception"] = _raise_exception
template = _env.from_string(_template_string)

developer_prompt = (
    "You are a tool-augmented QA agent. Cleverly leverage appropriate tools to answer the user's question."
)

tools_all: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Performs batched web searches: supply an array 'query'; "
                "the tool retrieves the top 10 results for each query in one call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Array of query strings. Include multiple complementary "
                            "search queries in a single call."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Parse webpage(s) and return the summary of the content according to the goal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": (
                            "The URL(s) of the webpage(s) to visit. "
                            "Can be a single URL or an array of URLs."
                        ),
                    },
                    "goal": {"type": "string", "description": "The goal of the visit for webpage(s)."},
                },
                "required": ["url", "goal"],
            },
        },
    },
]


def _normalize_completions_url(base_or_full: str) -> str:
    s = (base_or_full or "").strip()
    if not s:
        raise ValueError("Empty base_url / completions_url")
    if s.endswith("/completions"):
        return s
    if s.endswith("/v1"):
        return s + "/completions"
    if s.endswith("/v1/"):
        return s + "completions"
    if s.endswith("/"):
        return s + "v1/completions"
    return s + "/v1/completions"


def get_stream_response(
    completions_url: str,
    payload: Dict[str, Any],
    print_stream: bool = True,
    *,
    max_retries: int = 3,
    connect_timeout: int = 10,
    read_timeout: int = 60,
    max_total_seconds: int = 1200,
    max_idle_seconds: int = 45,
) -> Tuple[str, bool]:
    retry_backoff = 1.8
    retry_jitter = 0.4
    too_many_tokens_error = False
    result = ""

    for attempt in range(int(max_retries) + 1):
        start = time.monotonic()
        last = start
        chunks: List[str] = []
        got_done = False
        failed = False

        try:
            with requests.post(
                completions_url,
                json=payload,
                stream=True,
                timeout=(int(connect_timeout), int(read_timeout)),
            ) as response:
                response.raise_for_status()
                full_text_so_far = ""
                for raw_line in response.iter_lines(decode_unicode=False):
                    now = time.monotonic()
                    if now - start > float(max_total_seconds) or now - last > float(max_idle_seconds):
                        failed = True
                        break
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="replace")
                    if not line.startswith("data:"):
                        continue
                    last = now
                    json_str = line.split("data:", 1)[1].strip()
                    if not json_str:
                        continue
                    if json_str == "[DONE]":
                        got_done = not failed
                        break
                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict) and "error" in data:
                        err = data.get("error") or {}
                        message = (err.get("message") or "").strip()
                        if "maximum context length" in message or "context length" in message:
                            too_many_tokens_error = True
                        if message:
                            _plog(f"[LLM ERROR] {message}")
                        failed = True
                        break
                    choice0 = (data.get("choices") or [{}])[0] or {}
                    text_piece = choice0.get("text") or ""
                    if isinstance(text_piece, str) and text_piece:
                        if text_piece.startswith(full_text_so_far):
                            delta = text_piece[len(full_text_so_far) :]
                            full_text_so_far = text_piece
                        else:
                            delta = text_piece
                            full_text_so_far += text_piece
                        if delta:
                            chunks.append(delta)
                            if print_stream:
                                _plog(delta, end="", flush=True)
                    if choice0.get("finish_reason") or choice0.get("matched_stop"):
                        got_done = True
                        break
        except requests.exceptions.HTTPError as e:
            err_body = ""
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None)
            if resp is not None:
                try:
                    err_body = (resp.text or "").lower()
                except Exception:
                    pass
            _plog(f"[LLM ERROR] Request failed: {type(e).__name__}: {e}")
            if err_body:
                _plog(f"[LLM ERROR] body: {err_body[:300]}")
            if status == 400 or any(
                k in err_body
                for k in ("maximum context length", "context length", "too long", "max context", "prompt length", "badrequest")
            ):
                return "", True
            failed = True
        except requests.exceptions.RequestException as e:
            _plog(f"[LLM ERROR] Request failed: {type(e).__name__}: {e}")
            failed = True

        result = "".join(chunks)
        if got_done and not failed and result:
            return result, too_many_tokens_error
        if got_done and not failed and not result:
            _plog("[LLM] Empty stream response, retrying...")
        if too_many_tokens_error:
            return result, too_many_tokens_error
        if attempt < int(max_retries):
            time.sleep(min(10.0, (retry_backoff**attempt)) + random.uniform(0, retry_jitter))
            continue
        return result, too_many_tokens_error
    return result, too_many_tokens_error


_TOOL_CALLS_BLOCK_RE = re.compile(r"<tool_calls_begin>\s*(.*?)\s*</tool_calls_end>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_QWEN35_FUNCTION_RE = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL)
_QWEN35_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _parse_qwen35_arguments(params_text: str) -> Dict[str, Any]:
    arguments: Dict[str, Any] = {}
    for m in _QWEN35_PARAM_RE.finditer(params_text):
        key = m.group(1).strip()
        raw = m.group(2).strip()
        try:
            arguments[key] = json.loads(raw)
        except json.JSONDecodeError:
            arguments[key] = raw
    return arguments


def _param_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _tool_call_to_qwen35(name: str, arguments: Dict[str, Any]) -> str:
    parts = ["<tool_call>", f"<function={name}>"]
    for key, val in arguments.items():
        parts.extend([f"<parameter={key}>", _param_text(val), f"</parameter>"])
    parts.extend(["</function>", "</tool_call>"])
    return "\n".join(parts)


def _tool_calls_block(tool_calls: List[Dict[str, Any]]) -> str:
    blocks = []
    for tc in tool_calls:
        fn = (tc or {}).get("function") or {}
        name = (fn.get("name") or "").strip()
        args = fn.get("arguments") if isinstance(fn.get("arguments"), dict) else {}
        blocks.append(_tool_call_to_qwen35(name, args))
    return "\n\n".join(blocks)


def _assistant_content(reasoning: str, tool_calls: List[Dict[str, Any]], tail: str) -> str:
    parts: List[str] = []
    if reasoning:
        parts.append(f"<think>\n{reasoning}\n</think>")
    if tool_calls:
        parts.append(_tool_calls_block(tool_calls))
    elif tail:
        parts.append(tail)
    return "\n\n".join(parts)


def _parse_tool_calls_from_text(text: str) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    if not text:
        return "", [], "ERROR: No text to parse tool calls"

    tool_calls: List[Dict[str, Any]] = []
    errors: List[str] = []
    scan_text = _TOOL_CALLS_BLOCK_RE.sub(r"\1", text)
    scan_text = scan_text.replace("<tool_calls_begin>", "").replace("</tool_calls_end>", "")

    for m in _TOOL_CALL_RE.finditer(scan_text):
        inner = (m.group(1) or "").strip()
        if not inner:
            errors.append("ERROR: Found empty <tool_call> tag")
            continue
        fm = _QWEN35_FUNCTION_RE.search(inner)
        if not fm:
            errors.append(
                "ERROR: Invalid tool call. Use "
                "<tool_call><function=name><parameter=key>value</parameter></function></tool_call>."
            )
            continue
        name = fm.group(1).strip()
        if not name:
            errors.append("ERROR: Tool call missing function name")
            continue
        tool_calls.append(
            {"function": {"name": name, "arguments": _parse_qwen35_arguments(fm.group(2))}}
        )

    cleaned = _TOOL_CALL_RE.sub("", scan_text)
    return cleaned, tool_calls, "\n".join(errors) if errors and not tool_calls else None


def _has_answer_tag(text: str) -> bool:
    return bool(text and "</answer>" in text)


def _split_think_and_content(completion_text: str) -> Tuple[str, str]:
    s = (completion_text or "").lstrip()
    if s.startswith("<think>"):
        s = s[len("<think>") :]
    if "</think>" in s:
        reasoning, rest = s.split("</think>", 1)
        return reasoning.strip(), rest.lstrip("\n").lstrip()
    return "", s.strip()


def _filter_huggingface(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if "huggingface" not in line.lower())


def _execute_tool(tool_name: str, tool_args: Dict[str, Any], *, filter_huggingface: bool = False) -> str:
    if tool_name == "search":
        result = _search_web(tool_args.get("query"))
        return _filter_huggingface(result) if filter_huggingface and isinstance(result, str) else result
    if tool_name in ("visit", "visit_web"):
        timeout = int(os.getenv("VISIT_SERVER_TIMEOUT", 300))
        result_box: Dict[str, Any] = {}

        def _run_visit() -> None:
            result_box["value"] = asyncio.run(
                _visit_web(tool_args.get("url"), tool_args.get("goal"))
            )

        t = threading.Thread(target=_run_visit, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            return f"[visit] Timeout after {timeout} seconds."
        return str(result_box.get("value", "[visit] Failed to visit."))
    return "Unknown tool or call tool with incorrect format."


def _render_prompt(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], *, add_generation_prompt: bool) -> str:
    return template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=True,
    )


def call_llm_with_tool(item: Dict[str, Any], args, *, return_metrics: bool = False, return_trace: bool = False):
    query = item["query"]
    base_url = os.getenv("ABSEEKER_BASE_URL", "")
    if not base_url:
        raise ValueError("ABSEEKER_BASE_URL environment variable is required")
    completions_url = _normalize_completions_url(base_url)
    model_name = os.getenv("ABSEEKER_MODEL", "")
    if not model_name:
        raise ValueError("ABSEEKER_MODEL environment variable is required")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    trace: List[Dict[str, Any]] = []
    step_num = 0
    tool_count = 0
    tool_count_max = int(getattr(args, "tool_count_max", 200))
    local_max_tokens = int(getattr(args, "max_tokens", 16384))
    _GEN_PROMPT = "<|im_start|>assistant\n"
    disable_tools = False
    pending_assistant_prefix = ""
    _final_answer_retries = 0
    _parse_fail_streak = 0
    finish_reason = "unknown"
    error_info: Optional[Dict[str, Any]] = None
    context_full = False

    while True:
        tools_for_render: List[Dict[str, Any]] = []
        injected_prefix = pending_assistant_prefix
        if pending_assistant_prefix:
            prompt_text = _render_prompt(messages, tools_for_render, add_generation_prompt=False) + _GEN_PROMPT + injected_prefix
        else:
            prompt_text = _render_prompt(messages, tools_for_render, add_generation_prompt=True)
        pending_assistant_prefix = ""

        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "max_tokens": local_max_tokens,
            "temperature": float(getattr(args, "temperature", 0.7)),
            "top_p": float(getattr(args, "top_p", 0.95)),
            "stream": True,
            "skip_special_tokens": False,
        }
        # _plog(f"=== LLM Request {step_num + 1} ===", flush=True)
        # _plog(prompt_text, flush=True)
        # _plog(flush=True)
        completion_text, too_many_tokens_error = get_stream_response(
            completions_url, payload, print_stream=bool(getattr(args, "print_stream", False))
        )

        if too_many_tokens_error and not completion_text:
            finish_reason = "context_token_limit"
            context_full = True
            break

        if not completion_text and not too_many_tokens_error:
            finish_reason = "empty_response"
            break

        if too_many_tokens_error:
            new_local_max = int(local_max_tokens * 0.9)
            if new_local_max > 2048 and new_local_max < local_max_tokens:
                local_max_tokens = new_local_max
                continue
            if 128 < new_local_max <= 2048 and new_local_max < local_max_tokens:
                if not pending_assistant_prefix:
                    pending_assistant_prefix = "I have used too many tokens, so I will conclude my answer.\n"
                disable_tools = True
                local_max_tokens = new_local_max
                continue
            completion_text = (
                _GEN_PROMPT
                + "<think>\n\n</think>\n\n\n<answer>\nThe max context length has been reached.</answer><|im_end|>\n"
            )
            too_many_tokens_error = False

        if injected_prefix:
            completion_text = _GEN_PROMPT + injected_prefix + completion_text
        step_num += 1

        _plog(f"=== LLM Response {step_num} ===", flush=True)
        _plog(completion_text, flush=True)
        _plog(flush=True)

        reasoning_content, content_raw = _split_think_and_content(
            completion_text.replace("<|im_end|>", "").replace(_GEN_PROMPT, "")
        )
        has_answer = _has_answer_tag(content_raw)
        _, tool_calls, parse_error = _parse_tool_calls_from_text(content_raw)
        if disable_tools and tool_calls:
            tool_calls = []

        body = re.sub(r"<think>\s*.*?\s*</think>\s*", "", content_raw, flags=re.DOTALL).strip()
        hist_content = _assistant_content(reasoning_content, tool_calls, body)
        messages.append({"role": "assistant", "content": hist_content})
        if return_trace:
            trace.append(
                {
                    "step": step_num,
                    "type": "model_message",
                    "content": {"content": hist_content, "tool_calls": tool_calls},
                }
            )

        if has_answer:
            finish_reason = "answer"
            break

        if not tool_calls:
            tool_count += 1
            if tool_count >= tool_count_max and not disable_tools:
                finish_reason = "tool_count_limit_no_tool_or_answer"
                error_info = {
                    "type": "NoToolCallOrAnswerLimit",
                    "message": f"Stopped after tool_count_max={tool_count_max}.",
                }
                break
            if disable_tools:
                _final_answer_retries += 1
                if _final_answer_retries >= 3:
                    finish_reason = "no_final_answer_after_tools_disabled"
                    error_info = {"type": "NoFinalAnswerAfterToolsDisabled", "message": "No final answer."}
                    break
                pending_assistant_prefix = (
                    "Tools are disabled. Do NOT output any <tool_call> tags. "
                    "Provide my final answer as plain text immediately."
                )
                messages.pop()
                continue
            _parse_fail_streak += 1
            error_content = parse_error or (
                "ERROR: Tool call parsing failed. Use "
                "<tool_call><function=name><parameter=key>value</parameter></function></tool_call>."
            )
            _plog(f"\033[91m {error_content}\033[0m")
            if _parse_fail_streak >= 3:
                finish_reason = "giveup"
                error_info = {
                    "type": "ParseFailGiveUp",
                    "message": "Gave up after 3 consecutive parse failures.",
                }
                print("giveup", flush=True)
                _plog("giveup", flush=True)
                break
            messages.pop()
            messages.append({"role": "user", "content": f"<tool_response>\n{error_content}\n</tool_response>"})
            if return_trace:
                trace.append({"step": step_num, "type": "tool_response", "content": {"content": error_content}})
            continue

        _parse_fail_streak = 0
        for tc in tool_calls:
            fn = (tc or {}).get("function") or {}
            tool_name = (fn.get("name") or "").strip()
            tool_args = fn.get("arguments") or {}
            if not isinstance(tool_args, dict):
                tool_args = {}
            try:
                tool_output = _execute_tool(
                    tool_name, tool_args, filter_huggingface=bool(getattr(args, "filter_huggingface", False))
                )
            except Exception as e:
                tool_output = f"Error during tool execution: {type(e).__name__}: {e}"
            try:
                _print_tool_call(tool_name or "unknown", tool_args, tool_output)
            except Exception:
                pass
            tool_count += 1
            messages.append({"role": "user", "content": f"<tool_response>\n{tool_output}\n</tool_response>"})
            if return_trace:
                trace.append({"step": step_num, "type": "tool_call", "content": {"tool_name": tool_name, "tool_args": tool_args}})
                trace.append({"step": step_num, "type": "tool_response", "content": {"tool_name": tool_name, "tool_response": tool_output}})
            if tool_count >= tool_count_max:
                if not pending_assistant_prefix:
                    pending_assistant_prefix = (
                        "I have reached the tool usage limit. Do NOT call any more tools. "
                        "Provide my final answer as plain text NOW."
                    )
                disable_tools = True
                break

    full_traj = _render_prompt(messages, tools_for_render, add_generation_prompt=False)
    metrics = {"tool_calls": tool_count, "context_chars": len(full_traj), "finish_reason": finish_reason}
    if context_full:
        metrics["context_full"] = True
    if error_info is not None:
        metrics["error"] = error_info
    if return_metrics:
        return (full_traj, metrics, trace) if return_trace else (full_traj, metrics)
    return (full_traj, trace) if return_trace else full_traj


def _get_last_assistant_answer_from_messages(full_traj: str) -> str:
    if not full_traj:
        return ""
    parts = full_traj.split("<|im_start|>assistant")
    if len(parts) < 2:
        return full_traj.strip()
    last = parts[-1]
    if "<|im_end|>" in last:
        last = last.split("<|im_end|>", 1)[0]
    last = _TOOL_CALLS_BLOCK_RE.sub("", last)
    last = _TOOL_CALL_RE.sub("", last)
    last = re.sub(r"<tool_response>.*?</tool_response>", "", last, flags=re.DOTALL)
    if "</think>" in last:
        last = last.split("</think>", 1)[-1]
    m = re.search(r"<answer>(.*?)</answer>", last, re.DOTALL)
    if m:
        return m.group(1).strip()
    return last.strip()


def _estimate_tokens_from_chars(n_chars: int) -> int:
    return max(1, int(n_chars / 4))


def solve_query_with_tools(
    query: str,
    *,
    max_tokens: int = 16384,
    tool_count_max: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.95,
    print_stream: bool = False,
    filter_huggingface: bool = False,
    return_full_traj: bool = True,
    return_trace: bool = True,
) -> Dict[str, Any]:
    start = time.time()
    args = argparse.Namespace(
        max_tokens=int(max_tokens),
        tool_count_max=int(tool_count_max),
        temperature=float(temperature),
        top_p=float(top_p),
        print_stream=bool(print_stream),
        filter_huggingface=bool(filter_huggingface),
    )
    item = {"query": query}
    if return_trace:
        full_traj, metrics, trace = call_llm_with_tool(item, args, return_metrics=True, return_trace=True)
    else:
        full_traj, metrics = call_llm_with_tool(item, args, return_metrics=True, return_trace=False)
        trace = []

    elapsed = time.time() - start
    answer = _get_last_assistant_answer_from_messages(full_traj)
    result: Dict[str, Any] = {
        "answer": answer,
        "tool_calls": int(metrics.get("tool_calls", 0)),
        "elapsed_seconds": elapsed,
        "context_chars": int(metrics.get("context_chars", len(full_traj))),
        "context_est_tokens": _estimate_tokens_from_chars(int(metrics.get("context_chars", len(full_traj)))),
        "finish_reason": metrics.get("finish_reason", "unknown"),
    }
    if metrics.get("context_full"):
        result["context_full"] = True
    if metrics.get("finish_reason") == "empty_response":
        result["empty_response"] = True
    if metrics.get("error") is not None:
        result["error"] = metrics.get("error")
        result["answer"] = ""
    if return_full_traj:
        result["full_traj"] = full_traj
    if return_trace:
        result["trace"] = trace
    return result
