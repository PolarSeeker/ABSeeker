"""
recover_intermediate_steps_deepseek.py
Recover clue 推理脚本，主模型后端由环境变量控制（与 run_recover_clue.sh 一致）：
  VISIT_SERVER=local|oss|ds|gpugeek
  SUMMARY_MODEL_NAME=<model on that backend>
visit_web 摘要同样读取上述环境变量（见 tools/visit_web.py）。
"""
import argparse
import asyncio
import json
import logging
import os
import random
import sys
import tiktoken
import openai
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from utils.utils import build_tool_suite, process_tool_call, message_to_dict

load_dotenv(dotenv_path=PROJECT_ROOT / "configs" / ".env")

# ── LLM 后端（与 tools/visit_web.py 对齐，从环境变量读取）────────────────────
from tools.visit_web import (  # noqa: E402
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_URL,
    GPU_GEEK_API_KEY,
    GPU_GEEK_MODEL,
    GPU_GEEK_URL,
    GPT_OSS_API_KEY,
    GPT_OSS_MODEL,
    GPT_OSS_URL,
    SUMMARY_APIS,
)

DS_URL = DEEPSEEK_URL
DS_API_KEY = DEEPSEEK_API_KEY
DS_DEFAULT_MODEL = DEEPSEEK_MODEL

GPUGEEK_URL = GPU_GEEK_URL
GPUGEEK_API_KEY = GPU_GEEK_API_KEY
GPUGEEK_MODEL = GPU_GEEK_MODEL

OSS_URL = GPT_OSS_URL
OSS_API_KEY = GPT_OSS_API_KEY
OSS_MODEL = GPT_OSS_MODEL

LOCAL_APIS = SUMMARY_APIS
LOCAL_DEFAULT_MODEL = os.getenv("LOCAL_LLM_MODEL", "gpt-oss-120b")

MAX_LOOPS = 60
MAX_CONTEXT_TOKENS = 160000

SYSTEM_PROMPT = (
    "You are an expert investigator. Use `search_web` and `visit_web` to gather evidence. "
    "Do NOT provide the final answer directly. Extract intermediate clues (dates, places, events) "
    "from the question, search for them first, then piece together to verify the answer."
)


def build_prompt(query: str, answer: str) -> str:
    return (
        "Reconstruct cite-backed steps that prove the answer is correct.\n\n"
        f"Question: {query}\n"
        f"Known Answer: {answer}\n\n"
        "Your task: Find the intermediate clues that lead to the correct answer. "
        "These clues are usually conditions in the question (events, dates, places, entities, etc.). "
        "Search for them first, then verify the answer. "
        "You MUST explicitly mention these intermediate clues in your final answer and explain how they support the answer."
    )


def _resolve_llm_config() -> Tuple[str, str, str, bool]:
    """返回 (api_key, base_url, model, enable_thinking)。"""
    server = os.getenv("VISIT_SERVER", "local").lower()
    model_override = os.getenv("SUMMARY_MODEL_NAME", "").strip()

    if server == "ds":
        return DS_API_KEY, DS_URL, model_override or DS_DEFAULT_MODEL, True
    if server == "gpugeek":
        return GPUGEEK_API_KEY, GPUGEEK_URL, model_override or GPUGEEK_MODEL, True
    if server == "oss":
        return OSS_API_KEY, OSS_URL, model_override or OSS_MODEL, False
    # local
    url, api_key = random.choice(LOCAL_APIS)
    return api_key, url, model_override or LOCAL_DEFAULT_MODEL, False


def _make_llm_client() -> openai.AsyncOpenAI:
    """每次新建，不走代理，不共享缓存（多进程安全）。"""
    api_key, base_url, _, _ = _resolve_llm_config()
    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=600.0,
        max_retries=3,
    )


async def _call_llm(
    client: openai.AsyncOpenAI,
    messages: List[Dict],
    tool_schemas: Optional[List[Dict]] = None,
) -> openai.types.chat.ChatCompletion:
    _, _, model, enable_thinking = _resolve_llm_config()
    kwargs: Dict = {
        "model": model,
        "messages": messages,
    }
    if enable_thinking:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    if tool_schemas:
        kwargs["tools"] = tool_schemas
        kwargs["tool_choice"] = "auto"
    return await client.chat.completions.create(**kwargs)


# ── ReAct 循环
async def callllm(
    prompt: str,
    system_prompt: str,
    tools: List[str],
    tools_dir: str,
    logger: logging.Logger,
    session: aiohttp.ClientSession,
    execution_cwd: str = ".",
    log_path: Optional[Path] = None,
) -> Tuple[List[Dict], str]:
    messages: List[Dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
    ]
    tool_schemas, callable_tools = await build_tool_suite(tools_dir, tools)
    encoding = tiktoken.get_encoding("cl100k_base")

    def _token_count() -> int:
        return len(encoding.encode(
            json.dumps(messages + (tool_schemas or []), ensure_ascii=False),
            allowed_special=set(), disallowed_special=()
        ))

    client = _make_llm_client()
    loop_count = 0

    try:
        while loop_count < MAX_LOOPS:
            append_log(f"=== Loop {loop_count + 1} ===", log_path)
            # 接近上下文限制时强制结束
            if _token_count() >= MAX_CONTEXT_TOKENS * 0.9:
                messages.append({"role": "user", "content": "You have used too many tools, now conclude your answer."})
                completion = await _call_llm(client, messages, tool_schemas)
                # append_log(str(completion), log_path)
                msg_dict = message_to_dict(completion.choices[0].message)
                messages.append(msg_dict)
                break

            completion = await _call_llm(client, messages, tool_schemas)
            # append_log(str(completion), log_path)
            message = completion.choices[0].message
            msg_dict = message_to_dict(message)

            # 没有 tool_calls → 检查是否有实质内容
            if not message.tool_calls:
                actual_content = (msg_dict.get("content") or "").strip()
                has_reasoning = bool((msg_dict.get("reasoning_content") or "").strip())
                # 异常情况：有 reasoning_content 但 content 为空，且没有 tool_calls
                # 说明模型只在思考阶段"想了想下一步"，既没给出答案也没发出工具调用
                if not actual_content and has_reasoning:
                    messages.append(msg_dict)
                    append_log("[reasoning only, no content, no tool_calls - retrying same step]", log_path)
                    continue
                messages.append(msg_dict)
                break

            # 有 tool_calls → 执行工具
            # 每一轮最多重试 3 次：若 LLM 生成的工具参数 JSON 解析失败，重新叫 LLM；
            # 最终仍失败时，把解析错误作为 tool response 反馈给模型，继续下一轮。
            _final_in_retry = False
            for _turn_try in range(3):
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                    if tc.type == "function" and tc.function.name
                ]

                _tc_inputs = []
                _parse_errors = []
                for tc in message.tool_calls:
                    if tc.type != "function" or not tc.function.name:
                        continue
                    tool_args_raw = tc.function.arguments or "{}"
                    try:
                        _tc_inputs.append((tc, json.loads(tool_args_raw)))
                    except json.JSONDecodeError as _parse_err:
                        _parse_errors.append((tc, _parse_err, tool_args_raw))

                if _parse_errors and _turn_try < 2:
                    first_err = _parse_errors[0][1]
                    append_log(
                        f"[Tool JSON parse error (attempt {_turn_try + 1}/3), retrying LLM: {first_err}]",
                        log_path,
                    )
                    completion = await _call_llm(client, messages, tool_schemas)
                    message = completion.choices[0].message
                    msg_dict = message_to_dict(message)
                    if not message.tool_calls:
                        messages.append(msg_dict)
                        append_log("[Final response generated after parse retry]", log_path)
                        _final_in_retry = True
                        break
                    continue

                messages.append(msg_dict)
                for tc, tool_input in _tc_inputs:
                    tool_name = tc.function.name
                    tool_input["task_cwd"] = execution_cwd
                    tool_input["logger"] = logger
                    tool_input["session"] = session
                    tool_result = await process_tool_call(callable_tools, tool_name, tool_input, execution_cwd)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(tool_result)})

                for tc, parse_err, tool_args_raw in _parse_errors:
                    error_msg = (
                        f"[tool_error] Failed to parse tool arguments for {tc.function.name}: "
                        f"{parse_err}. Tool arguments must be valid JSON. Raw arguments: {tool_args_raw}"
                    )
                    append_log(f"[Tool JSON parse error: {parse_err}]", log_path)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": error_msg})

                if _parse_errors:
                    append_log(
                        f"[Tool JSON parse failed after {_turn_try + 1} attempt(s); "
                        "returned error to model and continuing]",
                        log_path,
                    )
                break

            if _final_in_retry:
                break  # 重试中 LLM 已给出最终答案，退出 while 循环

            loop_count += 1
        else:
            # 超过 max_loops
            messages.append({"role": "user", "content": "You have used too many tools, now conclude your answer."})
            completion = await _call_llm(client, messages, tool_schemas)
            append_log(str(completion), log_path)
            msg_dict = message_to_dict(completion.choices[0].message)
            messages.append(msg_dict)
    finally:
        await client.close()

    # 取最后一条没有 tool_calls 的 assistant 消息作为最终答案
    content = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and not msg.get("tool_calls"):
            content = msg.get("content") or msg.get("reasoning_content") or ""
            if content:
                break
    return messages, content


# ── 辅助函数 ─────────────────────────────────────────────────────────
def append_log(msg: str, log_path: Optional[Path]) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def log_completion(entry_id) -> None:
    print(f"Completed {entry_id}", flush=True)


def result_output_path(output_dir: Path, entry_id) -> Path:
    return output_dir / f"{entry_id}.json"


def has_existing_result(output_dir: Path, entry_id) -> bool:
    out_file = result_output_path(output_dir, entry_id)
    return out_file.is_file() and out_file.stat().st_size > 0


def filter_pending_entries(dataset: List[Dict], output_dir: Path) -> List[Dict]:
    pending: List[Dict] = []
    for entry in dataset:
        entry_id = entry.get("id")
        if has_existing_result(output_dir, entry_id):
            print(f"Skipping {entry_id}", flush=True)
        else:
            pending.append(entry)
    return pending


# ── 核心处理函数 ──────────────────────────────────────────────────────
async def recover_steps(
    entry: Dict,
    session: aiohttp.ClientSession,
    logger: logging.Logger,
    log_path: Optional[Path] = None,
) -> Optional[Dict]:
    gt_answer = entry.get("gt")
    if not gt_answer:
        logger.warning("Skipping id=%s: no gt answer.", entry.get("id"))
        return None

    prompt = build_prompt(entry["query"], gt_answer)

    messages, final_answer = await callllm(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        tools=["search_web", "visit_web"],
        tools_dir="./tools",
        logger=logger,
        session=session,
        execution_cwd=".",
        log_path=log_path,
    )

    full_conversation = json.dumps(messages, ensure_ascii=False)
    result = {
        "id": entry["id"],
        "query": entry["query"],
        "gt": gt_answer,
        "recovered": final_answer,
        "full_conversation": full_conversation,
    }

    append_log(f"Final answer: {final_answer}", log_path)

    return result


# ── main ──────────────────────────────────────────────────────────────
def main():
    global MAX_LOOPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path",  type=str, default="dataset/example.json")
    parser.add_argument("--output_dir",    type=str, default="results/example_recover")
    parser.add_argument("--ids",           type=str, default="")
    parser.add_argument("--sample_size",   type=int, default=None)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--max_loops",     type=int, default=60)
    args = parser.parse_args()
    MAX_LOOPS = args.max_loops

    _, _, llm_model, llm_thinking = _resolve_llm_config()
    print(
        f">> llm config: server={os.getenv('VISIT_SERVER', 'local')}, "
        f"model={llm_model}, thinking={llm_thinking}",
        flush=True,
    )

    dataset_path = Path(args.dataset_path)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with dataset_path.open("r", encoding="utf-8") as f:
        dataset: List[Dict] = json.load(f)[0:3000]

    id_set = {x.strip() for x in args.ids.split(",") if x.strip()}
    if id_set:
        dataset = [e for e in dataset if str(e.get("id", "")) in id_set]
    if args.sample_size:
        dataset = dataset[: args.sample_size]

    dataset = filter_pending_entries(dataset, output_dir)
    if not dataset:
        return

    def save_result(result: Dict) -> None:
        out_file = result_output_path(output_dir, result["id"])
        out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log_dir = output_dir / "log"
        append_log(f"Saved result for item {result['id']}.", log_dir / f"{result['id']}.log")
        log_completion(result["id"])

    # Run in a single OS process; use asyncio concurrency as logical workers.
    max_concurrency = max(1, args.num_processes)
    semaphore = asyncio.Semaphore(max_concurrency)
    logger = logging.getLogger("recover_clue")
    logger.setLevel(logging.INFO)

    async def process_entries():
        log_dir = output_dir / "log"
        log_dir.mkdir(parents=True, exist_ok=True)

        async with aiohttp.ClientSession() as session:
            async def run(e):
                async with semaphore:
                    entry_log_path = log_dir / f"{e['id']}.log"
                    # redirect_stdout/redirect_stderr are process-global; avoid them when running concurrent coroutines.
                    # 外层重试只针对真正的异常（网络、API等）；工具参数错误已在每轮内部处理
                    max_retries = 3
                    for attempt in range(1, max_retries + 1):
                        try:
                            result = await recover_steps(e, session, logger, log_path=entry_log_path)
                            return result
                        except Exception as ex:
                            err = str(ex)
                            if attempt < max_retries:
                                retry_msg = f"Retry id={e.get('id')} attempt {attempt + 1}/{max_retries}: {err}"
                                append_log(retry_msg, entry_log_path)
                                print(retry_msg, flush=True)
                                continue
                            fail_msg = f"Give up id={e.get('id')} after {max_retries} attempts: {err}"
                            append_log(fail_msg, entry_log_path)
                            print(fail_msg, flush=True)
                            return None

            tasks = [asyncio.create_task(run(e)) for e in dataset]
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result:
                    save_result(result)

    asyncio.run(process_entries())
    print("All entries have finished processing.", flush=True)


if __name__ == "__main__":
    main()
