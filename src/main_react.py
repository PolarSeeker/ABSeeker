"""ReAct inference via ABSeeker solve_query_with_tools (tools)."""
import argparse
import asyncio
import functools
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)


def load_env(env_path: str) -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if not os.environ.get(key):
                    os.environ[key] = value


load_env(os.path.join(PROJECT_ROOT, "configs", ".env"))

from llm_tool_abseeker import run_with_log_file, solve_query_with_tools

DEFAULT_ABSEEKER_BASE_URL = "http://127.0.0.1:30016/v1"


def setup_abseeker_env(model: str) -> None:
    os.environ.setdefault("ABSEEKER_BASE_URL", DEFAULT_ABSEEKER_BASE_URL)
    os.environ.setdefault("ABSEEKER_MODEL", model)


async def process_item(args, item: dict, result_dir: str) -> None:
    item_id = str(item.get("id", ""))
    question = item["query"]
    result_path = os.path.join(result_dir, f"result_{item_id}.json")
    log_path = os.path.join(result_dir, "logs", f"{item_id}.log")

    if os.path.exists(result_path):
        print(f"Skipping item {item_id} (result already exists)")
        return

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.time()
    print(f">> START id={item_id} query={question[:120]!r}")

    runner = functools.partial(
        run_with_log_file,
        log_path,
        solve_query_with_tools,
        question,
        max_tokens=args.max_tokens,
        tool_count_max=args.tool_count_max,
        temperature=args.temperature,
        top_p=args.top_p,
        print_stream=args.print_stream,
        filter_huggingface=args.filter_huggingface,
        return_full_traj=True,
        return_trace=True,
    )
    try:
        res = await asyncio.to_thread(runner)
    except Exception:
        print(f">> FAILED id={item_id}\n{traceback.format_exc()}")
        raise

    out = {"id": item_id, "query": question, **res, "wall_seconds": time.time() - t0}
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    if out.get("finish_reason") == "giveup":
        print(f">> giveup id={item_id}", flush=True)
    print(
        f">> DONE id={item_id} wall={out['wall_seconds']:.1f}s "
        f"tool_calls={out.get('tool_calls')} finish={out.get('finish_reason')}"
    )


async def main_async(args) -> None:
    load_env(os.path.join(PROJECT_ROOT, "configs", ".env"))
    visit_server = args.visit_server or os.getenv("VISIT_SERVER", "local")
    model = args.model or os.getenv("ABSEEKER_MODEL", "models/example")
    os.environ["VISIT_SERVER"] = visit_server
    setup_abseeker_env(model)
    print(
        f">> abseeker: base_url={os.environ['ABSEEKER_BASE_URL']}, "
        f"model={os.environ['ABSEEKER_MODEL']}"
    )
    print(
        f">> visit config: JINA_API_KEY={'set' if os.getenv('JINA_API_KEY') else 'MISSING'}, "
        f"proxy={os.getenv('VISIT_HTTP_PROXY', 'http://127.0.0.1:6478')}, "
        f"server={visit_server}"
    )

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if args.item_ids:
        ids = set(args.item_ids)
        dataset = [x for x in dataset if str(x.get("id")) in ids]
    if args.limit > 0:
        dataset = dataset[: args.limit]

    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    result_dir = os.path.join(PROJECT_ROOT, "result", f"{args.exp_name}_{dataset_name}_react")
    os.makedirs(result_dir, exist_ok=True)

    sem = asyncio.Semaphore(args.num_workers)

    async def run_one(item):
        async with sem:
            await process_item(args, item, result_dir)

    await asyncio.gather(*[run_one(item) for item in dataset])
    print(f"All {len(dataset)} items processed. Results: {result_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="react")
    parser.add_argument("--dataset", type=str, default="dataset/example.json")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--item_ids", type=str, nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tool_count_max", type=int, default=200)
    parser.add_argument("--print_stream", action="store_true", default=False)
    parser.add_argument("--filter_huggingface", action="store_true", default=True)
    parser.add_argument("--no-filter_huggingface", action="store_false", dest="filter_huggingface")
    parser.add_argument("--visit_server", type=str, default=None, choices=["local", "oss", "ds", "gpugeek"])
    parser.add_argument("--model", type=str, default=None)
    asyncio.run(main_async(parser.parse_args()))
