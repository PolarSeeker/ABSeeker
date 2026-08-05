#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from dotenv import load_dotenv

load_dotenv(dotenv_path=PROJECT_ROOT / "configs" / ".env")

from visit_web import (  # noqa: E402
    DEEPSEEK_MODEL,
    GPU_GEEK_MODEL,
    GPT_OSS_MODEL,
    call_server,
)


def _resolve_model_info() -> Tuple[str, str]:
    server = os.getenv("VISIT_SERVER", "local").lower()
    defaults = {
        "ds": DEEPSEEK_MODEL,
        "gpugeek": GPU_GEEK_MODEL,
        "oss": GPT_OSS_MODEL,
        "local": "gpt-oss-120b",
    }
    model = os.getenv("SUMMARY_MODEL_NAME", defaults.get(server, DEEPSEEK_MODEL))
    return server, model


def _parse_key_clues_response(response: str) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        return json.loads(response), None
    except json.JSONDecodeError as e:
        left = response.find("{")
        right = response.rfind("}")
        if left == -1 or right == -1 or left > right:
            return None, f"JSON parse error: {e}"
        try:
            return json.loads(response[left : right + 1]), None
        except json.JSONDecodeError as e2:
            return None, f"JSON parse error after trimming: {e2}"


def extract_clues_from_data(
    query: str,
    final_answer: str,
    *,
    gt_answer: str = "",
    max_retries: int = 3,
    item_id: Optional[str] = None,
) -> List[Dict]:
    """Extract core clues from query and final answer."""
    prefix = f"id={item_id} " if item_id else ""
    prompt = """
You are given a question and its answer with reasoning.
Extract the key intermediate clues (specific named entities, values, or facts) that are essential for deriving the answer.

Question:
{query}

Answer with reasoning:
{final_answer}

Task:
1. Identify every concrete entity, value, or fact mentioned that serves as an intermediate step toward the answer.
2. For EACH such entity/value/fact, create one clue entry.
3. Also include the ground truth final answer "{gt_answer}" as the last clue in key_clues.

CRITICAL: The "name" field MUST be the ACTUAL proper noun or specific value — NOT a description.

Return JSON:
{{
    "key_clues": [
        {{
            "clue_number": <number>,
            "name": "<specific_name>",
            "type": "<entity_type>",
            "description": "<brief description>"
        }}
    ]
}}

Return ONLY valid JSON.
""".format(
        query=query,
        final_answer=final_answer,
        gt_answer=gt_answer,
    )

    msgs = [{"role": "user", "content": prompt}]
    last_error = "unknown error"

    for attempt in range(1, max_retries + 1):
        response = call_server(msgs, max_retries=2)
        if not response:
            last_error = "empty response from LLM (API error or rate limit)"
            print(
                f"{prefix}extract attempt {attempt}/{max_retries} failed: {last_error}",
                flush=True,
            )
            if attempt < max_retries:
                time.sleep(2 * attempt)
            continue

        result, parse_error = _parse_key_clues_response(response)
        if parse_error:
            last_error = parse_error
            print(
                f"{prefix}extract attempt {attempt}/{max_retries} failed: {parse_error}",
                flush=True,
            )
            print(f"{prefix}raw response preview: {response[:300]!r}", flush=True)
            if attempt < max_retries:
                time.sleep(2 * attempt)
            continue

        key_clues = result.get("key_clues", [])
        if not key_clues and str(final_answer).strip():
            last_error = "LLM returned valid JSON but key_clues is empty"
            print(
                f"{prefix}extract attempt {attempt}/{max_retries} failed: {last_error}",
                flush=True,
            )
            if attempt < max_retries:
                time.sleep(2 * attempt)
            continue

        if any(isinstance(c, dict) and not str(c.get("type", "")).strip() for c in key_clues):
            last_error = "key_clues missing type field"
            print(f"{prefix}extract attempt {attempt}/{max_retries} failed: {last_error}", flush=True)
            if attempt < max_retries:
                time.sleep(2 * attempt)
            continue

        return key_clues

    print(f"{prefix}extract gave up after {max_retries} attempts: {last_error}", flush=True)
    return []


DEFAULT_INPUT_DIR = Path(PROJECT_ROOT) / "results" / "example_recover"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "extracted"


def main():
    import argparse
    from concurrent.futures import ThreadPoolExecutor

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--ids", type=str, default="")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    server, model = _resolve_model_info()
    print(f">> extract config: server={server}, model={model}, max_retries={args.max_retries}", flush=True)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    id_set = {x.strip() for x in args.ids.split(",") if x.strip()}
    output_dir.mkdir(parents=True, exist_ok=True)

    def process(p: Path):
        out = output_dir / f"{p.stem}_format.json"
        if out.exists():
            print(f"skipping {p.name}", flush=True)
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        item_id = str(data.get("id", p.stem))
        recovered = data.get("recovered", "") or ""
        if not str(recovered).strip():
            print(f"skip {p.name}: empty recovered (id={item_id})", flush=True)
            return
        key_clues = extract_clues_from_data(
            data["query"],
            recovered,
            gt_answer=data.get("gt", ""),
            max_retries=args.max_retries,
            item_id=item_id,
        )
        out.write_text(
            json.dumps({"key_clues": key_clues}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"complete {p.name} clues={len(key_clues)}", flush=True)

    if args.dataset_path:
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        id_set = {str(item["id"]) for item in dataset}
        files = [
            input_dir / f"{item_id}.json"
            for item_id in sorted(id_set, key=lambda x: int(x) if x.isdigit() else x)
            if (input_dir / f"{item_id}.json").is_file()
        ]
        print(f">> dataset: {args.dataset_path}, using first {len(dataset)} entries", flush=True)
    else:
        files = sorted(
            p
            for p in input_dir.glob("*.json")
            if not p.stem.endswith("_format") and (not id_set or p.stem in id_set)
        )
    with ThreadPoolExecutor(max_workers=args.num_processes) as ex:
        list(ex.map(process, files))
    print("All entries have finished processing.", flush=True)


if __name__ == "__main__":
    main()
