#!/usr/bin/env python3
"""
Score LLM trajectory steps based on recovered ground truth.
For each ID, reads the recovered clue and the trajectory to score.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Any

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from dotenv import load_dotenv

load_dotenv(dotenv_path=PROJECT_ROOT / "configs" / ".env")

from visit_web import call_server

CALL_SERVER_TIMEOUT_SECONDS = 360
CALL_SERVER_MAX_RETRIES = 3


def load_ground_truth_from_recovered(recovered_data: Dict[str, Any]) -> Dict[str, Any]:
    """Load ground truth from recovered data."""
    return {
        "id": recovered_data["id"],
        "final_answer": recovered_data.get("recovered", ""),
    }


def _strip_tool_response(content: str) -> str:
    """Remove <tool_response>...</tool_response> wrapper if present."""
    m = re.search(r'<tool_response>\s*(.*?)\s*</tool_response>', content, re.DOTALL)
    return m.group(1).strip() if m else content


def _parse_assistant_content(content: str) -> tuple:
    """Parse assistant content into (reasoning, tool_call). Extract <think>, content after think, and <tool_call>."""
    reasoning = ""
    tool_call = ""
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    tool_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', content, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        # Content after </think> but before tool_call (or to end) is final_answer, include in reasoning
        after_think_end = think_match.end()
        before_tool = tool_match.start() if tool_match else len(content)
        if before_tool > after_think_end:
            rest = content[after_think_end:before_tool].strip()
            if rest:
                reasoning = reasoning + "\n\n" + rest
    if tool_match:
        tool_call = tool_match.group(1).strip()
    if not reasoning:
        if tool_match:
            reasoning = content[: tool_match.start()].strip()
        else:
            reasoning = content
    return reasoning, tool_call


def build_steps_from_sft_messages(traj_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从 sft 格式的 messages 中构造要打分的 step 列表。
    规则：从第一个 user 开始，按 (user, assistant) 成对作为一组 step。
    """
    messages = traj_item.get("messages", [])
    steps: List[Dict[str, Any]] = []

    # 找到第一个 user 的索引，跳过 system 等
    start_idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            start_idx = i
            break
    if start_idx is None:
        return steps

    step_num = 1
    i = start_idx
    while i < len(messages) - 1:
        user_msg = messages[i]
        assistant_msg = messages[i + 1]
        if user_msg.get("role") != "user" or assistant_msg.get("role") != "assistant":
            i += 1
            continue

        user_content = _strip_tool_response(user_msg.get("content", ""))
        asst_content = assistant_msg.get("content", "")
        reasoning, tool_call = _parse_assistant_content(asst_content)
        steps.append(
            {
                "step": step_num,
                "user_content": user_content,
                "standard_reasoning": reasoning,
                "standard_tool_call": tool_call,
            }
        )
        step_num += 1
        i += 2

    return steps


def create_scoring_prompt(step, core_clues, query: str = "", correct_answer: str = ""):
    """Create prompt for LLM to score a step."""
    
    system_prompt = """You are a trajectory scorer. Your task is to evaluate each step in an AI agent's problem-solving trajectory based on the scoring rubric.

Focus on INTERMEDIATE STEPS and KEY ENTITIES. Base your scoring ONLY on the step's content (Context, Reasoning, and Tool Call)."""

    # Format core clues for display
    core_clues_text = "\n".join([
        f"  {i+1}. {clue.get('name', 'unknown')} ({clue.get('type', 'unknown')}): {clue.get('description', '')}"
        for i, clue in enumerate(core_clues)
    ])

    rubric = """
## Scoring Rubric:

Criteria 2-4 apply only to clues listed in "Core Clues" below. Each step starts from base score 1.0. Multiple criteria may apply in one step.

1. **Baseline (1.0)**:
   - Reasonable progress without an obvious error gets the base score 1.0.
   - Example: general searches or exploration without major mistakes.

2. **Finds or verifies a correct clue (+0.8)**:
   - The step discovers, investigates, or verifies a clue that matches the core clue set.

3. **Correctly rules out a wrong candidate (+0.4)**:
   - The step correctly excludes, dismisses, or identifies as wrong an irrelevant or incorrect candidate.

4. **Incorrectly dismisses a correct clue (-0.8)**:
   - The step wrongly excludes or dismisses a clue that is correct according to the core clues.
   - Exploring wrong candidates does NOT count as this; only incorrectly rejecting a correct clue does.

5. **Correct final answer (+1.0)**:
   - ONLY when the step explicitly gives the final answer (e.g. in <answer> tags or a clear final conclusion) and it matches the Correct Answer below.
   - Minor wording or formatting differences still count as correct.

6. **Wrong final answer (-1.0)**:
   - ONLY when the step explicitly submits an incorrect final answer.
   - If criterion 6 applies, do NOT apply criterion 5.

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
    "score": <float between 0.0 and 2.0; start from 1.0, add/subtract applied deltas, then clip to [0, 2.0]>,
    "criteria": <array of int 1-6, the rubric criterion numbers that apply, e.g. [1] or [1, 2, 5]>,
    "explanation": "<brief explanation. When applying criteria, MUST name the specific clue(s)/entity, e.g. 'clue X', 'entity Y'>"
}}

Important: Return ONLY valid JSON, no other text.
""".format(
        core_clues_text=core_clues_text,
        query=query,
        correct_answer=correct_answer,
        step_num=step.get('step', 'N/A'),
        user_content=step.get('user_content', ''),
        reasoning=step.get('standard_reasoning', ''),
        action=str(step.get('standard_tool_call', 'N/A'))
    )
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rubric}
    ]


def _extract_json(content: str) -> str:
    """Extract JSON from LLM response."""
    try:
        json.loads(content)
        return content
    except json.JSONDecodeError:
        left = content.find("{")
        right = content.rfind("}")
        if left != -1 and right != -1 and left <= right:
            return content[left : right + 1]
        return content


async def score_step_async(
    step, core_clues, query: str, correct_answer: str, logger
):
    """Score a single step using call_server with message list."""
    prompt_msgs = create_scoring_prompt(step, core_clues, query, correct_answer)

    for attempt in range(1, CALL_SERVER_MAX_RETRIES + 1):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(call_server, prompt_msgs),
                timeout=CALL_SERVER_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "call_server timeout at step %s (attempt %d/%d, timeout=%ss)",
                step.get("step", "N/A"),
                attempt,
                CALL_SERVER_MAX_RETRIES,
                CALL_SERVER_TIMEOUT_SECONDS,
            )
            if attempt >= CALL_SERVER_MAX_RETRIES:
                raise RuntimeError(
                    f"call_server timeout after {CALL_SERVER_MAX_RETRIES} attempts"
                )
            continue
        content = _extract_json(response or "")
        try:
            result = json.loads(content)
            score = float(result.get("score", 1.0))
            explanation = result.get("explanation", "")
            criteria = result.get("criteria", []) or []
            if not isinstance(criteria, list):
                criteria = [criteria]
            score = max(0.0, min(2.0, score))
            return score, explanation, criteria
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        
    # print(content)
    score, explanation, criteria = 1.0, "parse failed", []
    score = max(0.0, min(2.0, score))
    return score, explanation, criteria


async def score_single_id_async(
    ID: int, recover_dir: Path, traj_agg: Dict[int, Dict[str, Any]], output_dir: Path, logger
):
    """Score steps for a single ID."""
    recover_path = recover_dir / f"{ID}.json"
    if not recover_path.exists():
        alt_path = recover_dir / f"{ID}_format.json"
        if alt_path.exists():
            recover_path = alt_path
        else:
            print(f"Skipping item {ID} (recover file not found)")
            return
    output_path = output_dir / f"scored_result_{ID}.json"
    if output_path.exists():
        print(f"Skipping item {ID} (result already exists)")
        return

    with open(recover_path, "r", encoding="utf-8") as f:
        recovered_data = json.load(f)

    if {"full_conversation", "recovered", "id"}.issubset(recovered_data.keys()):
        ground_truth = load_ground_truth_from_recovered(recovered_data)
    else:
        ground_truth = {"id": ID, "final_answer": "", "reasoning": ""}
    core_clues = recovered_data.get("key_clues", [])
    item = traj_agg[ID]
    trajectory = build_steps_from_sft_messages(item)
    gt_answer = str(item.get("gt", ground_truth.get("final_answer", ""))).strip()
    trajectory_correctness = item.get("trajectory correctness", item.get("trajectory_correctness", ""))

    logger.info("[ID %s] Loaded %d core clues, %d steps to score", ID, len(core_clues), len(trajectory))

    scored_steps = []
    for i, step in enumerate(trajectory):
        step_num = step.get("step", i + 1)
        logger.info("[ID %s] Scoring step %d/%d (step %s)...", ID, i + 1, len(trajectory), step_num)

        if not core_clues:
            score, explanation, criteria = 1.0, "empty key_clues, default score 1.0", [1]
        else:
            try:
                score, explanation, criteria = await score_step_async(
                    step,
                    core_clues,
                    query=item.get("query", ""),
                    correct_answer=item.get("gt", ""),
                    logger=logger,
                )
            except RuntimeError as ex:
                print(f"Skipping item {ID} ({ex})", flush=True)
                logger.warning("[ID %s] Skipped due to timeout: %s", ID, ex)
                return

        scored_steps.append({
            "step": step_num,
            "score": score,
            "criteria": criteria,
            "explanation": explanation,
            "original_reasoning": step.get("standard_reasoning", ""),
            "context": step.get("user_content", ""),
            "tool_call": step.get("standard_tool_call", ""),
        })

        logger.info("[ID %s]   Score: %.3f", ID, score)

        scores = [s["score"] for s in scored_steps]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0

    scores = [s["score"] for s in scored_steps]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0
    output = {
        "id": ground_truth["id"],
        "trajectory correctness": trajectory_correctness,
        "correct_answer": gt_answer,
        "total_steps": len(trajectory),
        "statistics": {"average_score": avg_score, "min_score": min_score, "max_score": max_score},
        "steps": scored_steps,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved result for item {ID}")
    logger.info("[ID %s] Scoring complete! Average score: %.3f, Results saved to %s", ID, avg_score, output_path)


def main():
    """Main function to score steps."""
    parser = argparse.ArgumentParser(description="Score LLM trajectory steps based on recovered ground truth")
    parser.add_argument("--ids", type=str, required=True, help="Comma-separated IDs to score")
    parser.add_argument("--recover_dir", type=str, default="results/example_recover", help="Directory containing recovered files")
    parser.add_argument("--traj_file", type=str, required=True, help="Aggregated trajectory JSON file (sft format)")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for scored results")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes for concurrent scoring")
    args = parser.parse_args()
    
    # File paths
    recover_dir = PROJECT_ROOT / args.recover_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse IDs: if empty, run all IDs found in recover_dir/*.json
    id_set = {int(x.strip()) for x in args.ids.split(",") if x.strip()}
    if id_set:
        ids = sorted(id_set)
    else:
        ids_found = set()
        for p in recover_dir.glob("*.json"):
            stem = p.stem
            if stem.isdigit():
                ids_found.add(int(stem))
            elif stem.endswith("_format") and stem[:-7].isdigit():
                ids_found.add(int(stem[:-7]))
        ids = sorted(ids_found)

    IDS_SLICE = slice(0, 10)  # 改这里，如 slice(20, 40)；slice(None) = 全部
    # ids = ids[IDS_SLICE]

    # 加载聚合轨迹文件（sft 格式），按 id 索引
    traj_file_path = PROJECT_ROOT / args.traj_file
    with open(traj_file_path, "r", encoding="utf-8") as f:
        traj_list = json.load(f)
    traj_agg = {int(item["id"]): item for item in traj_list}

    logger = logging.getLogger("score_steps")
    logger.setLevel(logging.INFO)

    asyncio.run(
        _run_all_async(
            ids, recover_dir, traj_agg, output_dir, logger, max_concurrency=max(1, args.num_processes)
        )
    )
    print("All entries have finished processing.", flush=True)


async def _run_all_async(ids, recover_dir, traj_agg, output_dir, logger, max_concurrency=1):
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def run_one(ID):
        async with semaphore:
            await score_single_id_async(ID, recover_dir, traj_agg, output_dir, logger)

    await asyncio.gather(*(run_one(ID) for ID in ids))


if __name__ == "__main__":
    main()
