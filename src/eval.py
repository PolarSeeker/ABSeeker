import argparse
import asyncio
import glob
import json
import os
import sys

from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, "configs", ".env"))

from tools.visit_web import call_server


def extract_answer(standard_response):
    if not standard_response:
        return ""
    end = standard_response.rfind("</answer>")
    if end == -1:
        return ""
    start = standard_response.rfind("<answer>", 0, end)
    if start == -1:
        return ""
    return standard_response[start + len("<answer>") : end].strip()


def get_last_answer(result_file):
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            raw = f.read()
        if not raw.strip():
            return "", 0
        data = json.loads(raw)
    except Exception:
        return "", 0

    if isinstance(data, dict):
        answer = (data.get("answer") or "").strip()
        if not answer:
            answer = extract_answer(data.get("full_traj", ""))
        return answer, int(data.get("tool_calls", 0) or 0)

    if not isinstance(data, list):
        return "", 0

    if data and ("total_tokens_consumed" in data[-1] or "total_prompt_tokens" in data[-1]):
        total_steps = len(data) - 1
    else:
        total_steps = len(data)

    for item in reversed(data):
        if "standard_response" in item:
            answer = extract_answer(item["standard_response"])
            if answer:
                return answer, total_steps
        if "response" in item:
            answer = extract_answer(item["response"])
            if answer:
                return answer, total_steps
    return "", total_steps


def load_dataset(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return {item["id"]: (item["gt"], item["query"]) for item in dataset}


async def judge_answer(pred, gt, query):
    prompt = f"""You are an evaluator. Your ONLY task is to decide: does the Predicted Answer refer to the SAME entity or value as the Ground Truth?

- Compare only the main answer (e.g. the name, number, or key fact). Ignore any extra explanation or verification in the Predicted Answer.
- If the core answer matches or clearly refers to the same thing as the Ground Truth, set correct: true. Minor wording/format differences are still correct.
- If the Predicted Answer differs from the Ground Truth only by a very small surface variation (e.g. singular/plural, tiny typo, punctuation, spacing, capitalization), treat it as correct as long as it clearly refers to the same entity or value.
- If the Predicted Answer includes the Ground Truth answer but also mentions other possibilities/extra content, treat it as correct as long as the Ground Truth is clearly stated.
- If the Predicted Answer expresses the same entity or value in a different language (e.g. translation of a name or title) but clearly points to the same thing as the Ground Truth, you must also treat it as correct.
- If the Predicted Answer mentions an alternative name or alias that refers to the same entity as the Ground Truth, treat it as correct.
- Do NOT set correct: false just because the predicted answer's reasoning or verification has an error. We only care whether the final answer matches the ground truth.

Question: {query}

Ground Truth: {gt}

Predicted Answer: {pred}

Output only a JSON object with this format:
{{"correct": true/false, "reason": "brief explanation"}}"""

    msgs = [{"role": "user", "content": prompt}]
    response = await asyncio.to_thread(call_server, msgs)

    try:
        result = json.loads(response)
        return result.get("correct", False), result.get("reason", "")
    except Exception:
        left = response.find("{")
        right = response.rfind("}")
        if left != -1 and right != -1:
            try:
                result = json.loads(response[left : right + 1])
                return result.get("correct", False), result.get("reason", "")
            except Exception:
                pass
        return False, "Failed to parse LLM response"


async def process_item(result_file, dataset, semaphore, pbar, existing_results=None):
    async with semaphore:
        filename = os.path.basename(result_file)
        item_id = filename.replace("result_", "").replace(".json", "")
        item_id = str(int(item_id))

        if existing_results and item_id in existing_results:
            pbar.update(1)
            return existing_results[item_id]

        pred, total_steps = get_last_answer(result_file)
        gt, query = dataset[item_id]

        if not pred:
            pbar.update(1)
            return {
                "id": item_id,
                "correct": False,
                "model_answer": "",
                "gt": gt,
                "reason": "No answer found",
                "steps": total_steps,
            }

        is_correct, reason = await judge_answer(pred, gt, query)
        pbar.update(1)
        return {
            "id": item_id,
            "correct": is_correct,
            "model_answer": pred,
            "gt": gt,
            "reason": reason,
            "steps": total_steps,
        }


async def main_async(args):
    result_dir = args.result_dir
    dataset_path = args.dataset_path
    num_workers = args.num_workers
    item_id = args.item_id

    dataset = load_dataset(dataset_path)

    if item_id:
        result_file = os.path.join(result_dir, f"result_{item_id}.json")
        if os.path.exists(result_file):
            result_files = [result_file]
        else:
            print(f"Error: Result file not found: {result_file}")
            return
    else:
        result_files = sorted(glob.glob(os.path.join(result_dir, "result_*.json")))

    existing_results = {}
    skip_existing = args.skip_existing.lower() == "true"
    if skip_existing:
        eval_results_file = os.path.join(result_dir, "eval_results.json")
        if os.path.exists(eval_results_file):
            try:
                with open(eval_results_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if "results" in existing_data:
                        existing_results = {r.get("id"): r for r in existing_data["results"]}
                        print(f"Found {len(existing_results)} existing results, will skip if already evaluated")
            except Exception as e:
                print(f"Warning: Failed to load existing results: {e}")
    else:
        print("Skip existing is disabled, will re-evaluate all items")

    semaphore = asyncio.Semaphore(num_workers)
    with tqdm(total=len(result_files), desc="Evaluating") as pbar:
        tasks = [
            process_item(result_file, dataset, semaphore, pbar, existing_results)
            for result_file in result_files
        ]
        results = await asyncio.gather(*tasks)

    results.sort(key=lambda x: int(x.get("id", 0)) if str(x.get("id", "")).isdigit() else 0)
    avg_steps = sum(r.get("steps", 0) for r in results) / len(results) if results else 0

    total_tokens_consumed = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for rf in result_files:
        try:
            with open(rf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data and isinstance(data[-1], dict):
                last = data[-1]
                total_tokens_consumed += last.get("total_tokens_consumed", 0)
                total_prompt_tokens += last.get("total_prompt_tokens", 0)
                total_completion_tokens += last.get("total_completion_tokens", 0)
        except Exception:
            continue

    PROMPT_PRICE_PER_M = 0.26
    COMPLETION_PRICE_PER_M = 0.35
    total_cost = (total_prompt_tokens / 1e6 * PROMPT_PRICE_PER_M) + (
        total_completion_tokens / 1e6 * COMPLETION_PRICE_PER_M
    )
    n_questions = len(results) if results else 1
    avg_cost_per_question = total_cost / n_questions

    token_summary = {
        "total_tokens_consumed": total_tokens_consumed,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_cost_usd": round(total_cost, 4),
        "avg_cost_per_question_usd": round(avg_cost_per_question, 4),
    }

    correct_count = sum(1 for r in results if r.get("correct", False))
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"\nAccuracy: {correct_count}/{total_count} = {accuracy:.4f}")
    print(f"Average Steps: {avg_steps:.2f}")

    output_file = os.path.join(result_dir, "eval_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "token_summary": token_summary,
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_count": total_count,
                "avg_steps": avg_steps,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "result", "example_react"),
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "dataset", "example.json"),
    )
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--item_id", type=str, default=None)
    parser.add_argument("--skip_existing", type=str, default="true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
