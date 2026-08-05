"""
ABSeeker batch reward manager.

Uses anchor-clue per-turn scores instead of logprob info-gain with anchor-clue per-turn scores produced
by the agent loop's online LLM judge.  Per-turn scores are read from
  data_item.non_tensor_batch["abseeker_turn_scores"]
which is a list[float] | None written by the agent loop after each rollout.
(Fallback: extra_info["abseeker_turn_scores"] for legacy paths.)

When abseeker_turn_scores is None (agent loop not yet implemented, or single-turn
trajectory), only the final answer score is placed at the last response
token — identical to standard GRPO.  The training run is fully functional
in this degraded mode.

Phase 1: LLM judge (or F1) computes the final-answer outcome score.
         Identical to NaiveBatchRewardManager Phase 1.
Phase 2: abseeker_reward.compute_score maps outcome_score + turn_scores
         into a token-level reward tensor.
"""

import json
import os
import re
from collections import defaultdict

import torch
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.workers.reward_manager import register


@register("abseeker_batch")
class ABSeekerBatchRewardManager:
    """Reward manager for ABSeeker anchor-clue training."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **reward_kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.use_format_penalty = bool(reward_kwargs.get("use_format_penalty", False))
        self.format_penalty_deduction = float(reward_kwargs.get("format_penalty_deduction", 0.2))

    def __call__(self, data: DataProto, return_dict=False, default_batchsize=16, val_type="llm"):
        from verl.utils.reward_score import abseeker_reward as abseeker_module

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        batch_size = min(default_batchsize, len(data))

        # ── Build batched inputs ──────────────────────────────────────────
        batched_inputs = []
        for start_idx in range(0, len(data), batch_size):
            batch = data[start_idx: start_idx + batch_size]

            prompts, responses, ground_truths, data_sources, extra_infos = [], [], [], [], []
            valid_response_lengths = []

            for data_item in batch:
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

                valid_response_lengths.append(valid_response_length)
                prompts.append(prompt_str)
                responses.append(response_str)
                ground_truths.append(data_item.non_tensor_batch["reward_model"]["ground_truth"])
                data_sources.append(data_item.non_tensor_batch.get(self.reward_fn_key, "unknown"))
                extra_infos.append(data_item.non_tensor_batch.get("extra_info", None))

            batched_inputs.append({
                "start_idx": start_idx,
                "valid_response_lengths": valid_response_lengths,
                "prompts": prompts,
                "responses": responses,
                "ground_truths": ground_truths,
                "data_sources": data_sources,
                "extra_infos": extra_infos,
            })

        # ── Phase 1: outcome scores only for samples missing anchor turn_scores ─
        response_idx_dict = {}

        for inputs in batched_inputs:
            start_idx = inputs["start_idx"]
            fallback_idx = []
            for idx in range(len(inputs["responses"])):
                global_idx = start_idx + idx
                extra_info = inputs["extra_infos"][idx] or {}
                turn_scores = extra_info.get("abseeker_turn_scores")
                if turn_scores is None:
                    turn_scores = data[global_idx].non_tensor_batch.get("abseeker_turn_scores")
                if turn_scores is None or (hasattr(turn_scores, '__len__') and len(turn_scores) == 0):
                    fallback_idx.append(idx)

            if not fallback_idx:
                continue

            if val_type == "llm":
                from verl.utils.reward_score import llm_judge
                scores = llm_judge.compute_score_batch(
                    [inputs["prompts"][i] for i in fallback_idx],
                    [inputs["responses"][i] for i in fallback_idx],
                    [inputs["ground_truths"][i] for i in fallback_idx],
                    [inputs["data_sources"][i] for i in fallback_idx],
                    batch_size,
                    data.meta_info.get("validate", False),
                    self.tokenizer,
                )
            else:
                scores = self.compute_score(
                    data_source=[inputs["data_sources"][i] for i in fallback_idx],
                    prompt_str=[inputs["prompts"][i] for i in fallback_idx],
                    solution_str=[inputs["responses"][i] for i in fallback_idx],
                    ground_truth=[inputs["ground_truths"][i] for i in fallback_idx],
                    extra_info=[inputs["extra_infos"][i] for i in fallback_idx],
                    val_type=val_type,
                    batch_size=batch_size,
                    is_valid=data.meta_info.get("validate", False),
                    tokenizer=self.tokenizer,
                )

            for idx, score in zip(fallback_idx, scores):
                global_idx = start_idx + idx
                response_idx = int(inputs["valid_response_lengths"][idx]) - 1
                scalar = float(score) if isinstance(score, (int, float)) else float(score)
                reward_tensor[global_idx, response_idx] = scalar
                response_idx_dict[global_idx] = response_idx

                data_source = inputs["data_sources"][idx]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", inputs["prompts"][idx])
                    print("[response]", inputs["responses"][idx])
                    print("[data_source]", data_source,
                          "[ground_truth]", inputs["ground_truths"][idx])
                    print("[outcome_score]", scalar)

        # Save outcome scores before Phase 2 overwrites.
        outcome_scores = []
        for i in range(len(data)):
            idx = response_idx_dict.get(i, 0)
            outcome_scores.append(reward_tensor[i, idx].item())
        reward_extra_info["outcome_score"] = outcome_scores

        # ── Phase 2: overlay token-level per-turn scores ──────────────────
        # Reads abseeker_turn_scores from extra_info (written by agent loop).
        # Falls back gracefully when the field is absent.
        from verl.utils.reward_score.format_checker import (
            apply_format_deduction_to_turn_scores,
            deduct_outcome_for_answer_format,
        )

        _processed = _skipped = _placed_total = 0
        _fmt_penalty_count = 0
        work_scores_by_idx: dict[int, list[float]] = {}

        for inputs in batched_inputs:
            start_idx = inputs["start_idx"]
            for idx in range(len(inputs["responses"])):
                global_idx = start_idx + idx

                extra_info = inputs["extra_infos"][idx] or {}
                turn_scores = extra_info.get("abseeker_turn_scores")
                if turn_scores is None:
                    turn_scores = data[global_idx].non_tensor_batch.get("abseeker_turn_scores")

                messages = extra_info.get("messages")
                if messages is None:
                    messages = data[global_idx].non_tensor_batch.get("messages")
                turn_boundaries = extra_info.get("turn_boundaries")
                if turn_boundaries is None:
                    turn_boundaries = data[global_idx].non_tensor_batch.get("turn_boundaries")

                outcome_score = outcome_scores[global_idx]

                if turn_scores is None or (hasattr(turn_scores, '__len__') and len(turn_scores) == 0):
                    _skipped += 1
                    orig_len = int(inputs["valid_response_lengths"][idx])
                    if self.use_format_penalty:
                        new_score, penalized = deduct_outcome_for_answer_format(
                            outcome_score,
                            messages,
                            turn_boundaries,
                            self.format_penalty_deduction,
                        )
                        if penalized:
                            _fmt_penalty_count += 1
                            outcome_scores[global_idx] = new_score
                            if orig_len > 0:
                                reward_tensor[global_idx, orig_len - 1] = new_score
                    continue

                _processed += 1
                work_scores = list(turn_scores)
                if self.use_format_penalty:
                    work_scores, n_pen = apply_format_deduction_to_turn_scores(
                        work_scores,
                        messages,
                        turn_boundaries,
                        self.format_penalty_deduction,
                    )
                    _fmt_penalty_count += n_pen
                work_scores_by_idx[global_idx] = [float(s) for s in work_scores]

                orig_len = int(inputs["valid_response_lengths"][idx])
                orig_ids = data.batch["responses"][global_idx][:orig_len].tolist()

                token_scores = abseeker_module.compute_score(
                    solution_str=inputs["responses"][idx],
                    ground_truth=inputs["ground_truths"][idx],
                    data_source=inputs["data_sources"][idx],
                    turn_scores=work_scores,
                    tokenizer=self.tokenizer,
                    is_validation=False,
                    outcome_score=outcome_score,
                    original_token_ids=orig_ids,
                )

                reward_tensor[global_idx, :] = 0.0
                reward_tensor[global_idx, :orig_len] = torch.tensor(
                    token_scores[:orig_len],
                    dtype=reward_tensor.dtype,
                    device=reward_tensor.device,
                )
                _placed_total += int((reward_tensor[global_idx] != 0).sum().item())
                outcome_scores[global_idx] = float(work_scores[-1]) if work_scores else 0.0

        _fmt_status = ""
        if self.use_format_penalty:
            _fmt_status = f", format_deductions={_fmt_penalty_count} (deduct={self.format_penalty_deduction})"
        print(
            f"[A2S-Phase2] processed={_processed} skipped={_skipped} "
            f"total_nz_placed={_placed_total}{_fmt_status}",
            flush=True,
        )

        # ── Outcome accuracy via LLM judge (same proxy as anchor judge) ──
        from concurrent.futures import ThreadPoolExecutor

        from verl.utils.reward_score.abseeker_anchor_judge import judge_answer_correct

        def _extract_answer(resp):
            try:
                m = re.search(r"<answer>(.*?)</answer>", resp.split("assistant\n")[-1], re.DOTALL)
                return m.group(1).strip() if m else ""
            except Exception:
                return ""

        def _extract_query(prompt):
            parts = prompt.split("user\n")
            return parts[1].split("assistant\n")[0].strip() if len(parts) > 1 else prompt

        _acc_args = [("", "", "")] * len(data)
        for inputs in batched_inputs:
            start_idx = inputs["start_idx"]
            for idx in range(len(inputs["responses"])):
                global_idx = start_idx + idx
                _acc_args[global_idx] = (
                    _extract_query(inputs["prompts"][idx]),
                    str(inputs["ground_truths"][idx]),
                    _extract_answer(inputs["responses"][idx]),
                )
        with ThreadPoolExecutor(max_workers=min(32, len(data))) as _ex:
            outcome_accs = list(_ex.map(lambda a: judge_answer_correct(*a), _acc_args))
        reward_extra_info["outcome_acc"] = outcome_accs

        # ── Write scores back to trace JSON files ─────────────────────────
        for inputs in batched_inputs:
            start_idx = inputs["start_idx"]
            for idx in range(len(inputs["responses"])):
                global_idx = start_idx + idx
                extra_info = inputs["extra_infos"][idx] or {}
                trace_path = extra_info.get("_trace_path") or data[global_idx].non_tensor_batch.get("_trace_path")
                if not trace_path or not os.path.exists(trace_path):
                    continue
                try:
                    with open(trace_path) as f:
                        record = json.load(f)
                    record["outcome_score"] = outcome_scores[global_idx]
                    record["outcome_acc"] = outcome_accs[global_idx]
                    if global_idx in work_scores_by_idx:
                        record["turn_scores"] = work_scores_by_idx[global_idx]
                    record["ground_truth"] = inputs["ground_truths"][idx]
                    with open(trace_path, "w") as f:
                        json.dump(record, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[A2S] Failed to write scores to trace {trace_path}: {e}")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
