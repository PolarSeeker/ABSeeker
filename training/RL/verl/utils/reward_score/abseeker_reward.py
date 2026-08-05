"""
ABSeeker token-level reward scoring.

Per-turn scores come from an online LLM judge that evaluates each assistant
turn against pre-computed anchor clues derived from the ground truth.
These scores are stored in extra_info.abseeker_turn_scores by the agent loop.
turn_scores covers ALL turns including the final one.

Token mapping:
  - every turn end token  ← turn_scores[i]   (including final turn)
  - fallback when turn_scores is None: outcome_score (LLM judge) or F1
    is placed at the last response token only.
"""

from verl.utils.reward_score.info_gain import (
    _find_turn_boundaries_by_tokens,
    compute_f1,
)


def compute_score(
    solution_str,
    ground_truth,
    data_source,
    turn_scores,
    tokenizer,
    is_validation=False,
    outcome_score=None,
    original_token_ids=None,
):
    """Map per-turn anchor-clue scores to a token-level reward list.

    Args:
        solution_str: Decoded response string (all turns concatenated).
        ground_truth: Ground-truth answer string.
        data_source: Dataset identifier (used for F1 fallback).
        turn_scores: list[float] | None.  One score per assistant turn
            (including the final turn), produced by the online LLM judge.
            When provided, ALL turns use these scores; outcome_score is ignored.
            None or empty → fallback: only outcome_score / F1 at last token.
        tokenizer: HuggingFace tokenizer, required for turn boundary detection.
        is_validation: When True, also return F1/EM/noformatf1 metrics.
        outcome_score: Fallback final-answer score used only when turn_scores
            is None or empty (e.g. before agent loop implements per-turn scoring).
        original_token_ids: list[int] | None.  Original response token IDs
            (avoids BPE re-encode drift).  Required for correct placement.

    Returns:
        list[float] of length len(original_token_ids), or a dict with
        keys {"f1", "em", "noformatf1", "scores"} when is_validation=True.
    """
    if tokenizer is None:
        raise ValueError("tokenizer cannot be None")

    # Validation metrics always use F1 regardless of turn_scores.
    if is_validation:
        f1_score = compute_f1(solution_str, ground_truth, data_source, val_type="f1")
        em_score = compute_f1(solution_str, ground_truth, data_source, val_type="em")
        noformatf1 = compute_f1(solution_str, ground_truth, data_source, val_type="noformatf1")

    if original_token_ids is None:
        if is_validation:
            return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1, "scores": []}
        return []

    tokens_size = len(original_token_ids)
    scores = [0.0] * tokens_size

    if tokens_size == 0:
        if is_validation:
            return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1, "scores": scores}
        return scores

    # ── No per-turn scores: fallback to outcome_score / F1 ───────────────
    if not turn_scores:
        fallback = outcome_score if outcome_score is not None else compute_f1(
            solution_str, ground_truth, data_source, val_type="f1"
        )
        scores[-1] = fallback
        if is_validation:
            return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1, "scores": scores}
        return scores

    # ── Locate turn boundaries via special-token anchors ──────────────────
    # turn_ends: <|im_end|> positions for all non-final assistant turns.
    # Final turn has no <|im_end|> marker in the response → its score goes
    # to scores[-1].
    turn_ends = _find_turn_boundaries_by_tokens(original_token_ids, tokenizer)
    num_turns = len(turn_ends) + 1  # non-final turns + 1 final turn

    work_scores = list(turn_scores)
    if len(work_scores) > num_turns:
        print(
            f"abseeker_reward: turn_scores too long (truncating) — "
            f"detected_turns={num_turns}, turn_scores={len(work_scores)}"
        )
        work_scores = work_scores[:num_turns]

    # ── Place non-final turn scores ───────────────────────────────────────
    for i, end_pos in enumerate(turn_ends):
        if i >= len(work_scores):
            break
        if work_scores[i] is None:
            continue
        scores[end_pos] = float(work_scores[i])

    # ── Place final turn score at last token ──────────────────────────────
    # turn_scores[-1] is the LLM judge score for the final (answer) turn.
    scores[-1] = float(work_scores[-1]) if work_scores[-1] is not None else 0.0

    if is_validation:
        return {"f1": f1_score, "em": em_score, "noformatf1": noformatf1, "scores": scores}
    return scores
