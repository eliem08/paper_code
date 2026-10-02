"""Metrics for the agentic evaluation harness.

These cover MCP system-level metrics
(tool success rate, schema validity, latency) but add task-level
diagnostic metrics specific to the four agentic tasks.
"""

from __future__ import annotations

from statistics import mean, quantiles
from typing import Iterable

from breastpath_copilot.agents.types import AgentResult, Trajectory


def triage_accuracy(results: Iterable[AgentResult]) -> dict[str, float | int]:
    """Aggregate metrics for the triage task across many runs.

    Returns
    -------
    Mapping with:

    * ``n`` -- total runs evaluated.
    * ``success_rate`` -- fraction where the agent produced a final decision.
    * ``accuracy`` -- fraction where the decision matched the gold triage label
      (among runs with a final decision and a non-None ``correct``).
    * ``safe_on_refer`` -- among gold ``refer`` cases, fraction where the
      agent predicted ``refer`` or ``suspicious`` (i.e. did not under-call).
    * ``flag_rate`` -- fraction of decisions with ``flag_for_review=true``.
    """
    results = list(results)
    n = len(results)
    if n == 0:
        return {"n": 0}
    successes = [r for r in results if r.success and r.decision is not None]
    correct = [r for r in successes if r.correct is True]
    safe_on_refer_total = 0
    safe_on_refer_hit = 0
    flagged = 0
    for r in successes:
        breakdown = (r.decision or {}).get("_eval_breakdown", {}) if r.decision else {}
        if breakdown.get("expected_triage") == "refer":
            safe_on_refer_total += 1
            if breakdown.get("is_safe"):
                safe_on_refer_hit += 1
        if breakdown.get("flag_for_review"):
            flagged += 1
    return {
        "n": n,
        "success_rate": len(successes) / n,
        "accuracy": (len(correct) / len(successes)) if successes else 0.0,
        "safe_on_refer": (safe_on_refer_hit / safe_on_refer_total) if safe_on_refer_total else 0.0,
        "flag_rate": (flagged / len(successes)) if successes else 0.0,
    }


def evidence_report_metrics(results: Iterable[AgentResult]) -> dict[str, float | int]:
    """Aggregate metrics for the evidence-report task."""
    results = list(results)
    n = len(results)
    if n == 0:
        return {"n": 0}
    successes = [r for r in results if r.success and r.decision is not None]
    label_hits = 0
    evidence_provided = 0
    overlap_hits = 0
    correct = 0
    evidence_counts: list[int] = []
    for r in successes:
        bd = (r.decision or {}).get("_eval_breakdown", {})
        if bd.get("label_correct"):
            label_hits += 1
        if bd.get("evidence_provided"):
            evidence_provided += 1
        if bd.get("evidence_overlap_ok"):
            overlap_hits += 1
        if r.correct:
            correct += 1
        evidence_counts.append(int(bd.get("evidence_count", 0)))
    denom = max(len(successes), 1)
    return {
        "n": n,
        "success_rate": len(successes) / n,
        "label_accuracy": label_hits / denom,
        "evidence_provided_rate": evidence_provided / denom,
        "evidence_overlap_rate": overlap_hits / denom,
        "joint_accuracy": correct / denom,
        "mean_evidence_count": (sum(evidence_counts) / denom) if evidence_counts else 0.0,
    }


def cohort_routing_metrics(results: Iterable[AgentResult]) -> dict[str, float | int]:
    """Aggregate metrics for the cross-cohort routing task."""
    results = list(results)
    n = len(results)
    if n == 0:
        return {"n": 0}
    successes = [r for r in results if r.success and r.decision is not None]
    cohort_hits = 0
    expert_total = 0
    expert_hits = 0
    joint = 0
    for r in successes:
        bd = (r.decision or {}).get("_eval_breakdown", {})
        if bd.get("cohort_correct"):
            cohort_hits += 1
        if bd.get("expert_correct") is not None:
            expert_total += 1
            if bd.get("expert_correct"):
                expert_hits += 1
        if r.correct:
            joint += 1
    denom = max(len(successes), 1)
    return {
        "n": n,
        "success_rate": len(successes) / n,
        "cohort_accuracy": cohort_hits / denom,
        "expert_accuracy": (expert_hits / expert_total) if expert_total else 0.0,
        "joint_accuracy": joint / denom,
    }


def discrepancy_metrics(results: Iterable[AgentResult]) -> dict[str, float | int]:
    """Aggregate metrics for the discrepancy-resolution task."""
    results = list(results)
    n = len(results)
    if n == 0:
        return {"n": 0}
    successes = [r for r in results if r.success and r.decision is not None]
    label_hits = 0
    source_hits = 0
    joint = 0
    escalated_correctly = 0
    escalated_total = 0
    for r in successes:
        bd = (r.decision or {}).get("_eval_breakdown", {})
        if bd.get("label_correct"):
            label_hits += 1
        if bd.get("source_correct"):
            source_hits += 1
        if r.correct:
            joint += 1
        if bd.get("gold_correct_source") == "neither":
            escalated_total += 1
            if bd.get("flagged") and bd.get("trust_source") == "neither":
                escalated_correctly += 1
    denom = max(len(successes), 1)
    return {
        "n": n,
        "success_rate": len(successes) / n,
        "label_accuracy": label_hits / denom,
        "source_accuracy": source_hits / denom,
        "joint_accuracy": joint / denom,
        "neither_escalation_rate": (
            escalated_correctly / escalated_total if escalated_total else 0.0
        ),
    }


def trajectory_summary_stats(
    trajectories: Iterable[Trajectory],
) -> dict[str, float | int]:
    """Tool-use efficiency + reliability stats across many trajectories.

    Returns mean / p95 step counts, mean tool calls per run, tool success
    rate, and truncation rate.
    """
    trajectories = list(trajectories)
    n = len(trajectories)
    if n == 0:
        return {"n": 0}
    step_counts = [t.step_count for t in trajectories]
    truncated = sum(1 for t in trajectories if t.truncated)
    tool_calls = [len(t.tool_results) for t in trajectories]
    tool_ok = sum(sum(1 for r in t.tool_results if r.ok) for t in trajectories)
    tool_total = sum(tool_calls) or 1
    latencies = [
        r.latency_ms
        for t in trajectories
        for r in t.tool_results
        if r.latency_ms is not None
    ]
    p95 = (
        quantiles(latencies, n=20)[-1]
        if len(latencies) >= 20
        else (max(latencies) if latencies else 0.0)
    )
    return {
        "n": n,
        "mean_step_count": mean(step_counts) if step_counts else 0.0,
        "mean_tool_calls": mean(tool_calls) if tool_calls else 0.0,
        "tool_success_rate": tool_ok / tool_total,
        "truncation_rate": truncated / n,
        "mean_tool_latency_ms": mean(latencies) if latencies else 0.0,
        "p95_tool_latency_ms": p95,
    }
