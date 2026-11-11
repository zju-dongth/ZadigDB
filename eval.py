"""Query accuracy and hardware telemetry evaluation utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np

from zadigdb.pipeline import QueryExecutionResult


def _compute_prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = float(tp) / float(max(1, tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp) / float(max(1, tp + fn)) if (tp + fn) > 0 else 0.0
    if prec + rec <= 1e-9:
        return 0.0, 0.0, 0.0
    return prec * 100.0, rec * 100.0, (200.0 * prec * rec) / (prec + rec)


def _score_against_gt(
    result: QueryExecutionResult,
    gt_info: Optional[Mapping[str, Any]],
    slice_gt_entities: Optional[Mapping[int, int]],
) -> Tuple[float, float, float]:
    if not gt_info or not slice_gt_entities:
        return 0.0, 0.0, 0.0

    if result.query_class == "Q1" and "gt_target_entities" in gt_info:
        gt_entities = set(gt_info["gt_target_entities"])
        pred_entities = {
            slice_gt_entities[sid]
            for sid in result.matched_results
            if slice_gt_entities.get(sid, -1) >= 0
        }
        bg_fps = sum(1 for sid in result.matched_results if slice_gt_entities.get(sid, -1) < 0)
        tp = len(pred_entities & gt_entities)
        fp = len(pred_entities - gt_entities) + bg_fps
        fn = len(gt_entities - pred_entities)
        return _compute_prf1(tp, fp, fn)

    if result.query_class == "Q2" and "gt_target_entities" in gt_info:
        gt_entities = set(gt_info["gt_target_entities"])
        recovered: Set[int] = set()
        fp_chains = 0
        for chain in result.matched_results:
            e_ids = [slice_gt_entities.get(sid, -1) for sid in chain]
            valid = [e for e in e_ids if e >= 0]
            if not valid:
                fp_chains += 1
                continue
            dom = max(sorted(set(valid)), key=e_ids.count)
            if dom in gt_entities and (e_ids.count(dom) / float(len(e_ids))) >= 0.99:
                recovered.add(dom)
            else:
                fp_chains += 1
        return _compute_prf1(len(recovered), fp_chains, max(0, len(gt_entities) - len(recovered)))

    if "gt_comove_pairs" in gt_info:
        gt_pairs = set(gt_info["gt_comove_pairs"])
        pred_pairs: Set[Tuple[int, int]] = set()
        fp_pairs = 0
        for chain_a, chain_b, _ in result.matched_results:
            eids_a = [slice_gt_entities.get(sid, -1) for sid in chain_a]
            eids_b = [slice_gt_entities.get(sid, -1) for sid in chain_b]
            val_a = [e for e in eids_a if e >= 0]
            val_b = [e for e in eids_b if e >= 0]
            if not val_a or not val_b:
                fp_pairs += 1
                continue
            dom_a = max(sorted(set(val_a)), key=eids_a.count)
            dom_b = max(sorted(set(val_b)), key=eids_b.count)
            pur_a = eids_a.count(dom_a) / float(len(eids_a))
            pur_b = eids_b.count(dom_b) / float(len(eids_b))
            pair_key = (min(dom_a, dom_b), max(dom_a, dom_b))
            if (pair_key in gt_pairs or (dom_a, dom_b) in gt_pairs) and pur_a >= 0.99 and pur_b >= 0.99:
                pred_pairs.add((dom_a, dom_b) if (dom_a, dom_b) in gt_pairs else pair_key)
            else:
                fp_pairs += 1
        return _compute_prf1(len(pred_pairs), fp_pairs, max(0, len(gt_pairs) - len(pred_pairs)))

    return 0.0, 0.0, 0.0


def evaluate_query(
    result: QueryExecutionResult,
    gt_info: Optional[Mapping[str, Any]] = None,
    slice_gt_entities: Optional[Mapping[int, int]] = None,
) -> Dict[str, Any]:
    prec, rec, f1 = _score_against_gt(result, gt_info, slice_gt_entities)
    report = result.audit_report
    return {
        "query_id": result.query_id,
        "query_class": result.query_class,
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "f1": round(f1, 2),
        "latency_ms": round(float(result.latency_sec) * 1000.0, 2),
        "peak_vram_gb": round(float(result.peak_vram_gb), 2),
        "visual_tokens_materialized": int(result.tokens_evaluated),
        "token_savings_ratio": round(float(result.token_savings_ratio), 2),
        "buffer_hit_rate": round(float(result.cache_hit_rate), 2),
        "clause_repairs": int(report.repair_iterations) if report is not None else 0,
        "clauses_certified": bool(report.passed_all) if report is not None else True,
        "oom_occurred": bool(result.oom_occurred),
        "num_matches": len(result.matched_results),
    }


def evaluate_workload(
    results: Sequence[QueryExecutionResult],
    query_gt: Optional[Mapping[str, Mapping[str, Any]]] = None,
    slice_gt_entities: Optional[Mapping[int, int]] = None,
) -> Dict[str, Any]:
    per_query: List[Dict[str, Any]] = [
        evaluate_query(r, (query_gt or {}).get(r.query_id), slice_gt_entities) for r in results
    ]
    if not per_query:
        return {"num_queries": 0, "queries": []}

    latencies = np.asarray([q["latency_ms"] for q in per_query], dtype=np.float64)
    total_tokens = int(sum(q["visual_tokens_materialized"] for q in per_query))
    return {
        "num_queries": len(per_query),
        "precision": round(float(np.mean([q["precision"] for q in per_query])), 2),
        "recall": round(float(np.mean([q["recall"] for q in per_query])), 2),
        "f1": round(float(np.mean([q["f1"] for q in per_query])), 2),
        "latency_ms": round(float(np.mean(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "peak_vram_gb": round(float(max(q["peak_vram_gb"] for q in per_query)), 2),
        "visual_tokens_materialized": total_tokens,
        "mean_visual_tokens_materialized": round(float(total_tokens) / float(len(per_query)), 2),
        "token_savings_ratio": round(float(np.mean([q["token_savings_ratio"] for q in per_query])), 2),
        "buffer_hit_rate": round(float(np.mean([q["buffer_hit_rate"] for q in per_query])), 2),
        "clause_c1_c3_repairs": int(sum(q["clause_repairs"] for q in per_query)),
        "certified_queries": int(sum(1 for q in per_query if q["clauses_certified"])),
        "oom_count": int(sum(1 for q in per_query if q["oom_occurred"])),
        "queries": per_query,
    }
