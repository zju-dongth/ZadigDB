"""
Formal Multi-Camera Spatio-Temporal Semantic Algebraic Operators for ZadigDB:
  - SEM_FILTER: Progressive 3-tier semantic attribute & motion filtering with observability-aware escalation.
  - SEM_LINK: Cross-camera entity linkage over the Top-k_comp Competitive Provenance View.
  - SEM_COMOVE_JOIN: Coupled multi-entity co-movement join via headway belief propagation and DAG DP.
"""

from typing import Dict, List, Optional, Set, Tuple
import bisect
import math
import numpy as np

from .buffer_pool import ProgressiveVisualBufferPool
from .schema import (
    CameraTopology,
    CompetitiveProvenanceView,
    PhysicalOperatorNode,
    ProvenanceEdge,
    TokenResolutionTier,
    VisualDetectionSlice,
)

TEMPORAL_MOTION_PREDICATES: Set[str] = {
    "turning_left",
    "turning_right",
    "stopping",
    "going_straight",
}


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def conjunctive_predicate_score(scores_map: Dict[str, float], predicates: List[str]) -> float:
    """
    Evaluates a conjunctive multi-predicate SQL `SEM_FILTER` clause (`p_1 AND p_2 AND ...`).
    Combines minimum-conjunct satisfaction with mean confidence so partial matches cannot
    masquerade as full conjunction matches.
    """
    if not predicates:
        return 1.0
    vals = [float(scores_map.get(p, 0.5)) for p in predicates]
    if len(vals) == 1:
        return vals[0]
    return 0.60 * min(vals) + 0.40 * (sum(vals) / float(len(vals)))


def estimate_empirical_tier0_hubs(
    slices: List[VisualDetectionSlice],
    k_nn: int = 5,
    min_sim: float = 0.25,
    z_thresh: float = 1.15,
) -> Set[int]:
    """
    Unsupervised Tier-0 similarity-graph hubness detector (Section IV-B, I_hub(v) in {0, 1}).
    Computes empirical cross-camera k-occurrence in-degree N_{k_nn}(v) purely from Tier-0
    summary descriptors z_u^{(0)}.
    """
    n = len(slices)
    if n < 6:
        return set()

    mat = np.stack([s.tier0_embed for s in slices], axis=0).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat_norm = mat / np.maximum(norms, 1e-8)
    sim_matrix = mat_norm @ mat_norm.T

    cam_ids = np.array([s.cam_id for s in slices], dtype=np.int32)
    same_cam_mask = cam_ids[:, None] == cam_ids[None, :]
    sim_matrix[same_cam_mask] = -1.0

    k_eff = min(k_nn, n - 1)
    if k_eff <= 0:
        return set()

    # Identify top-k_eff cross-camera Tier-0 neighbors for each slice u
    top_indices = np.argpartition(-sim_matrix, kth=k_eff - 1, axis=1)[:, :k_eff]
    in_degrees = np.zeros(n, dtype=np.float32)
    row_idx = np.arange(n)[:, None]
    top_sims = sim_matrix[row_idx, top_indices]
    valid_neighbors = top_indices[top_sims >= min_sim]
    if valid_neighbors.size > 0:
        np.add.at(in_degrees, valid_neighbors, 1.0)

    mean_deg = float(np.mean(in_degrees))
    std_deg = float(np.std(in_degrees))
    cutoff = max(2.0, mean_deg + z_thresh * max(std_deg, 0.5))
    return {slices[i].slice_id for i in range(n) if float(in_degrees[i]) >= cutoff}


def _slice_conjunctive_score(
    s: VisualDetectionSlice,
    tier_scores: Dict[str, float],
    predicates: List[str],
) -> float:
    if not predicates:
        return 1.0
    vals = [
        float(
            tier_scores[p]
            if p in tier_scores
            else s.predicate_scores.get(p, 0.5)
        )
        for p in predicates
    ]
    if len(vals) == 1:
        return vals[0]
    return 0.60 * min(vals) + 0.40 * (sum(vals) / float(len(vals)))


class SpatioTemporalSemanticAlgebra:
    """
    Executes SEM_FILTER, SEM_LINK, and SEM_COMOVE_JOIN over unmaterialized multi-camera
    video detection slices backed by the Multi-Granularity Visual Token Buffer Pool.
    """

    def __init__(
        self,
        topology: CameraTopology,
        buffer_pool: ProgressiveVisualBufferPool,
        filter_threshold: float = 0.52,
        uncertain_margin: float = 0.14,
        link_appearance_weight: float = 0.90,
        link_spatiotemporal_weight: float = 0.10,
        link_min_edge_confidence: float = 0.36,
        comove_min_pair_confidence: float = 0.38,
        hub_regularization_weight: float = 0.35,
    ):
        self.topology = topology
        self.buffer_pool = buffer_pool
        self.filter_threshold = float(filter_threshold)
        self.uncertain_margin = float(uncertain_margin)
        self.link_appearance_weight = float(link_appearance_weight)
        self.link_spatiotemporal_weight = float(link_spatiotemporal_weight)
        self.link_min_edge_confidence = float(link_min_edge_confidence)
        self.comove_min_pair_confidence = float(comove_min_pair_confidence)
        self.hub_regularization_weight = float(hub_regularization_weight)
        self.total_forward_pairs: int = 0
        self.c2_pruned_pairs: int = 0
        self.evaluated_link_pairs: int = 0
        self._filter_score_cache: Dict[Tuple[int, Tuple[str, ...]], float] = {}

    def _get_slice_pred_score(
        self, s: VisualDetectionSlice, predicates: List[str]
    ) -> float:
        if not predicates:
            return 1.0
        cached = self._filter_score_cache.get((id(s), tuple(predicates)))
        if cached is not None:
            return cached
        if self.buffer_pool.is_tier2_resident(s.slice_id):
            return _slice_conjunctive_score(s, s.attribute_scores_t2, predicates)
        if self.buffer_pool.is_tier1_resident(s.slice_id):
            return _slice_conjunctive_score(s, s.attribute_scores_t1, predicates)
        return _slice_conjunctive_score(s, s.attribute_scores_t0, predicates)

    def _chain_satisfies_predicates(
        self,
        chain_ids: List[int],
        slice_map: Dict[int, VisualDetectionSlice],
        predicates: List[str],
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Verifies entity-level multi-view predicate consensus across a linked multi-camera
        trajectory chain, requiring at least one high-confidence anchor view (>= threshold)
        and consistent trajectory-wide mean confidence (>= threshold - 0.07).
        """
        if not predicates or not chain_ids:
            return True
        eff_thr = self.filter_threshold if threshold is None else float(threshold)
        scores = [
            self._get_slice_pred_score(slice_map[sid], predicates)
            for sid in chain_ids
            if sid in slice_map
        ]
        if not scores:
            return False
        max_sc = max(scores)
        mean_sc = sum(scores) / float(len(scores))
        return (max_sc >= eff_thr) and (mean_sc >= max(0.38, eff_thr - 0.07))

    def _is_forward_reachable(
        self, u: VisualDetectionSlice, v: VisualDetectionSlice, use_c2: bool
    ) -> bool:
        if u.cam_id == v.cam_id:
            return False
        use_colocated_mid = self.topology.co_located_slack_sec > 0.0
        max_horizon_dt = 600.0 if use_c2 else 180.0
        if use_colocated_mid:
            u_mid = 0.5 * (u.t_enter + u.t_exit)
            v_mid = 0.5 * (v.t_enter + v.t_exit)
            if (v_mid, v.t_enter, v.cam_id, v.slice_id) <= (
                u_mid,
                u.t_enter,
                u.cam_id,
                u.slice_id,
            ):
                return False
            if (v_mid - u_mid) > max_horizon_dt or (
                v.t_enter - u.t_exit
            ) < -self.topology.co_located_slack_sec:
                return False
        else:
            dt = v.t_enter - u.t_exit
            if dt <= 0.0 or dt > max_horizon_dt:
                return False
        if use_c2:
            return self.topology.are_adjacent(
                u.cam_id, v.cam_id
            ) and self.topology.is_causally_reachable(
                u.cam_id, v.cam_id, u.t_exit, v.t_enter
            )
        return True

    def _compute_c2_chain_lengths(
        self,
        slices_sub: List[VisualDetectionSlice],
    ) -> Dict[int, int]:
        """
        Relational Clause C2 Reachability Semi-Join DP:
        Computes for each slice s in `slices_sub` the maximum causal trajectory length
        `L_in(s) + L_out(s) - 1` achievable over the directed spatio-temporal reachability
        DAG G_{C2} using purely relational metadata `(cam_id, t_enter, t_exit)` (zero VLM tokens).
        """
        if len(slices_sub) <= 1:
            return {s.slice_id: 1 for s in slices_sub}

        ordered = sorted(
            slices_sub,
            key=lambda s: (0.5 * (s.t_enter + s.t_exit), s.t_enter, s.cam_id, s.slice_id),
        )
        by_cam: Dict[int, List[VisualDetectionSlice]] = {}
        for s in ordered:
            by_cam.setdefault(s.cam_id, []).append(s)
        cam_enter_sorted: Dict[int, Tuple[List[float], List[VisualDetectionSlice]]] = {}
        for cid, slist in by_cam.items():
            s_sorted = sorted(slist, key=lambda x: x.t_enter)
            cam_enter_sorted[cid] = ([x.t_enter for x in s_sorted], s_sorted)

        active_cams = list(by_cam.keys())
        adj_offsets: Dict[int, List[Tuple[int, float, float]]] = {}
        for c_src in active_cams:
            outs: List[Tuple[int, float, float]] = []
            for c_dst in active_cams:
                if c_dst != c_src and self.topology.are_adjacent(c_src, c_dst):
                    t_min0, t_max0 = self.topology.causal_window(c_src, c_dst, 0.0)
                    if t_max0 >= 0.0:
                        d_ij = self.topology.distance_m(c_src, c_dst)
                        d_eff = max(0.0, d_ij - self.topology.fov_overlap_m)
                        co_slk = self.topology.co_located_slack_sec if d_ij <= 180.0 else 0.0
                        off_min = (d_eff / max(self.topology.v_max_mps, 1e-3)) - co_slk - 2.0
                        off_max = t_max0 + 2.0
                        outs.append((c_dst, off_min, off_max))
            adj_offsets[c_src] = outs

        l_in: Dict[int, int] = {s.slice_id: 1 for s in ordered}
        out_neighbors: Dict[int, List[int]] = {s.slice_id: [] for s in ordered}

        for u in ordered:
            u_id = u.slice_id
            lu = l_in[u_id]
            for c_dst, off_min, off_max in adj_offsets.get(u.cam_id, []):
                t_lo = max(-2.0, u.t_exit + off_min)
                t_hi = u.t_exit + off_max
                enters_dst, slist_dst = cam_enter_sorted[c_dst]
                idx_lo = bisect.bisect_left(enters_dst, t_lo)
                idx_hi = bisect.bisect_right(enters_dst, t_hi)
                for k_idx in range(idx_lo, idx_hi):
                    v = slist_dst[k_idx]
                    if self._is_forward_reachable(u, v, True):
                        v_id = v.slice_id
                        out_neighbors[u_id].append(v_id)
                        if lu + 1 > l_in[v_id]:
                            l_in[v_id] = lu + 1

        l_out: Dict[int, int] = {s.slice_id: 1 for s in ordered}
        for u in reversed(ordered):
            u_id = u.slice_id
            nbrs = out_neighbors[u_id]
            if nbrs:
                l_out[u_id] = 1 + max(l_out[v_id] for v_id in nbrs)

        return {s.slice_id: (l_in[s.slice_id] + l_out[s.slice_id] - 1) for s in ordered}

    def execute_sem_filter(
        self,
        slices: List[VisualDetectionSlice],
        op: PhysicalOperatorNode,
        threshold: Optional[float] = None,
    ) -> List[VisualDetectionSlice]:
        """
        Executes SEM_FILTER(slices, predicates) using 3-tier margin-guided progressive
        token materialization (Tier-0 1-tok screen -> Tier-1 32-tok regional filter -> Tier-2 256-tok escalation).
        When feeding downstream multi-hop linkage (op.downstream_multihop), retains soft-margin
        slices that form causally reachable Tier-1 visual bridges to confirmed anchor detections.
        """
        eff_threshold = self.filter_threshold if threshold is None else float(threshold)
        target_cams = set(op.target_cameras) if op.target_cameras else None
        candidates = [
            s for s in slices if (target_cams is None or s.cam_id in target_cams)
        ]
        self.buffer_pool.record_tier0_scan(len(candidates))

        if not op.predicates:
            return candidates

        pred_key = tuple(op.predicates)
        num_preds = len(op.predicates)
        bridge_mode = bool(op.downstream_multihop)
        terminal_3tier = not bridge_mode
        static_preds = [p for p in op.predicates if p not in TEMPORAL_MOTION_PREDICATES]
        has_temporal_pred = len(static_preds) < len(op.predicates)
        t0_eval_preds = (
            static_preds if (terminal_3tier and static_preds) else list(op.predicates)
        )

        # Stage 1: Tier-0 (1-token) summary descriptor screening (static attributes only for terminal filters)
        t0_floor = (eff_threshold - 0.30) if bridge_mode else (eff_threshold - 0.25)
        c2_dag_floor = max(0.12, t0_floor - 0.10)
        stage1_records: List[Tuple[VisualDetectionSlice, float, float]] = []
        c2_dag_slices: List[VisualDetectionSlice] = []
        for s in candidates:
            t0_score = _slice_conjunctive_score(s, s.attribute_scores_t0, t0_eval_preds)
            if bridge_mode and op.use_causal_window_c2 and t0_score >= c2_dag_floor:
                c2_dag_slices.append(s)
            if t0_score >= t0_floor:
                u0 = 1.0 - min(1.0, abs(t0_score - eff_threshold) * 2.5)
                stage1_records.append((s, t0_score, u0))

        if not stage1_records:
            return []

        # Prioritize slices with highest Tier-0 plausibility within max_tier1_slices;
        # when Clause C1 truncates Stage-1 candidates on city-scale multi-hop queries, push down
        # Clause C2 reachability semi-join to prioritize slices participating in >= req_hops paths
        t1_quota = max(0, int(op.max_tier1_slices))
        req_hops = max(2, int(op.min_co_move_hops))
        c2_chain_len: Dict[int, int] = {}
        if bridge_mode and op.use_causal_window_c2 and len(stage1_records) > t1_quota:
            c2_chain_len = self._compute_c2_chain_lengths(c2_dag_slices)
            stage1_records.sort(
                key=lambda rec: (
                    1 if c2_chain_len.get(rec[0].slice_id, 1) >= req_hops else 0,
                    min(c2_chain_len.get(rec[0].slice_id, 1), req_hops),
                    rec[1],
                    rec[2],
                ),
                reverse=True,
            )
        else:
            stage1_records.sort(key=lambda rec: (rec[1], rec[2]), reverse=True)
        t1_batch = stage1_records[:t1_quota]
        t1_pin_ids = [s.slice_id for s, _, _ in t1_batch]

        # Stage 2: Tier-1 (32-token) regional sub-grid evaluation & observability-aware escalation
        esc_low_margin = (eff_threshold - 0.08) if terminal_3tier else (eff_threshold - self.uncertain_margin)
        soft_bridge_floor = max(0.28, eff_threshold - 0.18)
        high_margin = (1.01 if has_temporal_pred else 0.86) if terminal_3tier else 0.86
        final_scores: Dict[int, float] = {}
        slice_by_id: Dict[int, VisualDetectionSlice] = {}
        t1_embeds_local: Dict[int, np.ndarray] = {}
        t1_static_by_id: Dict[int, float] = {}
        ambiguous_t1: List[Tuple[VisualDetectionSlice, float, float]] = []

        self.buffer_pool.pin_tier1(t1_pin_ids)
        try:
            for s, _, u0 in t1_batch:
                emb1 = self.buffer_pool.access_tier1(
                    s, heat=max(0.45, u0), num_predicates=num_preds, predicates=op.predicates
                )
                t1_score = _slice_conjunctive_score(s, s.attribute_scores_t1, op.predicates)
                t1_stat_sc = (
                    _slice_conjunctive_score(s, s.attribute_scores_t1, t0_eval_preds)
                    if terminal_3tier
                    else t1_score
                )
                t1_static_by_id[s.slice_id] = t1_stat_sc
                if bridge_mode and t1_score >= soft_bridge_floor:
                    t1_embeds_local[s.slice_id] = emb1
                    slice_by_id[s.slice_id] = s
                if t1_score < esc_low_margin or (terminal_3tier and t1_stat_sc < esc_low_margin):
                    if bridge_mode and t1_score >= soft_bridge_floor:
                        final_scores[s.slice_id] = t1_score
                    continue
                if t1_score >= high_margin:
                    final_scores[s.slice_id] = t1_score
                else:
                    u1 = 1.0 - min(1.0, abs(t1_score - eff_threshold) * 2.0)
                    ambiguous_t1.append((s, t1_score, u1))

            # Stage 3: On-demand Tier-2 (256-token) escalation & IPO plan-context terminal verification
            t2_quota = max(0, int(op.max_tier2_slices))
            if c2_chain_len and len(ambiguous_t1) > t2_quota:
                ambiguous_t1.sort(
                    key=lambda rec: (
                        1 if c2_chain_len.get(rec[0].slice_id, 1) >= req_hops else 0,
                        0.65 * rec[2] + 0.35 * rec[1],
                    ),
                    reverse=True,
                )
            elif terminal_3tier and has_temporal_pred:
                ambiguous_t1.sort(
                    key=lambda rec: (
                        0.55 * rec[1] + 0.45 * t1_static_by_id.get(rec[0].slice_id, rec[1]),
                        rec[2],
                    ),
                    reverse=True,
                )
            else:
                ambiguous_t1.sort(key=lambda rec: (0.65 * rec[2] + 0.35 * rec[1]), reverse=True)

            eff_accept_threshold = (
                (eff_threshold + 0.06) if (terminal_3tier and has_temporal_pred) else eff_threshold
            )
            t2_used = 0
            t2_pin_ids = [s.slice_id for s, _, _ in ambiguous_t1[:t2_quota]]
            self.buffer_pool.pin_tier2(t2_pin_ids)
            try:
                for idx, (s, t1_sc, u1) in enumerate(ambiguous_t1):
                    if idx < t2_quota:
                        self.buffer_pool.access_tier2(
                            s, heat=max(0.70, u1), num_predicates=num_preds, predicates=op.predicates
                        )
                        t2_sc = _slice_conjunctive_score(s, s.attribute_scores_t2, op.predicates)
                        if terminal_3tier and has_temporal_pred:
                            t2_min_conj = min(
                                float(
                                    s.attribute_scores_t2[p]
                                    if p in s.attribute_scores_t2
                                    else s.predicate_scores.get(p, 0.5)
                                )
                                for p in op.predicates
                            )
                            fused_sc = 0.15 * t1_static_by_id.get(s.slice_id, t1_sc) + 0.85 * t2_sc
                            final_scores[s.slice_id] = (
                                fused_sc if t2_min_conj >= 0.45 else min(fused_sc, eff_accept_threshold - 0.01)
                            )
                        else:
                            final_scores[s.slice_id] = t2_sc
                        t2_used += 1
                    else:
                        final_scores[s.slice_id] = (
                            0.0 if (terminal_3tier and has_temporal_pred and t1_sc < 0.92) else t1_sc
                        )
            finally:
                self.buffer_pool.unpin_tier2(t2_pin_ids)
        finally:
            self.buffer_pool.unpin_tier1(t1_pin_ids)

        for sid, sc in final_scores.items():
            s_ref = slice_by_id.get(sid)
            if s_ref is not None:
                self._filter_score_cache[(id(s_ref), pred_key)] = sc
        if not bridge_mode:
            for s in candidates:
                if s.slice_id in final_scores:
                    self._filter_score_cache[(id(s), pred_key)] = final_scores[s.slice_id]

        strong_ids = {sid for sid, sc in final_scores.items() if sc >= eff_accept_threshold}
        if not bridge_mode or not strong_ids:
            return [s for s in candidates if s.slice_id in strong_ids]

        # Causal Stage-1 overflow escalation: if Stage-1 Tier-0 quota truncated candidate scan
        # on large streams, promote uninspected Stage-1 slices causally adjacent to confirmed
        # anchors to Tier-1 (evicting cold Tier-1 slots within the admitted Clause C1 pool)
        if len(stage1_records) > t1_quota:
            ov_by_cam: Dict[int, List[VisualDetectionSlice]] = {}
            for rec in stage1_records[t1_quota:]:
                ov_by_cam.setdefault(rec[0].cam_id, []).append(rec[0])
            all_ov_cams = list(ov_by_cam.keys())
            ov_promoted: Set[int] = set()
            ov_budget = min(128, max(32, t2_quota // 2))
            ov_frontier = [s for s in candidates if s.slice_id in strong_ids]
            for _ in range(2):
                if not ov_frontier or len(ov_promoted) >= ov_budget:
                    break
                best_sim_by_ov: Dict[int, Tuple[float, VisualDetectionSlice]] = {}
                for s_anc in ov_frontier:
                    adj_cams = (
                        [
                            c_id
                            for c_id in all_ov_cams
                            if c_id != s_anc.cam_id
                            and (
                                self.topology.are_adjacent(s_anc.cam_id, c_id)
                                or self.topology.are_adjacent(c_id, s_anc.cam_id)
                            )
                        ]
                        if op.use_causal_window_c2
                        else [c_id for c_id in all_ov_cams if c_id != s_anc.cam_id]
                    )
                    for c_id in adj_cams:
                        for s_ov in ov_by_cam[c_id]:
                            if s_ov.slice_id in ov_promoted or s_ov.slice_id in final_scores:
                                continue
                            if not (
                                self._is_forward_reachable(s_anc, s_ov, op.use_causal_window_c2)
                                or self._is_forward_reachable(s_ov, s_anc, op.use_causal_window_c2)
                            ):
                                continue
                            sim0 = _cosine_sim(s_ov.tier0_embed, s_anc.tier0_embed)
                            if sim0 >= -0.05:
                                prev_entry = best_sim_by_ov.get(s_ov.slice_id)
                                if prev_entry is None or sim0 > prev_entry[0]:
                                    best_sim_by_ov[s_ov.slice_id] = (sim0, s_ov)

                if not best_sim_by_ov:
                    break
                ranked_ov = sorted(
                    best_sim_by_ov.values(),
                    key=lambda item: (item[0], -item[1].slice_id),
                    reverse=True,
                )
                next_ov_frontier: List[VisualDetectionSlice] = []
                for _, s_ov in ranked_ov:
                    if len(ov_promoted) >= ov_budget:
                        break
                    ov_promoted.add(s_ov.slice_id)
                    emb_ov1 = self.buffer_pool.access_tier1(
                        s_ov, heat=0.60, num_predicates=num_preds, predicates=op.predicates
                    )
                    ov_sc = _slice_conjunctive_score(
                        s_ov, s_ov.attribute_scores_t1, op.predicates
                    )
                    if esc_low_margin <= ov_sc < high_margin and t2_used < t2_quota:
                        self.buffer_pool.access_tier2(
                            s_ov, heat=0.72, num_predicates=num_preds, predicates=op.predicates
                        )
                        ov_sc = _slice_conjunctive_score(
                            s_ov, s_ov.attribute_scores_t2, op.predicates
                        )
                        t2_used += 1
                    slice_by_id[s_ov.slice_id] = s_ov
                    if ov_sc >= soft_bridge_floor:
                        t1_embeds_local[s_ov.slice_id] = emb_ov1
                        final_scores[s_ov.slice_id] = ov_sc
                        self._filter_score_cache[(id(s_ov), pred_key)] = ov_sc
                    if ov_sc >= eff_threshold:
                        strong_ids.add(s_ov.slice_id)
                        next_ov_frontier.append(s_ov)
                ov_frontier = next_ov_frontier

        # Multi-hop bridge retention: rescue soft-margin views [threshold - 0.18, threshold)
        # that are causally reachable and Tier-1 visually consistent with strong anchor views
        soft_pool = [
            s
            for s in candidates
            if s.slice_id in final_scores
            and soft_bridge_floor <= final_scores[s.slice_id] < eff_threshold
            and s.slice_id in t1_embeds_local
        ]
        bridged_ids: Set[int] = set()
        if soft_pool:
            frontier = [
                s for s in candidates if s.slice_id in strong_ids and s.slice_id in t1_embeds_local
            ]
            for _ in range(2):
                if not frontier:
                    break
                next_frontier: List[VisualDetectionSlice] = []
                for s_soft in soft_pool:
                    if s_soft.slice_id in bridged_ids:
                        continue
                    emb_soft = t1_embeds_local[s_soft.slice_id]
                    for s_anc in frontier:
                        if s_soft.cam_id == s_anc.cam_id:
                            continue
                        if not (
                            self._is_forward_reachable(s_anc, s_soft, op.use_causal_window_c2)
                            or self._is_forward_reachable(s_soft, s_anc, op.use_causal_window_c2)
                        ):
                            continue
                        if _cosine_sim(emb_soft, t1_embeds_local[s_anc.slice_id]) >= 0.38:
                            bridged_ids.add(s_soft.slice_id)
                            next_frontier.append(s_soft)
                            break
                frontier = next_frontier

        accepted_ids = strong_ids | bridged_ids
        return [s for s in candidates if s.slice_id in accepted_ids]

    def execute_sem_link(
        self,
        slices: List[VisualDetectionSlice],
        op: PhysicalOperatorNode,
    ) -> Tuple[CompetitiveProvenanceView, Dict[int, List[int]]]:
        """
        Executes SEM_LINK over candidate slices, enforcing Clause C2 (Spatio-Temporal Causal Window)
        and Clause C3 (Top-k_comp Competitive Provenance View) with unsupervised Tier-0 hubness
        estimation and Tier-1 softmax reduced-cost slack escalation (Def. 2, Eqs. 2-3, Section IV-B, Eq. 7).
        Returns both the CompetitiveProvenanceView and reconstructed entity trajectories.
        """
        prov_view = CompetitiveProvenanceView(k_comp=max(1, op.k_comp))
        if not slices:
            return prov_view, {}

        use_colocated_mid = self.topology.co_located_slack_sec > 0.0
        if use_colocated_mid:
            slices_sorted = sorted(
                slices, key=lambda s: (0.5 * (s.t_enter + s.t_exit), s.t_enter, s.cam_id, s.slice_id)
            )
        else:
            slices_sorted = sorted(
                slices, key=lambda s: (s.t_enter, s.t_exit, s.cam_id, s.slice_id)
            )
        slice_map = {s.slice_id: s for s in slices_sorted}

        # Unsupervised Tier-0 similarity-graph hubness indicator I_hub(v) in {0, 1} (Section IV-B)
        empirical_hub_ids = estimate_empirical_tier0_hubs(slices_sorted)

        # Group slices by camera for fast causal lookup
        by_cam: Dict[int, List[VisualDetectionSlice]] = {}
        for s in slices_sorted:
            by_cam.setdefault(s.cam_id, []).append(s)

        for u in slices_sorted:
            u_mid = 0.5 * (u.t_enter + u.t_exit)
            candidate_targets: List[Tuple[VisualDetectionSlice, float]] = []
            for dst_cam, dst_list in by_cam.items():
                if dst_cam == u.cam_id:
                    continue
                topo_connected = self.topology.are_adjacent(u.cam_id, dst_cam)
                for v in dst_list:
                    max_horizon_dt = 600.0 if op.use_causal_window_c2 else 180.0
                    if use_colocated_mid:
                        v_mid = 0.5 * (v.t_enter + v.t_exit)
                        if (v_mid, v.t_enter, v.cam_id, v.slice_id) <= (
                            u_mid,
                            u.t_enter,
                            u.cam_id,
                            u.slice_id,
                        ):
                            continue
                        if (v_mid - u_mid) > max_horizon_dt or (
                            v.t_enter - u.t_exit
                        ) < -self.topology.co_located_slack_sec:
                            continue
                    else:
                        dt = v.t_enter - u.t_exit
                        if dt <= 0.0 or dt > max_horizon_dt:
                            continue

                    self.total_forward_pairs += 1
                    causally_valid = topo_connected and self.topology.is_causally_reachable(
                        u.cam_id, dst_cam, u.t_exit, v.t_enter
                    )
                    if not causally_valid:
                        self.c2_pruned_pairs += 1
                        if op.use_causal_window_c2:
                            continue

                    self.evaluated_link_pairs += 1
                    if self.buffer_pool.is_tier1_resident(
                        u.slice_id
                    ) and self.buffer_pool.is_tier1_resident(v.slice_id):
                        emb_u1_res = (
                            u.tier1_embed if u.tier1_embed is not None else u.tier0_embed
                        )
                        emb_v1_res = (
                            v.tier1_embed if v.tier1_embed is not None else v.tier0_embed
                        )
                        sim_init = _cosine_sim(emb_u1_res, emb_v1_res)
                        min_init_sim = 0.05
                    else:
                        sim_init = _cosine_sim(u.tier0_embed, v.tier0_embed)
                        min_init_sim = -0.35
                    if sim_init >= min_init_sim:
                        candidate_targets.append((v, sim_init))

            if not candidate_targets:
                continue

            # Sort by initial screening similarity and form shortlist proportional to k_comp
            candidate_targets.sort(key=lambda pair: pair[1], reverse=True)
            shortlist_cap = max(op.k_comp * 3, 10) if use_colocated_mid else max(op.k_comp * 3, 6)
            shortlist = candidate_targets[:shortlist_cap]
            step_t1_pins = [u.slice_id] + [v.slice_id for v, _ in shortlist]

            self.buffer_pool.pin_tier1(step_t1_pins)
            try:
                # Pass 1: Tier-1 (32 regional tokens) evaluation and softmax reduced-cost slack (Def. 2, Eqs. 2-3)
                emb_u1 = self.buffer_pool.access_tier1(u, heat=0.6)
                t1_records: List[Tuple[VisualDetectionSlice, float]] = []
                for v, _ in shortlist:
                    emb_v1 = self.buffer_pool.access_tier1(v, heat=0.6)
                    t1_records.append((v, _cosine_sim(emb_u1, emb_v1)))

                t1_records.sort(key=lambda pair: pair[1], reverse=True)
                max_t1_sim = t1_records[0][1]
                t1_exps = [math.exp(max(-10.0, min(10.0, 6.0 * (s1 - max_t1_sim)))) for _, s1 in t1_records]
                t1_denom = max(sum(t1_exps), 1e-8)
                best_p1 = t1_exps[0] / t1_denom
                second_best_p1 = (t1_exps[1] / t1_denom) if len(t1_exps) >= 2 else 0.0

                # Pass 2: Escalation priority U_esc^link(u, v) (Section IV-B, Eq. 7) & PruneDecided (Algorithm 3)
                scored_edges: List[Tuple[VisualDetectionSlice, float, TokenResolutionTier]] = []
                for rank_t1, ((v, sim_t1), exp_t1) in enumerate(zip(t1_records, t1_exps)):
                    p1_uv = exp_t1 / t1_denom
                    if rank_t1 == 0:
                        tilde_delta_t1 = (
                            max(0.0, best_p1 - second_best_p1) if len(shortlist) >= 2 else 1.0
                        )
                    else:
                        tilde_delta_t1 = max(0.0, best_p1 - p1_uv)
                    hub_ind = 1.0 if v.slice_id in empirical_hub_ids else 0.0
                    w_hub = min(1.0, max(0.0, self.hub_regularization_weight))
                    u_esc_link = (1.0 - w_hub) * (1.0 - tilde_delta_t1) + w_hub * hub_ind

                    sim = sim_t1
                    tier_used = TokenResolutionTier.TIER_1_COARSE
                    prune_decided = (tilde_delta_t1 > 0.75) and (hub_ind == 0.0)
                    if not prune_decided and (
                        rank_t1 < op.k_comp or (op.k_comp > 1 and u_esc_link >= 0.50)
                    ):
                        esc_heat = max(0.75, min(1.0, u_esc_link))
                        pair_t2_pins = [u.slice_id, v.slice_id]
                        self.buffer_pool.pin_tier2(pair_t2_pins)
                        try:
                            emb_u2 = self.buffer_pool.access_tier2(u, heat=esc_heat)
                            emb_v2 = self.buffer_pool.access_tier2(v, heat=esc_heat)
                        finally:
                            self.buffer_pool.unpin_tier2(pair_t2_pins)
                        sim_t2 = _cosine_sim(emb_u2, emb_v2)
                        # Section IV-B: Unsupervised hubness regularization on coarse-to-fine similarity drop
                        is_glare_hub = (v.slice_id in empirical_hub_ids) and (
                            sim_t2 <= sim_t1 + 0.01
                        )
                        hub_corr = -0.05 if is_glare_hub else 0.0
                        sim = 0.15 * sim_t1 + 0.85 * sim_t2 + hub_corr
                        tier_used = TokenResolutionTier.TIER_2_FINE

                    # Plateau kinematic travel-time modulation psi_kin(u, v) within Clause C2 causal window (Def. 2)
                    if op.use_causal_window_c2 and sim > 0.0:
                        t_min, t_max = self.topology.causal_window(u.cam_id, v.cam_id, u.t_exit)
                        if t_min >= 0.0 and t_max > t_min:
                            d_uv = self.topology.distance_m(u.cam_id, v.cam_id)
                            coloc_edge = (
                                self.topology.fov_overlap_m > 0.0
                                and d_uv <= self.topology.fov_overlap_m
                            )
                            span = max(2.0, t_max - t_min)
                            plateau_hi = (
                                min(t_max, u.t_exit + 9.0)
                                if coloc_edge
                                else (t_min + 0.65 * span)
                            )
                            if v.t_enter < t_min:
                                excess = (t_min - v.t_enter) / max(2.0, 0.25 * span)
                            elif v.t_enter > plateau_hi:
                                denom_hi = 6.0 if coloc_edge else max(2.0, 0.35 * span)
                                excess = (v.t_enter - plateau_hi) / denom_hi
                            else:
                                excess = 0.0
                            kin_score = math.exp(-0.45 * (excess ** 2))
                            w_sp = max(self.link_spatiotemporal_weight, 0.16) if coloc_edge else self.link_spatiotemporal_weight
                            w_ap = 1.0 - w_sp
                            sim = sim * (w_ap + w_sp * kin_score)

                    scored_edges.append((v, sim, tier_used))
            finally:
                self.buffer_pool.unpin_tier1(step_t1_pins)

            scored_edges.sort(key=lambda item: item[1], reverse=True)
            top_edges = scored_edges[: max(1, op.k_comp)]
            max_sim = top_edges[0][1] if top_edges else 0.0
            exp_scores = [math.exp(max(-10.0, min(10.0, 6.0 * (s - max_sim)))) for _, s, _ in top_edges]
            denom = max(sum(exp_scores), 1e-8)

            for (v, raw_sim, tier_used), e_sc in zip(top_edges, exp_scores):
                prob = e_sc / denom
                prov_view.add_edge(
                    ProvenanceEdge(
                        src_slice_id=u.slice_id,
                        dst_slice_id=v.slice_id,
                        src_cam=u.cam_id,
                        dst_cam=v.cam_id,
                        dt_sec=v.t_enter - u.t_exit,
                        raw_similarity=raw_sim,
                        transition_prob=prob,
                        reduced_cost_slack=max(0.0, (exp_scores[0] / denom) - prob),
                        tier_evaluated=tier_used,
                    )
                )

        prov_view.finalize_topk()
        enforce_preds = (
            list(op.predicates)
            if (not op.downstream_multihop and op.predicates)
            else []
        )
        trajectories = self._extract_best_trajectories(
            slices_sorted,
            slice_map,
            prov_view,
            required_predicates=enforce_preds,
            min_hops=max(2, int(op.min_co_move_hops)),
        )
        return prov_view, trajectories

    @staticmethod
    def _best_slice_embed(s: VisualDetectionSlice) -> np.ndarray:
        if s.tier2_embed is not None:
            return s.tier2_embed
        if s.tier1_embed is not None:
            return s.tier1_embed
        return s.tier0_embed

    def _extract_best_trajectories(
        self,
        slices_sorted: List[VisualDetectionSlice],
        slice_map: Dict[int, VisualDetectionSlice],
        prov_view: CompetitiveProvenanceView,
        required_predicates: Optional[List[str]] = None,
        min_hops: int = 2,
    ) -> Dict[int, List[int]]:
        """
        Extracts node-disjoint, distinct-camera maximum-a-posteriori (MAP) trajectories over
        the time-ordered provenance DAG via coherent bounded beam search, preventing camera
        re-visit cycles and weak cross-vehicle tail grafting.
        """
        use_colocated_mid = self.topology.co_located_slack_sec > 0.0
        base_edge_conf = self.link_min_edge_confidence
        tau_assoc = (base_edge_conf + 0.02) if use_colocated_mid else base_edge_conf
        eff_min_hops = max(2, int(min_hops))
        ordered_ids = [s.slice_id for s in slices_sorted if s.slice_id in slice_map]
        used: Set[int] = set()
        trajectories: Dict[int, List[int]] = {}
        traj_idx = 0
        beam_width = 6
        max_chain_len = 8

        def _chain_map_score(c_ids: List[int], c_sims: List[float]) -> float:
            mean_s = sum(c_sims) / float(len(c_sims))
            if len(c_ids) >= 3:
                end_s = _cosine_sim(
                    self._best_slice_embed(slice_map[c_ids[0]]),
                    self._best_slice_embed(slice_map[c_ids[-1]]),
                )
            else:
                end_s = c_sims[0]
            hop_factor = 1.0 if len(c_ids) >= eff_min_hops else 0.45
            return hop_factor * (0.82 * mean_s + 0.18 * max(0.0, end_s))

        while True:
            best_global_score = -1.0
            best_global_chain: Optional[List[int]] = None

            for start_uid in ordered_ids:
                if start_uid in used:
                    continue
                out0 = [
                    e
                    for e in prov_view.outgoing_edges.get(start_uid, [])
                    if e.dst_slice_id not in used
                    and e.dst_slice_id in slice_map
                    and slice_map[e.dst_slice_id].cam_id != slice_map[start_uid].cam_id
                    and e.raw_similarity >= tau_assoc
                ]
                if not out0:
                    continue

                # Beam items: (chain_ids, visited_cams, edge_sims)
                beams: List[Tuple[List[int], Set[int], List[float]]] = [
                    ([start_uid], {slice_map[start_uid].cam_id}, [])
                ]
                completed: List[Tuple[List[int], List[float]]] = []

                for _ in range(max_chain_len - 1):
                    nxt_beams: List[Tuple[List[int], Set[int], List[float]]] = []
                    for chain, vcams, esims in beams:
                        last_id = chain[-1]
                        mean_prev = (sum(esims) / float(len(esims))) if esims else 0.0
                        extended = False
                        for e in prov_view.outgoing_edges.get(last_id, []):
                            dst = e.dst_slice_id
                            if dst in used or dst not in slice_map:
                                continue
                            d_cam = slice_map[dst].cam_id
                            if d_cam in vcams or e.raw_similarity < tau_assoc:
                                continue
                            if esims and e.raw_similarity < (mean_prev - 0.16):
                                continue
                            if len(chain) >= eff_min_hops and e.raw_similarity < max(
                                tau_assoc + 0.08, mean_prev - 0.04
                            ):
                                continue
                            if len(chain) >= 2:
                                anchor_sim = _cosine_sim(
                                    self._best_slice_embed(slice_map[chain[0]]),
                                    self._best_slice_embed(slice_map[dst]),
                                )
                                if anchor_sim < (tau_assoc - 0.04):
                                    continue
                            new_chain = chain + [dst]
                            new_sims = esims + [e.raw_similarity]
                            nxt_beams.append((new_chain, vcams | {d_cam}, new_sims))
                            completed.append((new_chain, new_sims))
                            extended = True
                        if not extended and len(chain) >= 2:
                            completed.append((chain, esims))
                    if not nxt_beams:
                        break
                    nxt_beams.sort(
                        key=lambda b: _chain_map_score(b[0], b[2]),
                        reverse=True,
                    )
                    beams = nxt_beams[:beam_width]

                for cand_chain, cand_sims in completed:
                    if len(cand_chain) < 2:
                        continue
                    if required_predicates and not self._chain_satisfies_predicates(
                        cand_chain, slice_map, required_predicates
                    ):
                        continue
                    sc = _chain_map_score(cand_chain, cand_sims)
                    if sc > best_global_score:
                        best_global_score = sc
                        best_global_chain = cand_chain

            if best_global_chain is None or len(best_global_chain) < 2:
                break
            trajectories[traj_idx] = best_global_chain
            traj_idx += 1
            used.update(best_global_chain)

        return trajectories

    def execute_sem_comove_join(
        self,
        slices_a: List[VisualDetectionSlice],
        slices_b: List[VisualDetectionSlice],
        prov_a: CompetitiveProvenanceView,
        prov_b: CompetitiveProvenanceView,
        op: PhysicalOperatorNode,
    ) -> List[Tuple[List[int], List[int], float]]:
        """
        Executes SEM_COMOVE_JOIN(P_k^(A), P_k^(B)) with Damped Coupled Provenance Belief
        Propagation across Top-k_comp transition hypotheses and distinct-camera MAP
        DAG chain extraction (Algorithm 2).
        """
        use_colocated_mid = self.topology.co_located_slack_sec > 0.0
        base_pair_conf = self.comove_min_pair_confidence
        min_single_sim = (base_pair_conf - 0.01) if use_colocated_mid else (base_pair_conf - 0.03)
        min_coupled_sc = (base_pair_conf + 0.03) if use_colocated_mid else base_pair_conf
        map_a = {s.slice_id: s for s in slices_a}
        map_b = {s.slice_id: s for s in slices_b}

        def _t_ref(s: VisualDetectionSlice) -> float:
            return 0.5 * (s.t_enter + s.t_exit) if use_colocated_mid else s.t_enter

        # Index candidate slices by camera for co-presence headway matching
        b_by_cam: Dict[int, List[VisualDetectionSlice]] = {}
        for sb in slices_b:
            b_by_cam.setdefault(sb.cam_id, []).append(sb)

        # Pairwise co-presence states (u_a, u_b) at each camera
        co_pairs_by_cam: Dict[int, List[Tuple[int, int, float]]] = {}
        for sa in slices_a:
            ta_ref = _t_ref(sa)
            for sb in b_by_cam.get(sa.cam_id, []):
                if sa.slice_id == sb.slice_id:
                    continue
                headway = abs(ta_ref - _t_ref(sb))
                if headway <= op.join_window_sec:
                    co_pairs_by_cam.setdefault(sa.cam_id, []).append(
                        (sa.slice_id, sb.slice_id, headway)
                    )

        # Coupled Provenance Edge Join with Damped Mutual Belief Propagation
        coupled_adj: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]] = {}
        pair_nodes_set: Set[Tuple[int, int]] = set()
        for _, pairs in co_pairs_by_cam.items():
            for ua_id, ub_id, _ in pairs:
                ua_obj = map_a[ua_id]
                ub_obj = map_b[ub_id]
                edges_a = prov_a.outgoing_edges.get(ua_id, [])
                edges_b = prov_b.outgoing_edges.get(ub_id, [])
                if not edges_a or not edges_b:
                    continue

                for ea in edges_a:
                    if ea.raw_similarity < min_single_sim:
                        continue
                    for eb in edges_b:
                        if eb.raw_similarity < min_single_sim or ea.dst_cam != eb.dst_cam:
                            continue
                        va = map_a.get(ea.dst_slice_id)
                        vb = map_b.get(eb.dst_slice_id)
                        if va is None or vb is None or va.slice_id == vb.slice_id:
                            continue
                        if va.cam_id == ua_obj.cam_id:
                            continue
                        if use_colocated_mid and (
                            _t_ref(va) + _t_ref(vb),
                            va.slice_id,
                            vb.slice_id,
                        ) <= (
                            _t_ref(ua_obj) + _t_ref(ub_obj),
                            ua_id,
                            ub_id,
                        ):
                            continue
                        hw_dst = abs(_t_ref(va) - _t_ref(vb))
                        if hw_dst > op.join_window_sec:
                            continue

                        # Coupled headway consistency kernel Phi_couple (Algorithm 2)
                        dt_drift = abs(
                            (_t_ref(va) - _t_ref(vb)) - (_t_ref(ua_obj) - _t_ref(ub_obj))
                        )
                        phi_couple = math.exp(-3.5 * (dt_drift / max(2.0, op.join_window_sec)) ** 2)

                        mean_vis = 0.50 * (ea.raw_similarity + eb.raw_similarity)
                        min_vis = min(ea.raw_similarity, eb.raw_similarity)
                        vis_joint = 0.65 * mean_vis + 0.35 * min_vis
                        coupled_score = vis_joint * (0.84 + 0.16 * phi_couple)

                        if coupled_score >= min_coupled_sc:
                            src_pair = (ua_id, ub_id)
                            dst_pair = (va.slice_id, vb.slice_id)
                            coupled_adj.setdefault(src_pair, []).append((dst_pair, coupled_score))
                            pair_nodes_set.add(src_pair)
                            pair_nodes_set.add(dst_pair)

        if not pair_nodes_set:
            return []

        for p_k in coupled_adj:
            coupled_adj[p_k].sort(key=lambda item: item[1], reverse=True)

        # Algorithm 2: ExtractDAGChains via distinct-camera MAP beam search on coupled pair DAG
        min_hops = max(2, op.min_co_move_hops)
        max_len = min_hops + 2
        beam_width = 6
        max_drift = max(10.0, 0.85 * op.join_window_sec)
        topo_pairs = sorted(
            pair_nodes_set,
            key=lambda p: (_t_ref(map_a[p[0]]) + _t_ref(map_b[p[1]]), p[0], p[1]),
        )
        results: List[Tuple[List[int], List[int], float]] = []
        used_pairs: Set[Tuple[int, int]] = set()
        slice_use_counts: Dict[int, int] = {}
        max_slice_reuse = 2

        def _coupled_map_score(p_list: List[Tuple[int, int]], e_list: List[float]) -> float:
            mean_e = sum(e_list) / float(len(e_list))
            if len(p_list) >= 3:
                end_a = _cosine_sim(
                    self._best_slice_embed(map_a[p_list[0][0]]),
                    self._best_slice_embed(map_a[p_list[-1][0]]),
                )
                end_b = _cosine_sim(
                    self._best_slice_embed(map_b[p_list[0][1]]),
                    self._best_slice_embed(map_b[p_list[-1][1]]),
                )
            else:
                end_a = mean_e
                end_b = mean_e
            return 0.78 * mean_e + 0.11 * max(0.0, end_a) + 0.11 * max(0.0, end_b)

        while True:
            best_score = -1.0
            best_pair_chain: Optional[List[Tuple[int, int]]] = None
            best_edge_scores: List[float] = []

            for start_p in topo_pairs:
                if start_p in used_pairs or start_p not in coupled_adj:
                    continue
                if (
                    slice_use_counts.get(start_p[0], 0) >= max_slice_reuse
                    or slice_use_counts.get(start_p[1], 0) >= max_slice_reuse
                ):
                    continue
                start_cam = map_a[start_p[0]].cam_id
                init_signed_hw = _t_ref(map_a[start_p[0]]) - _t_ref(map_b[start_p[1]])

                # Beam items: (pair_chain, set_a, set_b, visited_cams, edge_scores)
                beams: List[
                    Tuple[List[Tuple[int, int]], Set[int], Set[int], Set[int], List[float]]
                ] = [([start_p], {start_p[0]}, {start_p[1]}, {start_cam}, [])]

                for _ in range(max_len - 1):
                    nxt_beams: List[
                        Tuple[List[Tuple[int, int]], Set[int], Set[int], Set[int], List[float]]
                    ] = []
                    for p_chain, sa_set, sb_set, vcams, escs in beams:
                        last_p = p_chain[-1]
                        mean_prev = (sum(escs) / float(len(escs))) if escs else 0.0
                        for nxt_p, e_sc in coupled_adj.get(last_p, [])[:beam_width]:
                            if nxt_p in used_pairs:
                                continue
                            va_id, vb_id = nxt_p
                            if (
                                slice_use_counts.get(va_id, 0) >= max_slice_reuse
                                or slice_use_counts.get(vb_id, 0) >= max_slice_reuse
                            ):
                                continue
                            if va_id in sa_set or vb_id in sb_set or va_id in sb_set or vb_id in sa_set:
                                continue
                            d_cam = map_a[va_id].cam_id
                            if d_cam in vcams:
                                continue
                            if escs and e_sc < (mean_prev - 0.14):
                                continue
                            if len(p_chain) >= min_hops and e_sc < max(
                                min_coupled_sc + 0.08, mean_prev - 0.04
                            ):
                                continue
                            if len(p_chain) >= 2:
                                anch_a = _cosine_sim(
                                    self._best_slice_embed(map_a[p_chain[0][0]]),
                                    self._best_slice_embed(map_a[va_id]),
                                )
                                anch_b = _cosine_sim(
                                    self._best_slice_embed(map_b[p_chain[0][1]]),
                                    self._best_slice_embed(map_b[vb_id]),
                                )
                                if anch_a < (min_single_sim - 0.04) or anch_b < (
                                    min_single_sim - 0.04
                                ):
                                    continue
                            cur_signed_hw = _t_ref(map_a[va_id]) - _t_ref(map_b[vb_id])
                            if abs(cur_signed_hw - init_signed_hw) > max_drift:
                                continue

                            new_p_chain = p_chain + [nxt_p]
                            new_escs = escs + [e_sc]
                            nxt_beams.append(
                                (
                                    new_p_chain,
                                    sa_set | {va_id},
                                    sb_set | {vb_id},
                                    vcams | {d_cam},
                                    new_escs,
                                )
                            )
                            if len(new_p_chain) >= min_hops:
                                if op.predicates or op.secondary_predicates:
                                    cand_a_ids = [p[0] for p in new_p_chain]
                                    cand_b_ids = [p[1] for p in new_p_chain]
                                    if op.predicates and not self._chain_satisfies_predicates(
                                        cand_a_ids, map_a, op.predicates
                                    ):
                                        continue
                                    if (
                                        op.secondary_predicates
                                        and not self._chain_satisfies_predicates(
                                            cand_b_ids, map_b, op.secondary_predicates
                                        )
                                    ):
                                        continue
                                sc_val = _coupled_map_score(new_p_chain, new_escs)
                                if sc_val > best_score:
                                    best_score = sc_val
                                    best_pair_chain = new_p_chain
                                    best_edge_scores = new_escs

                    if not nxt_beams:
                        break
                    nxt_beams.sort(
                        key=lambda b: _coupled_map_score(b[0], b[4]),
                        reverse=True,
                    )
                    beams = nxt_beams[:beam_width]

            if best_pair_chain is None or len(best_pair_chain) < min_hops:
                break

            used_pairs.update(best_pair_chain)
            used_pairs.update((b_id, a_id) for a_id, b_id in best_pair_chain)
            for a_id, b_id in best_pair_chain:
                slice_use_counts[a_id] = slice_use_counts.get(a_id, 0) + 1
                slice_use_counts[b_id] = slice_use_counts.get(b_id, 0) + 1
            chain_a = [p[0] for p in best_pair_chain]
            chain_b = [p[1] for p in best_pair_chain]
            mean_sc = float(np.mean(best_edge_scores)) if best_edge_scores else 0.5
            results.append((chain_a, chain_b, mean_sc))

        return results
