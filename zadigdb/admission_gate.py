"""
Clause-Level Admission Gate for ZadigDB.

Audits synthesized physical operator plans against three admission clauses (Section III-B, Algorithm 1):
  - Clause C1 (VRAM Ceiling): Guarantees analytical peak GPU allocation M_peak(P) <= B_VRAM < B_hard
    (parameterized GPU memory budget, instantiated at 22.5 GB < 24.0 GB on a single RTX 4090, Definition 3, Eq. 4).
  - Clause C2 (Causal Reachability Pushdown): Enforces road-network kinematic reachability W(c_i, c_j, t_i)
    with transition horizon >= tau_max(G) (Eq. 1).
  - Clause C3 (Coupled Provenance Recall): Retains k_comp competitive hypotheses on multi-camera queries
    (H(Q) >= 2) to prevent p^{2k} recall collapse (Theorem 1, Eqs. 5-6).
"""

from dataclasses import dataclass
import math
import time
from typing import Tuple

from .schema import (
    CameraTopology,
    OperatorType,
    PhysicalPlanDAG,
    TokenResolutionTier,
)


@dataclass
class AdmissionAuditReport:
    """Result of auditing a synthesized PhysicalPlanDAG against Clauses C1, C2, and C3."""
    passed_all: bool
    passed_c1: bool
    passed_c2: bool
    passed_c3: bool
    estimated_vram_gb: float
    vram_budget_gb: float
    pruned_transition_ratio_c2: float
    theoretical_top1_recall: float
    theoretical_coupled_recall: float
    semantic_error_bound: float
    repair_iterations: int
    optimization_time_ms: float = 0.0


def theoretical_recall_collapse_top1(
    per_hop_top1_acc: float, num_hops: int, is_coupled_join: bool = True
) -> float:
    """
    Theorem 1 (Part A, Eq. 5): Under Eager Materialize-then-Query (Top-1 argmax assignment),
    a single trajectory across k inter-camera hops has recall p^k, while a binary co-movement
    join (SEM_COMOVE_JOIN) suffers exponential recall collapse:
        R_{Mat-Top1}(k) = p^{2k}.
    """
    p = max(0.0, min(1.0, per_hop_top1_acc))
    k = max(1, int(num_hops))
    exponent = 2 * k if is_coupled_join else k
    return float(p ** exponent)


def theoretical_coupled_provenance_recall(
    per_hop_top1_acc: float,
    k_comp: int,
    num_hops: int,
    is_coupled_join: bool = True,
    coupling_consistency_gain: float = 0.85,
    rank_tail_alpha: float = 0.55,
) -> float:
    """
    Theorem 1 (Part B, Eq. 6): Under Clause C3 Top-k_comp Competitive Provenance Retention
    with Coupled Provenance Join (SEM_COMOVE_JOIN), retaining k_comp >= 2 hypotheses bounds
    per-hop candidate exclusion by:
        epsilon_{k_comp} <= (1 - p) * exp(-alpha * (k_comp - 1)),
    and coupled headway disambiguation bounds residual rank inversions by delta_couple(k_comp):
        R_{ZadigDB}(k) >= (1 - epsilon_{k_comp})^{2k} * (1 - delta_couple(k_comp)).
    """
    p = max(0.01, min(0.999, per_hop_top1_acc))
    k = max(1, int(num_hops))
    k_c = max(1, int(k_comp))
    if k_c == 1:
        return theoretical_recall_collapse_top1(p, k, is_coupled_join=is_coupled_join)

    alpha = max(0.05, float(rank_tail_alpha))
    eps_k = (1.0 - p) * math.exp(-alpha * (k_c - 1))
    exponent = 2 * k if is_coupled_join else k
    survival = (1.0 - eps_k) ** exponent
    delta_couple = (
        max(0.0, (1.0 - coupling_consistency_gain) * (1.0 - p) / math.sqrt(k_c))
        if is_coupled_join
        else 0.0
    )
    return float(max(0.0, min(1.0, survival * (1.0 - delta_couple))))


class ClauseAdmissionGate:
    """
    Clause-Level Admission Gate & Closed-Loop Plan Repair Engine (Section III-B & III-D, Algorithm 1).
    """

    def __init__(
        self,
        vram_budget_gb: float = 22.5,
        hard_vram_limit_gb: float = 24.0,
        base_model_weights_gb: float = 8.174,
        tier0_descriptor_bytes: float = 2048.0,
        kv_bytes_per_token: float = 131072.0,  # 128 KiB per visual token (Section IV-A)
        min_coupled_recall: float = 0.55,
        default_per_hop_p: float = 0.63,
        rank_tail_alpha: float = 0.55,
        coupling_consistency_gain: float = 0.85,
        delta_tier: float = 0.12,
        stagnation_epsilon: float = 0.015,
        min_compression_alpha: float = 0.25,
    ):
        self.vram_budget_gb = vram_budget_gb
        self.hard_vram_limit_gb = hard_vram_limit_gb
        self.base_model_weights_gb = base_model_weights_gb
        self.tier0_descriptor_bytes = tier0_descriptor_bytes
        self.kv_bytes_per_token = kv_bytes_per_token
        self.min_coupled_recall = min_coupled_recall
        self.default_per_hop_p = default_per_hop_p
        self.rank_tail_alpha = float(rank_tail_alpha)
        self.coupling_consistency_gain = float(coupling_consistency_gain)
        self.delta_tier = float(delta_tier)
        self.stagnation_epsilon = stagnation_epsilon
        self.min_compression_alpha = min_compression_alpha

    def estimate_plan_vram_gb(
        self,
        plan: PhysicalPlanDAG,
        total_tier0_slices: int,
    ) -> float:
        """
        Definition 3 & Theorem 2 (Part 1, Eqs. 4 & 9 - Strict VRAM Safety Invariant):
        Computes the analytical upper bound on peak GPU memory (in GB) for physical plan P:
            M_peak(P) = M_weights + N * tau_0 * beta_0 + (N_1 * tau_1 + N_2 * tau_2) * gamma_KV + M_prov(N, k_comp).
        """
        max_t1 = max((op.max_tier1_slices for op in plan.operators), default=512)
        max_t2 = max((op.max_tier2_slices for op in plan.operators), default=128)
        max_k = max((op.k_comp for op in plan.operators), default=4)

        tokens_t0 = total_tier0_slices * int(TokenResolutionTier.TIER_0_MICRO)
        tokens_t1 = max_t1 * int(TokenResolutionTier.TIER_1_COARSE)
        tokens_t2 = max_t2 * int(TokenResolutionTier.TIER_2_FINE)

        t0_gb = (tokens_t0 * self.tier0_descriptor_bytes) / (1024.0 ** 3)
        kv_gb = ((tokens_t1 + tokens_t2) * self.kv_bytes_per_token) / (1024.0 ** 3)
        prov_gb = (total_tier0_slices * max_k * 64.0) / (1024.0 ** 3) + 0.35

        return round(self.base_model_weights_gb + t0_gb + kv_gb + prov_gb, 3)

    def estimate_semantic_error_bound(
        self,
        plan: PhysicalPlanDAG,
        total_slices: int,
        lipschitz_constant: float = 0.18,
    ) -> float:
        """
        Theorem 2 (Part 2, Eq. 10 - Escalation Completeness & Discrepancy Bound):
        Bounds expected score discrepancy Delta_err(P) relative to exhaustive full-page evaluation:
            Delta_err(P) <= epsilon_win + L_f * Delta_tier * (1 - rho_esc(P)).
        """
        uses_c2 = all(op.use_causal_window_c2 for op in plan.operators)
        eps_win = 0.008 if uses_c2 else 0.145
        max_t1 = max((op.max_tier1_slices for op in plan.operators), default=512)
        max_t2 = max((op.max_tier2_slices for op in plan.operators), default=128)
        active_escalated = min(
            1.0,
            (max_t1 + 2.0 * max_t2) / max(1.0, self.min_compression_alpha * total_slices),
        )
        delta_tier = self.delta_tier
        err = eps_win + lipschitz_constant * delta_tier * max(0.0, 1.0 - active_escalated)
        return round(float(err), 4)

    def _estimate_coupled_recall(
        self,
        plan: PhysicalPlanDAG,
        num_hops: int,
    ) -> Tuple[bool, float, float]:
        """
        Evaluates Clause C3 (Algorithm 1, line 5):
        For single-camera queries (H(Q) < 2 or no SEM_LINK / SEM_COMOVE_JOIN operators),
        Clause C3 holds trivially with recall 1.0 without mutating k_comp.
        """
        multihop_ops = [
            op
            for op in plan.operators
            if op.op_type in (OperatorType.SEM_LINK, OperatorType.SEM_COMOVE_JOIN)
        ]
        eff_h = plan.num_hops if plan.num_hops > 0 else int(num_hops)
        if not multihop_ops or eff_h < 2:
            return True, 1.0, 1.0

        inter_cam_hops = max(1, eff_h - 1) if plan.num_hops >= 2 else max(1, int(num_hops))
        min_k = min(op.k_comp for op in multihop_ops)
        max_k = max(op.k_comp for op in multihop_ops)
        has_join = any(op.op_type == OperatorType.SEM_COMOVE_JOIN for op in multihop_ops)
        top1_rec = theoretical_recall_collapse_top1(
            self.default_per_hop_p, inter_cam_hops, is_coupled_join=has_join
        )
        coupled_rec = theoretical_coupled_provenance_recall(
            self.default_per_hop_p,
            min_k,
            inter_cam_hops,
            is_coupled_join=has_join,
            coupling_consistency_gain=self.coupling_consistency_gain,
            rank_tail_alpha=self.rank_tail_alpha,
        )
        return max_k <= 8 and coupled_rec >= self.min_coupled_recall, top1_rec, coupled_rec

    @staticmethod
    def _check_clause_c2(plan: PhysicalPlanDAG, tau_max_sec: float) -> bool:
        for op in plan.operators:
            if not op.use_causal_window_c2:
                return False
            if op.causal_window_sec is not None and (op.causal_window_sec + 1e-6) < tau_max_sec:
                return False
        return True

    def audit_and_repair(
        self,
        plan: PhysicalPlanDAG,
        topology: CameraTopology,
        total_slices: int,
        num_hops: int = 4,
        auto_repair: bool = True,
        stream_horizon_sec: float = 340.0,
        pre_synthesis_ms: float = 0.0,
    ) -> AdmissionAuditReport:
        """
        Audits a PhysicalPlanDAG against Clauses C1, C2, and C3 (Algorithm 1).
        If auto_repair=True, iteratively repairs violated clauses on CPU prior to execution.
        """
        t_start = time.perf_counter()
        repair_steps = 0
        tau_max_sec = round(topology.max_causal_window_sec(), 2)
        min_t1_floor = max(64, int(round(self.min_compression_alpha * 1024)))
        min_t2_floor = max(16, int(round(self.min_compression_alpha * 256)))
        prev_vram: float = float("inf")
        prev_contracted_c1: bool = False

        while True:
            est_vram = self.estimate_plan_vram_gb(plan, total_slices)
            passed_c1 = est_vram <= self.vram_budget_gb
            passed_c2 = self._check_clause_c2(plan, tau_max_sec)
            passed_c3, _, _ = self._estimate_coupled_recall(plan, num_hops)

            if (passed_c1 and passed_c2 and passed_c3) or not auto_repair or repair_steps >= 8:
                break
            if (
                prev_contracted_c1
                and not passed_c1
                and passed_c2
                and passed_c3
                and (prev_vram - est_vram) < self.stagnation_epsilon
            ):
                break

            prev_vram = est_vram
            prev_contracted_c1 = not passed_c1
            repair_steps += 1
            for op in plan.operators:
                if not passed_c1:
                    op.max_tier1_slices = min(1536, max(min_t1_floor, int(op.max_tier1_slices * 0.85)))
                    op.max_tier2_slices = min(384, max(min_t2_floor, int(op.max_tier2_slices * 0.78)))
                if not passed_c2:
                    op.use_causal_window_c2 = True
                    if op.causal_window_sec is None or op.causal_window_sec < tau_max_sec:
                        op.causal_window_sec = tau_max_sec
                if not passed_c3 and op.op_type in (OperatorType.SEM_LINK, OperatorType.SEM_COMOVE_JOIN):
                    op.k_comp = min(8, max(2, op.k_comp + 1))

        est_vram = self.estimate_plan_vram_gb(plan, total_slices)
        passed_c1 = est_vram <= self.vram_budget_gb
        if auto_repair and not passed_c1:
            floor_kv_gb = (
                (
                    min_t1_floor * int(TokenResolutionTier.TIER_1_COARSE)
                    + min_t2_floor * int(TokenResolutionTier.TIER_2_FINE)
                )
                * self.kv_bytes_per_token
            ) / (1024.0 ** 3)
            t0_gb = (
                total_slices * int(TokenResolutionTier.TIER_0_MICRO) * self.tier0_descriptor_bytes
            ) / (1024.0 ** 3)
            max_k = max((op.k_comp for op in plan.operators), default=4)
            prov_gb = (total_slices * max_k * 64.0) / (1024.0 ** 3) + 0.35
            m_fixed_gb = round(self.base_model_weights_gb + t0_gb + floor_kv_gb + prov_gb, 3)
            if self.vram_budget_gb >= m_fixed_gb:
                for op in plan.operators:
                    op.max_tier1_slices = min(op.max_tier1_slices, min_t1_floor)
                    op.max_tier2_slices = min(op.max_tier2_slices, min_t2_floor)
                est_vram = self.estimate_plan_vram_gb(plan, total_slices)
                passed_c1 = est_vram <= self.vram_budget_gb
        passed_c2 = self._check_clause_c2(plan, tau_max_sec)
        passed_c3, top1_rec, coupled_rec = self._estimate_coupled_recall(plan, num_hops)

        # Compute physical road-network topological density & kinematic causal window selectivity
        m = topology.num_cameras
        adj_mask = topology.adjacency > 0
        reachable_edges = float(adj_mask.sum())
        total_pairs = float(max(1, m * (m - 1)))
        topo_density = reachable_edges / total_pairs
        if passed_c2 and reachable_edges > 0:
            valid_dists = topology.dist_matrix[adj_mask]
            valid_dists = valid_dists[valid_dists > 0.0]
            if valid_dists.size > 0:
                spans = (
                    (valid_dists / max(topology.v_min_mps, 1e-3) + topology.tau_sig_sec)
                    - (valid_dists / max(topology.v_max_mps, 1e-3))
                    + 4.0
                )
                mean_span_sec = float(spans.mean())
            else:
                mean_span_sec = topology.tau_sig_sec + 4.0
            mean_forward_horizon = max(20.0, 0.5 * stream_horizon_sec)
            temporal_window_fraction = min(1.0, mean_span_sec / mean_forward_horizon)
            pruned_ratio = 1.0 - (topo_density * temporal_window_fraction)
        else:
            pruned_ratio = 0.0

        plan.estimated_vram_gb = est_vram
        plan.admitted_c1 = passed_c1
        plan.admitted_c2 = passed_c2
        plan.admitted_c3 = passed_c3
        plan.repair_steps = repair_steps
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        opt_time_ms = round(max(0.01, pre_synthesis_ms + elapsed_ms), 3)

        return AdmissionAuditReport(
            passed_all=(passed_c1 and passed_c2 and passed_c3),
            passed_c1=passed_c1,
            passed_c2=passed_c2,
            passed_c3=passed_c3,
            estimated_vram_gb=est_vram,
            vram_budget_gb=self.vram_budget_gb,
            pruned_transition_ratio_c2=round(max(0.0, min(0.999, pruned_ratio)), 4),
            theoretical_top1_recall=round(top1_rec, 4),
            theoretical_coupled_recall=round(coupled_rec, 4),
            semantic_error_bound=self.estimate_semantic_error_bound(plan, total_slices),
            repair_iterations=repair_steps,
            optimization_time_ms=opt_time_ms,
        )
