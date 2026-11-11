"""
Spatio-Temporal Data Model, Competitive Provenance View, Multi-Granularity
Visual Token Schema, and Physical Operator Plan DAG for ZadigDB.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math
import numpy as np


class TokenResolutionTier(int, Enum):
    """Three-tier visual token hierarchy per camera-time detection slice (Section IV-A)."""
    TIER_0_MICRO = 1       # 1 summary descriptor token (28x28 crop, resident in B0 index)
    TIER_1_COARSE = 32     # 32 regional sub-grid tokens (224x112 single-frame crop)
    TIER_2_FINE = 256      # 256 dense patch tokens (448x448 2x2 4-frame temporal montage)


class OperatorType(str, Enum):
    """Multi-camera spatio-temporal semantic algebraic operators (Section II-C)."""
    SEM_FILTER = "SEM_FILTER"               # Resolution-agnostic semantic attribute and motion filter
    SEM_LINK = "SEM_LINK"                   # Cross-camera entity linkage over Top-k provenance view
    SEM_COMOVE_JOIN = "SEM_COMOVE_JOIN"     # Coupled multi-entity co-movement join


@dataclass
class CameraTopology:
    """
    Spatio-Temporal Camera Topology Graph G = (C, E_G, d_G, W) (Section II-A, Definition 1).
    Encodes directed road-network shortest-path distances d_G(c_i, c_j), kinematic
    speed bounds [v_min, v_max], and traffic signal phase slack tau_sig.
    """
    num_cameras: int
    camera_ids: List[int]
    dist_matrix: np.ndarray                 # Shape (M, M), shortest-path road distance in meters
    adjacency: np.ndarray                   # Shape (M, M), binary directed road-network reachability
    v_min_mps: float = 5.5                  # Minimum urban arterial cruising speed (m/s)
    v_max_mps: float = 21.0                 # Maximum urban arterial cruising speed (m/s)
    tau_sig_sec: float = 18.0               # Signal phase delay slack (seconds)
    fov_overlap_m: float = 0.0              # Effective inter-camera FOV road-reach overlap (meters)
    co_located_slack_sec: float = 0.0       # Concurrent visibility slack for co-located junction cameras (seconds)
    congestion_speed_floor_mps: float = 0.8 # Minimum corridor speed floor under severe congestion (m/s)
    congestion_clearance_scale: float = 2.2 # Clearance scaling factor for congested signal queues
    _cam_to_idx: Dict[int, int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._cam_to_idx = {int(cid): idx for idx, cid in enumerate(self.camera_ids)}

    def distance_m(self, cam_src: int, cam_dst: int) -> float:
        """Returns shortest-path road distance d_G(cam_src, cam_dst) in meters (inf if unreachable)."""
        src_idx = self._cam_to_idx.get(cam_src)
        dst_idx = self._cam_to_idx.get(cam_dst)
        if src_idx is None or dst_idx is None:
            return float("inf")
        return float(self.dist_matrix[src_idx, dst_idx])

    def are_adjacent(self, cam_src: int, cam_dst: int) -> bool:
        """Returns True if directed road-network edge (cam_src -> cam_dst) exists in E_G."""
        src_idx = self._cam_to_idx.get(cam_src)
        dst_idx = self._cam_to_idx.get(cam_dst)
        if src_idx is None or dst_idx is None:
            return False
        return bool(self.adjacency[src_idx, dst_idx] != 0)

    def max_causal_window_sec(self) -> float:
        """
        Returns the maximum causal transition upper bound tau_max(G) across all directed
        road edges (c_i, c_j) in E_G: max(d_G(c_i, c_j) / v_min + tau_sig) (Eq. 1).
        """
        adj_mask = self.adjacency > 0
        if not np.any(adj_mask):
            return 0.0
        valid_dists = self.dist_matrix[adj_mask]
        valid_dists = valid_dists[np.isfinite(valid_dists) & (valid_dists > 0.0)]
        if valid_dists.size == 0:
            return 0.0
        return float(np.max(valid_dists) / max(self.v_min_mps, 1e-3) + self.tau_sig_sec)

    def causal_window(self, cam_src: int, cam_dst: int, t_exit: float) -> Tuple[float, float]:
        """
        Computes the Clause C2 kinematic causal transition window W(c_i, c_j, t_i) (Eq. 1):
            [t_i + d_eff(c_i, c_j) / v_max - tau_coloc,  t_i + d_G(c_i, c_j) / v_min + tau_sig]
        """
        if cam_src == cam_dst:
            return (t_exit, t_exit + 5.0)
        d_ij = self.distance_m(cam_src, cam_dst)
        if d_ij <= 0.0 or not math.isfinite(d_ij):
            return (-1.0, -2.0)
        d_eff = max(0.0, d_ij - self.fov_overlap_m)
        co_slack = self.co_located_slack_sec if d_ij <= 180.0 else 0.0
        eff_sig_slack = (
            min(self.tau_sig_sec, 12.0)
            if (self.fov_overlap_m > 0.0 and d_ij <= self.fov_overlap_m)
            else self.tau_sig_sec
        )
        t_earliest = max(0.0, t_exit + (d_eff / max(self.v_max_mps, 1e-3)) - co_slack)
        t_latest = t_exit + d_G_over_vmin(d_ij, self.v_min_mps) + eff_sig_slack
        return (t_earliest, t_latest)

    def is_causally_reachable(
        self, cam_src: int, cam_dst: int, t_exit: float, t_enter: float, margin_sec: float = 2.0
    ) -> bool:
        """Checks whether transition (cam_src, t_exit) -> (cam_dst, t_enter) satisfies Clause C2."""
        if cam_src == cam_dst:
            return abs(t_enter - t_exit) <= margin_sec
        if not self.are_adjacent(cam_src, cam_dst):
            return False
        t_min, t_max = self.causal_window(cam_src, cam_dst, t_exit)
        if t_min < 0.0 or t_max < t_min:
            return False
        return (t_min - margin_sec) <= t_enter <= (t_max + margin_sec)

    def adapt_signal_slack_from_telemetry(
        self,
        observed_mean_speed_mps: float,
        base_tau_sig_sec: float = 18.0,
        max_tau_sig_sec: float = 110.0,
    ) -> float:
        """
        Dynamically widens the Clause C2 traffic signal phase slack tau_sig under
        arterial congestion using corridor speed telemetry (Section V-F).
        """
        effective_speed = max(self.congestion_speed_floor_mps, float(observed_mean_speed_mps))
        if effective_speed < self.v_min_mps:
            congestion_factor = self.v_min_mps / effective_speed
            self.tau_sig_sec = min(max_tau_sig_sec, base_tau_sig_sec * congestion_factor)
        else:
            self.tau_sig_sec = base_tau_sig_sec
        return self.tau_sig_sec


def d_G_over_vmin(d_ij: float, v_min: float) -> float:
    return d_ij / max(v_min, 1e-3)


@dataclass
class VisualDetectionSlice:
    """
    A relational candidate detection slice tuple u = (slice_id, cam_id, t_enter, t_exit)
    in an unmaterialized multi-camera video stream (Section II-A).
    """
    slice_id: int
    cam_id: int
    t_enter: float
    t_exit: float
    tier0_embed: np.ndarray                                              # Shape (d,), Tier-0 1-token summary descriptor
    tier1_embed: Optional[np.ndarray] = None                             # Shape (d,), Tier-1 32-token regional descriptor
    tier2_embed: Optional[np.ndarray] = None                             # Shape (d,), Tier-2 256-token dense patch descriptor
    attribute_scores_t0: Dict[str, float] = field(default_factory=dict)  # Tier-0 predicate confidences
    attribute_scores_t1: Dict[str, float] = field(default_factory=dict)  # Tier-1 predicate confidences
    attribute_scores_t2: Dict[str, float] = field(default_factory=dict)  # Tier-2 predicate confidences
    predicate_scores: Dict[str, float] = field(default_factory=dict)     # Highest-tier evaluated predicate scores
    frame_paths: Tuple[str, ...] = field(default_factory=tuple)          # Chronological crop image paths


@dataclass
class ProvenanceEdge:
    """Directed transition hypothesis e = (u -> v) in the Top-k Competitive Provenance View (Definition 2)."""
    src_slice_id: int
    dst_slice_id: int
    src_cam: int
    dst_cam: int
    dt_sec: float
    raw_similarity: float
    transition_prob: float
    reduced_cost_slack: float
    tier_evaluated: TokenResolutionTier = TokenResolutionTier.TIER_1_COARSE


@dataclass
class CompetitiveProvenanceView:
    """
    Top-k_comp Competitive Provenance Graph View P_k = (U', E_k, s, p, delta) (Section II-B, Definition 2).
    Retains up to k_comp competitive outgoing transition hypotheses per slice u (Eqs. 2-3),
    preventing the p^{2k} recall collapse of greedy Top-1 trajectory materialization (Theorem 1).
    """
    k_comp: int
    outgoing_edges: Dict[int, List[ProvenanceEdge]] = field(default_factory=dict)
    incoming_edges: Dict[int, List[ProvenanceEdge]] = field(default_factory=dict)

    def add_edge(self, edge: ProvenanceEdge) -> None:
        self.outgoing_edges.setdefault(edge.src_slice_id, []).append(edge)
        self.incoming_edges.setdefault(edge.dst_slice_id, []).append(edge)

    def finalize_topk(self) -> None:
        """Retains top-k_comp edges per source slice and normalizes probabilities & slack (Eqs. 2-3)."""
        self.incoming_edges.clear()
        for src_id, edges in list(self.outgoing_edges.items()):
            if not edges:
                self.outgoing_edges[src_id] = []
                continue
            edges_sorted = sorted(edges, key=lambda e: e.transition_prob, reverse=True)[: self.k_comp]
            clamped = [max(e.transition_prob, 1e-6) for e in edges_sorted]
            total = sum(clamped)
            for e, c_val in zip(edges_sorted, clamped):
                e.transition_prob = c_val / total
            best_p = edges_sorted[0].transition_prob
            for e in edges_sorted:
                e.reduced_cost_slack = max(0.0, best_p - e.transition_prob)
                self.incoming_edges.setdefault(e.dst_slice_id, []).append(e)
            self.outgoing_edges[src_id] = edges_sorted

    def total_edges(self) -> int:
        return sum(len(v) for v in self.outgoing_edges.values())


@dataclass
class PhysicalOperatorNode:
    """A physical operator node in the synthesized execution plan DAG (Section III-A)."""
    node_id: str
    op_type: OperatorType
    predicates: List[str]
    target_cameras: List[int]
    k_comp: int = 4
    use_causal_window_c2: bool = True
    causal_window_sec: Optional[float] = None
    max_tier2_slices: int = 384
    max_tier1_slices: int = 1536
    join_window_sec: float = 15.0
    min_co_move_hops: int = 3
    secondary_predicates: List[str] = field(default_factory=list)
    downstream_multihop: bool = False


@dataclass
class PhysicalPlanDAG:
    """Synthesized Physical Query Plan DAG certified by the Clause-Level Admission Gate (Section III)."""
    query_id: str
    query_class: str                        # "Q1", "Q2", "Q3", or "Q4"
    sql_query: str
    operators: List[PhysicalOperatorNode]
    num_hops: int = 1
    estimated_vram_gb: float = 0.0
    admitted_c1: bool = False
    admitted_c2: bool = False
    admitted_c3: bool = False
    repair_steps: int = 0
