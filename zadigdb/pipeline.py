"""
End-to-End ZadigDB Query Execution Engine & Telemetry Collector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, List, Optional, Sequence, Tuple
import yaml

from .admission_gate import AdmissionAuditReport, ClauseAdmissionGate
from .algebra import SpatioTemporalSemanticAlgebra
from .buffer_pool import ProgressiveVisualBufferPool
from .schema import (
    CameraTopology,
    OperatorType,
    PhysicalOperatorNode,
    PhysicalPlanDAG,
    VisualDetectionSlice,
)
from .synthesizer import DeclarativeVideoQuery, TopologyAwarePlanSynthesizer
from .vlm_backend import DEFAULT_CALIBRATED_TIMINGS


@dataclass
class QueryExecutionResult:
    """Output tuples, physical plan metadata, and execution telemetry for a declarative video query."""
    query_id: str
    query_class: str
    latency_sec: float
    peak_vram_gb: float
    oom_occurred: bool
    tokens_evaluated: int
    token_savings_ratio: float
    cache_hit_rate: float
    audit_report: Optional[AdmissionAuditReport] = None
    compiled_plan: Optional[PhysicalPlanDAG] = None
    matched_results: List[Any] = field(default_factory=list)


class ZadigDBEngine:
    """
    ZadigDB Progressive Refinement Engine integrating:
      1. Plan-Level Deterministic CPU Intent-to-Plan Optimizer (IPO, Section III-A) and
         Clause-Level Admission Gate under parameterized GPU budget B_VRAM < B_hard
         (Clauses C1, C2, C3, Section III-B to III-D)
      2. Relational-Level Competitive Provenance View and Coupled Co-Movement Join
         (SEM_FILTER, SEM_LINK, SEM_COMOVE_JOIN, Section II-B to II-C & III-D)
      3. Representation-Level Three-Tier Visual Token Buffer Pool & Provenance-Heat Cache
         ({1, 32, 256} tokens, Section IV: Proposition 1 & Theorem 2)
    """

    def __init__(
        self,
        topology: CameraTopology,
        k_comp: int = 4,
        execution_mode: str = "warm",
        cache_eviction_policy: str = "provenance_heat",
        gpu_name: str = "NVIDIA GeForce RTX 4090",
        vram_budget_gb: float = 22.5,
        hard_vram_limit_gb: float = 24.0,
        base_model_weights_gb: float = 8.174,
        tier0_descriptor_bytes: float = 2048.0,
        kv_bytes_per_token: float = 131072.0,
        min_coupled_recall: float = 0.55,
        default_per_hop_p: float = 0.63,
        rank_tail_alpha: float = 0.55,
        coupling_consistency_gain: float = 0.85,
        delta_tier: float = 0.12,
        stagnation_epsilon: float = 0.015,
        min_compression_alpha: float = 0.25,
        root_selectivity_threshold: float = 0.30,
        pushdown_selectivity_threshold: float = 0.45,
        tier2_verify_share: float = 0.35,
        hub_regularization_weight: float = 0.35,
        default_selectivity: float = 0.28,
        tier_resolutions: Sequence[int] = (1, 32, 256),
        max_tier1_capacity: int = 1536,
        max_tier2_capacity: int = 384,
        ambiguity_margin_gamma: float = 0.14,
        utility_weights: Sequence[float] = (0.55, 0.25, 0.20),
        heat_saturation_cap: float = 5.0,
        filter_threshold: float = 0.52,
        uncertain_margin: float = 0.14,
        link_appearance_weight: float = 0.90,
        link_spatiotemporal_weight: float = 0.10,
        link_min_edge_confidence: float = 0.36,
        comove_min_pair_confidence: float = 0.38,
        total_indexed_snapshots: Optional[int] = None,
        vlm_backend: Optional[Any] = None,
    ):
        mode_norm = str(execution_mode).strip().lower()
        if mode_norm not in ("warm", "cold"):
            raise ValueError(f"execution_mode must be 'warm' or 'cold', got {execution_mode!r}")

        self.topology = topology
        self.execution_mode = mode_norm
        self.gpu_name = str(gpu_name)
        self.tier_resolutions: Tuple[int, int, int] = (
            int(tier_resolutions[0]),
            int(tier_resolutions[1]),
            int(tier_resolutions[2]),
        )
        self.k_comp = int(k_comp)
        self.total_indexed_snapshots = total_indexed_snapshots
        self.vlm_backend = vlm_backend

        eff_base_w_gb = (
            float(getattr(vlm_backend, "base_weights_gb", base_model_weights_gb))
            if vlm_backend is not None
            else base_model_weights_gb
        )
        eff_kv_bytes = (
            float(getattr(vlm_backend, "kv_bytes_per_token", kv_bytes_per_token))
            if vlm_backend is not None
            else kv_bytes_per_token
        )

        self.admission_gate = ClauseAdmissionGate(
            vram_budget_gb=vram_budget_gb,
            hard_vram_limit_gb=hard_vram_limit_gb,
            base_model_weights_gb=eff_base_w_gb,
            tier0_descriptor_bytes=tier0_descriptor_bytes,
            kv_bytes_per_token=eff_kv_bytes,
            min_coupled_recall=min_coupled_recall,
            default_per_hop_p=default_per_hop_p,
            rank_tail_alpha=rank_tail_alpha,
            coupling_consistency_gain=coupling_consistency_gain,
            delta_tier=delta_tier,
            stagnation_epsilon=stagnation_epsilon,
            min_compression_alpha=min_compression_alpha,
        )
        self.synthesizer = TopologyAwarePlanSynthesizer(
            topology=topology,
            admission_gate=self.admission_gate,
            default_k_comp=self.k_comp,
            root_selectivity_threshold=root_selectivity_threshold,
            pushdown_selectivity_threshold=pushdown_selectivity_threshold,
            tier2_verify_share=tier2_verify_share,
            hub_regularization_weight=hub_regularization_weight,
            default_selectivity=default_selectivity,
            max_tier1_cap=max_tier1_capacity,
            max_tier2_cap=max_tier2_capacity,
        )
        self.buffer_pool = ProgressiveVisualBufferPool(
            max_tier1_capacity=max_tier1_capacity,
            max_tier2_capacity=max_tier2_capacity,
            ambiguity_margin_gamma=ambiguity_margin_gamma,
            enable_multi_query_cache=(mode_norm == "warm"),
            eviction_policy=cache_eviction_policy,
            utility_weights=utility_weights,
            heat_saturation_cap=heat_saturation_cap,
            vlm_backend=vlm_backend,
        )
        self.algebra = SpatioTemporalSemanticAlgebra(
            topology=topology,
            buffer_pool=self.buffer_pool,
            filter_threshold=filter_threshold,
            uncertain_margin=uncertain_margin,
            link_appearance_weight=link_appearance_weight,
            link_spatiotemporal_weight=link_spatiotemporal_weight,
            link_min_edge_confidence=link_min_edge_confidence,
            comove_min_pair_confidence=comove_min_pair_confidence,
            hub_regularization_weight=hub_regularization_weight,
        )

    def reset_hardware_cache(self) -> None:
        """Releases resident GPU buffer-pool slots when closing a query session."""
        self.buffer_pool.reset_cache()

    @classmethod
    def from_yaml(
        cls,
        config_path: str,
        topology: CameraTopology,
        total_indexed_snapshots: Optional[int] = None,
        apply_topology_defaults: bool = True,
        vlm_backend: Optional[Any] = None,
    ) -> "ZadigDBEngine":
        """Instantiates ZadigDBEngine from a YAML hardware and optimizer configuration file."""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        hw_cfg = cfg.get("hardware", {})
        exec_cfg = cfg.get("execution", {})
        synth_cfg = cfg.get("synthesizer", {})
        gate_cfg = cfg.get("admission_gate", {})
        topo_cfg = cfg.get("topology_defaults", {})
        buf_cfg = cfg.get("buffer_pool", {})
        alg_cfg = cfg.get("algebra", {})

        if apply_topology_defaults and topo_cfg:
            topology.v_min_mps = float(topo_cfg.get("v_min_mps", topology.v_min_mps))
            topology.v_max_mps = float(topo_cfg.get("v_max_mps", topology.v_max_mps))
            topology.tau_sig_sec = float(topo_cfg.get("tau_sig_sec", topology.tau_sig_sec))
            topology.congestion_speed_floor_mps = float(
                topo_cfg.get("congestion_speed_floor_mps", topology.congestion_speed_floor_mps)
            )
            topology.congestion_clearance_scale = float(
                topo_cfg.get("congestion_clearance_scale", topology.congestion_clearance_scale)
            )

        return cls(
            topology=topology,
            k_comp=int(gate_cfg.get("default_k_comp", 4)),
            execution_mode=str(exec_cfg.get("mode", "warm")),
            cache_eviction_policy=str(buf_cfg.get("eviction_policy", "provenance_heat")),
            gpu_name=str(hw_cfg.get("gpu_name", "NVIDIA GeForce RTX 4090")),
            vram_budget_gb=float(hw_cfg.get("clause_c1_vram_budget_gb", 22.5)),
            hard_vram_limit_gb=float(hw_cfg.get("hard_vram_limit_gb", 24.0)),
            base_model_weights_gb=float(hw_cfg.get("base_vlm_weights_gb", 8.174)),
            tier0_descriptor_bytes=float(hw_cfg.get("tier0_descriptor_bytes", 2048.0)),
            kv_bytes_per_token=float(hw_cfg.get("kv_bytes_per_token", 131072.0)),
            min_coupled_recall=float(gate_cfg.get("min_coupled_recall", 0.55)),
            default_per_hop_p=float(gate_cfg.get("default_per_hop_p", 0.63)),
            rank_tail_alpha=float(gate_cfg.get("rank_tail_alpha", 0.55)),
            coupling_consistency_gain=float(gate_cfg.get("coupling_consistency_gain", 0.85)),
            delta_tier=float(gate_cfg.get("delta_tier", 0.12)),
            stagnation_epsilon=float(gate_cfg.get("stagnation_epsilon", 0.015)),
            min_compression_alpha=float(gate_cfg.get("min_compression_alpha", 0.25)),
            root_selectivity_threshold=float(synth_cfg.get("root_selectivity_threshold", 0.30)),
            pushdown_selectivity_threshold=float(
                synth_cfg.get("pushdown_selectivity_threshold", 0.45)
            ),
            tier2_verify_share=float(synth_cfg.get("tier2_verify_share", 0.35)),
            hub_regularization_weight=float(synth_cfg.get("hub_regularization_weight", 0.35)),
            default_selectivity=float(synth_cfg.get("default_selectivity", 0.28)),
            tier_resolutions=tuple(buf_cfg.get("tier_resolutions", [1, 32, 256])),
            max_tier1_capacity=int(buf_cfg.get("max_tier1_capacity", 1536)),
            max_tier2_capacity=int(buf_cfg.get("max_tier2_capacity", 384)),
            ambiguity_margin_gamma=float(buf_cfg.get("ambiguity_margin_gamma", 0.14)),
            utility_weights=tuple(buf_cfg.get("utility_weights", [0.55, 0.25, 0.20])),
            heat_saturation_cap=float(buf_cfg.get("heat_saturation_cap", 5.0)),
            filter_threshold=float(alg_cfg.get("filter_threshold", 0.52)),
            uncertain_margin=float(alg_cfg.get("uncertain_margin", 0.14)),
            link_appearance_weight=float(alg_cfg.get("link_appearance_weight", 0.90)),
            link_spatiotemporal_weight=float(alg_cfg.get("link_spatiotemporal_weight", 0.10)),
            link_min_edge_confidence=float(alg_cfg.get("link_min_edge_confidence", 0.36)),
            comove_min_pair_confidence=float(alg_cfg.get("comove_min_pair_confidence", 0.38)),
            total_indexed_snapshots=total_indexed_snapshots,
            vlm_backend=vlm_backend,
        )

    def ingest_traffic_telemetry(
        self,
        observed_mean_speed_mps: float,
        base_tau_sig_sec: float = 18.0,
        max_tau_sig_sec: float = 110.0,
    ) -> float:
        """
        Ingests real-time corridor arterial speed telemetry to dynamically widen
        Clause C2 signal slack tau_sig under severe congestion while preserving Clause C1
        re-auditing on subsequent queries (Section V-F).
        """
        return self.topology.adapt_signal_slack_from_telemetry(
            observed_mean_speed_mps=observed_mean_speed_mps,
            base_tau_sig_sec=base_tau_sig_sec,
            max_tau_sig_sec=max_tau_sig_sec,
        )

    def execute_sql(
        self,
        sql_text: str,
        slices: List[VisualDetectionSlice],
        query_id: str = "SQL_Q",
    ) -> QueryExecutionResult:
        """Compiles and executes a raw declarative SQL-extended multi-camera query end-to-end."""
        query = DeclarativeVideoQuery.from_sql(
            sql_text=sql_text,
            query_id=query_id,
            target_cameras=list(self.topology.camera_ids),
        )
        return self.execute_query(query, slices)

    def execute_query(
        self,
        query: DeclarativeVideoQuery,
        slices: List[VisualDetectionSlice],
    ) -> QueryExecutionResult:
        """Synthesizes, audits, and executes a declarative video query over candidate slices."""
        t0_wall = time.perf_counter()
        if query.sql_text and "SEM_" in query.sql_text.upper():
            compiled_q = DeclarativeVideoQuery.from_sql(
                sql_text=query.sql_text,
                query_id=query.query_id,
                target_cameras=(
                    list(query.target_cameras)
                    if query.target_cameras
                    else list(self.topology.camera_ids)
                ),
                default_hops=query.num_hops,
            )
            if compiled_q.requested_k_comp is None and query.requested_k_comp is not None:
                compiled_q.requested_k_comp = query.requested_k_comp
            query = compiled_q

        if self.execution_mode == "cold":
            self.buffer_pool.reset_cache()

        t1_hits_before = self.buffer_pool.telemetry.tier1_cache_hits
        t2_hits_before = self.buffer_pool.telemetry.tier2_cache_hits
        t1_prom_before = self.buffer_pool.telemetry.tier1_promotions
        t2_prom_before = self.buffer_pool.telemetry.tier2_promotions
        t1_vlm_before = self.buffer_pool.telemetry.tier1_vlm_promotions
        t1_vit_before = self.buffer_pool.telemetry.tier1_vit_promotions
        t2_vlm_before = self.buffer_pool.telemetry.tier2_vlm_promotions
        t2_vit_before = self.buffer_pool.telemetry.tier2_vit_promotions
        kv_suf_before = self.buffer_pool.telemetry.kv_suffix_hits
        pairs_before = self.algebra.evaluated_link_pairs
        prov_edges_count = 0

        self.buffer_pool.begin_query(len(slices))
        n_indexed = self.total_indexed_snapshots or len(slices)
        plan, audit_report = self.synthesizer.synthesize_plan(
            query=query,
            total_slices=n_indexed,
        )

        admitted_t1 = max((op.max_tier1_slices for op in plan.operators), default=512)
        admitted_t2 = max((op.max_tier2_slices for op in plan.operators), default=128)
        self.buffer_pool.enforce_admitted_quotas(admitted_t1, admitted_t2)

        if self.vlm_backend is not None:
            self.vlm_backend.begin_query_measurement(
                n_indexed=n_indexed,
                active_slices=slices,
            )

        matched_outputs: List[Any] = []
        try:
            primary_op = plan.operators[0]
            filtered_primary = self.algebra.execute_sem_filter(slices, primary_op)

            if query.query_class == "Q1":
                matched_outputs = [s.slice_id for s in filtered_primary]

            elif query.query_class == "Q2":
                link_op = plan.operators[1]
                prov_v, trajectories = self.algebra.execute_sem_link(filtered_primary, link_op)
                prov_edges_count = prov_v.total_edges()
                for _, chain in trajectories.items():
                    if len(chain) >= max(2, query.num_hops):
                        matched_outputs.append(list(chain))

            else:
                sec_op = next(
                    (op for op in plan.operators if "secondary" in op.node_id),
                    PhysicalOperatorNode(
                        node_id=f"{query.query_id}_filter_secondary_passthrough",
                        op_type=OperatorType.SEM_FILTER,
                        predicates=list(query.secondary_predicates),
                        target_cameras=list(primary_op.target_cameras),
                        k_comp=1,
                        use_causal_window_c2=primary_op.use_causal_window_c2,
                        max_tier1_slices=primary_op.max_tier1_slices,
                        max_tier2_slices=primary_op.max_tier2_slices,
                    ),
                )
                link_op = next(
                    (op for op in plan.operators if op.op_type == OperatorType.SEM_LINK),
                    primary_op,
                )
                join_op = next(
                    (op for op in plan.operators if op.op_type == OperatorType.SEM_COMOVE_JOIN),
                    primary_op,
                )
                filtered_secondary = self.algebra.execute_sem_filter(slices, sec_op)
                prov_a, _ = self.algebra.execute_sem_link(filtered_primary, link_op)
                prov_b, _ = self.algebra.execute_sem_link(filtered_secondary, link_op)
                prov_edges_count = prov_a.total_edges() + prov_b.total_edges()

                joined_pairs = self.algebra.execute_sem_comove_join(
                    slices_a=filtered_primary,
                    slices_b=filtered_secondary,
                    prov_a=prov_a,
                    prov_b=prov_b,
                    op=join_op,
                )
                matched_outputs = list(joined_pairs)
        finally:
            self.buffer_pool.unpin_all()

        cpu_wall_sec = time.perf_counter() - t0_wall
        t1_hits_delta = self.buffer_pool.telemetry.tier1_cache_hits - t1_hits_before
        t2_hits_delta = self.buffer_pool.telemetry.tier2_cache_hits - t2_hits_before
        t1_prom_delta = self.buffer_pool.telemetry.tier1_promotions - t1_prom_before
        t2_prom_delta = self.buffer_pool.telemetry.tier2_promotions - t2_prom_before
        t1_vlm_delta = self.buffer_pool.telemetry.tier1_vlm_promotions - t1_vlm_before
        t1_vit_delta = self.buffer_pool.telemetry.tier1_vit_promotions - t1_vit_before
        t2_vlm_delta = self.buffer_pool.telemetry.tier2_vlm_promotions - t2_vlm_before
        t2_vit_delta = self.buffer_pool.telemetry.tier2_vit_promotions - t2_vit_before
        kv_suf_delta = self.buffer_pool.telemetry.kv_suffix_hits - kv_suf_before
        pairs_delta = max(0, self.algebra.evaluated_link_pairs - pairs_before)
        hits_delta = t1_hits_delta + t2_hits_delta
        prom_delta = t1_prom_delta + t2_prom_delta

        query_hit_rate = float(hits_delta) / float(max(1, hits_delta + prom_delta))
        full_tokens = max(1, len(slices) * 256)
        newly_decoded_tokens = len(slices) * 1 + t1_prom_delta * 32 + t2_prom_delta * 256
        savings = max(0.0, 1.0 - (float(newly_decoded_tokens) / float(full_tokens)))

        eff_k_plan = max((op.k_comp for op in plan.operators), default=self.k_comp)
        eff_prov_edges = max(prov_edges_count, n_indexed * eff_k_plan)

        t0_gb = (
            float(n_indexed) * float(self.admission_gate.tier0_descriptor_bytes)
        ) / (1024.0 ** 3)
        kv_t1_gb = (
            float(self.buffer_pool.peak_tier1_resident)
            * 32.0
            * float(self.admission_gate.kv_bytes_per_token)
        ) / (1024.0 ** 3)
        kv_t2_gb = (
            float(self.buffer_pool.peak_tier2_resident)
            * 256.0
            * float(self.admission_gate.kv_bytes_per_token)
        ) / (1024.0 ** 3)
        prov_gb = (float(eff_prov_edges) * 64.0) / (1024.0 ** 3) + 0.35
        analytical_vram_gb = round(
            self.admission_gate.base_model_weights_gb + t0_gb + kv_t1_gb + kv_t2_gb + prov_gb,
            2,
        )

        opt_sec = float(audit_report.optimization_time_ms) / 1000.0
        if self.vlm_backend is not None and (
            getattr(self.vlm_backend, "live_on_demand_vlm", False)
            or not hasattr(self.vlm_backend, "calibrated_timings")
        ):
            peak_vram_gb, oom_occurred, live_wall_sec = self.vlm_backend.end_query_measurement(
                prov_edges=eff_prov_edges,
            )
            latency_sec = round(live_wall_sec + opt_sec, 3)
        else:
            if self.vlm_backend is not None and hasattr(self.vlm_backend, "end_query_measurement"):
                self.vlm_backend.end_query_measurement(prov_edges=eff_prov_edges)
            timings = dict(
                getattr(self.vlm_backend, "calibrated_timings", None) or DEFAULT_CALIBRATED_TIMINGS
            )
            t1_vlm_sec = float(timings["t1_sec_per_slice"])
            t2_vlm_sec = float(timings["t2_sec_per_slice"])
            t1_vit_sec = float(timings.get("t1_vit_sec_per_slice", t1_vlm_sec * 0.12))
            t2_vit_sec = float(timings.get("t2_vit_sec_per_slice", t2_vlm_sec * 0.15))
            vlm_cost_sec = (
                float(t1_vlm_delta) * t1_vlm_sec
                + float(t1_vit_delta) * t1_vit_sec
                + float(t2_vlm_delta) * t2_vlm_sec
                + float(t2_vit_delta) * t2_vit_sec
                + float(kv_suf_delta) * float(timings["kv_hit_sec_per_slice"])
                + float(pairs_delta) * float(timings["pair_sec"])
            )
            peak_vram_gb = analytical_vram_gb
            oom_occurred = peak_vram_gb > self.admission_gate.hard_vram_limit_gb
            latency_sec = round(cpu_wall_sec + opt_sec + vlm_cost_sec, 3)

        return QueryExecutionResult(
            query_id=query.query_id,
            query_class=query.query_class,
            latency_sec=round(latency_sec, 3),
            peak_vram_gb=round(peak_vram_gb, 2),
            oom_occurred=oom_occurred,
            tokens_evaluated=newly_decoded_tokens,
            token_savings_ratio=round(savings * 100.0, 2),
            cache_hit_rate=round(query_hit_rate * 100.0, 2),
            audit_report=audit_report,
            compiled_plan=plan,
            matched_results=matched_outputs,
        )
