"""
Deterministic Intent-to-Plan Optimizer (IPO) & Topology-Aware Physical Plan Synthesizer for ZadigDB.

Deterministically compiles declarative SQL-extended multi-camera video queries (Q1-Q4, Section II-C)
without any LLM invocation into physical operator DAGs over SEM_FILTER, SEM_LINK, and SEM_COMOVE_JOIN,
parameterized by road-network topology G and certified by the Clause-Level Admission Gate (Section III, Algorithm 1).
"""

from dataclasses import dataclass
import re
import time
from typing import List, Optional, Tuple

from .admission_gate import AdmissionAuditReport, ClauseAdmissionGate
from .schema import (
    CameraTopology,
    OperatorType,
    PhysicalOperatorNode,
    PhysicalPlanDAG,
)


_STOPWORDS = {"with", "and", "or", "in", "at", "on", "the", "a", "an"}


def _extract_predicate_tokens(raw_phrase: str) -> List[str]:
    tokens = [tok.strip().lower() for tok in re.split(r"[^a-zA-Z0-9_]+", raw_phrase) if tok.strip()]
    return [t for t in tokens if t not in _STOPWORDS]


@dataclass
class DeclarativeVideoQuery:
    """Declarative SQL-extended multi-camera semantic query specification (Section II-C)."""
    query_id: str
    query_class: str                        # "Q1", "Q2", "Q3", or "Q4"
    sql_text: str
    primary_predicates: List[str]
    secondary_predicates: List[str]
    target_cameras: List[int]
    num_hops: int = 4
    join_window_sec: float = 15.0
    requested_k_comp: Optional[int] = None

    @property
    def min_hops(self) -> int:
        return self.num_hops

    @classmethod
    def from_sql(
        cls,
        sql_text: str,
        query_id: str = "SQL_Q",
        target_cameras: Optional[List[int]] = None,
        default_hops: int = 4,
    ) -> "DeclarativeVideoQuery":
        """
        Parses a declarative SQL-extended multi-camera query string containing
        SEM_FILTER(...), SEM_LINK(...), SEM_COMOVE_JOIN(...), CAMERA_ID IN (...),
        and GROUP BY / HAVING clauses (including SELECT COUNT(...) AS <alias> ... HAVING <alias> >= k)
        into a structured DeclarativeVideoQuery specification.
        """
        sql_upper = sql_text.upper()
        is_comove = "SEM_COMOVE_JOIN" in sql_upper

        # Identify primary and secondary stream aliases in SEM_COMOVE_JOIN if present
        alias_primary: Optional[str] = None
        alias_secondary: Optional[str] = None
        if is_comove:
            link_aliases = re.findall(
                r"SEM_LINK\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)",
                sql_text,
                flags=re.IGNORECASE,
            )
            if len(link_aliases) >= 2:
                alias_primary = link_aliases[0].lower()
                alias_secondary = link_aliases[1].lower()
            else:
                direct_join = re.search(
                    r"SEM_COMOVE_JOIN\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*)",
                    sql_text,
                    flags=re.IGNORECASE,
                )
                if direct_join and direct_join.group(1).upper() != "SEM_LINK":
                    alias_primary = direct_join.group(1).lower()
                    alias_secondary = direct_join.group(2).lower()

        filter_clauses = re.findall(
            r"SEM_FILTER\s*\(\s*([^,]+?)\s*,\s*['\"]([^'\"]+)['\"]\s*\)",
            sql_text,
            flags=re.IGNORECASE,
        )
        primary_raw: List[str] = []
        secondary_raw: List[str] = []
        seen_aliases: List[str] = []

        for raw_alias, phrase in filter_clauses:
            alias_tok = raw_alias.strip().split(".")[0].strip().lower()
            toks = _extract_predicate_tokens(phrase)
            if not is_comove:
                primary_raw.extend(toks)
            elif alias_primary is not None and alias_secondary is not None:
                if alias_tok == alias_primary:
                    primary_raw.extend(toks)
                elif alias_tok == alias_secondary:
                    secondary_raw.extend(toks)
                elif not primary_raw:
                    primary_raw.extend(toks)
                else:
                    secondary_raw.extend(toks)
            else:
                if alias_tok not in seen_aliases:
                    seen_aliases.append(alias_tok)
                if seen_aliases.index(alias_tok) == 0:
                    primary_raw.extend(toks)
                else:
                    secondary_raw.extend(toks)

        primary_preds = list(dict.fromkeys(primary_raw))
        secondary_preds = list(dict.fromkeys(secondary_raw))

        win_match = re.search(r"window\s*(?:=|<=)\s*([0-9.]+)\s*s?", sql_text, flags=re.IGNORECASE)
        join_window_sec = float(win_match.group(1)) if win_match else 15.0

        k_matches = [int(x) for x in re.findall(r"\b(?:k|k_comp)\s*=\s*(\d+)", sql_text, flags=re.IGNORECASE)]
        requested_k_comp = max(k_matches) if k_matches else None

        cam_in_match = re.search(
            r"\bcam(?:era)?_id\s+IN\s*\(\s*([0-9,\s]+)\s*\)", sql_text, flags=re.IGNORECASE
        )
        if cam_in_match:
            parsed_cams = [
                int(tok.strip())
                for tok in cam_in_match.group(1).split(",")
                if tok.strip().isdigit()
            ]
        else:
            parsed_cams = list(target_cameras or [])

        count_alias_match = re.search(
            r"(?:HOP_COUNT|COUNT)\s*\([^)]*\)\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)",
            sql_text,
            flags=re.IGNORECASE,
        )
        count_alias = count_alias_match.group(1) if count_alias_match else None

        having_match = re.search(
            r"HAVING\s+(?:hops|cams|HOP_COUNT(?:\s*\([^)]*\))?|COUNT\s*\([^)]*\)|[A-Za-z_][A-Za-z0-9_]*)\s*>=\s*([0-9]+)",
            sql_text,
            flags=re.IGNORECASE,
        )
        if having_match:
            num_hops = int(having_match.group(1))
        elif count_alias and re.search(
            rf"\b{re.escape(count_alias)}\s*>=\s*([0-9]+)", sql_text, flags=re.IGNORECASE
        ):
            alias_m = re.search(
                rf"\b{re.escape(count_alias)}\s*>=\s*([0-9]+)", sql_text, flags=re.IGNORECASE
            )
            num_hops = int(alias_m.group(1))  # type: ignore[union-attr]
        else:
            hop_match = re.search(
                r"(?:hops|HOP_COUNT(?:\s*\([^)]*\))?|COUNT\s*\([^)]*\))\s*>=\s*(\d+)",
                sql_text,
                flags=re.IGNORECASE,
            )
            num_hops = (
                int(hop_match.group(1))
                if hop_match
                else (default_hops if ("SEM_LINK" in sql_upper or is_comove) else 1)
            )

        if is_comove:
            if (
                "GROUP BY" in sql_upper
                or "HAVING" in sql_upper
                or len(primary_preds) >= 3
                or len(secondary_preds) >= 3
            ):
                q_class = "Q4"
            else:
                q_class = "Q3"
        elif "SEM_LINK" in sql_upper:
            q_class = "Q2"
        else:
            q_class = "Q1"

        return cls(
            query_id=query_id,
            query_class=q_class,
            sql_text=sql_text.strip(),
            primary_predicates=primary_preds,
            secondary_predicates=secondary_preds,
            target_cameras=parsed_cams,
            num_hops=num_hops,
            join_window_sec=join_window_sec,
            requested_k_comp=requested_k_comp,
        )


class TopologyAwarePlanSynthesizer:
    """
    Deterministic Intent-to-Plan Optimizer (IPO, Section III-A): compiles declarative SQL-extended
    multi-camera queries without any LLM invocation into physical operator DAGs and submits them
    to the Clause-Level Admission Gate for verification and repair (Algorithm 1).
    """

    def __init__(
        self,
        topology: CameraTopology,
        admission_gate: Optional[ClauseAdmissionGate] = None,
        default_k_comp: int = 4,
        root_selectivity_threshold: float = 0.30,
        pushdown_selectivity_threshold: float = 0.45,
        tier2_verify_share: float = 0.35,
        hub_regularization_weight: float = 0.35,
        default_selectivity: float = 0.28,
        max_tier1_cap: int = 1536,
        max_tier2_cap: int = 384,
    ):
        self.topology = topology
        self.admission_gate = admission_gate or ClauseAdmissionGate()
        self.default_k_comp = default_k_comp
        self.root_selectivity_threshold = root_selectivity_threshold
        self.pushdown_selectivity_threshold = pushdown_selectivity_threshold
        self.tier2_verify_share = tier2_verify_share
        self.hub_regularization_weight = hub_regularization_weight
        self.default_selectivity = default_selectivity
        self.max_tier1_cap = max_tier1_cap
        self.max_tier2_cap = max_tier2_cap
        self.last_estimated_selectivity: Tuple[float, float] = (default_selectivity, default_selectivity)

    def estimate_selectivity(self, predicates: List[str], is_pushdown: bool = False) -> float:
        """Estimates predicate selectivity prior for IPO operator ordering (Section III-A)."""
        if not predicates:
            return 1.0
        raw_sel = float(self.default_selectivity) * (0.72 ** max(0, len(predicates) - 1))
        cap = self.pushdown_selectivity_threshold if is_pushdown else self.root_selectivity_threshold
        return min(max(0.02, raw_sel), max(cap, raw_sel))

    def synthesize_plan(
        self,
        query: DeclarativeVideoQuery,
        total_slices: int,
    ) -> tuple[PhysicalPlanDAG, AdmissionAuditReport]:
        """
        Deterministically synthesizes a physical plan DAG tailored to query class (Q1-Q4),
        selectivity priors, and road-network topology G, then audits/repairs it via Clause C1-C3.
        """
        t_synth_start = time.perf_counter()
        ops: List[PhysicalOperatorNode] = []

        sel_primary = self.estimate_selectivity(list(query.primary_predicates), is_pushdown=True)
        sel_secondary = (
            self.estimate_selectivity(list(query.secondary_predicates), is_pushdown=True)
            if query.secondary_predicates
            else sel_primary
        )
        self.last_estimated_selectivity = (sel_primary, sel_secondary)
        eff_sel = max(0.20, 0.75 * self.default_selectivity + 0.25 * min(1.0, max(sel_primary, sel_secondary)))

        t1_ratio = round(0.50 * eff_sel, 6)
        t2_ratio = round(0.10 * self.tier2_verify_share, 6)
        t1_cap = min(self.max_tier1_cap, max(24, int(t1_ratio * total_slices)))
        t2_cap = min(self.max_tier2_cap, max(12, int(t2_ratio * total_slices)))
        eff_k = (
            min(8, max(1, int(query.requested_k_comp)))
            if query.requested_k_comp is not None
            else self.default_k_comp
        )
        eff_cams = list(query.target_cameras) if query.target_cameras else list(self.topology.camera_ids)
        is_multihop = query.query_class in ("Q2", "Q3", "Q4")
        c2_win_sec = round(self.topology.max_causal_window_sec(), 2)

        # Push down selective semantic filter first (with HAVING camera-count constraint when multi-hop)
        req_hops = max(2, query.num_hops) if is_multihop else 1
        ops.append(
            PhysicalOperatorNode(
                node_id=f"{query.query_id}_filter_primary",
                op_type=OperatorType.SEM_FILTER,
                predicates=list(query.primary_predicates),
                target_cameras=eff_cams,
                k_comp=1,
                use_causal_window_c2=True,
                causal_window_sec=c2_win_sec,
                max_tier1_slices=t1_cap,
                max_tier2_slices=t2_cap,
                min_co_move_hops=req_hops,
                downstream_multihop=is_multihop,
            )
        )

        if query.query_class == "Q2":
            ops.append(
                PhysicalOperatorNode(
                    node_id=f"{query.query_id}_link",
                    op_type=OperatorType.SEM_LINK,
                    predicates=list(query.primary_predicates),
                    target_cameras=eff_cams,
                    k_comp=eff_k,
                    use_causal_window_c2=True,
                    causal_window_sec=c2_win_sec,
                    max_tier1_slices=t1_cap,
                    max_tier2_slices=t2_cap,
                    min_co_move_hops=req_hops,
                    downstream_multihop=False,
                )
            )
        elif query.query_class in ("Q3", "Q4"):
            ops.append(
                PhysicalOperatorNode(
                    node_id=f"{query.query_id}_filter_secondary",
                    op_type=OperatorType.SEM_FILTER,
                    predicates=list(query.secondary_predicates),
                    target_cameras=eff_cams,
                    k_comp=1,
                    use_causal_window_c2=True,
                    causal_window_sec=c2_win_sec,
                    max_tier1_slices=t1_cap,
                    max_tier2_slices=t2_cap,
                    min_co_move_hops=req_hops,
                    downstream_multihop=True,
                )
            )
            ops.append(
                PhysicalOperatorNode(
                    node_id=f"{query.query_id}_link_candidates",
                    op_type=OperatorType.SEM_LINK,
                    predicates=list(query.primary_predicates),
                    target_cameras=eff_cams,
                    k_comp=eff_k,
                    use_causal_window_c2=True,
                    causal_window_sec=c2_win_sec,
                    max_tier1_slices=t1_cap,
                    max_tier2_slices=t2_cap,
                    min_co_move_hops=req_hops,
                    downstream_multihop=True,
                )
            )
            ops.append(
                PhysicalOperatorNode(
                    node_id=f"{query.query_id}_comove_join",
                    op_type=OperatorType.SEM_COMOVE_JOIN,
                    predicates=list(query.primary_predicates),
                    secondary_predicates=list(query.secondary_predicates),
                    target_cameras=eff_cams,
                    k_comp=eff_k,
                    use_causal_window_c2=True,
                    causal_window_sec=c2_win_sec,
                    max_tier1_slices=t1_cap,
                    max_tier2_slices=t2_cap,
                    join_window_sec=query.join_window_sec,
                    min_co_move_hops=req_hops,
                    downstream_multihop=False,
                )
            )

        plan = PhysicalPlanDAG(
            query_id=query.query_id,
            query_class=query.query_class,
            sql_query=query.sql_text,
            operators=ops,
            num_hops=req_hops,
        )

        synth_ms = (time.perf_counter() - t_synth_start) * 1000.0
        report = self.admission_gate.audit_and_repair(
            plan=plan,
            topology=self.topology,
            total_slices=total_slices,
            num_hops=req_hops,
            auto_repair=True,
            pre_synthesis_ms=synth_ms,
        )
        return plan, report

    def synthesize_from_sql(
        self,
        sql_text: str,
        total_slices: int,
        query_id: str = "SQL_Q",
    ) -> tuple[DeclarativeVideoQuery, PhysicalPlanDAG, AdmissionAuditReport]:
        """Compiles a raw declarative SQL string into a certified PhysicalPlanDAG and report."""
        parsed_query = DeclarativeVideoQuery.from_sql(
            sql_text=sql_text,
            query_id=query_id,
            target_cameras=list(self.topology.camera_ids),
        )
        plan, report = self.synthesize_plan(
            query=parsed_query,
            total_slices=total_slices,
        )
        return parsed_query, plan, report
