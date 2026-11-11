"""
Multi-Granularity Visual Token Buffer Pool & Provenance-Heat Multi-Query Cache for ZadigDB.

Manages progressive visual token materialization across Tier-0 (1 summary token),
Tier-1 (32 regional sub-grid tokens), and Tier-2 (256 dense patch tokens) under a
parameterized GPU admission budget B_VRAM < B_hard (instantiated at 22.5 GB < 24.0 GB on
RTX 4090), with active-slot pinning, margin-guided escalation (Proposition 1),
provenance-heat replacement, and cross-query token reuse (Section IV, Algorithm 3, Theorem 2).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import numpy as np

from .schema import TokenResolutionTier, VisualDetectionSlice


@dataclass
class BufferSlotEntry:
    """Metadata for a resident visual slice page in the Tier-1 or Tier-2 GPU buffer pool."""
    slice_id: Any
    cam_id: int
    tier: TokenResolutionTier
    last_access_step: int
    access_count: int = 1
    provenance_heat: float = 0.5
    gpu_tensor: Optional[Any] = field(default=None, repr=False)


@dataclass
class BufferPoolTelemetry:
    """Per-query and cumulative telemetry of the Multi-Granularity Visual Token Buffer Pool."""
    tier0_lookups: int = 0
    tier1_promotions: int = 0
    tier2_promotions: int = 0
    tier1_vlm_promotions: int = 0
    tier1_vit_promotions: int = 0
    tier2_vlm_promotions: int = 0
    tier2_vit_promotions: int = 0
    tier1_cache_hits: int = 0
    tier2_cache_hits: int = 0
    kv_suffix_hits: int = 0
    evictions: int = 0
    total_tokens_evaluated: int = 0
    full_exhaust_tokens: int = 0

    @property
    def token_savings_ratio(self) -> float:
        if self.full_exhaust_tokens <= 0:
            return 0.0
        return max(0.0, 1.0 - (self.total_tokens_evaluated / float(self.full_exhaust_tokens)))

    @property
    def cache_hit_rate(self) -> float:
        total_req = (
            self.tier1_promotions
            + self.tier2_promotions
            + self.tier1_cache_hits
            + self.tier2_cache_hits
        )
        if total_req <= 0:
            return 0.0
        return float(self.tier1_cache_hits + self.tier2_cache_hits) / float(total_req)


class ProgressiveVisualBufferPool:
    """
    Three-Tier Multi-Granularity Visual Token Buffer Pool with Margin-Guided Tier Escalation,
    Active-Slot Pinning (Pi_pin^(r)), and Provenance-Heat Multi-Query Caching (Section IV, Algorithm 3).
    """

    def __init__(
        self,
        max_tier1_capacity: int = 1536,
        max_tier2_capacity: int = 384,
        ambiguity_margin_gamma: float = 0.14,
        enable_multi_query_cache: bool = True,
        eviction_policy: str = "provenance_heat",
        utility_weights: Sequence[float] = (0.55, 0.25, 0.20),
        heat_saturation_cap: float = 5.0,
        vlm_backend: Optional[Any] = None,
    ):
        if eviction_policy not in ("provenance_heat", "lru"):
            raise ValueError(f"Unsupported eviction_policy: {eviction_policy}")
        if len(utility_weights) != 3:
            raise ValueError("utility_weights must contain exactly 3 floats (heat, freq, recency).")

        self.max_tier1_capacity = max_tier1_capacity
        self.max_tier2_capacity = max_tier2_capacity
        self.ambiguity_margin_gamma = float(ambiguity_margin_gamma)
        self.enable_multi_query_cache = enable_multi_query_cache
        self.eviction_policy = eviction_policy
        self.utility_weights: Tuple[float, float, float] = (
            float(utility_weights[0]),
            float(utility_weights[1]),
            float(utility_weights[2]),
        )
        self.heat_saturation_cap = float(heat_saturation_cap)
        self.vlm_backend = vlm_backend

        self._step_counter: int = 0
        self._tier1_pool: Dict[Any, BufferSlotEntry] = {}
        self._tier2_pool: Dict[Any, BufferSlotEntry] = {}
        self._pinned_tier1: Set[Any] = set()
        self._pinned_tier2: Set[Any] = set()
        self._tier1_scored_preds: Dict[Any, Set[str]] = {}
        self._tier2_scored_preds: Dict[Any, Set[str]] = {}
        self._tier1_has_kv_prefill: Set[Any] = set()
        self._tier2_has_kv_prefill: Set[Any] = set()
        self._query_promoted_t1: Set[Any] = set()
        self._query_promoted_t2: Set[Any] = set()
        self._query_hit_t1: Set[Any] = set()
        self._query_hit_t2: Set[Any] = set()
        self._provenance_heat: Dict[Any, float] = {}
        self._predicate_reuse_freq: Dict[str, int] = {}
        self.peak_tier1_resident: int = 0
        self.peak_tier2_resident: int = 0
        self.telemetry = BufferPoolTelemetry()

    def pin_tier1(self, slice_ids: Iterable[Any]) -> None:
        """Pins active working-set slice IDs in B_1 against eviction (Algorithm 3, Pi_pin^(1))."""
        self._pinned_tier1.update(slice_ids)

    def unpin_tier1(self, slice_ids: Iterable[Any]) -> None:
        """Unpins slice IDs in B_1 after operator step completion."""
        self._pinned_tier1.difference_update(slice_ids)

    def pin_tier2(self, slice_ids: Iterable[Any]) -> None:
        """Pins active working-set slice IDs in B_2 against eviction (Algorithm 3, Pi_pin^(2))."""
        self._pinned_tier2.update(slice_ids)

    def unpin_tier2(self, slice_ids: Iterable[Any]) -> None:
        """Unpins slice IDs in B_2 after operator step completion."""
        self._pinned_tier2.difference_update(slice_ids)

    def unpin_all(self) -> None:
        """Clears all active working-set pins across B_1 and B_2."""
        self._pinned_tier1.clear()
        self._pinned_tier2.clear()

    def reset_cache(self) -> None:
        """Resets resident B_1/B_2 caches, pin sets, heat statistics, and backend KV slots."""
        self._tier1_pool.clear()
        self._tier2_pool.clear()
        self._pinned_tier1.clear()
        self._pinned_tier2.clear()
        self._tier1_scored_preds.clear()
        self._tier2_scored_preds.clear()
        self._tier1_has_kv_prefill.clear()
        self._tier2_has_kv_prefill.clear()
        self._query_promoted_t1.clear()
        self._query_promoted_t2.clear()
        self._query_hit_t1.clear()
        self._query_hit_t2.clear()
        self._provenance_heat.clear()
        self._predicate_reuse_freq.clear()
        self.peak_tier1_resident = 0
        self.peak_tier2_resident = 0
        if self.vlm_backend is not None and hasattr(self.vlm_backend, "clear_slots"):
            self.vlm_backend.clear_slots()

    def clear_pools(self) -> None:
        """Alias for `reset_cache()` releasing resident Tier-1 and Tier-2 GPU buffer slots."""
        self.reset_cache()

    def begin_query(self, num_candidate_slices: int) -> None:
        """Prepares buffer pool for a new query; resets caches when multi-query retention is disabled."""
        self._step_counter += 1
        self.telemetry.full_exhaust_tokens += num_candidate_slices * int(
            TokenResolutionTier.TIER_2_FINE
        )
        if not self.enable_multi_query_cache:
            self.reset_cache()
        self._query_promoted_t1.clear()
        self._query_promoted_t2.clear()
        self._query_hit_t1.clear()
        self._query_hit_t2.clear()
        self.peak_tier1_resident = len(self._tier1_pool)
        self.peak_tier2_resident = len(self._tier2_pool)

    def enforce_admitted_quotas(self, admitted_t1: int, admitted_t2: int) -> None:
        """Enforces Clause C1 admitted physical plan quotas (N_1, N_2) on the live GPU buffer pools."""
        self.max_tier1_capacity = max(16, int(admitted_t1))
        self.max_tier2_capacity = max(8, int(admitted_t2))
        if len(self._tier1_pool) > self.max_tier1_capacity:
            self._evict_lowest_utility(
                self._tier1_pool,
                max_keep=self.max_tier1_capacity,
                pinned_ids=self._pinned_tier1,
                tier_level=1,
                capacity_ref=self.max_tier1_capacity,
            )
        if len(self._tier2_pool) > self.max_tier2_capacity:
            self._evict_lowest_utility(
                self._tier2_pool,
                max_keep=self.max_tier2_capacity,
                pinned_ids=self._pinned_tier2,
                tier_level=2,
                capacity_ref=self.max_tier2_capacity,
            )
        self.peak_tier1_resident = len(self._tier1_pool)
        self.peak_tier2_resident = len(self._tier2_pool)

    def record_tier0_scan(self, count: int) -> None:
        self.telemetry.tier0_lookups += count
        self.telemetry.total_tokens_evaluated += count * int(TokenResolutionTier.TIER_0_MICRO)

    def is_tier1_resident(self, slice_id: Any) -> bool:
        """Returns True if `slice_id` currently resides in the Tier-1 GPU buffer pool."""
        return slice_id in self._tier1_pool

    def is_tier2_resident(self, slice_id: Any) -> bool:
        """Returns True if `slice_id` currently resides in the Tier-2 GPU buffer pool."""
        return slice_id in self._tier2_pool

    def _promote_on_backend(
        self,
        slice_obj: VisualDetectionSlice,
        tier: TokenResolutionTier,
        num_predicates: int = 0,
        predicates: Optional[Sequence[str]] = None,
    ) -> Any:
        if self.vlm_backend is None:
            return None
        if hasattr(self.vlm_backend, "materialize_tier"):
            return self.vlm_backend.materialize_tier(
                slice_obj,
                tier=tier,
                retain_slot=True,
                predicates=predicates,
            )
        try:
            return self.vlm_backend.promote_slice_to_gpu(
                slice_obj, tier, num_predicates=num_predicates
            )
        except TypeError:
            return self.vlm_backend.promote_slice_to_gpu(slice_obj, tier)

    def _sync_slice_predicates(
        self,
        slice_obj: VisualDetectionSlice,
        tier_level: int,
        predicates: Optional[Sequence[str]],
        is_cache_hit: bool = False,
    ) -> None:
        """
        Ensures all requested `predicates` are scored at `tier_level` for `slice_obj`.
        On warm cache hits with refined SQL predicates, evaluates only missing predicates
        via `vlm_backend.score_cached_predicates` or offline tier score tables.
        """
        sid = slice_obj.slice_id
        tier_dict = (
            slice_obj.attribute_scores_t2
            if tier_level == 2
            else slice_obj.attribute_scores_t1
        )
        scored_set = (
            self._tier2_scored_preds.setdefault(sid, set())
            if tier_level == 2
            else self._tier1_scored_preds.setdefault(sid, set())
        )
        t2_done = self._tier2_scored_preds.get(sid, set())

        req = list(dict.fromkeys(predicates)) if predicates else []
        if not req:
            return

        for p in req:
            self._predicate_reuse_freq[p] = self._predicate_reuse_freq.get(p, 0) + 1

        if not is_cache_hit:
            for p in req:
                if p in tier_dict:
                    if tier_level == 2 or p not in t2_done:
                        slice_obj.predicate_scores[p] = float(tier_dict[p])
                    scored_set.add(p)

        missing = [
            p
            for p in req
            if (p not in scored_set) or (p not in slice_obj.predicate_scores)
        ]
        if not missing:
            return

        if is_cache_hit:
            prefill_set = (
                self._tier2_has_kv_prefill if tier_level == 2 else self._tier1_has_kv_prefill
            )
            if sid not in prefill_set:
                prefill_set.add(sid)
                if tier_level == 2:
                    self.telemetry.tier2_vlm_promotions += 1
                else:
                    self.telemetry.tier1_vlm_promotions += 1
            else:
                self.telemetry.kv_suffix_hits += 1

        if self.vlm_backend is not None and hasattr(self.vlm_backend, "score_cached_predicates"):
            newly_scored = self.vlm_backend.score_cached_predicates(
                slice_obj, tier_level, missing
            )
            if newly_scored:
                tier_dict.update(newly_scored)

        for p in missing:
            if p in tier_dict:
                val = float(tier_dict[p])
                if tier_level == 2 or p not in t2_done:
                    slice_obj.predicate_scores[p] = val
                scored_set.add(p)

    def _evict_lowest_utility(
        self,
        pool: Dict[Any, BufferSlotEntry],
        max_keep: int,
        pinned_ids: Optional[Set[Any]] = None,
        tier_level: int = 1,
        capacity_ref: Optional[int] = None,
    ) -> None:
        """
        Evicts lowest-utility unpinned pages w in B_r \\ Pi_pin^(r) until len(pool) <= max_keep
        (Section IV-C, Algorithm 3, Eq. 8). Never evicts a pinned slice if any unpinned entry exists.
        """
        cap_for_age = capacity_ref if capacity_ref is not None else max(1, max_keep + 1)
        age_scale = max(1.0, float(cap_for_age) * 2.0)
        w_heat, w_freq, w_rec = self.utility_weights
        sat_cap = max(1.0, self.heat_saturation_cap)

        while len(pool) > max_keep and pool:
            unpinned = (
                [sid for sid in pool.keys() if sid not in pinned_ids]
                if pinned_ids
                else list(pool.keys())
            )
            candidates = unpinned if unpinned else list(pool.keys())
            if self.eviction_policy == "lru":
                victim_id = min(candidates, key=lambda sid: pool[sid].last_access_step)
            else:
                # Eq. (8): U_cache(w, t) provenance-heat + access frequency - recency decay
                victim_id = min(
                    candidates,
                    key=lambda sid: (
                        w_heat * min(1.0, pool[sid].provenance_heat)
                        + w_freq * min(1.0, pool[sid].access_count / sat_cap)
                        - w_rec * (max(0, self._step_counter - pool[sid].last_access_step) / age_scale)
                    ),
                )
            del pool[victim_id]
            if tier_level == 2:
                self._tier2_scored_preds.pop(victim_id, None)
                self._tier2_has_kv_prefill.discard(victim_id)
            else:
                self._tier1_scored_preds.pop(victim_id, None)
                self._tier1_has_kv_prefill.discard(victim_id)
            if self.vlm_backend is not None and hasattr(self.vlm_backend, "release_slot"):
                self.vlm_backend.release_slot(victim_id, tier_level)
            self.telemetry.evictions += 1

    def access_tier1(
        self,
        slice_obj: VisualDetectionSlice,
        heat: float = 0.5,
        num_predicates: int = 0,
        predicates: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Fetches or escalates a slice to a Tier-1 regional sub-grid slot (32 tokens)."""
        sid = slice_obj.slice_id
        eff_n_preds = len(predicates) if predicates is not None else num_predicates
        self._step_counter += 1
        self._provenance_heat[sid] = max(self._provenance_heat.get(sid, 0.0), float(heat))

        if sid in self._tier1_pool:
            entry = self._tier1_pool[sid]
            entry.last_access_step = self._step_counter
            entry.access_count += 1
            entry.provenance_heat = max(entry.provenance_heat, heat)
            if sid not in self._query_promoted_t1 and sid not in self._query_hit_t1:
                self._query_hit_t1.add(sid)
                self.telemetry.tier1_cache_hits += 1
            if self.vlm_backend is not None and hasattr(self.vlm_backend, "record_cache_hit_access"):
                self.vlm_backend.record_cache_hit_access(
                    entry.gpu_tensor, int(TokenResolutionTier.TIER_1_COARSE)
                )
            self._sync_slice_predicates(
                slice_obj, tier_level=1, predicates=predicates, is_cache_hit=True
            )
            return slice_obj.tier1_embed if slice_obj.tier1_embed is not None else slice_obj.tier0_embed

        if len(self._tier1_pool) >= self.max_tier1_capacity:
            self._evict_lowest_utility(
                self._tier1_pool,
                max_keep=max(0, self.max_tier1_capacity - 1),
                pinned_ids=self._pinned_tier1,
                tier_level=1,
                capacity_ref=self.max_tier1_capacity,
            )

        gpu_slot = self._promote_on_backend(
            slice_obj,
            TokenResolutionTier.TIER_1_COARSE,
            num_predicates=eff_n_preds,
            predicates=predicates,
        )
        self._tier1_pool[sid] = BufferSlotEntry(
            slice_id=sid,
            cam_id=slice_obj.cam_id,
            tier=TokenResolutionTier.TIER_1_COARSE,
            last_access_step=self._step_counter,
            access_count=1,
            provenance_heat=heat,
            gpu_tensor=gpu_slot,
        )
        self._query_promoted_t1.add(sid)
        if eff_n_preds > 0:
            self._tier1_has_kv_prefill.add(sid)
            self.telemetry.tier1_vlm_promotions += 1
        else:
            self.telemetry.tier1_vit_promotions += 1
        self._sync_slice_predicates(slice_obj, tier_level=1, predicates=predicates)
        self.peak_tier1_resident = max(self.peak_tier1_resident, len(self._tier1_pool))
        self.telemetry.tier1_promotions += 1
        self.telemetry.total_tokens_evaluated += int(TokenResolutionTier.TIER_1_COARSE)
        return slice_obj.tier1_embed if slice_obj.tier1_embed is not None else slice_obj.tier0_embed

    def access_tier2(
        self,
        slice_obj: VisualDetectionSlice,
        heat: float = 0.8,
        num_predicates: int = 0,
        predicates: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Fetches or escalates a slice to a Tier-2 dense patch page (256 tokens)."""
        sid = slice_obj.slice_id
        eff_n_preds = len(predicates) if predicates is not None else num_predicates
        self._step_counter += 1
        self._provenance_heat[sid] = max(self._provenance_heat.get(sid, 0.0), float(heat))

        if sid in self._tier2_pool:
            entry = self._tier2_pool[sid]
            entry.last_access_step = self._step_counter
            entry.access_count += 1
            entry.provenance_heat = max(entry.provenance_heat, heat)
            if sid not in self._query_promoted_t2 and sid not in self._query_hit_t2:
                self._query_hit_t2.add(sid)
                self.telemetry.tier2_cache_hits += 1
            if self.vlm_backend is not None and hasattr(self.vlm_backend, "record_cache_hit_access"):
                self.vlm_backend.record_cache_hit_access(
                    entry.gpu_tensor, int(TokenResolutionTier.TIER_2_FINE)
                )
            self._sync_slice_predicates(
                slice_obj, tier_level=2, predicates=predicates, is_cache_hit=True
            )
            return slice_obj.tier2_embed if slice_obj.tier2_embed is not None else slice_obj.tier0_embed

        if len(self._tier2_pool) >= self.max_tier2_capacity:
            self._evict_lowest_utility(
                self._tier2_pool,
                max_keep=max(0, self.max_tier2_capacity - 1),
                pinned_ids=self._pinned_tier2,
                tier_level=2,
                capacity_ref=self.max_tier2_capacity,
            )

        gpu_slot = self._promote_on_backend(
            slice_obj,
            TokenResolutionTier.TIER_2_FINE,
            num_predicates=eff_n_preds,
            predicates=predicates,
        )
        self._tier2_pool[sid] = BufferSlotEntry(
            slice_id=sid,
            cam_id=slice_obj.cam_id,
            tier=TokenResolutionTier.TIER_2_FINE,
            last_access_step=self._step_counter,
            access_count=1,
            provenance_heat=heat,
            gpu_tensor=gpu_slot,
        )
        self._query_promoted_t2.add(sid)
        if eff_n_preds > 0:
            self._tier2_has_kv_prefill.add(sid)
            self.telemetry.tier2_vlm_promotions += 1
        else:
            self.telemetry.tier2_vit_promotions += 1
        self._sync_slice_predicates(slice_obj, tier_level=2, predicates=predicates)
        self.peak_tier2_resident = max(self.peak_tier2_resident, len(self._tier2_pool))
        self.telemetry.tier2_promotions += 1
        self.telemetry.total_tokens_evaluated += int(TokenResolutionTier.TIER_2_FINE)
        return slice_obj.tier2_embed if slice_obj.tier2_embed is not None else slice_obj.tier0_embed

    def escalate_margin_candidates(
        self,
        slices: List[VisualDetectionSlice],
        uncertainty_scores: Dict[Any, float],
        max_t1: int,
        max_t2: int,
        num_predicates: int = 0,
        predicates: Optional[Sequence[str]] = None,
    ) -> Tuple[Set[Any], Set[Any]]:
        """
        Margin-Guided Tier Escalation (Section IV-B, Proposition 1, Eq. 7, Algorithm 3):
        Ranks candidate slices by predicate/provenance uncertainty margin, pins the active
        batch against self-eviction, and escalates up to (max_t1, max_t2) slices.
        """
        ranked = sorted(
            slices,
            key=lambda s: uncertainty_scores.get(s.slice_id, 0.0),
            reverse=True,
        )
        t1_batch = ranked[: min(len(ranked), max_t1)]
        t2_batch = ranked[: min(len(ranked), max_t2)]
        t1_selected = {s.slice_id for s in t1_batch}
        t2_selected = {s.slice_id for s in t2_batch}

        self.pin_tier1(t1_selected)
        self.pin_tier2(t2_selected)
        try:
            for s in t1_batch:
                heat = uncertainty_scores.get(s.slice_id, 0.5)
                self.access_tier1(
                    s, heat=heat, num_predicates=num_predicates, predicates=predicates
                )
            for s in t2_batch:
                heat = uncertainty_scores.get(s.slice_id, 0.8)
                self.access_tier2(
                    s, heat=heat, num_predicates=num_predicates, predicates=predicates
                )
        finally:
            self.unpin_tier1(t1_selected)
            self.unpin_tier2(t2_selected)
        return t1_selected, t2_selected
