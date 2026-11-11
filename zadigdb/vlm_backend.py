"""
Multi-Granularity Visual Token Materialization and GPU Visual-KV Slot Backend for ZadigDB.

Provides:
  1. Multi-granularity image construction aligned with Qwen2.5-VL's 14x14 ViT + 2x2 spatial merger:
       - Tier-0 (28x28   ->   1 summary descriptor token)
       - Tier-1 (224x112 ->  32 regional sub-grid tokens)
       - Tier-2 (448x448 -> 256 dense patch tokens via 2x2 4-frame temporal tracklet montage)
  2. Live 4-bit NF4 `Qwen2.5-VL-7B-Instruct` visual feature extraction and zero-shot Yes/No scoring.
  3. Per-(slice_id, tier) `VisualKVSlot` residency with reusable visual-prefix KV states and
     incremental predicate suffix scoring (`logits_to_keep=1`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .schema import TokenResolutionTier, VisualDetectionSlice


DEFAULT_CALIBRATED_TIMINGS: Dict[str, float] = {
    "t0_sec_per_slice": 0.004244,
    "t1_sec_per_slice": 0.385956,
    "t2_sec_per_slice": 0.421210,
    "t1_vit_sec_per_slice": 0.081918,
    "t2_vit_sec_per_slice": 0.100970,
    "kv_hit_sec_per_slice": 0.113028,
    "pair_sec": 0.000006,
}

# Canonical 18 visual & behavioral predicates grounded in CityFlow-NL descriptions.
DEFAULT_PREDICATE_PROMPTS: Dict[str, str] = {
    # Colors (8)
    "white": "Is the primary body color of the target vehicle white or off-white?",
    "black": "Is the primary body color of the target vehicle black?",
    "silver": "Is the primary body color of the target vehicle silver or light metallic?",
    "gray": "Is the primary body color of the target vehicle gray or dark charcoal?",
    "red": "Is the primary body color of the target vehicle red, maroon, or burgundy?",
    "blue": "Is the primary body color of the target vehicle blue or navy?",
    "green": "Is the primary body color of the target vehicle green?",
    "yellow": "Is the primary body color of the target vehicle yellow, gold, or beige?",
    # Vehicle body types (6)
    "suv": "Is the target vehicle an SUV, crossover, or jeep?",
    "sedan": "Is the target vehicle a sedan, coupe, or hatchback passenger car?",
    "van": "Is the target vehicle a van or minivan?",
    "pickup": "Is the target vehicle a pickup truck with an open or covered bed?",
    "truck": "Is the target vehicle a commercial truck, box truck, or cargo truck?",
    "wagon": "Is the target vehicle a station wagon or estate car?",
    # Intersection maneuvers / behaviors across chronological frames (4)
    "going_straight": "Across the frames, is the vehicle driving straight ahead along the road?",
    "turning_left": "Across the frames, is the vehicle making a left turn at the intersection?",
    "turning_right": "Across the frames, is the vehicle making a right turn at the intersection?",
    "stopping": "Across the frames, is the vehicle braking, slowing down, or stopped at the intersection?",
}


def _resolve_tier(tier: Any) -> Tuple[int, TokenResolutionTier]:
    """Maps a TokenResolutionTier enum or integer tier index/token count to (tier_idx, enum_val)."""
    if isinstance(tier, TokenResolutionTier):
        if tier == TokenResolutionTier.TIER_0_MICRO:
            return 0, TokenResolutionTier.TIER_0_MICRO
        if tier == TokenResolutionTier.TIER_1_COARSE:
            return 1, TokenResolutionTier.TIER_1_COARSE
        return 2, TokenResolutionTier.TIER_2_FINE
    val = int(tier)
    if val in (1, 32):
        return 1, TokenResolutionTier.TIER_1_COARSE
    if val in (2, 256):
        return 2, TokenResolutionTier.TIER_2_FINE
    return 0, TokenResolutionTier.TIER_0_MICRO


@dataclass
class VisualKVSlot:
    """
    Resident GPU visual-token slot for an admitted `(slice_id, tier)` buffer page (Section IV-A).
    Holds ViT patch embeddings, pooled L2-normalized descriptor, optional pre-computed multimodal
    prefix KV states (`visual_past_key_values`), and incrementally evaluated predicate scores.
    """
    slice_id: str
    tier: int
    vit_patches_bf16: Optional[Any] = field(default=None, repr=False)
    pooled_vec: Optional[np.ndarray] = field(default=None, repr=False)
    visual_past_key_values: Optional[Any] = field(default=None, repr=False)
    prefix_seq_len: int = 0
    rope_deltas: Optional[Any] = field(default=None, repr=False)
    is_montage: bool = False
    scored_predicates: Dict[str, float] = field(default_factory=dict)


def build_tier_image(frame_paths: Sequence[str], tier: TokenResolutionTier) -> Any:
    """
    Constructs a physical PIL RGB image whose pixel dimensions map to exact Qwen2.5-VL
    merged visual token counts (patch_size=14, spatial_merge_size=2 -> 28x28 px/token):
      - TIER_0_MICRO  (1 token):   28 x 28 single mid-track crop
      - TIER_1_COARSE (32 tokens): 224 x 112 single mid-track crop (8 x 4 = 32 tokens)
      - TIER_2_FINE   (256 tokens): 448 x 448 2x2 chronological 4-frame montage (16 x 16 = 256 tokens)
    """
    from PIL import Image

    if not frame_paths:
        raise ValueError("frame_paths must contain at least one valid image path.")

    mid_idx = len(frame_paths) // 2
    if tier == TokenResolutionTier.TIER_0_MICRO:
        with Image.open(frame_paths[mid_idx]) as im:
            return im.convert("RGB").resize((28, 28), Image.Resampling.BICUBIC)

    if tier == TokenResolutionTier.TIER_1_COARSE:
        with Image.open(frame_paths[mid_idx]) as im:
            return im.convert("RGB").resize((224, 112), Image.Resampling.BICUBIC)

    # TIER_2_FINE: 2x2 chronological 4-frame temporal tracklet montage (448 x 448 -> 256 merged tokens)
    n = len(frame_paths)
    sample_indices = [
        0,
        min(n - 1, max(0, int(round((n - 1) * 0.33)))),
        min(n - 1, max(0, int(round((n - 1) * 0.67)))),
        n - 1,
    ]
    canvas = Image.new("RGB", (448, 448))
    offsets = [(0, 0), (224, 0), (0, 224), (224, 224)]
    for idx_pos, (ox, oy) in zip(sample_indices, offsets):
        with Image.open(frame_paths[idx_pos]) as im:
            tile = im.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
            canvas.paste(tile, (ox, oy))
    return canvas


class Qwen25VLHardwareBackend:
    """
    Single-GPU CUDA Execution Backend powered by 4-bit NF4 `Qwen2.5-VL-7B-Instruct` (Section IV).

    Supports:
      - Multi-granularity (`1 / 32 / 256` merged visual tokens) ViT feature encoding & predicate scoring.
      - Per-`(slice_id, tier)` `VisualKVSlot` caching with reusable visual-prefix KV states and
        incremental warm predicate evaluation (`score_cached_predicates`).
    """

    def __init__(
        self,
        model_path: str,
        device_id: int = 0,
        kv_bytes_per_token: int = 131072,
        hard_vram_limit_gb: float = 24.0,
        predicate_prompts: Optional[Dict[str, str]] = None,
        live_on_demand_vlm: bool = False,
    ):
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

        self._torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Qwen25VLHardwareBackend.")

        self.device = torch.device(f"cuda:{int(device_id)}")
        self.model_path = model_path
        self.kv_bytes_per_token = int(kv_bytes_per_token)
        self.hard_vram_limit_gb = float(hard_vram_limit_gb)
        self.predicate_prompts: Dict[str, str] = {
            **DEFAULT_PREDICATE_PROMPTS,
            **(predicate_prompts or {}),
        }

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["visual"],
        )
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                min_pixels=28 * 28,
                max_pixels=448 * 448,
                use_fast=False,
            )
        except TypeError:
            from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen2_5_VLProcessor

            img_proc = Qwen2VLImageProcessor.from_pretrained(
                model_path,
                min_pixels=28 * 28,
                max_pixels=448 * 448,
            )
            tok_inst = AutoTokenizer.from_pretrained(model_path)
            chat_tmpl = None
            chat_tmpl_path = os.path.join(model_path, "chat_template.json")
            if os.path.exists(chat_tmpl_path):
                with open(chat_tmpl_path, "r", encoding="utf-8") as cf:
                    chat_tmpl = json.load(cf).get("chat_template")
            if not chat_tmpl:
                chat_tmpl = getattr(tok_inst, "chat_template", None)
            vp_cls = Qwen2_5_VLProcessor.get_possibly_dynamic_module("BaseVideoProcessor")
            vp_stub = object.__new__(vp_cls)
            vp_stub.model_input_names = ["pixel_values_videos", "video_grid_thw"]
            self.processor = Qwen2_5_VLProcessor(
                image_processor=img_proc,
                tokenizer=tok_inst,
                video_processor=vp_stub,
                chat_template=chat_tmpl,
            )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_cfg,
            device_map={"": int(device_id)},
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

        tok = self.processor.tokenizer
        tok.padding_side = "left"
        self._yes_ids = sorted(
            {
                tok.encode(w, add_special_tokens=False)[0]
                for w in ("Yes", " Yes", "yes", "YES")
                if len(tok.encode(w, add_special_tokens=False)) == 1
            }
        )
        self._no_ids = sorted(
            {
                tok.encode(w, add_special_tokens=False)[0]
                for w in ("No", " No", "no", "NO")
                if len(tok.encode(w, add_special_tokens=False)) == 1
            }
        )
        if not self._yes_ids or not self._no_ids:
            raise RuntimeError("Failed to resolve single-token Yes/No IDs in Qwen2.5-VL tokenizer.")

        free_b, total_b = torch.cuda.mem_get_info(self.device)
        used_driver_b = max(0, int(total_b - free_b))
        alloc_b = int(torch.cuda.memory_allocated(self.device))
        self._cuda_context_overhead_bytes: int = max(0, used_driver_b - alloc_b)
        self.base_weights_gb = round(float(used_driver_b) / (1024.0 ** 3), 3)

        self._prompt_template_cache: Dict[Tuple[bool, str], str] = {}
        self._gpu_slot_cache: Dict[Tuple[str, int], VisualKVSlot] = {}
        self._tier0_index_buf: Optional[Any] = None
        self._query_oom: bool = False
        self._query_wall_t0: float = 0.0
        self.live_on_demand_vlm: bool = bool(live_on_demand_vlm)
        self.calibrated_timings: Dict[str, float] = dict(DEFAULT_CALIBRATED_TIMINGS)

    def _resolve_question(self, pred_name: str) -> str:
        if pred_name in self.predicate_prompts:
            return self.predicate_prompts[pred_name]
        clean = pred_name.replace("_", " ").strip()
        return f"Does this vehicle match the attribute '{clean}'?"

    def extract_visual_embeddings_batch(
        self,
        images: Sequence[Any],
    ) -> Tuple[List[np.ndarray], float]:
        """
        Runs `Qwen2.5-VL-7B-Instruct`'s vision tower (`get_image_features`) over a batch of
        same-tier PIL images on GPU, returning L2-normalized embeddings and per-image GPU latency (s).
        """
        torch = self._torch
        if not images:
            return [], 0.0

        inp = self.processor.image_processor(images=list(images), return_tensors="pt")
        pv = inp["pixel_values"].to(self.device)
        thw = inp["image_grid_thw"].to(self.device)

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)
        ev_start.record()
        with torch.no_grad():
            vis_out = self.model.get_image_features(pixel_values=pv, image_grid_thw=thw)
        ev_end.record()
        torch.cuda.synchronize(self.device)
        elapsed_sec = float(ev_start.elapsed_time(ev_end)) / 1000.0

        token_seqs = vis_out.pooler_output if hasattr(vis_out, "pooler_output") else vis_out
        embeds: List[np.ndarray] = []
        for seq in token_seqs:
            seq_f = seq.detach().float()
            mean_v = seq_f.mean(dim=0)
            if seq_f.shape[0] > 1:
                max_v = seq_f.max(dim=0).values
                pooled = 0.70 * mean_v + 0.30 * max_v
            else:
                pooled = mean_v
            vec = pooled.cpu().numpy().astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec = vec / norm
            embeds.append(vec)

        return embeds, elapsed_sec / float(len(images))

    def _encode_single_tier_crop(self, image: Any, tier_idx: int) -> Tuple[Any, np.ndarray]:
        """Encodes one tier crop through `visual` to obtain `vit_patches_bf16` and pooled vector."""
        torch = self._torch
        inp = self.processor.image_processor(images=[image], return_tensors="pt")
        pv = inp["pixel_values"].to(self.device)
        thw = inp["image_grid_thw"].to(self.device)
        with torch.no_grad():
            vis_out = self.model.get_image_features(pixel_values=pv, image_grid_thw=thw)
        token_seqs = vis_out.pooler_output if hasattr(vis_out, "pooler_output") else vis_out
        patches_bf16 = token_seqs[0].detach().to(device=self.device, dtype=torch.bfloat16)
        seq_f = patches_bf16.float()
        mean_v = seq_f.mean(dim=0)
        pooled = (0.70 * mean_v + 0.30 * seq_f.max(dim=0).values) if seq_f.shape[0] > 1 else mean_v
        vec = pooled.cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        _ = tier_idx
        return patches_bf16, vec

    def _format_predicate_prompt(self, question: str, is_montage: bool) -> str:
        cache_key = (bool(is_montage), question)
        cached = self._prompt_template_cache.get(cache_key)
        if cached is not None:
            return cached
        prefix = (
            "Inspect this 2x2 chronological 4-frame sequence (top-left, top-right, bottom-left, bottom-right) of the target traffic vehicle."
            if is_montage
            else "Inspect this traffic camera vehicle crop."
        )
        formatted = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": f"{prefix} {question} Answer strictly Yes or No.",
                        },
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        if "<|image_pad|>" not in formatted:
            raise RuntimeError("Multimodal chat template failed to emit <|image_pad|> placeholder.")
        self._prompt_template_cache[cache_key] = formatted
        return formatted

    def _split_visual_prefix_and_suffix(self, question: str, is_montage: bool) -> Tuple[str, str]:
        full_prompt = self._format_predicate_prompt(question, is_montage=is_montage)
        marker = "<|vision_end|>"
        idx = full_prompt.find(marker)
        if idx < 0:
            raise RuntimeError("Multimodal chat template missing <|vision_end|> boundary.")
        cut = idx + len(marker)
        return full_prompt[:cut], full_prompt[cut:]

    def _prefill_visual_prefix_kv(self, image: Any, is_montage: bool) -> Tuple[Any, int, Any]:
        """Prefills the visual token prefix once (`use_cache=True`) for reusable KV-slot caching."""
        torch = self._torch
        prefix_text, _ = self._split_visual_prefix_and_suffix("Is this a vehicle?", is_montage=is_montage)
        inputs = self.processor(
            text=[prefix_text],
            images=[image],
            padding=False,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, use_cache=True, logits_to_keep=1)
        seq_len = int(inputs["input_ids"].shape[1])
        inner_model = getattr(self.model, "model", None)
        rope_deltas = getattr(inner_model, "rope_deltas", None)
        if rope_deltas is not None:
            rope_deltas = rope_deltas.detach().clone()
        return out.past_key_values, seq_len, rope_deltas

    def _score_suffix_on_slot(self, slot: VisualKVSlot, question: str) -> float:
        """Evaluates a single predicate suffix over a cached `VisualKVSlot` (`logits_to_keep=1`)."""
        torch = self._torch
        _, suffix_text = self._split_visual_prefix_and_suffix(question, is_montage=slot.is_montage)
        suffix_ids = self.processor.tokenizer(
            suffix_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

        kv_cache = slot.visual_past_key_values
        if hasattr(kv_cache, "crop"):
            kv_cache.crop(slot.prefix_seq_len)
        inner_model = getattr(self.model, "model", None)
        if inner_model is not None and slot.rope_deltas is not None:
            inner_model.rope_deltas = slot.rope_deltas

        suffix_len = int(suffix_ids.shape[1])
        total_len = slot.prefix_seq_len + suffix_len
        attn_mask = torch.ones((1, total_len), dtype=torch.long, device=self.device)
        cache_pos = torch.arange(
            slot.prefix_seq_len, total_len, dtype=torch.long, device=self.device
        )
        delta_val = (
            int(slot.rope_deltas.reshape(-1)[0].item())
            if slot.rope_deltas is not None
            else 0
        )
        pos_ids = (cache_pos + delta_val).view(1, 1, suffix_len).expand(3, 1, suffix_len).contiguous()
        with torch.no_grad():
            out = self.model(
                input_ids=suffix_ids,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                cache_position=cache_pos,
                past_key_values=kv_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            last_logits = out.logits[:, -1, :].float()
            yes_lse = torch.logsumexp(last_logits[:, self._yes_ids], dim=-1)
            no_lse = torch.logsumexp(last_logits[:, self._no_ids], dim=-1)
            prob = float(torch.sigmoid(yes_lse - no_lse).item())
        if hasattr(kv_cache, "crop"):
            kv_cache.crop(slot.prefix_seq_len)
        return round(prob, 5)

    def score_predicates_for_image(
        self,
        image: Any,
        predicate_prompts: Dict[str, str],
        batch_size: int = 24,
        is_montage: bool = False,
    ) -> Tuple[Dict[str, float], float]:
        """
        Scores zero-shot semantic predicates on a single tier-formatted PIL image using
        calibrated `Qwen2.5-VL-7B-Instruct` prefill Yes/No next-token probabilities.
        """
        torch = self._torch
        if not predicate_prompts:
            return {}, 0.0

        keys = list(predicate_prompts.keys())
        scores: Dict[str, float] = {}
        total_gpu_ms = 0.0

        for i in range(0, len(keys), batch_size):
            sub_keys = keys[i : i + batch_size]
            texts = [
                self._format_predicate_prompt(predicate_prompts[k], is_montage=is_montage)
                for k in sub_keys
            ]
            imgs = [image] * len(sub_keys)
            inputs = self.processor(
                text=texts,
                images=imgs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()
            with torch.no_grad():
                out = self.model(**inputs, use_cache=False, logits_to_keep=1)
                last_logits = out.logits[:, -1, :].float()
                yes_lse = torch.logsumexp(last_logits[:, self._yes_ids], dim=-1)
                no_lse = torch.logsumexp(last_logits[:, self._no_ids], dim=-1)
                probs = torch.sigmoid(yes_lse - no_lse).cpu().numpy()
            ev_end.record()
            torch.cuda.synchronize(self.device)
            total_gpu_ms += float(ev_start.elapsed_time(ev_end))

            for k, p_val in zip(sub_keys, probs):
                scores[k] = round(float(p_val), 5)

        return scores, total_gpu_ms / 1000.0

    def materialize_tier(
        self,
        slice_obj: VisualDetectionSlice,
        tier: Any,
        retain_slot: bool = True,
        predicates: Optional[Sequence[str]] = None,
    ) -> Optional[VisualKVSlot]:
        """
        Materializes Tier-1 (32 tokens) or Tier-2 (256 tokens) visual features and predicate
        log-odds for `slice_obj`. When `retain_slot=True`, caches a `VisualKVSlot` with reusable
        visual-prefix KV states in `self._gpu_slot_cache[(slice_id, tier)]`.
        """
        torch = self._torch
        tier_idx, tier_enum = _resolve_tier(tier)
        sid_key = str(slice_obj.slice_id)
        slot_key = (sid_key, tier_idx)
        is_montage = tier_enum == TokenResolutionTier.TIER_2_FINE

        req_preds = list(dict.fromkeys(predicates)) if predicates is not None else []
        prompt_map = {p: self._resolve_question(p) for p in req_preds}

        existing = self._gpu_slot_cache.get(slot_key)
        if existing is not None and retain_slot:
            if req_preds:
                self.score_cached_predicates(slice_obj, tier_idx, req_preds)
            return existing

        try:
            if self.live_on_demand_vlm and slice_obj.frame_paths:
                img = build_tier_image(slice_obj.frame_paths, tier_enum)
                vit_patches, pooled_vec = self._encode_single_tier_crop(img, tier_idx)
                if tier_idx == 1:
                    slice_obj.tier1_embed = pooled_vec
                elif tier_idx == 2:
                    slice_obj.tier2_embed = pooled_vec

                if retain_slot:
                    if prompt_map:
                        past_kv, pref_len, rope_d = self._prefill_visual_prefix_kv(
                            img, is_montage=is_montage
                        )
                    else:
                        past_kv, pref_len, rope_d = None, 0, None
                    slot = VisualKVSlot(
                        slice_id=sid_key,
                        tier=tier_idx,
                        vit_patches_bf16=vit_patches,
                        pooled_vec=pooled_vec,
                        visual_past_key_values=past_kv,
                        prefix_seq_len=pref_len,
                        rope_deltas=rope_d,
                        is_montage=is_montage,
                        scored_predicates={},
                    )
                    if prompt_map:
                        raw_sc = {
                            p: self._score_suffix_on_slot(slot, q_txt)
                            for p, q_txt in prompt_map.items()
                        }
                        slot.scored_predicates.update(raw_sc)
                        self._write_scores_to_slice(slice_obj, tier_idx, raw_sc)
                    self._gpu_slot_cache[slot_key] = slot
                    return slot

                if prompt_map:
                    raw_sc, _ = self.score_predicates_for_image(
                        img, prompt_map, is_montage=is_montage
                    )
                    self._write_scores_to_slice(slice_obj, tier_idx, raw_sc)
                return None

            # Pre-extracted offline descriptor path
            emb_src = (
                slice_obj.tier2_embed
                if (tier_idx == 2 and slice_obj.tier2_embed is not None)
                else (
                    slice_obj.tier1_embed
                    if (tier_idx == 1 and slice_obj.tier1_embed is not None)
                    else slice_obj.tier0_embed
                )
            )
            emb_np = np.asarray(emb_src, dtype=np.float32)
            tier_scores = (
                slice_obj.attribute_scores_t2
                if tier_idx == 2
                else slice_obj.attribute_scores_t1
            )
            scored_init = {
                p: float(tier_scores[p]) for p in req_preds if p in tier_scores
            }
            if scored_init:
                self._write_scores_to_slice(slice_obj, tier_idx, scored_init)
            if not retain_slot:
                return None
            slot = VisualKVSlot(
                slice_id=sid_key,
                tier=tier_idx,
                vit_patches_bf16=None,
                pooled_vec=emb_np,
                visual_past_key_values=None,
                prefix_seq_len=0,
                rope_deltas=None,
                is_montage=is_montage,
                scored_predicates=scored_init,
            )
            self._gpu_slot_cache[slot_key] = slot
            return slot

        except torch.cuda.OutOfMemoryError:
            self._query_oom = True
            return None

    @staticmethod
    def _write_scores_to_slice(
        slice_obj: VisualDetectionSlice,
        tier_idx: int,
        scores: Dict[str, float],
    ) -> None:
        if tier_idx == 1:
            slice_obj.attribute_scores_t1.update(scores)
        elif tier_idx == 2:
            slice_obj.attribute_scores_t2.update(scores)
        else:
            slice_obj.attribute_scores_t0.update(scores)
        slice_obj.predicate_scores.update(scores)

    def score_cached_predicates(
        self,
        slice_obj: VisualDetectionSlice,
        tier: Any,
        predicates: Sequence[str],
    ) -> Dict[str, float]:
        """
        Incrementally evaluates newly requested predicates for an already warm-cached
        `(slice_id, tier)` slot using its cached `visual_past_key_values` (`logits_to_keep=1`),
        updating `slice_obj` without re-encoding visual tokens.
        """
        if not predicates:
            return {}
        tier_idx, tier_enum = _resolve_tier(tier)
        sid_key = str(slice_obj.slice_id)
        slot = self._gpu_slot_cache.get((sid_key, tier_idx))
        tier_dict = (
            slice_obj.attribute_scores_t2
            if tier_idx == 2
            else slice_obj.attribute_scores_t1
        )

        resolved: Dict[str, float] = {}
        missing: List[str] = []
        for p in dict.fromkeys(predicates):
            if slot is not None and p in slot.scored_predicates:
                resolved[p] = slot.scored_predicates[p]
            elif p in tier_dict and not self.live_on_demand_vlm:
                resolved[p] = float(tier_dict[p])
            else:
                missing.append(p)

        if missing:
            if (
                slot is not None
                and slot.visual_past_key_values is None
                and self.live_on_demand_vlm
                and slice_obj.frame_paths
            ):
                img_lazy = build_tier_image(slice_obj.frame_paths, tier_enum)
                pkv_l, plen_l, rd_l = self._prefill_visual_prefix_kv(
                    img_lazy, is_montage=slot.is_montage
                )
                slot.visual_past_key_values = pkv_l
                slot.prefix_seq_len = plen_l
                slot.rope_deltas = rd_l

            if slot is not None and slot.visual_past_key_values is not None:
                raw_new = {
                    p: self._score_suffix_on_slot(slot, self._resolve_question(p))
                    for p in missing
                }
                slot.scored_predicates.update(raw_new)
                resolved.update(raw_new)
            elif self.live_on_demand_vlm and slice_obj.frame_paths:
                img = build_tier_image(slice_obj.frame_paths, tier_enum)
                prompt_map = {p: self._resolve_question(p) for p in missing}
                raw_new, _ = self.score_predicates_for_image(
                    img,
                    prompt_map,
                    is_montage=(tier_enum == TokenResolutionTier.TIER_2_FINE),
                )
                if slot is not None:
                    slot.scored_predicates.update(raw_new)
                resolved.update(raw_new)
            else:
                for p in missing:
                    if p in tier_dict:
                        val = float(tier_dict[p])
                        resolved[p] = val
                        if slot is not None:
                            slot.scored_predicates[p] = val

        if resolved:
            self._write_scores_to_slice(slice_obj, tier_idx, resolved)
        return resolved

    def release_slot(self, slice_id: Any, tier: Any) -> None:
        """Releases a cached `VisualKVSlot` from GPU memory upon buffer pool eviction."""
        tier_idx, _ = _resolve_tier(tier)
        self._gpu_slot_cache.pop((str(slice_id), tier_idx), None)

    def clear_slots(self) -> None:
        """Clears all cached `VisualKVSlot` entries from GPU memory."""
        self._gpu_slot_cache.clear()
        if self.live_on_demand_vlm and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def promote_slice_to_gpu(
        self,
        slice_obj: VisualDetectionSlice,
        tier: TokenResolutionTier,
        num_predicates: int = 0,
        predicates: Optional[Sequence[str]] = None,
    ) -> Optional[VisualKVSlot]:
        """Materializes and retains a `VisualKVSlot` for `slice_obj` at `tier`."""
        _ = num_predicates
        return self.materialize_tier(
            slice_obj=slice_obj,
            tier=tier,
            retain_slot=True,
            predicates=predicates,
        )

    def record_cache_hit_access(self, gpu_slot: Any, num_tokens: int) -> None:
        """Records a warm cache hit on a resident `VisualKVSlot` without VLM re-encoding."""
        _ = num_tokens
        if isinstance(gpu_slot, VisualKVSlot) and gpu_slot.vit_patches_bf16 is not None:
            _ = gpu_slot.vit_patches_bf16.shape

    def begin_query_measurement(
        self,
        n_indexed: int,
        active_slices: Sequence[VisualDetectionSlice],
    ) -> None:
        """Initializes per-query GPU memory & wall-clock measurement while preserving warm slots."""
        _ = n_indexed
        torch = self._torch
        self._query_oom = False
        if not self.live_on_demand_vlm:
            self._query_wall_t0 = time.perf_counter()
            return
        if active_slices and self._tier0_index_buf is None:
            t0_mat = np.stack([s.tier0_embed for s in active_slices], axis=0)
            self._tier0_index_buf = torch.as_tensor(
                t0_mat, dtype=torch.bfloat16, device=self.device
            )

        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._query_wall_t0 = time.perf_counter()

    def end_query_measurement(
        self,
        prov_edges: int = 0,
    ) -> Tuple[float, bool, float]:
        """
        Synchronizes CUDA stream, reads `torch.cuda.max_memory_allocated()`, and returns
        `(peak_vram_gb, oom_occurred, latency_sec)`.
        """
        _ = prov_edges
        if not self.live_on_demand_vlm:
            wall_elapsed = time.perf_counter() - self._query_wall_t0
            return float(self.base_weights_gb), False, round(wall_elapsed, 3)
        torch = self._torch
        torch.cuda.synchronize(self.device)
        wall_elapsed = time.perf_counter() - self._query_wall_t0
        cuda_peak_bytes = int(torch.cuda.max_memory_allocated(self.device)) + int(
            self._cuda_context_overhead_bytes
        )
        peak_vram_gb = round(float(cuda_peak_bytes) / (1024.0 ** 3), 2)
        oom_occurred = bool(self._query_oom or peak_vram_gb > self.hard_vram_limit_gb)
        if oom_occurred:
            torch.cuda.empty_cache()

        return peak_vram_gb, oom_occurred, round(wall_elapsed, 3)
