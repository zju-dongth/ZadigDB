"""Observation slice, road-network camera topology, and query workload loaders."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from zadigdb.schema import CameraTopology, VisualDetectionSlice
from zadigdb.synthesizer import DeclarativeVideoQuery


def _decode_embedding(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, str):
        return np.frombuffer(base64.b64decode(value), dtype=np.float32).copy()
    return np.asarray(value, dtype=np.float32)


def _read_records(file_path: str) -> List[Dict[str, Any]]:
    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    for key in ("slices", "records", "queries", "workload"):
        if isinstance(payload, dict) and isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"Unsupported record container in {file_path!r}")


def load_topology(
    topology_file: str,
    *,
    v_min_mps: float = 5.5,
    v_max_mps: float = 21.0,
    tau_sig_sec: float = 18.0,
    congestion_speed_floor_mps: float = 0.8,
    congestion_clearance_scale: float = 2.2,
) -> CameraTopology:
    with open(topology_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cam_ids = [int(c) for c in meta["camera_ids"]]
    return CameraTopology(
        num_cameras=len(cam_ids),
        camera_ids=cam_ids,
        dist_matrix=np.asarray(meta["dist_matrix"], dtype=np.float32),
        adjacency=np.asarray(meta["adjacency"], dtype=np.int32),
        v_min_mps=float(meta.get("v_min_mps", v_min_mps)),
        v_max_mps=float(meta.get("v_max_mps", v_max_mps)),
        tau_sig_sec=float(meta.get("tau_sig_sec", tau_sig_sec)),
        fov_overlap_m=float(meta.get("fov_overlap_m", 0.0)),
        co_located_slack_sec=float(meta.get("co_located_slack_sec", 0.0)),
        congestion_speed_floor_mps=float(meta.get("congestion_speed_floor_mps", congestion_speed_floor_mps)),
        congestion_clearance_scale=float(meta.get("congestion_clearance_scale", congestion_clearance_scale)),
    )


def load_slices(slices_file: str) -> Tuple[List[VisualDetectionSlice], Dict[int, int]]:
    slices: List[VisualDetectionSlice] = []
    slice_gt_entities: Dict[int, int] = {}
    for rec in _read_records(slices_file):
        sid = int(rec["slice_id"])
        t0_embed = _decode_embedding(rec["tier0_embed"])
        if t0_embed is None:
            raise ValueError(f"Slice {sid} is missing tier0_embed.")
        frame_paths = rec.get("crop_paths") or rec.get("frame_paths") or ()
        slices.append(
            VisualDetectionSlice(
                slice_id=sid,
                cam_id=int(rec["cam_id"]),
                t_enter=float(rec["t_enter"]),
                t_exit=float(rec["t_exit"]),
                tier0_embed=t0_embed,
                tier1_embed=_decode_embedding(rec.get("tier1_embed")),
                tier2_embed=_decode_embedding(rec.get("tier2_embed")),
                attribute_scores_t0={str(k): float(v) for k, v in (rec.get("attribute_scores_t0") or {}).items()},
                attribute_scores_t1={str(k): float(v) for k, v in (rec.get("attribute_scores_t1") or {}).items()},
                attribute_scores_t2={str(k): float(v) for k, v in (rec.get("attribute_scores_t2") or {}).items()},
                frame_paths=tuple(str(p) for p in frame_paths),
            )
        )
        if rec.get("gt_entity_id") is not None:
            slice_gt_entities[sid] = int(rec["gt_entity_id"])
    return slices, slice_gt_entities


def load_queries(
    queries_file: str,
    target_cameras: Optional[List[int]] = None,
) -> Tuple[List[DeclarativeVideoQuery], Dict[str, Dict[str, Any]]]:
    queries: List[DeclarativeVideoQuery] = []
    query_gt: Dict[str, Dict[str, Any]] = {}
    for idx, rec in enumerate(_read_records(queries_file)):
        qid = str(rec.get("query_id", f"Q_{idx:03d}"))
        sql_text = str(rec.get("sql") or rec.get("sql_text") or "").strip()
        queries.append(
            DeclarativeVideoQuery.from_sql(
                sql_text=sql_text,
                query_id=qid,
                target_cameras=target_cameras,
            )
        )
        gt_entry: Dict[str, Any] = {}
        if rec.get("gt_target_entities") is not None:
            gt_entry["gt_target_entities"] = [int(x) for x in rec["gt_target_entities"]]
        if rec.get("gt_comove_pairs") is not None:
            gt_entry["gt_comove_pairs"] = [(int(p[0]), int(p[1])) for p in rec["gt_comove_pairs"]]
        if gt_entry:
            query_gt[qid] = gt_entry
    return queries, query_gt
