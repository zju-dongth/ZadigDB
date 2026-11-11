"""Main pipeline entrypoint for ZadigDB multi-camera video query execution."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional
import yaml

from eval import evaluate_workload
from load_data import load_queries, load_slices, load_topology
from zadigdb import Qwen25VLHardwareBackend, ZadigDBEngine


def _resolve_path(path_str: str, base_dir: str) -> str:
    if not path_str or os.path.isabs(path_str):
        return path_str
    if os.path.exists(path_str):
        return os.path.normpath(path_str)
    repo_root = os.path.dirname(base_dir) if os.path.basename(base_dir) == "configs" else base_dir
    for cand_root in (repo_root, base_dir):
        candidate = os.path.normpath(os.path.join(cand_root, path_str))
        if os.path.exists(candidate):
            return candidate
    return os.path.normpath(os.path.join(repo_root, path_str))


def _resolve_vlm_model_path(raw_model_path: str, base_dir: str) -> str:
    if not raw_model_path:
        return ""
    resolved = _resolve_path(raw_model_path, base_dir)
    if os.path.exists(resolved) or raw_model_path.startswith((".", "/", "~")):
        return os.path.expanduser(resolved)
    return raw_model_path


def run_pipeline(
    cfg_file: str,
    sql_override: Optional[str] = None,
    mode_override: Optional[str] = None,
    allow_vlm: bool = False,
) -> Dict[str, Any]:
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg_dir = os.path.dirname(os.path.abspath(cfg_file))
    ds_cfg = cfg.get("dataset", {})
    hw_cfg = cfg.get("hardware", {})
    topo_defaults = cfg.get("topology_defaults", {})

    topology_path = _resolve_path(str(ds_cfg.get("topology_file", "")), cfg_dir)
    slices_path = _resolve_path(str(ds_cfg.get("slices_file", "")), cfg_dir)
    queries_path = _resolve_path(str(ds_cfg.get("queries_file", "")), cfg_dir)
    vlm_model_path = _resolve_vlm_model_path(str(ds_cfg.get("vlm_model_path", "")).strip(), cfg_dir)

    topology = load_topology(
        topology_path,
        v_min_mps=float(topo_defaults.get("v_min_mps", 5.5)),
        v_max_mps=float(topo_defaults.get("v_max_mps", 21.0)),
        tau_sig_sec=float(topo_defaults.get("tau_sig_sec", 18.0)),
        congestion_speed_floor_mps=float(topo_defaults.get("congestion_speed_floor_mps", 0.8)),
        congestion_clearance_scale=float(topo_defaults.get("congestion_clearance_scale", 2.2)),
    )
    slices, slice_gt_entities = load_slices(slices_path)

    vlm_backend: Optional[Qwen25VLHardwareBackend] = None
    if allow_vlm:
        if not vlm_model_path:
            raise ValueError("dataset.vlm_model_path must be set when --allow-vlm is enabled.")
        vlm_backend = Qwen25VLHardwareBackend(
            model_path=vlm_model_path,
            kv_bytes_per_token=int(hw_cfg.get("kv_bytes_per_token", 131072)),
            hard_vram_limit_gb=float(hw_cfg.get("hard_vram_limit_gb", 24.0)),
            live_on_demand_vlm=True,
        )

    engine = ZadigDBEngine.from_yaml(
        config_path=cfg_file,
        topology=topology,
        total_indexed_snapshots=len(slices),
        apply_topology_defaults=False,
        vlm_backend=vlm_backend,
    )
    if mode_override is not None:
        mode_clean = mode_override.strip().lower()
        if mode_clean not in ("warm", "cold"):
            raise ValueError(f"Unsupported execution mode: {mode_override!r}")
        engine.execution_mode = mode_clean
        engine.buffer_pool.enable_multi_query_cache = mode_clean == "warm"

    try:
        if sql_override is not None:
            results = [engine.execute_sql(sql_override, slices, query_id="SQL_CLI")]
            return evaluate_workload(results, slice_gt_entities=slice_gt_entities)
        queries, query_gt = load_queries(queries_path, target_cameras=list(topology.camera_ids))
        results: List[Any] = [engine.execute_query(q, slices) for q in queries]
        return evaluate_workload(results, query_gt=query_gt, slice_gt_entities=slice_gt_entities)
    finally:
        engine.reset_hardware_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="ZadigDB Multi-Camera Video Query Pipeline")
    parser.add_argument("--cfg_file", type=str, default="configs/default_4090.yaml")
    parser.add_argument("--sql", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["warm", "cold"], default=None)
    parser.add_argument("--allow-vlm", dest="allow_vlm", action="store_true")
    args = parser.parse_args()
    summary = run_pipeline(
        cfg_file=args.cfg_file,
        sql_override=args.sql,
        mode_override=args.mode,
        allow_vlm=args.allow_vlm,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
