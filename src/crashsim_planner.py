#!/usr/bin/env python3
"""
crashsim_planner.py

Run planner rollouts for ego only on the nuScenes-Crash dataset (metadata-only nuScenes format):
- Surrounding agents: directly use the future trajectories recorded in nuScenes-Crash
  (no model-based simulation)
- Ego: roll out the specified planner (Lane-graph/IDM/PDM-Closed/replay) and replace ego future

References:
- Data loading and `create_planner` style in `src/adv_gen_eval_planner.py`
- Crash dataset schema conventions in `src/test_nusc_crash.py` / `src/build_nusc_crash.py`
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data as Graph
from torch_geometric.data import DataLoader as GraphDataLoader

import datasets.nuscenes_utils as nutils
from datasets.map_env import NUSC_MAP_SIZES, NuScenesMapEnv
from datasets.utils import MeanStdNormalizer, NUSC_NORM_STATS
from losses.adv_gen_nusc import check_single_veh_coll_min_box_gap
from planners.planner import PlannerConfig
from utils.common import dict2obj, mkdir
from utils.logger import Logger, throw_err
from utils.torch import get_device


EVALUATION_PIPELINE_VERSION = "population-grounded-interpretable-llm-v3.8"
# Bump when deterministic prior / population evidence formulas change so
# --eval_llm_only auto-rebuilds stale interpretable_eval.json.
DETERMINISTIC_EVIDENCE_VERSION = "population-prior-v3.3"


REQUIRED_TABLES = (
    "scene.json",
    "sample.json",
    "ego_pose.json",
    "sample_annotation.json",
    "instance.json",
    "log.json",
    "map.json",
)


def _read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected list in {path}, got {type(data)}")
    return data


def _index_by_token(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        tok = r.get("token")
        if isinstance(tok, str) and tok:
            out[tok] = r
    return out


def _yaw_from_quat_wxyz(q: Sequence[float]) -> float:
    # Written by build_nusc_crash: yaw-only rotation, quaternion is (w, 0, 0, z)
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

def _maybe_flip_singapore_xyhh(
    xyhh: np.ndarray,
    *,
    map_name: str,
    flip_singapore: bool,
) -> np.ndarray:
    """
    Align with the convention in `datasets/nuscenes_dataset.py`:
    if `flip_singapore` is enabled, mirror the y-axis for singapore-* maps and flip hsin.
    """
    if not (bool(flip_singapore) and isinstance(map_name, str) and map_name.startswith("singapore-")):
        return xyhh
    mheight = float(NUSC_MAP_SIZES[map_name][0])
    out = xyhh.copy()
    out[:, 1] = mheight - out[:, 1]
    out[:, 3] = -out[:, 3]
    return out


@dataclass(frozen=True)
class NuscCrashMeta:
    dataroot: str
    version: str
    meta_dir: str

    scene: List[Dict[str, Any]]
    sample: List[Dict[str, Any]]
    ego_pose: List[Dict[str, Any]]
    sample_annotation: List[Dict[str, Any]]
    instance: List[Dict[str, Any]]
    log: List[Dict[str, Any]]
    map: List[Dict[str, Any]]

    scene_by_token: Dict[str, Dict[str, Any]]
    sample_by_token: Dict[str, Dict[str, Any]]
    ego_by_ts: Dict[int, Dict[str, Any]]
    inst_by_token: Dict[str, Dict[str, Any]]
    log_by_token: Dict[str, Dict[str, Any]]

    sample_list_by_scene: Dict[str, List[Dict[str, Any]]]
    anns_by_sample: Dict[str, List[Dict[str, Any]]]
    anns_by_instance: Dict[str, List[Dict[str, Any]]]


def load_nusc_crash_meta(dataroot: str, version: str) -> NuscCrashMeta:
    version = version.replace("v1.0-", "")
    meta_dir = os.path.join(os.path.abspath(dataroot), f"v1.0-{version}")
    if not os.path.isdir(meta_dir):
        raise FileNotFoundError(f"meta dir not found: {meta_dir}")
    for t in REQUIRED_TABLES:
        p = os.path.join(meta_dir, t)
        if not os.path.exists(p):
            raise FileNotFoundError(f"required table not found: {p}")

    scene = _read_json(os.path.join(meta_dir, "scene.json"))
    sample = _read_json(os.path.join(meta_dir, "sample.json"))
    ego_pose = _read_json(os.path.join(meta_dir, "ego_pose.json"))
    sample_annotation = _read_json(os.path.join(meta_dir, "sample_annotation.json"))
    instance = _read_json(os.path.join(meta_dir, "instance.json"))
    log = _read_json(os.path.join(meta_dir, "log.json"))
    map_rows = _read_json(os.path.join(meta_dir, "map.json"))

    scene_by_token = _index_by_token(scene)
    sample_by_token = _index_by_token(sample)
    log_by_token = _index_by_token(log)
    inst_by_token = _index_by_token(instance)

    ego_by_ts: Dict[int, Dict[str, Any]] = {}
    for ep in ego_pose:
        ts = ep.get("timestamp")
        if isinstance(ts, int):
            ego_by_ts[ts] = ep

    anns_by_sample: Dict[str, List[Dict[str, Any]]] = {}
    anns_by_instance: Dict[str, List[Dict[str, Any]]] = {}
    for a in sample_annotation:
        st = a.get("sample_token", "")
        it = a.get("instance_token", "")
        if isinstance(st, str) and st:
            anns_by_sample.setdefault(st, []).append(a)
        if isinstance(it, str) and it:
            anns_by_instance.setdefault(it, []).append(a)

    sample_list_by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for s in sample:
        sc = s.get("scene_token")
        if isinstance(sc, str) and sc:
            sample_list_by_scene.setdefault(sc, []).append(s)
    for sc, lst in sample_list_by_scene.items():
        lst.sort(key=lambda r: int(r.get("timestamp", 0)))

    return NuscCrashMeta(
        dataroot=os.path.abspath(dataroot),
        version=version,
        meta_dir=meta_dir,
        scene=scene,
        sample=sample,
        ego_pose=ego_pose,
        sample_annotation=sample_annotation,
        instance=instance,
        log=log,
        map=map_rows,
        scene_by_token=scene_by_token,
        sample_by_token=sample_by_token,
        ego_by_ts=ego_by_ts,
        inst_by_token=inst_by_token,
        log_by_token=log_by_token,
        sample_list_by_scene=sample_list_by_scene,
        anns_by_sample=anns_by_sample,
        anns_by_instance=anns_by_instance,
    )


class NuScenesCrashGraphDataset(torch.utils.data.Dataset):
    """
    Convert nuScenes-Crash metadata scenes into sequential data of (scene_graph, map_idx).
    The output shape matches `NuScenesDataset.get_scene_item`.
    """

    def __init__(
        self,
        meta: NuscCrashMeta,
        map_env: NuScenesMapEnv,
        *,
        npast: int = 4,
        nfuture: int = 12,
        seq_interval: int = 1,
        categories: Sequence[str] = ("car", "truck"),
        reduce_cats: bool = False,
        dt: float = 0.5,
    ) -> None:
        super().__init__()
        self.meta = meta
        self.map_env = map_env
        self.map_list = self.map_env.map_list

        self.dt = float(dt)
        self.npast = int(npast)
        self.nfuture = int(nfuture)
        self.seq_len = self.npast + self.nfuture
        self.seq_interval = int(seq_interval)

        # categories -> one-hot like NuScenesDataset
        all_cats = ["car", "truck", "bus", "motorcycle", "trailer", "cyclist", "pedestrian", "emergency", "construction"]
        all_cat2key = {
            "car": ["vehicle.car"],
            "truck": ["vehicle.truck"],
            "bus": ["vehicle.bus"],
            "motorcycle": ["vehicle.motorcycle"],
            "trailer": ["vehicle.trailer"],
            "cyclist": ["vehicle.bicycle"],
            "pedestrian": ["human.pedestrian"],
            "emergency": ["vehicle.emergency"],
            "construction": ["vehicle.construction"],
        }
        self.categories = list(categories)
        for c in self.categories:
            if c not in all_cats:
                raise ValueError(f"unknown category {c!r}")

        self.key2cat: Dict[str, str] = {}
        for cat in self.categories:
            for k in all_cat2key[cat]:
                self.key2cat[k] = cat

        if reduce_cats:
            reduce_map = {
                "vehicle.car": "car",
                "vehicle.truck": "truck",
                "vehicle.bus": "truck",
                "vehicle.motorcycle": "motorcycle",
                "vehicle.trailer": "truck",
                "vehicle.bicycle": "cyclist",
                "human.pedestrian": "pedestrian",
                "vehicle.emergency": "car",
                "vehicle.construction": "truck",
            }
            self.key2cat = {k: reduce_map[k] for k in self.key2cat.keys()}
            self.categories = sorted(list(set(self.key2cat.values())))

        iden = torch.eye(len(self.categories), dtype=torch.int)
        self.cat2vec = {self.categories[i]: iden[i] for i in range(len(self.categories))}
        self.vec2cat = {tuple(iden[i].tolist()): self.categories[i] for i in range(len(self.categories))}

        # normalizers (same logic as NuScenesDataset)
        ninfo = NUSC_NORM_STATS[tuple(sorted(self.categories))]
        norm_mean = [ninfo["lscale"][0], ninfo["lscale"][0], ninfo["h"][0], ninfo["h"][0], ninfo["s"][0], ninfo["hdot"][0]]
        norm_std = [ninfo["lscale"][1], ninfo["lscale"][1], ninfo["h"][1], ninfo["h"][1], ninfo["s"][1], ninfo["hdot"][1]]
        self.normalizer = MeanStdNormalizer(torch.Tensor(norm_mean), torch.Tensor(norm_std))

        att_norm_mean = [ninfo["l"][0], ninfo["w"][0]]
        att_norm_std = [ninfo["l"][1], ninfo["w"][1]]
        self.veh_att_normalizer = MeanStdNormalizer(torch.Tensor(att_norm_mean), torch.Tensor(att_norm_std))

        # build deterministic seq map: (scene_token, start_idx)
        self.seq_map: List[Tuple[str, int]] = []
        for sc in self.meta.scene:
            sc_tok = sc["token"]
            samples = self.meta.sample_list_by_scene.get(sc_tok, [])
            T = len(samples)
            # Crash dataset scenes are often exactly T = npast + nfuture (e.g., 4+12=16),
            # so we must allow T == seq_len to generate at least the subseq at sidx=0.
            if T < self.seq_len:
                continue
            for sidx in range(0, T - self.seq_len + 1, self.seq_interval):
                self.seq_map.append((sc_tok, int(sidx)))

        self.data_len = len(self.seq_map)

    def __len__(self) -> int:
        return self.data_len

    def get_state_normalizer(self) -> MeanStdNormalizer:
        return self.normalizer

    def get_att_normalizer(self) -> MeanStdNormalizer:
        return self.veh_att_normalizer

    def _scene_to_map(self, scene_token: str) -> Tuple[str, int]:
        sc = self.meta.scene_by_token[scene_token]
        log = self.meta.log_by_token.get(sc.get("log_token", ""), {})
        map_name = log.get("location")
        if not isinstance(map_name, str) or not map_name:
            raise RuntimeError(f"scene missing map location: scene_token={scene_token}")
        if map_name not in self.map_list:
            raise RuntimeError(f"map {map_name!r} not in map_env.map_list={self.map_list}")
        return map_name, self.map_list.index(map_name)

    def _agent_traj_over_scene(
        self,
        samples: Sequence[Dict[str, Any]],
        *,
        instance_token: Optional[str],
        map_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the trajectory and visibility of an agent over the entire scene timeline:
        - traj: (T, 6) = (x, y, hcos, hsin, s, hdot); missing frames are NaN
        - vis: (T,) int in {0, 1}
        """
        T = len(samples)
        xyhh = np.full((T, 4), np.nan, dtype=np.float64)

        if instance_token is None:
            # Ego: align with ego_pose by timestamp
            for t, s in enumerate(samples):
                ts = s.get("timestamp")
                if not isinstance(ts, int):
                    continue
                ep = self.meta.ego_by_ts.get(ts)
                if ep is None:
                    continue
                x, y = float(ep["translation"][0]), float(ep["translation"][1])
                yaw = _yaw_from_quat_wxyz(ep["rotation"])
                xyhh[t] = [x, y, float(np.cos(yaw)), float(np.sin(yaw))]
        else:
            # Other instances: align via sample_annotation by sample_token
            sample_tok_to_t = {s["token"]: i for i, s in enumerate(samples)}
            anns = self.meta.anns_by_instance.get(instance_token, [])
            for a in anns:
                st = a.get("sample_token")
                if not isinstance(st, str) or st not in sample_tok_to_t:
                    continue
                t = int(sample_tok_to_t[st])
                x, y = float(a["translation"][0]), float(a["translation"][1])
                yaw = _yaw_from_quat_wxyz(a["rotation"])
                xyhh[t] = [x, y, float(np.cos(yaw)), float(np.sin(yaw))]

        # Important: the map and trajectories must share the same coordinate system (flip_singapore)
        xyhh = _maybe_flip_singapore_xyhh(xyhh, map_name=map_name, flip_singapore=bool(self.map_env.flip_singapore))

        vis = np.isfinite(xyhh[:, 0]) & np.isfinite(xyhh[:, 1])
        # speed / hdot (handle NaNs via existing helpers)
        t_sec = np.arange(T, dtype=np.float64) * float(self.dt)
        vel = nutils.velocity(xyhh[:, :2], t_sec)  # (T,2)
        s = np.linalg.norm(vel, axis=1)
        h = np.arctan2(xyhh[:, 3], xyhh[:, 2])
        hdot = nutils.heading_change_rate(h, t_sec)

        traj6 = np.concatenate([xyhh, s.reshape((-1, 1)), hdot.reshape((-1, 1))], axis=1)
        traj6[~vis] = np.nan
        return traj6, vis.astype(np.int32)

    def _instance_lw_and_sem(
        self,
        instance_token: str,
        samples: Sequence[Dict[str, Any]],
    ) -> Tuple[np.ndarray, torch.Tensor]:
        # lw: (2,) length,width; sem: one-hot based on category_name prefix
        anns = self.meta.anns_by_instance.get(instance_token, [])
        first = None
        for a in anns:
            if "size" in a and isinstance(a.get("size"), list) and len(a["size"]) == 3:
                first = a
                break
        if first is None:
            # fallback
            lw = np.array([4.8, 2.0], dtype=np.float64)
            sem = self.cat2vec.get("car", torch.tensor([1], dtype=torch.int))
            return lw, sem

        width = float(first["size"][0])
        length = float(first["size"][1])
        lw = np.array([length, width], dtype=np.float64)

        cat_name = first.get("category_name", "vehicle.car")
        key = ".".join(str(cat_name).split(".")[:2])
        high_cat = self.key2cat.get(key, "car")
        sem = self.cat2vec[high_cat]
        return lw, sem

    def get_scene_item(self, scene_token: str, sidx: int) -> Tuple[Graph, int]:
        samples = self.meta.sample_list_by_scene.get(scene_token, [])
        if not samples:
            raise RuntimeError(f"scene has no samples: {scene_token}")

        eidx = int(sidx) + self.seq_len
        midx = int(sidx) + self.npast
        if eidx > len(samples):
            raise IndexError("sequence exceeds scene length")

        map_name, map_idx = self._scene_to_map(scene_token)

        # collect instances present in this scene (deterministic order)
        inst_tokens: List[str] = []
        for s in samples:
            for a in self.meta.anns_by_sample.get(s["token"], []):
                it = a.get("instance_token")
                if isinstance(it, str) and it and it not in inst_tokens:
                    inst_tokens.append(it)
        inst_tokens = sorted(inst_tokens)

        # ego always at row 0
        ego_traj, ego_vis = self._agent_traj_over_scene(samples, instance_token=None, map_name=map_name)
        past = [ego_traj[sidx:midx]]
        future = [ego_traj[midx:eidx]]
        sem = [self.cat2vec["car"].numpy()]
        lw = [np.array([4.8, 2.0], dtype=np.float64)]
        past_vis = [ego_vis[sidx:midx]]
        fut_vis = [ego_vis[midx:eidx]]
        agent_tokens = ["ego"]

        for it in inst_tokens:
            tr, vis = self._agent_traj_over_scene(samples, instance_token=it, map_name=map_name)
            it_lw, it_sem = self._instance_lw_and_sem(it, samples)
            past.append(tr[sidx:midx])
            future.append(tr[midx:eidx])
            sem.append(it_sem.numpy())
            lw.append(it_lw)
            past_vis.append(vis[sidx:midx])
            fut_vis.append(vis[midx:eidx])
            agent_tokens.append(it)

        past_t = torch.tensor(np.stack(past, axis=0), dtype=torch.float32)
        future_t = torch.tensor(np.stack(future, axis=0), dtype=torch.float32)
        sem_t = torch.tensor(np.stack(sem, axis=0), dtype=torch.float32)
        lw_t = torch.tensor(np.stack(lw, axis=0), dtype=torch.float32)
        past_vis_t = torch.tensor(np.stack(past_vis, axis=0), dtype=torch.float32)
        fut_vis_t = torch.tensor(np.stack(fut_vis, axis=0), dtype=torch.float32)

        # normalize like NuScenesDataset
        past_gt = self.normalizer.normalize(past_t)
        past_n = self.normalizer.normalize(past_t)
        future_gt = self.normalizer.normalize(future_t)
        future_n = self.normalizer.normalize(future_t)
        lw_n = self.veh_att_normalizer.normalize(lw_t)

        NA = past_n.size(0)
        if NA > 1:
            node_list = list(range(NA))
            edge_index_list = [(i, j) for i in node_list for j in node_list if i != j]
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)

        graph_prop_dict = {
            "x": torch.empty((NA,)),
            "pos": torch.empty((NA,)),
            "edge_index": edge_index,
            "past": past_n,
            "past_gt": past_gt,
            "future": future_n,
            "future_gt": future_gt,
            "sem": sem_t,
            "lw": lw_n,
            "past_vis": past_vis_t,
            "future_vis": fut_vis_t,
            "agent_tokens": json.dumps(agent_tokens),
            "scene_name": scene_token,
            "sidx": int(sidx),
        }
        return Graph(**graph_prop_dict), int(map_idx)

    def __getitem__(self, idx: int) -> Tuple[Graph, torch.Tensor]:
        scene_token, sidx = self.seq_map[idx]
        g, map_idx = self.get_scene_item(scene_token, sidx)
        return g, torch.tensor([map_idx], dtype=torch.long)


def create_planner(config, planner_type: str, dataset: NuScenesCrashGraphDataset, map_env, device, ckpt: Optional[str] = None):
    """
    Create an ego planner.

    Kept consistent with `src/adv_gen_ddp.py:create_planner` so the same `cfg.planner`
    values behave the same across scripts.
    """
    if planner_type in ("ego", "replay"):
        return None

    if planner_type == "Lane-graph":
        from planners.hardcode_goalcond_nusc import CONFIG_DICT, HardcodeNuscPlanner

        planner_cfg = getattr(config, "planner_cfg", "default")
        assert planner_cfg in CONFIG_DICT
        return HardcodeNuscPlanner(map_env, PlannerConfig(**CONFIG_DICT[planner_cfg]))
    if planner_type == "IDM":
        from planners.hardcode_goalcond_nusc import CONFIG_DICT, IDMPlanner

        planner_cfg = getattr(config, "planner_cfg", "default")
        assert planner_cfg in CONFIG_DICT
        return IDMPlanner(map_env, PlannerConfig(**CONFIG_DICT[planner_cfg]))
    if planner_type == "PDM-Closed":
        from planners.hardcode_goalcond_nusc import CONFIG_DICT, PDMClosedPlanner

        planner_cfg = "final_tuned_val_1"
        assert planner_cfg in CONFIG_DICT
        return PDMClosedPlanner(map_env, PlannerConfig(**CONFIG_DICT[planner_cfg]))
    if planner_type == "IL":
        from planners import IL

        planner = IL(config, dataset, map_env, device, ckpt)
        planner.planner.eval()
        return planner
    if planner_type == "IL-multi":
        from planners import MultiIL

        planner = MultiIL(config, dataset, map_env, device, ckpt)
        planner.planner.eval()
        return planner
    if planner_type == "Diffusion":
        from planners import DiffusionPlanner

        planner = DiffusionPlanner(config, dataset, map_env, device, ckpt=ckpt)
        planner.planner.eval()
        return planner

    throw_err(f"Unknown planner type: {planner_type}")


def parse_cfg():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # i/o
    p.add_argument("--out", type=str, default="./out/crashsim_planner", help="Output directory")
    p.add_argument(
        "--dataroot",
        type=str,
        default="",
        help="nuScenes-Crash root directory (contains v1.0-<version>/ and optional maps/). Required for rollout; optional with --eval_only or --joint.",
    )
    p.add_argument("--version", type=str, default="trainval-crash", help="Version suffix (i.e., v1.0-<version>)")

    # dataset
    p.add_argument("--seq_interval", type=int, default=1)
    p.add_argument("--past_len", type=int, default=4)
    p.add_argument("--future_len", type=int, default=12)
    p.add_argument("--agent_types", type=str, nargs="+", default=["car", "truck"])
    p.add_argument("--reduce_cats", action="store_true", default=False)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_seqs", type=int, default=-1, help="Maximum number of subsequences to process (-1 for all)")

    # map env
    p.add_argument("--map_obs_size_pix", type=int, default=256)
    p.add_argument("--map_obs_bounds", type=float, nargs=4, default=[-17.0, -38.5, 60.0, 38.5])
    p.add_argument("--map_layers", type=str, nargs="+", default=["drivable_area", "carpark_area", "road_divider", "lane_divider"])
    p.add_argument("--pix_per_m", type=int, default=4)
    p.add_argument("--load_lanegraph", action="store_true", default=True)
    p.add_argument("--lanegraph_res_meters", type=float, default=1.0)
    p.add_argument(
        "--map_flip_singapore",
        action="store_true",
        default=False,
        help=(
            "If enabled, mirror the y-axis for singapore-* maps and trajectories "
            "(must match the coordinate convention used during training/data building)."
        ),
    )

    # planner
    p.add_argument("--planner", type=str, default="Lane-graph", choices=["Lane-graph", "IDM", "PDM-Closed", "replay", "IL", "IL-multi", "Diffusion"])
    p.add_argument("--planner_cfg", type=str, default="default", help="Lane-graph/IDM planner config key")
    p.add_argument(
        "--planner_ckpt",
        type=str,
        default="/path/to/planner_model/model.pth",
        help="Checkpoint to load for learned planners (IL, IL-multi, Diffusion). "
             "If empty or missing, learned planners run UNTRAINED.",
    )

    # learned planner model architecture (IL / IL-multi; defaults match utils/config.py)
    p.add_argument("--map_feat_size", type=int, default=64, help="Feature size for map crop encoding.")
    p.add_argument("--past_feat_size", type=int, default=64, help="Feature size for past trajectory encoding.")
    p.add_argument("--future_feat_size", type=int, default=64, help="Feature size for future traj encoding.")
    p.add_argument("--latent_size", type=int, default=32, help="CVAE latent space dim (IL-multi).")
    p.add_argument(
        "--model_output_bicycle", action="store_true", default=True,
        help="Use kinematic bicycle output parameterization for learned planners.",
    )
    p.add_argument(
        "--no_model_output_bicycle", dest="model_output_bicycle", action="store_false",
        help="Predict waypoints directly instead of bicycle parameters.",
    )
    p.add_argument(
        "--conv_kernel_list", type=int, nargs="+", default=[7, 5, 5, 3, 3, 3],
        help="Kernel size for each layer of the map encoder.",
    )
    p.add_argument(
        "--conv_stride_list", type=int, nargs="+", default=[2, 2, 2, 2, 2, 2],
        help="Stride for each layer of the map encoder.",
    )
    p.add_argument(
        "--conv_filter_list", type=int, nargs="+", default=[16, 32, 64, 64, 128, 128],
        help="Num filters output from each layer of the map encoder.",
    )

    # Diffusion planner (minimal DiT + EDM diffusion re-implementation)
    p.add_argument("--diff_dim", type=int, default=128, help="DiT hidden size.")
    p.add_argument("--diff_depth", type=int, default=4, help="Number of DiT blocks.")
    p.add_argument("--diff_heads", type=int, default=4, help="Number of DiT attention heads.")
    p.add_argument("--diff_sample_steps", type=int, default=32, help="EDM Heun sampler steps.")
    p.add_argument("--diff_num_samples", type=int, default=1, help="Diffusion draws averaged per rollout.")
    p.add_argument("--diff_sigma_data", type=float, default=1.0, help="EDM sigma_data (normalized data std).")
    p.add_argument("--diff_sigma_min", type=float, default=0.002, help="EDM sigma_min.")
    p.add_argument("--diff_sigma_max", type=float, default=80.0, help="EDM sigma_max.")
    p.add_argument("--diff_rho", type=float, default=7.0, help="EDM sampling schedule rho.")

    p.add_argument("--viz", action="store_true", default=True)
    p.add_argument(
        "--no_viz", dest="viz", action="store_false",
        help="Disable per-scene rollout visualization outputs.",
    )
    p.add_argument("--viz_video", action="store_true", default=False)

    # Interpretable AV evaluation: deterministic evidence + optional two-stage LLM agents
    p.add_argument(
        "--eval_only", action="store_true", default=False,
        help=(
            "Skip planner rollout and run interpretable evaluation directly on an existing "
            "results.jsonl (default: {out}/results.jsonl, override with --eval_results)."
        ),
    )
    p.add_argument(
        "--eval_results", type=str, default="",
        help="Path to results.jsonl for --eval_only (default: {out}/results.jsonl).",
    )
    p.add_argument(
        "--eval_baseline_results", type=str, default="",
        help="Optional baseline results.jsonl for a paired, scene-matched deterministic comparison.",
    )
    p.add_argument(
        "--eval_report", action="store_true", default=True,
        help="After rollout, generate deterministic evidence and optional LLM diagnosis/guidance.",
    )
    p.add_argument(
        "--no_eval_report", dest="eval_report", action="store_false",
        help="Disable post-rollout interpretable evaluation.",
    )
    p.add_argument(
        "--eval_use_llm", action="store_true", default=False,
        help="Run evidence-grounded LLM failure analysis and planner improvement guidance (requires API base and key).",
    )
    p.add_argument(
        "--no_eval_use_llm", dest="eval_use_llm", action="store_false",
        help="Generate deterministic evaluation only, without LLM diagnosis or improvement guidance.",
    )
    p.add_argument(
        "--eval_llm_only", action="store_true", default=False,
        help=(
            "With --eval_only: skip planner rollout; run LLM analysis using existing "
            "results.jsonl. Reuses interpretable_eval.json when it matches "
            "DETERMINISTIC_EVIDENCE_VERSION; otherwise rebuilds the deterministic prior."
        ),
    )
    p.add_argument(
        "--eval_rebuild_report", action="store_true", default=False,
        help=(
            "Force rebuild of interpretable_eval.json (deterministic population evidence "
            "and capability prior) from results.jsonl before LLM analysis. Useful with "
            "--eval_llm_only after changing prior scoring formulas."
        ),
    )
    p.add_argument(
        "--eval_api_base", type=str,
        default="https://your-api-endpoint/v1",
        help="LLM API base URL (OpenAI-compatible).",
    )
    p.add_argument(
        "--eval_api_key", type=str,
        default="",
        help="LLM api key (eval agent). Can be left empty to disable LLM analysis.",
    )
    p.add_argument(
        "--eval_model", type=str,
        default="gpt-4o",
        help="LLM model name used by the failure-analysis and improvement-guidance agents.",
    )
    p.add_argument("--eval_max_failures_in_prompt", type=int, default=120,
                   help="Optional cap on representative risk cases handed to the LLM "
                   "(0 = no cap; use all stratified selections).")
    p.add_argument("--eval_stratum_cases_per_cell", type=int, default=5,
                   help="Representative cases per failure_mechanism x ODD (speed x density) cell.")
    p.add_argument("--eval_max_odd_cells_in_prompt", type=int, default=30,
                   help="Max ODD subgroup cells handed to the LLM agents.")
    p.add_argument(
        "--eval_llm_temperature", type=float, default=0.25,
        help="LLM sampling temperature (modest value balances reasoning vs prior consistency).",
    )
    p.add_argument("--eval_llm_max_tokens", type=int, default=6000)
    p.add_argument("--eval_timeout", type=float, default=600.0)
    p.add_argument("--eval_min_group_size", type=int, default=5,
                   help="Minimum support before an ODD subgroup is treated as reliable.")
    p.add_argument("--eval_near_miss_clearance_m", type=float, default=1.0,
                   help="SAT box-separation margin threshold for a near miss.")
    p.add_argument("--eval_brake_accel_threshold", type=float, default=-0.3,
                   help="Acceleration threshold (m/s^2) used to detect braking.")
    p.add_argument("--eval_accel_into_conflict_threshold", type=float, default=0.5,
                   help="Acceleration threshold (m/s^2) for acceleration into conflict.")
    p.add_argument("--eval_severe_jerk_mps3", type=float, default=5.0,
                   help="Diagnostic comfort threshold for absolute jerk.")
    p.add_argument("--eval_severe_lat_accel_mps2", type=float, default=3.0,
                   help="Diagnostic comfort threshold for lateral acceleration.")
    p.add_argument("--eval_severe_decel_mps2", type=float, default=-4.0,
                   help="Diagnostic threshold for hard braking.")

    # Cross-dataset (nuScenes × nuCrash) joint analysis
    p.add_argument(
        "--joint",
        action="store_true",
        default=False,
        help=(
            "Pair nuScenes/nuCrash runs by (planner, seed), reuse or rebuild single-dataset "
            "evaluation stages, and write a cross-dataset joint analysis under "
            "comparison/joint/."
        ),
    )
    p.add_argument(
        "--runs_dir",
        type=str,
        default="",
        help="Experiment root containing nuScenes/nuCrash run dirs (required with --joint).",
    )
    p.add_argument(
        "--joint_planner",
        type=str,
        default="",
        help="Optional planner filter for --joint.",
    )
    p.add_argument(
        "--joint_seed",
        type=int,
        default=None,
        help="Optional seed filter for --joint.",
    )
    p.add_argument(
        "--joint_use_llm",
        action="store_true",
        default=False,
        help=(
            "With --joint: call an LLM for a joint synthesis overlay. "
            "Default is deterministic comparison from paired single-dataset outputs."
        ),
    )
    p.add_argument(
        "--joint_regenerate_upstream",
        action="store_true",
        default=False,
        help=(
            "With --joint: when deterministic evidence or single-dataset LLM outputs "
            "are invalid, rebuild them from existing results.jsonl (never reruns valid stages)."
        ),
    )

    # Theory-grounded scientific attribution (GLMM/GEE odds ratios, SHAP, FDR+BCa,
    # extreme value theory, survival analysis). Requires statsmodels/scikit-learn;
    # SHAP and lifelines are optional (graceful fallback/skip).
    p.add_argument("--eval_enable_advanced_stats", dest="eval_enable_advanced_stats",
                   action="store_true", help="Enable theory-grounded scientific attribution layer.")
    p.add_argument("--no_eval_advanced_stats", dest="eval_enable_advanced_stats",
                   action="store_false", help="Disable the scientific attribution layer.")
    p.set_defaults(eval_enable_advanced_stats=True)
    p.add_argument("--eval_stats_min_events", type=int, default=8,
                   help="Minimum collision events required to fit scientific models.")
    p.add_argument("--eval_stats_min_samples", type=int, default=30,
                   help="Minimum interactive sequences required for the scientific layer.")
    p.add_argument("--eval_bootstrap_n", type=int, default=2000,
                   help="Bootstrap resamples for BCa confidence intervals.")
    p.add_argument("--eval_fdr_alpha", type=float, default=0.05,
                   help="Benjamini-Hochberg FDR level for multiple-comparison control.")
    p.add_argument("--eval_evt_quantile", type=float, default=0.90,
                   help="Peaks-over-threshold quantile for the extreme value (GPD) fit.")
    p.add_argument("--eval_evt_min_exceedances", type=int, default=25,
                   help="Minimum tail exceedances required for the EVT fit.")
    p.add_argument("--eval_shap_max_background", type=int, default=200,
                   help="Max background samples for SHAP TreeExplainer.")
    p.add_argument("--eval_stats_cv_folds", type=int, default=5,
                   help="Cross-validation folds for surrogate-model AUC.")
    p.add_argument("--eval_stats_seed", type=int, default=0,
                   help="Random seed for the scientific attribution layer.")
    # viz params (aesthetics / framing)
    p.add_argument(
        "--viz_bounds",
        type=float,
        nargs=4,
        default=[-60.0, -60.0, 60.0, 60.0],
        help="Visualization crop bounds in meters: [xmin, ymin, xmax, ymax], applied in ego-crop coordinates",
    )
    p.add_argument(
        "--viz_crop_t",
        type=int,
        default=-1,
        help=(
            "Which timestep's ego pose to use as the crop center. -1 means the last past frame; "
            "0..(past_len+future_len-1) selects an explicit timestep."
        ),
    )

    args = p.parse_args()
    cfg_dict = vars(args)
    return dict2obj(cfg_dict), cfg_dict


def _parse_agent_tokens(scene_graph: Graph) -> Optional[List[str]]:
    """Parse the ``agent_tokens`` attribute attached to a scene graph.

    ``torch_geometric`` batching may turn non-tensor attrs into a Python list of
    length ``B``; this script runs with ``batch_size=1`` but the parsing is
    kept robust to either form.
    """
    raw = getattr(scene_graph, "agent_tokens", None)
    if isinstance(raw, (list, tuple)) and len(raw) == 1:
        raw = raw[0]
    if isinstance(raw, str):
        try:
            tokens = json.loads(raw)
        except Exception:
            return None
    elif isinstance(raw, list):
        tokens = raw
    else:
        return None
    if not isinstance(tokens, list):
        return None
    return [str(t) for t in tokens]


def _detect_adv_agent_meta(scene_graph: Graph, meta: NuscCrashMeta) -> Dict[str, Any]:
    """
    Locate the adversarial/attacking agent in the current graph and return its
    index, instance token and ``behavior_tag`` (when available).

    Convention: ``build_nusc_crash.py`` writes ``external_adv_token`` (and a
    ``behavior_tag`` next to it) to the attacker ``instance.json`` record,
    plus a ``behavior_tag`` on the corresponding ``scene.json`` row. These
    fields are unknown to the official devkit so they do not affect loading.
    """
    out: Dict[str, Any] = {"adv_idx": None, "adv_token": None, "behavior_tag": None}
    tokens = _parse_agent_tokens(scene_graph)
    if not tokens:
        return out
    for idx, tok in enumerate(tokens):
        if idx == 0 or not tok:
            continue
        inst = meta.inst_by_token.get(tok)
        if not isinstance(inst, dict):
            continue
        ext = inst.get("external_adv_token")
        if isinstance(ext, str) and ext:
            out["adv_idx"] = int(idx)
            out["adv_token"] = tok
            tag = inst.get("behavior_tag")
            if isinstance(tag, str) and tag.strip():
                out["behavior_tag"] = tag.strip()
            break

    if out["behavior_tag"] is None:
        # Fall back to the scene-level behavior_tag (also written by build_nusc_crash.py).
        scene_tok = getattr(scene_graph, "scene_name", None)
        if isinstance(scene_tok, (list, tuple)) and len(scene_tok) == 1:
            scene_tok = scene_tok[0]
        if isinstance(scene_tok, str):
            sc = meta.scene_by_token.get(scene_tok)
            if isinstance(sc, dict):
                tag = sc.get("behavior_tag")
                if isinstance(tag, str) and tag.strip():
                    out["behavior_tag"] = tag.strip()
    return out


def _detect_adv_agent_index(scene_graph: Graph, meta: NuscCrashMeta) -> Optional[int]:
    """Backward-compatible thin wrapper around :func:`_detect_adv_agent_meta`."""
    info = _detect_adv_agent_meta(scene_graph, meta)
    idx = info.get("adv_idx")
    if isinstance(idx, int):
        return idx
    return None


def _extract_scene_odd_meta(scene_token: str, meta: NuscCrashMeta) -> Dict[str, Any]:
    """Extract ODD (Operational Design Domain) metadata for a scene."""
    out: Dict[str, Any] = {
        "odd_location": None,
        "odd_region": None,
        "odd_vehicle": None,
        "odd_date_captured": None,
    }
    sc = meta.scene_by_token.get(scene_token)
    if not isinstance(sc, dict):
        return out
    log = meta.log_by_token.get(sc.get("log_token", ""), {})
    if not isinstance(log, dict):
        return out
    location = log.get("location")
    if isinstance(location, str) and location.strip():
        out["odd_location"] = location.strip()
        if location.startswith("boston-"):
            out["odd_region"] = "boston"
        elif location.startswith("singapore-"):
            out["odd_region"] = "singapore"
    vehicle = log.get("vehicle")
    if isinstance(vehicle, str) and vehicle.strip():
        out["odd_vehicle"] = vehicle.strip()
    date_captured = log.get("date_captured")
    if isinstance(date_captured, str) and date_captured.strip():
        out["odd_date_captured"] = date_captured.strip()
    return out


def _parse_location_from_scene_name(scene_name: Optional[str]) -> Optional[str]:
    """Fallback: parse map location from crash scene name."""
    if not isinstance(scene_name, str) or not scene_name.strip():
        return None
    m = re.match(r"crash-\d+-(.+?)_val_", scene_name.strip())
    return m.group(1) if m else None


def _odd_region_from_location(location: Optional[str]) -> Optional[str]:
    if not isinstance(location, str) or not location.strip():
        return None
    if location.startswith("boston-"):
        return "boston"
    if location.startswith("singapore-"):
        return "singapore"
    return None


def _odd_speed_bin(speed_mps: Any) -> Optional[str]:
    if not isinstance(speed_mps, (int, float)) or not math.isfinite(float(speed_mps)):
        return None
    v = float(speed_mps)
    if v < 3.0:
        return "low (<3 m/s)"
    if v < 8.0:
        return "medium (3–8 m/s)"
    return "high (≥8 m/s)"


def _odd_density_bin(na: Any) -> Optional[str]:
    if not isinstance(na, (int, float)):
        return None
    n = int(na)
    if n <= 3:
        return "sparse (≤3 agents)"
    if n <= 6:
        return "medium (4–6 agents)"
    return "dense (≥7 agents)"


def _enrich_record_odd_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Fill ODD bins on a result record (supports legacy JSONL without odd_* keys)."""
    if not record.get("odd_location"):
        loc = _parse_location_from_scene_name(record.get("scene_name"))
        if loc:
            record["odd_location"] = loc
            record["odd_region"] = _odd_region_from_location(loc)
    if not record.get("odd_speed_bin"):
        record["odd_speed_bin"] = _odd_speed_bin(record.get("ego_avg_speed_mps"))
    if not record.get("odd_density_bin"):
        record["odd_density_bin"] = _odd_density_bin(record.get("NA"))
    return record


def _eval_fmt(value: Any, fmt: str = ".4f") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    if fmt.endswith("%"):
        return format(float(value), fmt)
    return format(float(value), fmt)


def _eval_group_stats(group_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(group_records)
    n_with_other = sum(1 for r in group_records if bool(r.get("has_other_agents", True)))
    n_coll = sum(1 for r in group_records if bool(r.get("did_coll", False)))
    coll_rate = (n_coll / n_with_other) if n_with_other > 0 else None
    return {"n": n, "n_with_other": n_with_other, "n_coll": n_coll, "coll_rate": coll_rate}


def compute_odd_analysis(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate collision stats by ODD dimensions (location, region, speed, density)."""
    enriched = [_enrich_record_odd_fields(dict(r)) for r in records]
    coll_records = [r for r in enriched if bool(r.get("did_coll", False))]
    n_coll = len(coll_records)

    def _by_dimension(dim_key: str) -> List[Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in enriched:
            val = r.get(dim_key)
            label = str(val).strip() if val is not None and str(val).strip() else "unknown"
            groups[label].append(r)
        rows: List[Dict[str, Any]] = []
        for label, rs in groups.items():
            gs = _eval_group_stats(rs)
            share = (gs["n_coll"] / n_coll) if n_coll > 0 else None
            rows.append({"dimension": dim_key, "value": label, "failure_share": share, **gs})
        rows.sort(
            key=lambda x: (
                x.get("coll_rate") if isinstance(x.get("coll_rate"), (int, float)) else -1.0,
                x.get("n_coll") or 0,
            ),
            reverse=True,
        )
        return rows

    dimensions = {
        "odd_location": _by_dimension("odd_location"),
        "odd_region": _by_dimension("odd_region"),
        "odd_speed_bin": _by_dimension("odd_speed_bin"),
        "odd_density_bin": _by_dimension("odd_density_bin"),
    }

    # Cross-tab: behavior × location for collision cases.
    behavior_location: List[Dict[str, Any]] = []
    cross: Dict[Tuple[str, str], int] = Counter()
    for r in coll_records:
        tag = r.get("behavior_tag") or "unknown"
        loc = r.get("odd_location") or "unknown"
        cross[(str(tag), str(loc))] += 1
    for (tag, loc), cnt in cross.most_common(12):
        behavior_location.append(
            {
                "behavior_tag": tag,
                "odd_location": loc,
                "n_coll": cnt,
                "failure_share": (cnt / n_coll) if n_coll > 0 else None,
            }
        )

    # Coverage summary.
    coverage = {
        dim: len({r.get(dim) for r in enriched if r.get(dim) is not None})
        for dim in ("odd_location", "odd_region", "odd_speed_bin", "odd_density_bin")
    }

    return {
        "n_total": len(enriched),
        "n_coll": n_coll,
        "dimensions": dimensions,
        "behavior_by_location": behavior_location,
        "coverage": coverage,
    }


def compute_failure_profile_summary(
    records: List[Dict[str, Any]],
    attribution: Dict[str, Any],
) -> Dict[str, Any]:
    """Deterministic failure-profile synthesis from rollout metrics + attribution."""
    enriched = [_enrich_record_odd_fields(dict(r)) for r in records]
    coll_records = [r for r in enriched if bool(r.get("did_coll", False))]
    safe_records = [
        r
        for r in enriched
        if bool(r.get("has_other_agents", True)) and not bool(r.get("did_coll", False))
    ]
    n_with_other = sum(1 for r in enriched if bool(r.get("has_other_agents", True)))
    n_coll = len(coll_records)
    overall_coll_rate = (n_coll / n_with_other) if n_with_other > 0 else None

    behavior_risk = attribution.get("behavior_risk") or []
    top_behaviors = behavior_risk[:5]
    feature_sep = attribution.get("feature_separation") or []
    top_features = feature_sep[:5]
    timing = attribution.get("collision_timing") or {}
    fb = attribution.get("feature_based_attribution") or {}

    min_dist_coll = [
        float(r["min_dist_m"])
        for r in coll_records
        if isinstance(r.get("min_dist_m"), (int, float)) and math.isfinite(float(r["min_dist_m"]))
    ]
    min_dist_safe = [
        float(r["min_dist_m"])
        for r in safe_records
        if isinstance(r.get("min_dist_m"), (int, float)) and math.isfinite(float(r["min_dist_m"]))
    ]
    mean_min_coll = sum(min_dist_coll) / len(min_dist_coll) if min_dist_coll else None
    mean_min_safe = sum(min_dist_safe) / len(min_dist_safe) if min_dist_safe else None

    ttc_vals = [
        float(r["ttc_sec"])
        for r in coll_records
        if isinstance(r.get("ttc_sec"), (int, float)) and math.isfinite(float(r["ttc_sec"]))
    ]
    median_ttc = sorted(ttc_vals)[len(ttc_vals) // 2] if ttc_vals else timing.get("median_ttc_sec")

    cp = fb.get("collision_control_pattern") or {}
    roots = fb.get("root_hypotheses") or []

    # Template narrative (deterministic).
    narrative_parts: List[str] = []
    if overall_coll_rate is not None:
        narrative_parts.append(
            f"Overall collision rate is {overall_coll_rate:.1%} ({n_coll}/{n_with_other} scenes with other agents)."
        )
    if top_behaviors:
        lead = top_behaviors[0]
        tag = lead.get("behavior_tag", "?")
        cr = lead.get("coll_rate")
        fs = lead.get("failure_share")
        if cr is not None and fs is not None:
            narrative_parts.append(
                f"The dominant failure mode is '{tag}' (coll_rate={cr:.1%}, "
                f"failure_share={fs:.1%} of all collisions)."
            )
    if timing.get("mid_share") is not None:
        narrative_parts.append(
            f"Collisions concentrate in the mid-to-late rollout window "
            f"(early={timing.get('early_share', 0):.1%}, mid={timing.get('mid_share', 0):.1%}, "
            f"late={timing.get('late_share', 0):.1%}); median TTC≈{_eval_fmt(median_ttc, '.2f')} s."
        )
    if mean_min_coll is not None and mean_min_safe is not None:
        narrative_parts.append(
            f"Collision cases show much closer proximity at min distance "
            f"(mean {mean_min_coll:.2f} m vs {mean_min_safe:.2f} m in safe cases)."
        )
    brake_rate = cp.get("brake_before_collision_rate")
    accel_rate = cp.get("accelerate_into_collision_rate")
    if brake_rate is not None or accel_rate is not None:
        narrative_parts.append(
            f"Pre-collision control: brake-before-collision={_eval_fmt(brake_rate, '.1%')}, "
            f"accelerate-into-collision={_eval_fmt(accel_rate, '.1%')}."
        )

    return {
        "overall_coll_rate": overall_coll_rate,
        "n_coll": n_coll,
        "n_with_other": n_with_other,
        "top_behaviors": top_behaviors,
        "top_features": top_features,
        "collision_timing": timing,
        "collision_control_pattern": cp,
        "root_hypotheses": roots,
        "mean_min_dist_collided": mean_min_coll,
        "mean_min_dist_safe": mean_min_safe,
        "median_ttc_sec": median_ttc,
        "narrative": " ".join(narrative_parts),
    }


def render_failure_profile_section(summary: Dict[str, Any]) -> str:
    """Render deterministic failure-profile markdown."""
    if not summary:
        return ""
    lines = ["## Failure profile (data-driven)", ""]
    narrative = summary.get("narrative")
    if isinstance(narrative, str) and narrative.strip():
        lines.append(narrative.strip())
        lines.append("")

    top_behaviors = summary.get("top_behaviors") or []
    if top_behaviors:
        lines.append("### Top failure behaviors")
        lines.append("")
        lines.append("| rank | behavior_tag | n | n_coll | coll_rate | failure_share |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for i, item in enumerate(top_behaviors, start=1):
            lines.append(
                "| {r} | {tag} | {n} | {nc} | {cr} | {fs} |".format(
                    r=i,
                    tag=item.get("behavior_tag", "?"),
                    n=item.get("n", 0),
                    nc=item.get("n_coll", 0),
                    cr=_eval_fmt(item.get("coll_rate"), ".4f"),
                    fs=_eval_fmt(item.get("failure_share"), ".4f"),
                )
            )
        lines.append("")

    top_features = summary.get("top_features") or []
    if top_features:
        lines.append("### Key discriminative features (collided vs safe)")
        lines.append("")
        lines.append("| feature | mean_coll | mean_safe | delta | cohen's d | effect |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for item in top_features:
            lines.append(
                "| {feat} | {mc} | {ms} | {d} | {cd} | {eff} |".format(
                    feat=item.get("feature", "?"),
                    mc=_eval_fmt(item.get("mean_coll"), ".3f"),
                    ms=_eval_fmt(item.get("mean_no_coll"), ".3f"),
                    d=_eval_fmt(item.get("delta"), ".3f"),
                    cd=_eval_fmt(item.get("cohens_d"), ".3f"),
                    eff=item.get("effect_label", "n/a"),
                )
            )
        lines.append("")

    roots = summary.get("root_hypotheses") or []
    if roots:
        lines.append("### Root hypotheses")
        lines.append("")
        for h in roots:
            if isinstance(h, str) and h.strip():
                lines.append(f"- {h.strip()}")
        lines.append("")

    timing = summary.get("collision_timing") or {}
    if timing.get("n_coll_with_step"):
        lines.append("### Collision timing")
        lines.append("")
        lines.append(
            f"- Distribution: early={_eval_fmt(timing.get('early_share'), '.2%')}, "
            f"mid={_eval_fmt(timing.get('mid_share'), '.2%')}, "
            f"late={_eval_fmt(timing.get('late_share'), '.2%')}"
        )
        lines.append(
            f"- Median coll_step={_eval_fmt(timing.get('median_coll_step'), '.1f')}, "
            f"median TTC={_eval_fmt(timing.get('median_ttc_sec'), '.2f')} s"
        )
        lines.append("")

    return "\n".join(lines)


def render_odd_section(odd_analysis: Dict[str, Any]) -> str:
    """Render ODD (Operational Design Domain) analysis markdown."""
    if not odd_analysis:
        return ""
    lines = ["## ODD analysis (Operational Design Domain)", ""]
    lines.append(
        "Breakdown by map location, geographic region, ego speed bin, and agent-density bin. "
        "`failure_share` = fraction of all collision cases contributed by that ODD cell."
    )
    lines.append("")

    dimensions = odd_analysis.get("dimensions") or {}
    dim_titles = {
        "odd_location": "Map location",
        "odd_region": "Geographic region",
        "odd_speed_bin": "Ego speed bin (avg rollout speed)",
        "odd_density_bin": "Agent density bin (NA)",
    }
    for dim_key, title in dim_titles.items():
        rows = dimensions.get(dim_key) or []
        if not rows:
            continue
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| value | n | n_coll | coll_rate | 95% CI | failure_share | support |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in rows:
            lines.append(
                "| {val} | {n} | {nc} | {cr} | {ci} | {fs} | {support} |".format(
                    val=row.get("value", "?"),
                    n=row.get("n", 0),
                    nc=row.get("n_coll", 0),
                    cr=_eval_fmt(row.get("coll_rate"), ".4f"),
                    ci=(
                        f"{_eval_fmt(row.get('ci95_low'), '.2%')}–{_eval_fmt(row.get('ci95_high'), '.2%')}"
                        if row.get("ci95_low") is not None else "n/a"
                    ),
                    fs=_eval_fmt(row.get("failure_share"), ".4f"),
                    support=("low" if row.get("low_support") is True else "adequate" if row.get("low_support") is False else "n/a"),
                )
            )
        lines.append("")

    bl = odd_analysis.get("behavior_by_location") or []
    if bl:
        lines.append("### Collision hotspots: behavior × location")
        lines.append("")
        lines.append("| behavior_tag | odd_location | n_coll | failure_share |")
        lines.append("|---|---|---:|---:|")
        for row in bl:
            lines.append(
                "| {tag} | {loc} | {nc} | {fs} |".format(
                    tag=row.get("behavior_tag", "?"),
                    loc=row.get("odd_location", "?"),
                    nc=row.get("n_coll", 0),
                    fs=_eval_fmt(row.get("failure_share"), ".4f"),
                )
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-sample planner metrics
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Interpretable evaluation layer (deterministic evidence + optional LLM)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InterpretabilityThresholds:
    """Transparent thresholds used by the deterministic interpretation layer.

    These thresholds do not redefine the underlying planner metrics. They only
    turn continuous evidence into readable diagnostic flags and scorecards.
    Every threshold is exposed through the CLI and written into the report.
    """

    near_miss_clearance_m: float = 1.0
    brake_accel_threshold: float = -0.3
    accel_into_conflict_threshold: float = 0.5
    severe_jerk_mps3: float = 5.0
    severe_lat_accel_mps2: float = 3.0
    severe_decel_mps2: float = -4.0
    min_group_size: int = 5
    stratum_cases_per_cell: int = 5
    max_failures_in_prompt: int = 120

    @classmethod
    def from_cfg(cls, cfg: Any) -> "InterpretabilityThresholds":
        return cls(
            near_miss_clearance_m=float(getattr(cfg, "eval_near_miss_clearance_m", 1.0)),
            brake_accel_threshold=float(getattr(cfg, "eval_brake_accel_threshold", -0.3)),
            accel_into_conflict_threshold=float(getattr(cfg, "eval_accel_into_conflict_threshold", 0.5)),
            severe_jerk_mps3=float(getattr(cfg, "eval_severe_jerk_mps3", 5.0)),
            severe_lat_accel_mps2=float(getattr(cfg, "eval_severe_lat_accel_mps2", 3.0)),
            severe_decel_mps2=float(getattr(cfg, "eval_severe_decel_mps2", -4.0)),
            min_group_size=max(1, int(getattr(cfg, "eval_min_group_size", 5))),
            stratum_cases_per_cell=max(1, int(getattr(cfg, "eval_stratum_cases_per_cell", 5))),
            max_failures_in_prompt=max(0, int(getattr(cfg, "eval_max_failures_in_prompt", 120))),
        )


def _as_finite_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
        if math.isfinite(out):
            return out
    return None


def _finite_metric_values(records: Sequence[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for record in records:
        value = _as_finite_float(record.get(key))
        if value is not None:
            values.append(value)
    return values


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _median_or_none(values: Sequence[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    """Wilson score interval for a binomial rate without scipy."""
    if n <= 0:
        return None, None
    p = float(k) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _rate_with_ci(k: int, n: int) -> Dict[str, Any]:
    low, high = _wilson_interval(k, n)
    return {
        "count": int(k),
        "denominator": int(n),
        "rate": (float(k) / float(n)) if n > 0 else None,
        "ci95_low": low,
        "ci95_high": high,
    }


def _cohens_d(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    pooled_num = (len(av) - 1) * float(np.var(av, ddof=1)) + (len(bv) - 1) * float(np.var(bv, ddof=1))
    pooled_den = len(av) + len(bv) - 2
    if pooled_den <= 0:
        return None
    pooled = math.sqrt(max(0.0, pooled_num / pooled_den))
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(av) - np.mean(bv)) / pooled)


def _effect_label(d: Optional[float]) -> str:
    if d is None:
        return "insufficient support"
    a = abs(float(d))
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def _scene_key(record: Dict[str, Any]) -> Tuple[str, int]:
    scene = str(record.get("scene_token") or record.get("scene_name") or "")
    sidx = record.get("sidx")
    try:
        sidx_int = int(sidx) if sidx is not None else -1
    except (TypeError, ValueError):
        sidx_int = -1
    return scene, sidx_int


def _infer_conflict_geometry(record: Dict[str, Any]) -> str:
    rel_heading = _as_finite_float(record.get("rel_heading_at_min_dist"))
    long_gap = _as_finite_float(record.get("rel_long_gap_at_min_dist"))
    lat_gap = _as_finite_float(record.get("rel_lat_gap_at_min_dist"))
    if rel_heading is not None and abs(rel_heading) >= math.radians(45.0):
        return "crossing_or_opposing"
    if long_gap is not None and lat_gap is not None:
        if abs(lat_gap) > abs(long_gap):
            return "lateral_side_conflict"
        return "front_longitudinal" if long_gap >= 0.0 else "rear_longitudinal"
    return "unresolved_geometry"


def _infer_failure_mechanism(record: Dict[str, Any], thresholds: InterpretabilityThresholds) -> str:
    if not bool(record.get("has_other_agents", True)):
        return "no_interacting_agent"
    did_coll = bool(record.get("did_coll", False))
    near_miss = bool(record.get("did_near_crash", False))
    accel_into = bool(record.get("accelerate-into-collision", False))
    brake_before = record.get("brake-before-collision")
    ttc = _as_finite_float(record.get("ttc_sec"))
    impact_speed = _as_finite_float(record.get("relative_speed_at_collision_mps"))
    geometry = _infer_conflict_geometry(record)

    if not did_coll:
        if near_miss:
            return "near_miss_recovery"
        return "safe_resolution"
    if accel_into:
        return "acceleration_into_conflict"
    if brake_before is False and (ttc is None or ttc <= 3.0):
        return "late_or_absent_braking"
    if brake_before is True and impact_speed is not None and impact_speed >= 3.0:
        return "insufficient_collision_mitigation"
    if geometry == "crossing_or_opposing":
        return "crossing_conflict_handling"
    if geometry == "lateral_side_conflict":
        return "lateral_conflict_handling"
    if geometry in ("front_longitudinal", "rear_longitudinal"):
        return "longitudinal_gap_management"
    return "unresolved_collision_mechanism"


def _linear_score(value: Optional[float], good: float, bad: float, higher_is_better: bool) -> Optional[float]:
    if value is None:
        return None
    if math.isclose(good, bad):
        return None
    if higher_is_better:
        raw = (value - bad) / (good - bad)
    else:
        raw = (bad - value) / (bad - good)
    return float(100.0 * min(1.0, max(0.0, raw)))


def _diagnostic_scorecard(record: Dict[str, Any], thresholds: InterpretabilityThresholds) -> Dict[str, Any]:
    """Create transparent, non-normative diagnostic scores in [0, 100]."""
    did_coll = bool(record.get("did_coll", False))
    near_miss = bool(record.get("did_near_crash", False))
    clearance = _as_finite_float(record.get("min_clearance_proxy_m"))
    if did_coll:
        safety = 0.0
    elif near_miss:
        denom = max(1e-6, thresholds.near_miss_clearance_m)
        safety = 40.0 + 20.0 * min(1.0, max(0.0, (clearance or 0.0) / denom))
    else:
        safety = 100.0

    jerk = _as_finite_float(record.get("ego_max_abs_jerk"))
    lat_acc = _as_finite_float(record.get("ego_lat_accel_max"))
    decel = _as_finite_float(record.get("ego_max_decel"))
    jerk_score = _linear_score(jerk, good=0.0, bad=max(1e-6, thresholds.severe_jerk_mps3), higher_is_better=False)
    lat_score = _linear_score(lat_acc, good=0.0, bad=max(1e-6, thresholds.severe_lat_accel_mps2), higher_is_better=False)
    decel_mag = abs(min(0.0, decel)) if decel is not None else None
    decel_score = _linear_score(
        decel_mag,
        good=0.0,
        bad=max(1e-6, abs(thresholds.severe_decel_mps2)),
        higher_is_better=False,
    )
    comfort_parts = [x for x in (jerk_score, lat_score, decel_score) if x is not None]
    comfort = _mean_or_none(comfort_parts)

    progress = _as_finite_float(record.get("ego_progress_ratio"))
    path_eff = _as_finite_float(record.get("ego_path_efficiency"))
    progress_score = _linear_score(progress, good=1.0, bad=0.0, higher_is_better=True)
    path_score = _linear_score(path_eff, good=1.0, bad=0.5, higher_is_better=True)
    efficiency_parts = [x for x in (progress_score, path_score) if x is not None]
    efficiency = _mean_or_none(efficiency_parts)

    available = [(safety, 0.60)]
    if comfort is not None:
        available.append((comfort, 0.20))
    if efficiency is not None:
        available.append((efficiency, 0.20))
    weight_sum = sum(w for _, w in available)
    overall = sum(v * w for v, w in available) / weight_sum if weight_sum > 0 else None
    return {
        "safety": safety,
        "comfort": comfort,
        "efficiency": efficiency,
        "overall": overall,
        "weights": {"safety": 0.60, "comfort": 0.20, "efficiency": 0.20},
        "interpretation": "diagnostic only; raw metrics remain authoritative",
    }


def _build_scene_evidence(record: Dict[str, Any], thresholds: InterpretabilityThresholds) -> List[str]:
    evidence: List[str] = []
    if bool(record.get("did_coll", False)):
        ttc = _as_finite_float(record.get("ttc_sec"))
        impact = _as_finite_float(record.get("relative_speed_at_collision_mps"))
        text = "Collision occurred"
        if ttc is not None:
            text += f" at {ttc:.2f} s"
        if impact is not None:
            text += f" with relative-speed proxy {impact:.2f} m/s"
        evidence.append(text + ".")
    elif bool(record.get("did_near_crash", False)):
        clearance = _as_finite_float(record.get("min_clearance_proxy_m"))
        evidence.append(
            "Near miss detected"
            + (f" with SAT separation margin {clearance:.2f} m." if clearance is not None else ".")
        )
    else:
        evidence.append("No collision or threshold-defined near miss was observed.")

    evidence.append(f"Conflict geometry: {_infer_conflict_geometry(record)}.")
    if record.get("brake-before-collision") is not None:
        evidence.append(f"Brake before collision: {bool(record.get('brake-before-collision'))}.")
    if record.get("accelerate-into-collision") is not None:
        evidence.append(f"Acceleration into conflict: {bool(record.get('accelerate-into-collision'))}.")
    jerk = _as_finite_float(record.get("ego_max_abs_jerk"))
    if jerk is not None and jerk >= thresholds.severe_jerk_mps3:
        evidence.append(f"Comfort flag: max absolute jerk {jerk:.2f} m/s^3 exceeds {thresholds.severe_jerk_mps3:.2f}.")
    lat_acc = _as_finite_float(record.get("ego_lat_accel_max"))
    if lat_acc is not None and lat_acc >= thresholds.severe_lat_accel_mps2:
        evidence.append(f"Comfort flag: lateral acceleration {lat_acc:.2f} m/s^2 exceeds {thresholds.severe_lat_accel_mps2:.2f}.")
    return evidence


def _enrich_interpretability_record(
    record: Dict[str, Any], thresholds: InterpretabilityThresholds
) -> Dict[str, Any]:
    _enrich_record_odd_fields(record)
    if "brake_before_collision" not in record:
        record["brake_before_collision"] = record.get("brake-before-collision")
    if "accelerate_into_collision" not in record:
        record["accelerate_into_collision"] = record.get("accelerate-into-collision")
    record["conflict_geometry"] = _infer_conflict_geometry(record)
    record["failure_mechanism"] = _infer_failure_mechanism(record, thresholds)
    if bool(record.get("did_coll", False)):
        record["outcome_class"] = "collision"
    elif bool(record.get("did_near_crash", False)):
        record["outcome_class"] = "near_miss"
    elif bool(record.get("has_other_agents", True)):
        record["outcome_class"] = "safe_interaction"
    else:
        record["outcome_class"] = "no_interaction"
    record["diagnostic_scorecard"] = _diagnostic_scorecard(record, thresholds)
    record["evidence"] = _build_scene_evidence(record, thresholds)
    return record


def compute_deterministic_attribution(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    interactive = [r for r in records if bool(r.get("has_other_agents", True))]
    collided = [r for r in interactive if bool(r.get("did_coll", False))]
    safe = [r for r in interactive if not bool(r.get("did_coll", False))]
    n_coll = len(collided)

    behavior_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in interactive:
        behavior_groups[str(r.get("behavior_tag") or "unknown")].append(r)
    behavior_risk: List[Dict[str, Any]] = []
    for tag, group in behavior_groups.items():
        k = sum(bool(r.get("did_coll", False)) for r in group)
        rate = _rate_with_ci(k, len(group))
        behavior_risk.append(
            {
                "behavior_tag": tag,
                "n": len(group),
                "n_coll": k,
                "coll_rate": rate["rate"],
                "ci95_low": rate["ci95_low"],
                "ci95_high": rate["ci95_high"],
                "failure_share": (k / n_coll) if n_coll > 0 else None,
            }
        )
    behavior_risk.sort(key=lambda x: ((x.get("coll_rate") or 0.0), x.get("n_coll") or 0), reverse=True)

    feature_keys = [
        "min_clearance_proxy_m",
        "min_dist_m",
        "rel_speed_at_min_dist",
        "ego_avg_speed_mps",
        "ego_max_decel",
        "ego_max_abs_jerk",
        "ego_lat_accel_max",
        "ego_progress_ratio",
        "ego_path_efficiency",
        "ego_ade_m",
    ]
    feature_separation: List[Dict[str, Any]] = []
    for key in feature_keys:
        a = _finite_metric_values(collided, key)
        b = _finite_metric_values(safe, key)
        d = _cohens_d(a, b)
        if not a or not b:
            continue
        feature_separation.append(
            {
                "feature": key,
                "n_coll": len(a),
                "n_no_coll": len(b),
                "mean_coll": _mean_or_none(a),
                "mean_no_coll": _mean_or_none(b),
                "delta": (_mean_or_none(a) - _mean_or_none(b)) if a and b else None,
                "cohens_d": d,
                "effect_label": _effect_label(d),
            }
        )
    feature_separation.sort(key=lambda x: abs(x.get("cohens_d") or 0.0), reverse=True)

    timing_records = [r for r in collided if _as_finite_float(r.get("coll_step")) is not None]
    timing_bins = Counter()
    normalized_steps: List[float] = []
    for r in timing_records:
        step = float(r["coll_step"])
        ft = max(1.0, float(r.get("FT") or 1.0))
        frac = min(1.0, max(0.0, step / ft))
        normalized_steps.append(frac)
        timing_bins["early" if frac < 1.0 / 3.0 else "mid" if frac < 2.0 / 3.0 else "late"] += 1
    n_timing = len(timing_records)
    ttc_values = _finite_metric_values(collided, "ttc_sec")
    collision_timing = {
        "n_coll_with_step": n_timing,
        "early_share": timing_bins["early"] / n_timing if n_timing else None,
        "mid_share": timing_bins["mid"] / n_timing if n_timing else None,
        "late_share": timing_bins["late"] / n_timing if n_timing else None,
        "median_coll_step": _median_or_none([float(r["coll_step"]) for r in timing_records]),
        "median_ttc_sec": _median_or_none(ttc_values),
        "median_normalized_collision_time": _median_or_none(normalized_steps),
    }

    brake_known = [r for r in collided if r.get("brake-before-collision") is not None]
    accel_known = [r for r in collided if r.get("accelerate-into-collision") is not None]
    control_pattern = {
        "n_brake_known": len(brake_known),
        "brake_before_collision_rate": (
            sum(bool(r.get("brake-before-collision")) for r in brake_known) / len(brake_known)
            if brake_known else None
        ),
        "n_accel_known": len(accel_known),
        "accelerate_into_collision_rate": (
            sum(bool(r.get("accelerate-into-collision")) for r in accel_known) / len(accel_known)
            if accel_known else None
        ),
    }

    mechanism_counts = Counter(str(r.get("failure_mechanism") or "unknown") for r in collided)
    root_hypotheses: List[str] = []
    if mechanism_counts:
        for mechanism, count in mechanism_counts.most_common(3):
            share = count / max(1, n_coll)
            root_hypotheses.append(
                f"{mechanism} accounts for {count}/{n_coll} collision cases ({share:.1%}); inspect the linked scene evidence before treating this as causal."
            )
    return {
        "behavior_risk": behavior_risk,
        "feature_separation": feature_separation,
        "collision_timing": collision_timing,
        "feature_based_attribution": {
            "collision_control_pattern": control_pattern,
            "mechanism_counts": dict(mechanism_counts),
            "root_hypotheses": root_hypotheses,
        },
    }


def _augment_odd_uncertainty(odd_analysis: Dict[str, Any], thresholds: InterpretabilityThresholds) -> None:
    for rows in (odd_analysis.get("dimensions") or {}).values():
        for row in rows:
            low, high = _wilson_interval(int(row.get("n_coll") or 0), int(row.get("n_with_other") or 0))
            row["ci95_low"] = low
            row["ci95_high"] = high
            row["low_support"] = int(row.get("n_with_other") or 0) < thresholds.min_group_size


def compute_data_quality(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "did_coll", "min_dist_m", "min_clearance_proxy_m", "ego_avg_speed_mps",
        "ego_max_abs_jerk", "ego_progress_ratio", "behavior_tag", "odd_location",
    ]
    n = len(records)
    missing = {
        key: (sum(r.get(key) is None for r in records) / n if n else None)
        for key in keys
    }
    scene_keys = [_scene_key(r) for r in records]
    duplicates = len(scene_keys) - len(set(scene_keys))
    return {
        "n_records": n,
        "n_interactive": sum(bool(r.get("has_other_agents", True)) for r in records),
        "n_no_interaction": sum(not bool(r.get("has_other_agents", True)) for r in records),
        "n_unknown_behavior": sum(not r.get("behavior_tag") for r in records),
        "duplicate_scene_keys": duplicates,
        "missing_fraction": missing,
    }


def compute_paired_comparison(
    candidate: List[Dict[str, Any]], baseline: List[Dict[str, Any]]
) -> Dict[str, Any]:
    cand_map = {_scene_key(r): r for r in candidate}
    base_map = {_scene_key(r): r for r in baseline}
    common = sorted(set(cand_map) & set(base_map))
    transitions = Counter()
    clearance_delta: List[float] = []
    progress_delta: List[float] = []
    jerk_delta: List[float] = []
    for key in common:
        c = cand_map[key]
        b = base_map[key]
        c_coll, b_coll = bool(c.get("did_coll", False)), bool(b.get("did_coll", False))
        if b_coll and not c_coll:
            transitions["collision_avoided"] += 1
        elif not b_coll and c_coll:
            transitions["collision_introduced"] += 1
        elif c_coll and b_coll:
            transitions["collision_in_both"] += 1
        else:
            transitions["safe_in_both"] += 1
        for metric, target in (
            ("min_clearance_proxy_m", clearance_delta),
            ("ego_progress_ratio", progress_delta),
        ):
            cv, bv = _as_finite_float(c.get(metric)), _as_finite_float(b.get(metric))
            if cv is not None and bv is not None:
                target.append(cv - bv)
        cv, bv = _as_finite_float(c.get("ego_max_abs_jerk")), _as_finite_float(b.get("ego_max_abs_jerk"))
        if cv is not None and bv is not None:
            jerk_delta.append(cv - bv)
    return {
        "n_candidate": len(candidate),
        "n_baseline": len(baseline),
        "n_paired": len(common),
        "coverage_candidate": len(common) / len(candidate) if candidate else None,
        "transitions": dict(transitions),
        "net_collision_avoidance": transitions["collision_avoided"] - transitions["collision_introduced"],
        "mean_delta_clearance_m": _mean_or_none(clearance_delta),
        "mean_delta_progress_ratio": _mean_or_none(progress_delta),
        "mean_delta_max_abs_jerk_mps3": _mean_or_none(jerk_delta),
        "delta_direction": {
            "clearance": "higher is better",
            "progress_ratio": "closer to 1 is usually better; signed mean is descriptive",
            "max_abs_jerk": "lower is better",
        },
    }


def _mean_score(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals: List[float] = []
    for r in records:
        score = r.get("diagnostic_scorecard") or {}
        value = _as_finite_float(score.get(key))
        if value is not None:
            vals.append(value)
    return _mean_or_none(vals)


def _risk_case_sort_key(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """Deterministic priority for ranking risk cases within a stratum or globally."""
    return (
        2 if bool(record.get("did_coll", False)) else 1 if bool(record.get("did_near_crash", False)) else 0,
        -(_as_finite_float(record.get("ttc_sec")) or 1e9),
        (_as_finite_float(record.get("relative_speed_at_collision_mps")) or 0.0),
        -(_as_finite_float(record.get("diagnostic_scorecard", {}).get("overall")) or 0.0),
    )


def _failure_mechanism_odd_stratum_key(record: Dict[str, Any]) -> Tuple[str, str, str]:
    """Stratum = failure_mechanism x (odd_speed_bin, odd_density_bin)."""
    return (
        str(record.get("failure_mechanism") or "unknown"),
        str(record.get("odd_speed_bin") or "unknown"),
        str(record.get("odd_density_bin") or "unknown"),
    )


def select_stratified_representative_cases(
    records: Sequence[Dict[str, Any]],
    *,
    per_stratum: int = 5,
    adverse_only: bool = True,
    max_total: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Select representative risk cases by failure_mechanism x ODD stratification.

    Within each stratum, cases are ranked by ``_risk_case_sort_key`` and up to ``per_stratum``
    records are kept. Strata are processed in sorted key order so case numbering is reproducible.
    """
    per_stratum = max(1, int(per_stratum))
    pool = list(records)
    if adverse_only:
        pool = [
            r for r in pool
            if bool(r.get("did_coll", False)) or bool(r.get("did_near_crash", False))
        ]

    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in pool:
        buckets[_failure_mechanism_odd_stratum_key(record)].append(record)

    selected: List[Dict[str, Any]] = []
    for stratum in sorted(buckets.keys()):
        group = sorted(buckets[stratum], key=_risk_case_sort_key, reverse=True)
        selected.extend(group[:per_stratum])

    if max_total is not None and max_total > 0:
        selected = selected[:max_total]
    return selected


def _compute_advanced_statistics(enriched: List[Dict[str, Any]], cfg: Any) -> Dict[str, Any]:
    """Best-effort theory-grounded scientific attribution (isolated, never fatal)."""
    if not bool(getattr(cfg, "eval_enable_advanced_stats", True)):
        return {"enabled": False, "reason": "advanced statistics disabled via --no_eval_advanced_stats"}
    try:
        import agent.eval.crashsim_eval_stats as stats_layer
    except Exception as e:
        return {"enabled": False, "reason": f"crashsim_eval_stats unavailable: {e}"}
    try:
        sci_config = stats_layer.ScientificConfig.from_cfg(cfg)
        return stats_layer.compute_advanced_scientific_attribution(enriched, sci_config)
    except Exception as e:
        return {"enabled": True, "available": False, "reason": f"advanced statistics failed: {e}"}


def _deterministic_report_is_current(report: Any) -> bool:
    """True when interpretable_eval.json matches the current prior/evidence formula."""
    if not isinstance(report, dict):
        return False
    if str(report.get("schema_version")) != "3.0":
        return False
    if str(report.get("deterministic_evidence_version") or "") != DETERMINISTIC_EVIDENCE_VERSION:
        return False
    if not (
        report.get("population_statistics")
        and report.get("failure_statistics")
        and report.get("scenario_statistics")
        and report.get("capability_population_reference")
        and report.get("representative_cases")
    ):
        return False
    patterns = (report.get("failure_statistics") or {}).get("patterns") or {}
    if patterns:
        sample_pattern = next(iter(patterns.values()), None)
        if isinstance(sample_pattern, dict) and "adverse_rate" not in sample_pattern:
            return False
    cap_ref = report.get("capability_population_reference") or {}
    if isinstance(cap_ref, dict) and cap_ref:
        sample_dim = next(iter(cap_ref.values()), None)
        if isinstance(sample_dim, dict):
            if sample_dim.get("role") != "deterministic_heuristic_prior_only":
                return False
            if "outcome_ceiling_applied" not in sample_dim:
                return False
    return True


def build_interpretable_evaluation(
    records: List[Dict[str, Any]],
    thresholds: InterpretabilityThresholds,
    baseline_records: Optional[List[Dict[str, Any]]] = None,
    context: Optional[Dict[str, Any]] = None,
    cfg: Any = None,
) -> Dict[str, Any]:
    enriched = [_enrich_interpretability_record(dict(r), thresholds) for r in records]
    # Population evidence and representative-case selection are intentionally
    # isolated in a post-rollout-only helper.  Capability judgments must use the
    # complete evaluated population; selected cases are contextual illustrations.
    from agent.eval.crashsim_evidence import build_structured_evidence

    representative_per_group = max(1, thresholds.stratum_cases_per_cell)
    structured_evidence = build_structured_evidence(
        enriched, representative_per_group=representative_per_group
    )
    interactive = [r for r in enriched if bool(r.get("has_other_agents", True))]
    n_interactive = len(interactive)
    n_coll = sum(bool(r.get("did_coll", False)) for r in interactive)
    n_near = sum(bool(r.get("did_near_crash", False)) and not bool(r.get("did_coll", False)) for r in interactive)
    attribution = compute_deterministic_attribution(enriched)
    odd_analysis = compute_odd_analysis(enriched)
    _augment_odd_uncertainty(odd_analysis, thresholds)
    failure_profile = compute_failure_profile_summary(enriched, attribution)
    mechanism_records = [r for r in interactive if bool(r.get("did_coll", False))]
    mechanism_counts = Counter(str(r.get("failure_mechanism") or "unknown") for r in mechanism_records)

    max_representative = thresholds.max_failures_in_prompt
    representative_groups = structured_evidence["representative_cases"]
    representative_cases = [
        record
        for group_name in ("typical_failure", "severe_failure", "successful_avoidance")
        for record in representative_groups.get(group_name, [])
    ]
    if max_representative > 0:
        representative_cases = representative_cases[:max_representative]
    # Backward-compatible report/debug alias. It is no longer a "highest-risk"
    # sample and must never be treated as the basis for population capability.
    risk_cases = representative_cases

    baseline_enriched = None
    paired = None
    if baseline_records is not None:
        baseline_enriched = [_enrich_interpretability_record(dict(r), thresholds) for r in baseline_records]
        paired = compute_paired_comparison(enriched, baseline_enriched)

    advanced_statistics = _compute_advanced_statistics(enriched, cfg) if cfg is not None else None

    return {
        "schema_version": "3.0",
        "deterministic_evidence_version": DETERMINISTIC_EVIDENCE_VERSION,
        "method": "population deterministic evidence + diverse contextual cases + evidence-grounded LLM",
        "context": context or {},
        "thresholds": {
            "near_miss_clearance_m": thresholds.near_miss_clearance_m,
            "brake_accel_threshold": thresholds.brake_accel_threshold,
            "accel_into_conflict_threshold": thresholds.accel_into_conflict_threshold,
            "severe_jerk_mps3": thresholds.severe_jerk_mps3,
            "severe_lat_accel_mps2": thresholds.severe_lat_accel_mps2,
            "severe_decel_mps2": thresholds.severe_decel_mps2,
            "min_group_size": thresholds.min_group_size,
        },
        "overall": {
            "n_records": len(enriched),
            "n_interactive": n_interactive,
            "collision": _rate_with_ci(n_coll, n_interactive),
            "near_miss_excluding_collisions": _rate_with_ci(n_near, n_interactive),
            "mean_diagnostic_score": {
                "safety": _mean_score(interactive, "safety"),
                "comfort": _mean_score(interactive, "comfort"),
                "efficiency": _mean_score(interactive, "efficiency"),
                "overall": _mean_score(interactive, "overall"),
            },
            "mean_min_box_separation_margin_m": _mean_or_none(_finite_metric_values(interactive, "min_box_separation_margin_m")),
            "median_ttc_collision_sec": _median_or_none(_finite_metric_values([r for r in interactive if r.get("did_coll")], "ttc_sec")),
        },
        "failure_mechanisms": [
            {"mechanism": k, "count": v, "share": v / max(1, len(mechanism_records))}
            for k, v in mechanism_counts.most_common()
        ],
        "population_statistics": structured_evidence["population_statistics"],
        "failure_statistics": structured_evidence["failure_statistics"],
        "scenario_statistics": structured_evidence["scenario_statistics"],
        "capability_population_reference": structured_evidence["capability_population_reference"],
        "representative_cases": representative_groups,
        "missing_evidence_fields": structured_evidence["missing_fields"],
        "attribution": attribution,
        "advanced_statistics": advanced_statistics,
        "failure_profile": failure_profile,
        "odd_analysis": odd_analysis,
        "paired_baseline_comparison": paired,
        "data_quality": compute_data_quality(enriched),
        "top_risk_cases": risk_cases,
        "representative_risk_cases": representative_cases,
        "representative_case_selection": {
            "method": "diverse_typical_severe_successful",
            "strata": ["crash_category", "failure_mechanism", "interaction_density"],
            "per_group_cap": representative_per_group,
            "groups": ["typical_failure", "severe_failure", "successful_avoidance"],
            "adverse_only": False,
            "n_selected": len(representative_cases),
            "max_total_cap": max_representative if max_representative and max_representative > 0 else None,
            "top_risk_cases_alias_of": "representative_risk_cases",
            "details": representative_groups.get("selection_method") or {},
        },
        "scene_records": enriched,
    }


def _generate_evaluation_figures(report: Dict[str, Any], out_dir: str) -> Dict[str, str]:
    """Render deterministic-evidence charts as PNGs for embedding in the report.

    Returns a mapping ``{figure_key: relative_path}`` where the path is relative
    to the markdown file (i.e. ``figures/<name>.png``). Figures are best-effort:
    if matplotlib is unavailable or a panel lacks data, that figure is skipped
    and the markdown simply omits the corresponding image.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib not installed / no backend
        Logger.log(f"[eval][viz] skipped figure generation ({exc})")
        return {}

    fig_dir = os.path.join(out_dir, "figures")
    mkdir(fig_dir)
    figures: Dict[str, str] = {}

    def _save(fig, name: str) -> None:
        path = os.path.join(fig_dir, name)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        figures[name.rsplit(".", 1)[0]] = os.path.join("figures", name)

    palette = {
        "collision": "#d1495b",
        "near_miss": "#edae49",
        "safe": "#66a182",
        "bar": "#2e75b6",
        "pos": "#d1495b",
        "neg": "#2e75b6",
    }

    overall = report.get("overall") or {}
    coll = overall.get("collision") or {}
    near = overall.get("near_miss_excluding_collisions") or {}
    scores = overall.get("mean_diagnostic_score") or {}

    # 1) Overview: outcome distribution + mean diagnostic scores
    try:
        n_interactive = int(overall.get("n_interactive", 0) or 0)
        n_coll = int(coll.get("count", 0) or 0)
        n_near = int(near.get("count", 0) or 0)
        n_safe = max(0, n_interactive - n_coll - n_near)
        if n_interactive > 0:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            labels = ["collision", "near-miss", "safe"]
            counts = [n_coll, n_near, n_safe]
            colors = [palette["collision"], palette["near_miss"], palette["safe"]]
            bars = axes[0].bar(labels, counts, color=colors)
            axes[0].set_title("Outcome distribution (interactive sequences)")
            axes[0].set_ylabel("count")
            for b, c in zip(bars, counts):
                pct = (c / n_interactive) if n_interactive else 0.0
                axes[0].text(b.get_x() + b.get_width() / 2, b.get_height(),
                             f"{c}\n{pct:.1%}", ha="center", va="bottom", fontsize=9)

            score_keys = ["safety", "comfort", "efficiency", "overall"]
            score_vals = [_as_finite_float(scores.get(k)) or 0.0 for k in score_keys]
            sbars = axes[1].bar(score_keys, score_vals, color=palette["bar"])
            axes[1].set_ylim(0, 100)
            axes[1].set_title("Mean diagnostic scores (0–100)")
            axes[1].set_ylabel("score")
            for b, v in zip(sbars, score_vals):
                axes[1].text(b.get_x() + b.get_width() / 2, b.get_height(),
                             f"{v:.1f}", ha="center", va="bottom", fontsize=9)
            fig.tight_layout()
            _save(fig, "overview.png")
    except Exception as exc:
        Logger.log(f"[eval][viz] overview figure failed ({exc})")

    # 2) Collision failure mechanisms
    try:
        mechs = report.get("failure_mechanisms") or []
        if mechs:
            mechs = sorted(mechs, key=lambda r: r.get("count", 0))
            names = [str(r.get("mechanism")) for r in mechs]
            counts = [int(r.get("count", 0) or 0) for r in mechs]
            fig, ax = plt.subplots(figsize=(9, max(2.4, 0.5 * len(names) + 1)))
            bars = ax.barh(names, counts, color=palette["collision"])
            ax.set_title("Collision failure mechanisms")
            ax.set_xlabel("number of collision cases")
            for b, c, r in zip(bars, counts, mechs):
                ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                        f" {c} ({_eval_fmt(r.get('share'), '.1%')})", va="center", fontsize=9)
            fig.tight_layout()
            _save(fig, "failure_mechanisms.png")
    except Exception as exc:
        Logger.log(f"[eval][viz] mechanisms figure failed ({exc})")

    # 3) Behavior-level collision rate with 95% Wilson CI
    try:
        behavior = ((report.get("attribution") or {}).get("behavior_risk") or [])
        behavior = [r for r in behavior if r.get("coll_rate") is not None][:14]
        if behavior:
            behavior = sorted(behavior, key=lambda r: r.get("coll_rate") or 0.0)
            names = [str(r.get("behavior_tag")) for r in behavior]
            rates = [float(r.get("coll_rate") or 0.0) for r in behavior]
            lows = [float(r.get("ci95_low") or rates[i]) for i, r in enumerate(behavior)]
            highs = [float(r.get("ci95_high") or rates[i]) for i, r in enumerate(behavior)]
            xerr = np.array([[max(0.0, rates[i] - lows[i]) for i in range(len(rates))],
                             [max(0.0, highs[i] - rates[i]) for i in range(len(rates))]])
            fig, ax = plt.subplots(figsize=(9, max(2.8, 0.45 * len(names) + 1)))
            ax.barh(names, rates, color=palette["bar"], xerr=xerr,
                    error_kw={"ecolor": "#444444", "capsize": 3, "elinewidth": 1})
            ax.set_title("Collision rate by behavior (95% Wilson CI)")
            ax.set_xlabel("collision rate")
            ax.set_xlim(0, 1)
            fig.tight_layout()
            _save(fig, "behavior_risk.png")
    except Exception as exc:
        Logger.log(f"[eval][viz] behavior figure failed ({exc})")

    # 4) Feature separation (Cohen's d)
    try:
        features = ((report.get("attribution") or {}).get("feature_separation") or [])[:10]
        features = [r for r in features if r.get("cohens_d") is not None]
        if features:
            features = sorted(features, key=lambda r: r.get("cohens_d") or 0.0)
            names = [str(r.get("feature")) for r in features]
            ds = [float(r.get("cohens_d") or 0.0) for r in features]
            colors = [palette["pos"] if d >= 0 else palette["neg"] for d in ds]
            fig, ax = plt.subplots(figsize=(9, max(2.6, 0.5 * len(names) + 1)))
            bars = ax.barh(names, ds, color=colors)
            ax.axvline(0.0, color="#888888", linewidth=0.8)
            ax.set_title("Collision vs safe: standardized mean difference (Cohen's d)")
            ax.set_xlabel("Cohen's d  (collision mean − safe mean, pooled SD)")
            for b, d in zip(bars, ds):
                ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                        f" {d:+.2f}", va="center",
                        ha="left" if d >= 0 else "right", fontsize=9)
            fig.tight_layout()
            _save(fig, "feature_separation.png")
    except Exception as exc:
        Logger.log(f"[eval][viz] feature figure failed ({exc})")

    # 5) ODD collision-rate breakdown (up to 4 dimensions)
    try:
        dimensions = ((report.get("odd_analysis") or {}).get("dimensions") or {})
        dim_titles = {
            "odd_location": "Map location",
            "odd_region": "Geographic region",
            "odd_speed_bin": "Ego speed bin",
            "odd_density_bin": "Agent density bin",
        }
        active = [(k, t) for k, t in dim_titles.items() if dimensions.get(k)]
        if active:
            ncol = 2 if len(active) > 1 else 1
            nrow = int(math.ceil(len(active) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 3.4 * nrow), squeeze=False)
            for idx, (dim_key, title) in enumerate(active):
                ax = axes[idx // ncol][idx % ncol]
                rows = sorted(dimensions.get(dim_key) or [], key=lambda r: r.get("coll_rate") or 0.0)
                vals = [str(r.get("value")) for r in rows]
                rates = [float(r.get("coll_rate") or 0.0) for r in rows]
                bars = ax.barh(vals, rates, color=palette["bar"])
                ax.set_title(f"Collision rate — {title}")
                ax.set_xlim(0, 1)
                for b, r_row in zip(bars, rows):
                    ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                            f" {_eval_fmt(r_row.get('coll_rate'), '.1%')} (n={r_row.get('n', 0)})",
                            va="center", fontsize=8)
            for j in range(len(active), nrow * ncol):
                axes[j // ncol][j % ncol].axis("off")
            fig.tight_layout()
            _save(fig, "odd_breakdown.png")
    except Exception as exc:
        Logger.log(f"[eval][viz] ODD figure failed ({exc})")

    if figures:
        Logger.log(f"[eval][viz] wrote {len(figures)} figure(s) to {fig_dir}")
    return figures


def _render_interpretable_markdown(report: Dict[str, Any], figures: Optional[Dict[str, str]] = None) -> str:
    figures = figures or {}

    def _img(key: str, alt: str) -> List[str]:
        rel = figures.get(key)
        if not rel:
            return []
        return [f"![{alt}]({rel})", ""]

    overall = report.get("overall") or {}
    coll = overall.get("collision") or {}
    near = overall.get("near_miss_excluding_collisions") or {}
    scores = overall.get("mean_diagnostic_score") or {}
    lines = ["# Interpretable AV Evaluation", ""]
    lines.append(
        "This report is generated from deterministic rollout evidence. The optional LLM layer may summarize these results, "
        "but it does not compute or alter the metrics, failure labels, confidence intervals, or scene rankings."
    )
    lines.extend(["", "## Executive summary", ""])
    lines.append(
        f"- Evaluated **{overall.get('n_records', 0)}** sequences, including **{overall.get('n_interactive', 0)}** with other agents."
    )
    lines.append(
        f"- Collision rate: **{_eval_fmt(coll.get('rate'), '.2%')}** "
        f"(95% Wilson CI {_eval_fmt(coll.get('ci95_low'), '.2%')}–{_eval_fmt(coll.get('ci95_high'), '.2%')}; "
        f"{coll.get('count', 0)}/{coll.get('denominator', 0)})."
    )
    lines.append(
        f"- Near-miss rate excluding collisions: **{_eval_fmt(near.get('rate'), '.2%')}** "
        f"(95% Wilson CI {_eval_fmt(near.get('ci95_low'), '.2%')}–{_eval_fmt(near.get('ci95_high'), '.2%')})."
    )
    lines.append(
        f"- Mean diagnostic scores: safety={_eval_fmt(scores.get('safety'), '.1f')}, "
        f"comfort={_eval_fmt(scores.get('comfort'), '.1f')}, efficiency={_eval_fmt(scores.get('efficiency'), '.1f')}, "
        f"overall={_eval_fmt(scores.get('overall'), '.1f')}."
    )
    lines.append("- Diagnostic scores are transparent threshold-based summaries; raw metrics are authoritative.")
    lines.append("")
    lines.extend(_img("overview", "Outcome distribution and mean diagnostic scores"))

    lines.extend(["", "## Collision failure mechanisms", ""])
    lines.extend(_img("failure_mechanisms", "Collision failure mechanisms"))
    lines.extend(["| mechanism | count | share of collision cases |", "|---|---:|---:|"])
    for row in report.get("failure_mechanisms") or []:
        lines.append(f"| {row.get('mechanism')} | {row.get('count', 0)} | {_eval_fmt(row.get('share'), '.2%')} |")

    behavior = ((report.get("attribution") or {}).get("behavior_risk") or [])
    if behavior:
        lines.extend(["", "## Behavior-level risk attribution", ""])
        lines.extend(_img("behavior_risk", "Collision rate by behavior with 95% CI"))
        lines.extend(["| behavior | n | collisions | rate | 95% CI | failure share |", "|---|---:|---:|---:|---:|---:|"])
        for row in behavior:
            lines.append(
                f"| {row.get('behavior_tag')} | {row.get('n', 0)} | {row.get('n_coll', 0)} | "
                f"{_eval_fmt(row.get('coll_rate'), '.2%')} | "
                f"{_eval_fmt(row.get('ci95_low'), '.2%')}–{_eval_fmt(row.get('ci95_high'), '.2%')} | "
                f"{_eval_fmt(row.get('failure_share'), '.2%')} |"
            )

    features = ((report.get("attribution") or {}).get("feature_separation") or [])[:8]
    if features:
        lines.extend(["", "## Evidence separating collision and safe cases", ""])
        lines.extend(_img("feature_separation", "Feature separation by Cohen's d"))
        lines.extend(["| feature | collision mean | safe mean | delta | Cohen's d | effect |", "|---|---:|---:|---:|---:|---|"])
        for row in features:
            lines.append(
                f"| {row.get('feature')} | {_eval_fmt(row.get('mean_coll'), '.3f')} | "
                f"{_eval_fmt(row.get('mean_no_coll'), '.3f')} | {_eval_fmt(row.get('delta'), '.3f')} | "
                f"{_eval_fmt(row.get('cohens_d'), '.3f')} | {row.get('effect_label')} |"
            )

    odd_md = render_odd_section(report.get("odd_analysis") or {})
    if odd_md:
        odd_img = _img("odd_breakdown", "ODD collision-rate breakdown")
        if odd_img:
            odd_lines = odd_md.split("\n")
            # insert the figure right after the ODD section intro paragraph
            insert_at = 4 if len(odd_lines) >= 4 else len(odd_lines)
            odd_lines = odd_lines[:insert_at] + odd_img + odd_lines[insert_at:]
            odd_md = "\n".join(odd_lines)
        lines.extend(["", odd_md])

    advanced = report.get("advanced_statistics")
    if advanced:
        try:
            import agent.eval.crashsim_eval_stats as stats_layer
            adv_md = stats_layer.render_advanced_stats_section(advanced)
            if adv_md:
                lines.extend(["", adv_md])
        except Exception:
            pass

    paired = report.get("paired_baseline_comparison")
    if paired:
        trans = paired.get("transitions") or {}
        lines.extend(["", "## Paired baseline comparison", ""])
        lines.append(f"- Paired sequences: {paired.get('n_paired', 0)}.")
        lines.append(
            f"- Collision transitions: avoided={trans.get('collision_avoided', 0)}, introduced={trans.get('collision_introduced', 0)}, "
            f"in both={trans.get('collision_in_both', 0)}, safe in both={trans.get('safe_in_both', 0)}."
        )
        lines.append(f"- Net collision avoidance: {paired.get('net_collision_avoidance', 0)}.")
        lines.append(f"- Mean SAT separation-margin delta: {_eval_fmt(paired.get('mean_delta_clearance_m'), '.3f')} m (higher is better).")
        lines.append(f"- Mean max-jerk delta: {_eval_fmt(paired.get('mean_delta_max_abs_jerk_mps3'), '.3f')} m/s³ (lower is better).")

    lines.extend(["", "## Representative risk cases (stratified; LLM evidence pool)", ""])
    selection = report.get("representative_case_selection") or {}
    if selection:
        lines.append(
            f"- Selection: **{selection.get('method', 'unknown')}**; "
            f"per stratum={selection.get('per_stratum', '?')}; "
            f"n_selected={selection.get('n_selected', 0)}. "
            f"(`top_risk_cases` in JSON mirrors this list for report/debug.)"
        )
    rep_cases = report.get("representative_risk_cases") or report.get("top_risk_cases") or []
    for idx, r in enumerate(rep_cases, start=1):
        score = (r.get("diagnostic_scorecard") or {}).get("overall")
        lines.append(
            f"### case_{idx:03d}: {r.get('scene_name') or r.get('scene_token') or 'unknown scene'} "
            f"(sidx={r.get('sidx')}, outcome={r.get('outcome_class')}, mechanism={r.get('failure_mechanism')}, "
            f"odd={r.get('odd_speed_bin')}/{r.get('odd_density_bin')}, score={_eval_fmt(score, '.1f')})"
        )
        lines.append("")
        for item in r.get("evidence") or []:
            lines.append(f"- {item}")
        lines.append("")

    dq = report.get("data_quality") or {}
    lines.extend(["## Data quality and coverage", ""])
    lines.append(f"- Duplicate scene keys: {dq.get('duplicate_scene_keys', 0)}.")
    lines.append(f"- Unknown behavior tags: {dq.get('n_unknown_behavior', 0)}.")
    lines.append(f"- No-interaction sequences: {dq.get('n_no_interaction', 0)}.")
    missing = dq.get("missing_fraction") or {}
    lines.append("- Missing metric fractions: " + ", ".join(f"{k}={_eval_fmt(v, '.1%')}" for k, v in missing.items()) + ".")

    thresholds = report.get("thresholds") or {}
    lines.extend(["", "## Interpretation thresholds", ""])
    for key, value in thresholds.items():
        lines.append(f"- `{key}` = {value}")
    lines.append("")
    lines.append(
        "Failure mechanisms are evidence-linked diagnostic labels, not causal proof. A causal claim requires controlled counterfactual rollouts or intervention-based tests."
    )
    return "\n".join(lines)


def write_interpretable_evaluation(report: Dict[str, Any], out_dir: str) -> Dict[str, str]:
    mkdir(out_dir)
    json_path = os.path.join(out_dir, "interpretable_eval.json")
    md_path = os.path.join(out_dir, "interpretable_eval.md")
    evidence_path = os.path.join(out_dir, "scene_evidence.jsonl")
    failures_path = os.path.join(out_dir, "failure_cases.csv")

    report_without_records = dict(report)
    records = list(report_without_records.pop("scene_records", []))
    figures = _generate_evaluation_figures(report_without_records, out_dir)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_without_records, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_interpretable_markdown(report_without_records, figures))
    with open(evidence_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    failure_rows = [r for r in records if r.get("outcome_class") in ("collision", "near_miss")]
    fieldnames = [
        "scene_index", "scene_token", "scene_name", "sidx", "planner", "behavior_tag",
        "odd_location", "odd_speed_bin", "odd_density_bin", "outcome_class",
        "failure_mechanism", "conflict_geometry", "ttc_sec", "min_clearance_proxy_m",
        "relative_speed_at_collision_mps", "ego_max_decel", "ego_max_abs_jerk",
        "ego_progress_ratio", "diagnostic_overall_score", "evidence",
    ]
    with open(failures_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in failure_rows:
            row = {key: r.get(key) for key in fieldnames}
            row["diagnostic_overall_score"] = (r.get("diagnostic_scorecard") or {}).get("overall")
            row["evidence"] = " | ".join(r.get("evidence") or [])
            writer.writerow(row)

    paths = {
        "json": json_path,
        "markdown": md_path,
        "scene_evidence": evidence_path,
        "failure_cases": failures_path,
    }
    advanced = report_without_records.get("advanced_statistics")
    if advanced:
        stats_path = os.path.join(out_dir, "scientific_attribution.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(advanced, f, ensure_ascii=False, indent=2)
        paths["scientific_attribution"] = stats_path
    return paths


# ---------------------------------------------------------------------------
# Evidence-grounded LLM agents: failure analysis + planner improvement guidance
# ---------------------------------------------------------------------------
# The LLM never computes or overrides deterministic metrics, failure labels,
# confidence intervals, or scene rankings. It reasons over the auditable
# evidence package produced above: it synthesizes cross-scene failure patterns,
# challenges diagnostic labels when warranted, isolates vulnerable ODD cells,
# and turns the diagnosis into prioritized, testable planner improvements.
def _llm_chat_completions_url(api_base: str) -> str:
    base = str(api_base or "").strip().rstrip("/")
    if not base:
        raise ValueError("LLM API base is empty; set --eval_api_base")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _read_http_response_text(resp: Any) -> str:
    """Read an HTTP response body as UTF-8 text, decompressing if needed.

    Some OpenAI-compatible gateways ignore ``Accept-Encoding: identity`` and
    still return gzip/deflate payloads. ``urllib`` does not decompress
    automatically, so a naive ``resp.read().decode("utf-8")`` fails with an
    ``invalid start byte`` error on the gzip magic bytes (0x1f 0x8b).
    """
    raw = resp.read()
    encoding = ""
    try:
        encoding = (resp.headers.get("Content-Encoding") or "").strip().lower()
    except Exception:
        encoding = ""
    if not encoding and raw[:2] == b"\x1f\x8b":
        encoding = "gzip"
    try:
        if encoding == "gzip":
            raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
        elif encoding == "deflate":
            try:
                raw = zlib.decompress(raw)
            except zlib.error:
                raw = zlib.decompress(raw, -zlib.MAX_WBITS)
    except Exception:
        pass
    return raw.decode("utf-8", errors="replace")


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(raw[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("LLM response does not contain a valid JSON object")


def _openai_compatible_json_call(
    *,
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: float,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Call an OpenAI-compatible chat endpoint and require a JSON object.

    The function first requests JSON mode. Some compatible endpoints do not
    support ``response_format``; in that case it retries once without it while
    keeping the JSON-only instruction in the prompt.
    """
    url = _llm_chat_completions_url(api_base)
    base_payload: Dict[str, Any] = {
        "model": str(model),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    errors: List[str] = []
    for use_json_mode in (True, False):
        payload = dict(base_payload)
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
                response_obj = json.loads(_read_http_response_text(resp))
            choices = response_obj.get("choices") or []
            if not choices:
                raise ValueError(f"LLM response has no choices: {response_obj}")
            content = ((choices[0].get("message") or {}).get("content"))
            if isinstance(content, list):
                content = "".join(
                    str(item.get("text", "")) if isinstance(item, dict) else str(item)
                    for item in content
                )
            return _extract_json_object(str(content or ""))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = _read_http_response_text(e)[:1000]
            except Exception:
                pass
            errors.append(f"HTTP {e.code}: {detail}")
            if not use_json_mode:
                break
        except Exception as e:
            errors.append(str(e))
            if not use_json_mode:
                break
    raise RuntimeError("LLM request failed: " + " | ".join(errors))


def _compact_odd_cells(report: Dict[str, Any], max_cells: int) -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    dimensions = ((report.get("odd_analysis") or {}).get("dimensions") or {})
    for dimension, rows in dimensions.items():
        for row in rows or []:
            cells.append(
                {
                    "dimension": dimension,
                    "value": row.get("value"),
                    "n": row.get("n"),
                    "n_with_other": row.get("n_with_other"),
                    "n_coll": row.get("n_coll"),
                    "collision_rate": row.get("coll_rate"),
                    "ci95_low": row.get("ci95_low"),
                    "ci95_high": row.get("ci95_high"),
                    "failure_share": row.get("failure_share"),
                    "low_support": row.get("low_support"),
                }
            )
    cells.sort(
        key=lambda row: (
            bool(not row.get("low_support", False)),
            _as_finite_float(row.get("collision_rate")) or -1.0,
            int(row.get("n_coll") or 0),
        ),
        reverse=True,
    )
    return cells[: max(0, int(max_cells))]


CAPABILITY_LEVELS: Tuple[str, ...] = (
    "critical_weakness",
    "major_weakness",
    "mixed",
    "generally_capable",
    "strong",
)

CAPABILITY_LEVEL_RUBRIC: Dict[str, str] = {
    "critical_weakness": (
        "Repeated, high-support evidence shows severe capability failure associated with "
        "unsafe outcomes or inability to respond."
    ),
    "major_weakness": (
        "Clear recurring evidence shows a substantial limitation, but the capability is not "
        "uniformly absent."
    ),
    "mixed": (
        "Evidence shows both competent and deficient behavior, or available evidence is "
        "conflicting."
    ),
    "generally_capable": (
        "The capability usually performs adequately, with limited or low-severity weaknesses."
    ),
    "strong": (
        "Consistent evidence supports reliable performance with no major recurring weakness in "
        "the evaluated cases."
    ),
}

CAPABILITY_DIMENSIONS: Tuple[str, ...] = (
    "risk_anticipation",
    "interaction_reasoning",
    "longitudinal_response",
    "lateral_response",
    "comfort_and_stability",
    "mobility_efficiency",
)

CAPABILITY_DIMENSION_DISPLAY_NAMES: Dict[str, str] = {
    "risk_anticipation": "Risk anticipation",
    "interaction_reasoning": "Interaction reasoning",
    "longitudinal_response": "Longitudinal response",
    "lateral_response": "Lateral response",
    "comfort_and_stability": "Comfort and stability",
    "mobility_efficiency": "Mobility efficiency",
}

# Prompt-only hints: which case signals are relevant when selecting evidence_refs per dimension.
CAPABILITY_DIMENSION_EVIDENCE_HINTS: Dict[str, str] = {
    "risk_anticipation": (
        "On hard sets prefer delayed_or_absent_response.adverse_rate + minimum_ttc.median over "
        "collision_rate alone; case hints: late_or_absent_braking, insufficient_collision_mitigation; "
        "metrics: ttc_sec, brake_onset_ttc_sec, min_clearance_proxy_m"
    ),
    "interaction_reasoning": (
        "conflict_geometry in {crossing_or_opposing, merge_or_weave}; mechanisms: crossing_conflict_handling, "
        "lateral_conflict_handling; metrics: relative_heading_at_collision_rad, min_clearance_proxy_m"
    ),
    "longitudinal_response": (
        "mechanisms: late_or_absent_braking, longitudinal_gap_management, acceleration_into_conflict; "
        "metrics: brake-before-collision, brake_onset_ttc_sec, ego_max_decel, mean_accel_pre_collision"
    ),
    "lateral_response": (
        "mechanisms: lateral_conflict_handling; conflict_geometry: lateral_side_conflict; "
        "metrics: ego_lat_accel_max, min_clearance_proxy_m"
    ),
    "comfort_and_stability": (
        "metrics: ego_max_abs_jerk, ego_lat_accel_max, ego_max_decel; do NOT cite collision cases as "
        "positive comfort evidence unless assessment explains control smoothness despite adverse outcome"
    ),
    "mobility_efficiency": (
        "metrics: ego_progress_ratio, ego_ade_m; for generally_capable/strong also cite overall.mean_diagnostic_score "
        "in assessment text; collision-only refs are insufficient sole evidence for strong efficiency"
    ),
}

CONFIDENCE_LEVELS: Tuple[str, ...] = ("high", "medium", "low")

# Deterministic metric-key -> unit mapping. The LLM may propose a unit, but this mapping is
# authoritative: post-processing always overwrites/verifies it so units can never be invented.
METRIC_UNIT_MAP: Dict[str, Optional[str]] = {
    "ttc_sec": "s",
    "brake_onset_ttc_sec": "s",
    "min_clearance_proxy_m": "m",
    "relative_speed_at_collision_mps": "m/s",
    "relative_heading_at_collision_rad": "rad",
    "mean_accel_pre_collision": "m/s^2",
    "ego_max_decel": "m/s^2",
    "ego_max_abs_jerk": "m/s^3",
    "ego_lat_accel_max": "m/s^2",
    "ego_progress_ratio": None,
    "ego_ade_m": "m",
    "brake-before-collision": None,
    "accelerate-into-collision": None,
}


def _build_llm_evidence_package(report: Dict[str, Any], cfg) -> Dict[str, Any]:
    max_cases = max(0, int(getattr(cfg, "eval_max_failures_in_prompt", 120)))
    max_odd = max(1, int(getattr(cfg, "eval_max_odd_cells_in_prompt", 30)))
    from agent.eval.crashsim_evidence import compact_case

    representative_groups = report.get("representative_cases") or {}
    selected_by_type: Dict[str, List[Dict[str, Any]]] = {
        "typical_failure": [],
        "severe_failure": [],
        "successful_avoidance": [],
    }
    selected_cases: List[Dict[str, Any]] = []
    case_index = 1
    for case_type in selected_by_type:
        for record in representative_groups.get(case_type) or []:
            if max_cases > 0 and len(selected_cases) >= max_cases:
                break
            compact = compact_case(record, f"case_{case_index:03d}", case_type)
            selected_by_type[case_type].append(compact)
            selected_cases.append(compact)
            case_index += 1

    attribution = report.get("attribution") or {}
    feature_based = attribution.get("feature_based_attribution") or {}
    from agent.eval.crashsim_evidence import build_risk_anticipation_discrimination

    population_stats = report.get("population_statistics") or {}
    failure_stats = report.get("failure_statistics") or {}
    risk_discrimination = build_risk_anticipation_discrimination(
        population_stats, failure_stats
    )
    package: Dict[str, Any] = {
        "evaluation_scope": "single planner",
        "pipeline_version": EVALUATION_PIPELINE_VERSION,
        "planner_name": (report.get("context") or {}).get("planner"),
        "planner_context": report.get("context") or {},
        "population_statistics": population_stats,
        "failure_statistics": failure_stats,
        "scenario_statistics": report.get("scenario_statistics") or {},
        "capability_population_reference": report.get("capability_population_reference") or {},
        "capability_population_reference_role": (
            "optional magnitude reference only — judge levels from the full package; "
            "do not treat reference_level as an answer to copy"
        ),
        "risk_anticipation_discrimination": risk_discrimination,
        "representative_cases": selected_by_type,
        "missing_evidence_fields": report.get("missing_evidence_fields") or {},
        # Compatibility aliases for older renderers/validators.
        "overall": report.get("overall") or {},
        "failure_mechanisms": report.get("failure_mechanisms") or [],
        "behavior_risk": (attribution.get("behavior_risk") or [])[:10],
        "feature_separation": (attribution.get("feature_separation") or [])[:10],
        "collision_timing": attribution.get("collision_timing") or {},
        "collision_control_pattern": feature_based.get("collision_control_pattern") or {},
        "deterministic_root_hypotheses": feature_based.get("root_hypotheses") or [],
        "odd_high_risk_cells": _compact_odd_cells(report, max_odd),
        "data_quality": report.get("data_quality") or {},
        "representative_risk_cases": selected_cases,
        "representative_case_selection": report.get("representative_case_selection") or {},
        "interpretation_constraints": [
            "Use only supplied evidence IDs, values, and case IDs; never invent metrics.",
            (
                "Primary job: cross-scenario reasoning — compare ODDs, crash categories, and "
                "typical/severe/successful cases; explain tensions between signals; judge "
                "capabilities. Do NOT paraphrase population tables as the assessment."
            ),
            (
                "assessment prose ≠ supporting_evidence. Put exact numbers in "
                "supporting_evidence; in assessment lead with comparative claims "
                "(where the limitation concentrates vs where it does not)."
            ),
            (
                "supporting_evidence must span ≥2 grounding families among "
                "{safety, temporal, interaction, control, efficiency}; same-family pairs "
                "(jerk+unstable_evasive; progress+overly_conservative; collision+clearance) "
                "do not count. Prefer complementary pairs (safety+temporal, control+safety, "
                "efficiency+safety/interaction)."
            ),
            (
                "capability_population_reference.reference_level is an optional magnitude "
                "reference only. It is NOT an answer key — synthesize the full package; "
                "do not copy reference_level by default; prior components alone are "
                "invalid supporting_evidence."
            ),
            (
                "risk_anticipation on hard/crash sets: use "
                "risk_anticipation_discrimination — elevated collision_rate alone must NOT "
                "yield critical_weakness; require anticipation-specific collapse "
                "(delayed_or_absent_response.adverse_rate and/or jointly severe short TTC). "
                "soft_advisory_band is a check only, never an answer to copy."
            ),
            "Representative cases illustrate mechanisms; they cannot alone set a capability level.",
            "Do not rank planners; separate observation from unverified mechanism hypotheses.",
            "Treat low-support ODD cells cautiously; do not invent absent operating conditions.",
        ],
    }
    # Feed the deterministic paired baseline comparison to the LLM when available,
    # so improvement guidance can account for regressions vs. an existing planner.
    paired = report.get("paired_baseline_comparison")
    if paired:
        package["paired_baseline_comparison"] = paired

    # Feed the theory-grounded scientific attribution (odds ratios, SHAP, FDR-controlled
    # effects, EVT tail risk, survival hazards) so the LLM reasons over rigorous, uncertainty-
    # quantified evidence rather than raw correlations.
    advanced = report.get("advanced_statistics")
    if advanced and advanced.get("enabled", False):
        try:
            import agent.eval.crashsim_eval_stats as stats_layer
            compact = stats_layer.compact_for_llm(advanced)
            if compact:
                package["scientific_attribution"] = compact
                package["interpretation_constraints"].append(
                    "Scientific-attribution results (odds ratios, SHAP, FDR-controlled effects, EVT, "
                    "survival hazards) adjust for clustering and multiplicity; prefer FDR-significant "
                    "effects whose intervals exclude the null, but still treat them as strong "
                    "associations, not causal proof."
                )
        except Exception:
            pass
    return package


def _evidence_package_for_prompt(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy evidence for LLM prompts; keep prior levels as optional magnitude context.

    ``reference_level`` and ``risk_anticipation_discrimination`` remain visible as
    soft checks, but prompts must not instruct the model to copy them.
    """
    import copy

    pkg = copy.deepcopy(dict(evidence))
    refs = pkg.get("capability_population_reference")
    if isinstance(refs, dict):
        for payload in refs.values():
            if not isinstance(payload, dict):
                continue
            payload["role"] = (
                "optional magnitude reference for this dimension — not an answer key; "
                "judge from the full evidence package"
            )
    pkg["capability_population_reference_role"] = (
        "optional magnitude reference only; do not copy reference_level by default"
    )
    return pkg


def _capability_evidence_selection_prompt_block(evidence: Dict[str, Any]) -> str:
    """Build prompt text for stratified, dimension-aware capability evidence_refs selection."""
    selection = evidence.get("representative_case_selection") or {}
    per_stratum = max(1, int(selection.get("per_stratum") or 5))
    min_refs = min(per_stratum, 5)
    min_strata = min(3, max(1, per_stratum // 2 + 1))

    hint_lines = [
        f"- {dim}: {CAPABILITY_DIMENSION_EVIDENCE_HINTS.get(dim, 'use dimension-relevant metrics and mechanisms')}"
        for dim in CAPABILITY_DIMENSIONS
    ]
    return (
        "CAPABILITY EVIDENCE-REF SELECTION (mandatory for the six capability_assessment dimensions only). "
        "The representative_risk_cases pool is stratified by failure_mechanism x odd_speed_bin x odd_density_bin "
        f"(per_stratum={per_stratum}). For EACH dimension independently:\n"
        f"  Step 1 — Filter: from the FULL case list, keep cases whose mechanism, conflict_geometry, outcome, "
        "or metrics are relevant to that dimension (see dimension hints below).\n"
        f"  Step 2 — Cover strata: evidence_refs MUST include cases from at least {min_strata} distinct "
        "(diagnostic_failure_mechanism, odd.speed_bin, odd.density_bin) combinations when the filtered pool allows.\n"
        f"  Step 3 — Minimum count: cite at least {min_refs} distinct case IDs in evidence_refs when the filtered "
        "pool has that many relevant cases; otherwise cite all relevant cases found.\n"
        "  Step 4 — Metric binding: provide at least one supporting_metrics entry per evidence_ref, copying "
        "case_id, metric key, and value exactly from that case's metrics object (use JSON native types for "
        "numbers and booleans, not strings).\n"
        "  Step 5 — No lazy reuse: do NOT copy failure_patterns.supporting_case_ids into capability_assessment "
        "without checking dimension relevance; avoid citing the same case in more than two dimensions unless "
        "the assessment explains why it supports each dimension.\n"
        "  Step 6 — Level/outcome consistency: for critical_weakness or major_weakness, prefer collision or "
        "near_miss cases where the cited metrics/mechanisms show deficiency for that dimension; for "
        "generally_capable or strong, do NOT use collision-only cases as sole positive evidence — cite "
        "near_miss cases and/or ground the level in overall aggregate statistics while noting that the "
        "representative pool is adverse-only; for mixed, cite at least one deficient and one relatively "
        "competent case when available.\n"
        "Dimension relevance hints:\n"
        + "\n".join(hint_lines)
        + "\n"
    )


def _failure_analysis_system_prompt() -> str:
    return (
        "You are an autonomous-driving evaluation scientist. The deterministic module already "
        "computed WHAT HAPPENED (rates, patterns, ODD cells, cases). Your job is WHAT IT MEANS: "
        "cross-scenario diagnosis and capability judgment for ONE planner. "
        "You are an analyst, not a table restatement engine.\n\n"
        "REASONING TASK (do this for every capability assessment):\n"
        "1) Contrast — where the strength/limitation concentrates vs where it is weaker or absent "
        "(ODD density/speed, crash category, interaction type, behavior_risk cells).\n"
        "2) Tension — reconcile conflicting signals (e.g. progress vs completion; low exclusive "
        "pattern rate vs high collision; comfort collapse vs successful near-misses).\n"
        "3) Implication — state what this implies about the planner capability (not a metric dump).\n"
        "4) Uncertainty — what missing/low-support evidence would change the level.\n\n"
        "ANTI-RESTATEMENT RULE (strict):\n"
        "Do NOT write assessments that mainly restate metrics "
        "('collision rate is X%, delayed rate is Y%, TTC is Z'). "
        "Exact numbers belong only in supporting_evidence. "
        "In assessment prose, use qualitative magnitudes (elevated/modest/extreme) and lead with "
        "comparative operating-condition claims (where the failure concentrates vs where it does not).\n\n"
        "DEPTH REQUIREMENT:\n"
        "Each capability assessment must contain at least one contrast across ODDs/crash categories "
        "and one explicit tension between two evidence families. "
        "failure_analysis must contrast typical_failure vs severe_failure vs successful_avoidance "
        "and explain co-occurrence with operating conditions; mechanism claims stay unverified.\n\n"
        "CROSS-SCENARIO RULE:\n"
        "Use scenario_statistics, odd_high_risk_cells, behavior_risk, and representative_cases. "
        "Name concrete conditions present in the package. No extreme single case may dominate. "
        "Do not invent absent conditions.\n\n"
        "MAGNITUDE REFERENCE (optional):\n"
        "capability_population_reference.reference_level is a soft ordinal hint — use only as a "
        "magnitude check; do not copy it by default. If your level differs, set "
        "disagrees_with_deterministic_prior=true and briefly explain.\n\n"
        "RISK ANTICIPATION HARD-SET RULE (strict):\n"
        "Consult risk_anticipation_discrimination. On hard/crash sets with elevated "
        "collision_rate, do NOT assign critical_weakness from collision alone. "
        "critical_weakness requires anticipation-specific collapse "
        "(high delayed_or_absent_response.adverse_rate jointly with high collision). "
        "When collision is elevated, discriminate primarily with delayed/absent adverse_rate "
        "and minimum_ttc.median. soft_advisory_band is a check only — never copy it.\n\n"
        "experiment_result_summary: one planner-specific sentence on the capability "
        "trade-off and its safety-relevant behavioral implication. "
        "Do NOT reuse a shared template across planners "
        "(avoid identical 'strong X but limited by Y / yielding Z' wording). "
        "Ground phrasing in the six capability dimensions and their attested ODD/"
        "scenario foci; do not invent unsupported mechanisms "
        "(e.g. negotiation internals or implementation details).\n\n"
        "EPISTEMIC RULES:\n"
        "Never invent metrics/IDs/planner internals. Separate observation, interpretation, "
        "capability judgment, and unverified hypotheses. Do not claim causality. Do not rank "
        "planners.\n\n"
        "OUTPUT LEVELS: for each of the six dimensions choose exactly one of "
        "{critical_weakness, major_weakness, mixed, generally_capable, strong}. "
        "Rubric: critical_weakness = " + CAPABILITY_LEVEL_RUBRIC["critical_weakness"] + " "
        "major_weakness = " + CAPABILITY_LEVEL_RUBRIC["major_weakness"] + " "
        "mixed = " + CAPABILITY_LEVEL_RUBRIC["mixed"] + " "
        "generally_capable = " + CAPABILITY_LEVEL_RUBRIC["generally_capable"] + " "
        "strong = " + CAPABILITY_LEVEL_RUBRIC["strong"] + "\n\n"
        "CITATION COMPLIANCE (secondary to reasoning):\n"
        "supporting_evidence: ≥2 exact IDs from DISTINCT grounding families "
        "{safety, temporal, interaction, control, efficiency} with values copied exactly. "
        "Same-family pairs do not count. Prior components alone are invalid support.\n\n"
        "Return one valid JSON object only."
    )


def _failure_analysis_user_prompt(evidence: Dict[str, Any]) -> str:
    capability_entry_schema = {
        "level": "critical_weakness|major_weakness|mixed|generally_capable|strong",
        "assessment": (
            "2–4 sentences of cross-scenario reasoning: contrast (where it fails vs holds) + "
            "tension between signals + capability implication + uncertainty. "
            "Do NOT open with restated percentages; numbers live in supporting_evidence. "
            "Name ≥1 package operating condition when available."
        ),
        "supporting_evidence": [
            {
                "evidence_id": "exact path from population/failure/scenario/ODD/behavior evidence",
                "value": 0.25,
            },
            {
                "evidence_id": (
                    "second path from a DIFFERENT grounding family "
                    "(safety|temporal|interaction|control|efficiency)"
                ),
                "value": 0.1,
            },
        ],
        "confidence": "high|medium|low",
        "disagrees_with_deterministic_prior": False,
        "disagreement_rationale": (
            "optional; if level differs from capability_population_reference.reference_level, "
            "briefly explain the package-based reason (prior is magnitude reference only)"
        ),
    }
    schema = {
        "experiment_result_summary": (
            "one planner-specific sentence on the capability trade-off and "
            "safety-relevant behavior; no shared boilerplate across planners; "
            "no unsupported mechanisms beyond the six capability dimensions"
        ),
        "observed_evidence": [
            {
                "evidence_id": "pivotal evidence path that actually drove a judgment",
                "value": "exact source value",
            }
        ],
        "capability_assessment": {
            **{dim: dict(capability_entry_schema) for dim in CAPABILITY_DIMENSIONS},
        },
        "failure_analysis": (
            "cross-scenario narrative: contrast typical_failure / severe_failure / "
            "successful_avoidance; link mechanisms to ODD co-occurrence; label mechanisms unverified"
        ),
        "limitations": ["missing or low-support evidence that constrains interpretation"],
    }
    return (
        "Return the required JSON object.\n"
        "Reason first: for each capability write assessment with "
        "contrast→tension→implication→uncertainty (NO percentage laundry-list), then choose "
        "level from the full package. "
        "capability_population_reference.reference_level is an optional magnitude reference "
        "only — do not copy it by default.\n"
        "observed_evidence: at most 5 pivotal citations.\n"
        "supporting_evidence: ≥2 distinct grounding families; copy IDs/values exactly; "
        "avoid same-family pairs; comfort pairs control+non-control; mobility pairs "
        "efficiency+non-efficiency.\n"
        "For risk_anticipation: follow risk_anticipation_discrimination — do not assign "
        "critical_weakness merely because the set is hard; require anticipation-specific "
        "collapse (delayed/absent + collision jointly); soft_advisory_band is advisory only.\n"
        "Use representative_cases only inside failure_analysis.\n\n"
        f"REQUIRED_SCHEMA:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"EVIDENCE_PACKAGE:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )


def _improvement_guidance_system_prompt() -> str:
    return (
        "You are an autonomous-driving Planner Improvement Agent. Convert the population-grounded "
        "diagnosis into specific, testable, prioritized guidance. "
        "Prefer guidance that targets recurring cross-scenario weaknesses "
        "(ODD / crash-category / interaction concentrations) rather than restating rates.\n\n"
        "ANTI-RESTATEMENT RULE (strict):\n"
        "observed_limitation, possible_mechanism, and recommended_improvement must NOT be "
        "metric paraphrases ('collision rate is X%'). Put exact IDs/values in "
        "supporting_evidence_ids / evaluation_metrics only. Prose must explain WHERE the "
        "limitation concentrates, WHAT capability fails, and WHY a concrete intervention is "
        "the discriminating next step.\n\n"
        "ACTIONABLE IMPROVEMENT RULE (strict):\n"
        "recommended_improvement must be concrete and "
        "plausible given the diagnosed failure pattern — not abstract slogans. "
        "FORBIDDEN vague phrases include: 'improve risk anticipation', 'enhance safety "
        "capability', 'improve interaction reasoning', 'prioritize robustness'. "
        "Connect: observed failure pattern → planner limitation → targeted improvement "
        "direction. Prefer directions such as training-data augmentation, additional "
        "supervision signals, prediction/risk-representation changes, decision objectives, "
        "trajectory optimization, control constraints, or interaction modeling — when "
        "supported by evidence. Do NOT invent planner internals absent from evaluation "
        "results. Emphasize ONE primary improvement direction. "
        "Do NOT append validation/preservation trailers such as "
        "'while evaluating naturalistic … to avoid degradation', "
        "'while checking that naturalistic … does not regress', or "
        "'while preserving … under naturalistic conditions'. "
        "Keep guidance planner-specific and free of shared boilerplate.\n\n"
        "MECHANISM RULE:\n"
        "possible_mechanism must describe a candidate explanation with may / could / "
        "is consistent with. Explain observed behavior (delayed response, insufficient "
        "braking, unstable evasive behavior, inadequate interaction handling). Never claim "
        "a specific planner module is verified unless directly available.\n\n"
        "BEHAVIOR FOCUS:\n"
        "observed_limitation and performance_limiting_scenarios must describe collision "
        "interaction patterns (conflict type, response timing, braking/lateral avoidance, "
        "control stability, density/speed when characterizing behavior) — not raw field "
        "labels or map locations.\n\n"
        "DEPTH REQUIREMENT:\n"
        "Provide at least three prioritized improvement_guidance items covering distinct "
        "affected capabilities or distinct operating-condition concentrations when evidence "
        "supports them. Each item: observed limitation (cross-scene), unverified mechanism, "
        "affected capability, target component/module, recommended improvement, "
        "performance-limiting scenarios, evaluation_metrics, regression_constraints.\n\n"
        "Use ONLY metric names / evidence paths present in the supplied evidence package. "
        "Never invent generic metrics. Do not invent planner internals; when unavailable, name a "
        "capability-level component and state that limitation. Return JSON only."
    )


def _improvement_guidance_user_prompt(
    evidence: Dict[str, Any], failure_analysis: Dict[str, Any]
) -> str:
    schema = {
        "improvement_guidance": [
            {
                "priority": 1,
                "observed_limitation": (
                    "2–3 sentences: recurring cross-scene limitation with operating-condition "
                    "contrast; NO raw-rate restatement"
                ),
                "supporting_evidence_ids": ["exact deterministic evidence path"],
                "possible_mechanism": (
                    "2–3 sentences: write directly with may/might/could/is consistent with; "
                    "no 'One unverified interpretation is...' framing; "
                    "link limitation to capability without claiming causality"
                ),
                "affected_capability": "one of the six capability dimensions",
                "target_component_or_module": (
                    "specific target, or capability-level target if internals unavailable"
                ),
                "recommended_improvement": (
                    "2–3 sentences: ONE concrete technical action linked to the "
                    "diagnosed failure pattern (e.g. response-time supervision, "
                    "trajectory smoothing, interaction modeling) + why it "
                    "discriminates this limitation; NOT a capability slogan"
                ),
                "performance_limiting_scenarios": [
                    {
                        "scenario_description": (
                            "concise behavior-oriented statement: conflict type + "
                            "interaction/response pattern (+ density/speed when useful); "
                            "never field labels like 'location: ...' or 'speed bin: ...'"
                        ),
                        "source_evidence_ids": [
                            "exact odd_high_risk_cells.<dim>=<value>.collision_rate "
                            "OR failure_statistics.patterns.<name>.adverse_rate "
                            "(NO wildcards such as odd_high_risk_cells.*)"
                        ],
                    }
                ],
                "evaluation_metrics": [
                    {
                        "metric": "exact evidence path or registry metric from the package",
                        "expected_direction": "lower|higher",
                        "role": "primary|secondary",
                        "source_evidence_id": "exact deterministic evidence path",
                    }
                ],
                "regression_constraints": [
                    {
                        "metric": "exact evidence path or registry metric from the package",
                        "requirement": "non-inferior",
                    }
                ],
                "confidence": "high|medium|low",
            }
        ],
    }
    return (
        "Generate ≥3 prioritized, actionable guidance items that address DISTINCT recurring "
        "cross-scenario weaknesses from the diagnosis. Rank by expected diagnostic/value impact "
        "(priority=1 is highest). Each item must satisfy every schema field. "
        "supporting_evidence_ids, evaluation_metrics.source_evidence_id, and "
        "performance_limiting_scenarios.source_evidence_ids must refer to supplied deterministic "
        "population/ODD/scenario/failure/behavior evidence. Do not use a representative case as "
        "the sole rationale. "
        "Prose fields must reason about operating-condition concentration and capability failure — "
        "do NOT mainly restate percentages. "
        "Label scenarios as Performance-limiting scenarios and phrase them as collision-behavior "
        "patterns (not raw evidence field dumps). "
        "Do not invent metric names absent from the evidence package. "
        "performance_limiting_scenarios.source_evidence_ids must be EXACT citable paths "
        "(e.g. odd_high_risk_cells.odd_density_bin=dense (≥7 agents).collision_rate); "
        "never emit wildcards such as odd_high_risk_cells.*.\n\n"
        f"REQUIRED_SCHEMA:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"FAILURE_ANALYSIS:\n{json.dumps(failure_analysis, ensure_ascii=False, indent=2)}\n\n"
        f"SOURCE_EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )


def _collect_case_reference_warnings(obj: Any, valid_case_ids: Sequence[str]) -> List[str]:
    valid = set(valid_case_ids)
    seen: List[str] = []

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                _walk(child)
        elif isinstance(value, list):
            for child in value:
                _walk(child)
        elif isinstance(value, str):
            seen.extend(re.findall(r"case_\d{3}", value))

    _walk(obj)
    return sorted({f"unknown case reference: {ref}" for ref in seen if ref not in valid})


def _values_match(candidate: Any, source: Any, tol: float = 1e-8) -> bool:
    """Compare an LLM-returned metric value against the source evidence value.

    Numeric values are compared with a small absolute/relative tolerance; booleans and other
    types must match exactly (bool is checked before numeric since ``bool`` is an ``int`` subclass).
    String forms of numbers (common LLM output) are coerced before numeric comparison.
    """
    if isinstance(candidate, bool) or isinstance(source, bool):
        return candidate is source
    if candidate is None or source is None:
        return candidate is source
    if isinstance(candidate, str) and isinstance(source, (int, float)):
        try:
            candidate = float(candidate)
        except (TypeError, ValueError):
            return False
    if isinstance(source, str) and isinstance(candidate, (int, float)):
        try:
            source = float(source)
        except (TypeError, ValueError):
            return False
    if isinstance(candidate, (int, float)) and isinstance(source, (int, float)):
        try:
            return math.isclose(float(candidate), float(source), rel_tol=tol, abs_tol=tol)
        except (TypeError, ValueError):
            return False
    return candidate == source


def _validate_capability_assessments(
    failure_analysis: Dict[str, Any], evidence_package: Dict[str, Any]
) -> Dict[str, Any]:
    """Deterministically validate/normalize the six ``capability_assessment`` dimensions.

    Invalid metric/case references are dropped and recorded as structured warnings.
    Missing/older fields are backfilled for backward compatibility.
    """
    cases = evidence_package.get("representative_risk_cases") or []
    case_metrics: Dict[str, Dict[str, Any]] = {}
    valid_case_ids: set = set()
    for item in cases:
        cid = item.get("case_id")
        if not cid:
            continue
        valid_case_ids.add(cid)
        case_metrics[cid] = item.get("metrics") or {}

    warnings: List[str] = []
    invalid_case_references: List[str] = []
    invalid_metric_references: List[str] = []
    metric_value_mismatches: List[str] = []
    unit_corrections: List[str] = []

    n_supporting_metrics_total = 0
    n_valid_supporting_metrics = 0
    n_valid_capabilities = 0

    # Auditability counters (kept separate from validation-removal counters above).
    n_case_refs_total = 0
    n_case_refs_valid = 0
    n_metric_key_valid = 0  # case_id + metric key both exist, irrespective of value match
    n_value_match_among_valid_key = 0
    n_coverage_dims = 0

    raw_capability = (failure_analysis or {}).get("capability_assessment")
    if not isinstance(raw_capability, dict):
        raw_capability = {}
        warnings.append("capability_assessment missing or malformed; all dimensions defaulted")

    validated: Dict[str, Any] = {}
    for dim in CAPABILITY_DIMENSIONS:
        raw = raw_capability.get(dim)
        if not isinstance(raw, dict):
            warnings.append(f"{dim}: capability dimension missing; defaulted to empty/mixed/low")
            raw = {}

        assessment_text = raw.get("assessment")
        if not isinstance(assessment_text, str):
            assessment_text = "" if assessment_text is None else str(assessment_text)

        level = raw.get("capability_level")
        if level not in CAPABILITY_LEVELS:
            if level is None:
                warnings.append(f"{dim}: missing capability_level; inferred 'mixed'")
            else:
                warnings.append(f"{dim}: invalid capability_level {level!r} replaced with 'mixed'")
            level = "mixed"

        confidence = raw.get("confidence")
        if confidence not in CONFIDENCE_LEVELS:
            if confidence is None:
                warnings.append(f"{dim}: missing confidence; defaulted to 'low'")
            else:
                warnings.append(f"{dim}: invalid confidence {confidence!r} replaced with 'low'")
            confidence = "low"

        evidence_refs_raw = raw.get("evidence_refs")
        if evidence_refs_raw is None:
            warnings.append(f"{dim}: missing evidence_refs; defaulted to empty list")
            evidence_refs_raw = []
        elif not isinstance(evidence_refs_raw, list):
            warnings.append(f"{dim}: evidence_refs was not a list; defaulted to empty list")
            evidence_refs_raw = []

        valid_evidence_refs: List[str] = []
        for ref in evidence_refs_raw:
            ref_str = str(ref)
            n_case_refs_total += 1
            if ref_str in valid_case_ids:
                n_case_refs_valid += 1
                if ref_str not in valid_evidence_refs:
                    valid_evidence_refs.append(ref_str)
            else:
                invalid_case_references.append(f"{dim}.evidence_refs: unknown case_id '{ref_str}'")

        supporting_metrics_raw = raw.get("supporting_metrics")
        if supporting_metrics_raw is None:
            warnings.append(f"{dim}: missing supporting_metrics; defaulted to empty list")
            supporting_metrics_raw = []
        elif not isinstance(supporting_metrics_raw, list):
            warnings.append(f"{dim}: supporting_metrics was not a list; defaulted to empty list")
            supporting_metrics_raw = []

        valid_supporting_metrics: List[Dict[str, Any]] = []
        for entry in supporting_metrics_raw:
            n_supporting_metrics_total += 1
            if not isinstance(entry, dict):
                invalid_metric_references.append(f"{dim}.supporting_metrics: entry is not an object")
                continue
            cid = entry.get("case_id")
            n_case_refs_total += 1
            metric_key = entry.get("metric")
            value = entry.get("value")
            if cid not in valid_case_ids:
                invalid_case_references.append(f"{dim}.supporting_metrics: unknown case_id '{cid}'")
                continue
            n_case_refs_valid += 1
            metrics_for_case = case_metrics.get(cid) or {}
            if metric_key not in metrics_for_case:
                invalid_metric_references.append(
                    f"{dim}.supporting_metrics: unknown metric key '{metric_key}' for case '{cid}'"
                )
                continue
            n_metric_key_valid += 1
            source_value = metrics_for_case.get(metric_key)
            if not _values_match(value, source_value):
                metric_value_mismatches.append(
                    f"{dim}.supporting_metrics: value mismatch for case '{cid}' metric '{metric_key}' "
                    f"(reported {value!r}, source {source_value!r})"
                )
                continue
            n_value_match_among_valid_key += 1
            expected_unit = METRIC_UNIT_MAP.get(metric_key)
            provided_unit = entry.get("unit")
            if provided_unit != expected_unit:
                unit_corrections.append(
                    f"{dim}.supporting_metrics: unit for '{metric_key}' corrected from "
                    f"{provided_unit!r} to {expected_unit!r}"
                )
            valid_supporting_metrics.append(
                {"case_id": cid, "metric": metric_key, "value": source_value, "unit": expected_unit}
            )
            n_valid_supporting_metrics += 1

        if not valid_evidence_refs and not valid_supporting_metrics and confidence != "low":
            warnings.append(f"{dim}: no valid supporting evidence remains; confidence downgraded to 'low'")
            confidence = "low"

        validated[dim] = {
            "assessment": assessment_text,
            "capability_level": level,
            "evidence_refs": valid_evidence_refs,
            "supporting_metrics": valid_supporting_metrics,
            "confidence": confidence,
        }

        if valid_evidence_refs and valid_supporting_metrics:
            n_coverage_dims += 1

        if assessment_text.strip() and level in CAPABILITY_LEVELS and confidence in CONFIDENCE_LEVELS:
            n_valid_capabilities += 1

    capability_validation = {
        "n_capabilities": len(CAPABILITY_DIMENSIONS),
        "n_valid_capabilities": n_valid_capabilities,
        "n_supporting_metrics": n_supporting_metrics_total,
        "n_valid_supporting_metrics": n_valid_supporting_metrics,
        "invalid_case_references": sorted(set(invalid_case_references)),
        "invalid_metric_references": sorted(set(invalid_metric_references)),
        "metric_value_mismatches": sorted(set(metric_value_mismatches)),
        "unit_corrections": sorted(set(unit_corrections)),
        "warnings": warnings,
    }

    llm_auditability = {
        "valid_case_reference_rate": (n_case_refs_valid / n_case_refs_total) if n_case_refs_total else 1.0,
        "valid_metric_reference_rate": (
            n_metric_key_valid / n_supporting_metrics_total
        ) if n_supporting_metrics_total else 1.0,
        "metric_value_match_rate": (
            n_value_match_among_valid_key / n_metric_key_valid
        ) if n_metric_key_valid else 1.0,
        "capability_evidence_coverage": n_coverage_dims / len(CAPABILITY_DIMENSIONS),
        "capability_schema_completeness": n_valid_capabilities / len(CAPABILITY_DIMENSIONS),
    }

    return {
        "capability_assessment": validated,
        "capability_validation": capability_validation,
        "llm_auditability": llm_auditability,
    }


def _flatten_citable_evidence(
    evidence_package: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build evidence_id -> value map for LLM citation validation.

    Allows population/failure/scenario stats, ODD cells, behavior risk, and
    capability-reference components — not only the deterministic prior components.
    """
    index: Dict[str, Any] = {}

    def _index_row(path: str, item: Mapping[str, Any]) -> None:
        for leaf_key in (
            "collision_rate", "coll_rate", "n", "n_coll", "value",
            "rate", "median", "mean", "p90", "count", "share",
            "adverse_rate", "population_rate", "classified_failure_share",
        ):
            if leaf_key in item and not isinstance(item[leaf_key], (dict, list)):
                index[f"{path}.{leaf_key}"] = item[leaf_key]
        if "dimension" in item and "value" in item:
            dim = item.get("dimension")
            cell = item.get("value")
            if dim is not None and cell is not None:
                for leaf_key in ("collision_rate", "coll_rate", "n_coll", "n"):
                    if leaf_key in item and not isinstance(item[leaf_key], (dict, list)):
                        index[f"odd_high_risk_cells.{dim}={cell}.{leaf_key}"] = item[leaf_key]
        tag = item.get("behavior") or item.get("behavior_tag")
        if tag is not None:
            for leaf_key in ("coll_rate", "collision_rate", "n_coll", "n"):
                if leaf_key in item and not isinstance(item[leaf_key], (dict, list)):
                    index[f"behavior_risk.{tag}.{leaf_key}"] = item[leaf_key]

    def _walk(node: Any, prefix: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                _walk(value, path)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                item_path = f"{prefix}[{i}]" if prefix else f"[{i}]"
                if isinstance(item, Mapping):
                    _index_row(item_path, item)
                    _walk(item, item_path)
                elif item is None or isinstance(item, (bool, int, float, str)):
                    index[item_path] = item
        elif prefix and (node is None or isinstance(node, (bool, int, float, str))):
            index[prefix] = node

    for top_key in (
        "population_statistics",
        "failure_statistics",
        "scenario_statistics",
        "odd_high_risk_cells",
        "behavior_risk",
        "feature_separation",
        "collision_timing",
        "overall",
        "data_quality",
        "missing_evidence_fields",
    ):
        if top_key in evidence_package:
            _walk(evidence_package.get(top_key), top_key)

    # Always include capability_population_reference.components evidence_ids.
    references = evidence_package.get("capability_population_reference") or {}
    if isinstance(references, Mapping):
        for dim, payload in references.items():
            if not isinstance(payload, Mapping):
                continue
            for item in payload.get("components") or []:
                if isinstance(item, Mapping) and item.get("evidence_id"):
                    index[str(item["evidence_id"])] = item.get("value")
            if "reference_level" in payload:
                index[f"capability_population_reference.{dim}.reference_level"] = payload.get(
                    "reference_level"
                )
            if "population_reference_score" in payload:
                index[
                    f"capability_population_reference.{dim}.population_reference_score"
                ] = payload.get("population_reference_score")
    return index


def _as_rate(value: Any) -> Optional[float]:
    rate = _as_finite_float(value)
    if rate is None:
        return None
    return float(rate)


def _risk_anticipation_magnitude_level(evidence_package: Mapping[str, Any]) -> Optional[str]:
    """Compatibility wrapper around evidence.risk_anticipation_joint_magnitude_level."""
    from agent.eval.crashsim_evidence import risk_anticipation_joint_magnitude_level

    return risk_anticipation_joint_magnitude_level(
        evidence_package.get("population_statistics") or {},
        evidence_package.get("failure_statistics") or {},
    )


def _validate_population_capability_assessments(
    failure_analysis: Dict[str, Any], evidence_package: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate evidence citations; measure (but do not enforce) prior agreement.

    Consistency with ``capability_population_reference.reference_level`` is an
    audit metric only. The LLM is expected to reason over the full package and
    may disagree with the deterministic prior when broader evidence supports it.

    Hard-set risk_anticipation collapse is addressed at prompt/evidence time via
    ``risk_anticipation_discrimination`` (joint magnitude advisory). This
    validator never rewrites LLM levels.
    """
    raw_assessment = (failure_analysis or {}).get("capability_assessment") or {}
    references = evidence_package.get("capability_population_reference") or {}
    citable = _flatten_citable_evidence(evidence_package)
    level_rank = {level: index for index, level in enumerate(CAPABILITY_LEVELS)}
    risk_soft_band = _risk_anticipation_magnitude_level(evidence_package)
    validated: Dict[str, Any] = {}
    warnings: List[str] = []
    invalid_evidence_references: List[str] = []
    value_mismatches: List[str] = []
    consistency: Dict[str, Any] = {}
    n_valid = 0
    n_evidence = 0
    n_valid_evidence = 0
    n_disagreements = 0
    n_risk_overharsh_vs_soft_band = 0

    for dimension in CAPABILITY_DIMENSIONS:
        raw = raw_assessment.get(dimension)
        raw = raw if isinstance(raw, dict) else {}
        level = raw.get("level", raw.get("capability_level"))
        if level not in CAPABILITY_LEVELS:
            warnings.append(f"{dimension}: missing/invalid level")
            level = None
        assessment = raw.get("assessment")
        assessment = assessment if isinstance(assessment, str) else ""
        confidence = raw.get("confidence")
        if confidence not in CONFIDENCE_LEVELS:
            warnings.append(f"{dimension}: missing/invalid confidence")
            confidence = "low"

        raw_support = raw.get("supporting_evidence") or []
        valid_support: List[Dict[str, Any]] = []
        for item in raw_support:
            n_evidence += 1
            if not isinstance(item, dict):
                invalid_evidence_references.append(
                    f"{dimension}: supporting_evidence entry is not an object"
                )
                continue
            evidence_id = item.get("evidence_id")
            if evidence_id not in citable:
                invalid_evidence_references.append(
                    f"{dimension}: unknown population evidence_id {evidence_id!r}"
                )
                continue
            source_value = citable[evidence_id]
            if not _values_match(item.get("value"), source_value):
                value_mismatches.append(
                    f"{dimension}: {evidence_id} reported {item.get('value')!r}, "
                    f"source {source_value!r}"
                )
                continue
            valid_support.append(
                {"evidence_id": evidence_id, "value": source_value}
            )
            n_valid_evidence += 1

        # Soft audit only: flag over-harsh risk_anticipation vs joint band; never rewrite.
        overharsh_vs_soft_band = False
        if (
            dimension == "risk_anticipation"
            and level in level_rank
            and risk_soft_band in level_rank
            and level_rank[level] < level_rank[risk_soft_band]
        ):
            overharsh_vs_soft_band = True
            n_risk_overharsh_vs_soft_band += 1
            warnings.append(
                f"{dimension}: LLM level {level!r} is stricter than soft advisory "
                f"band {risk_soft_band!r} (prompt-time discrimination card; level kept)"
            )
            if confidence == "high":
                confidence = "medium"

        reference = references.get(dimension) or {}
        reference_level = reference.get("reference_level")
        if level in level_rank and reference_level in level_rank:
            distance = abs(level_rank[level] - level_rank[reference_level])
            consistency_score = max(0.0, 1.0 - distance / 4.0)
        else:
            distance = None
            consistency_score = None

        disagrees = bool(raw.get("disagrees_with_deterministic_prior", False))
        if level is not None and reference_level is not None:
            disagrees = level != reference_level
        disagreement_rationale = raw.get("disagreement_rationale")
        if not isinstance(disagreement_rationale, str):
            disagreement_rationale = ""
        if not disagrees:
            disagreement_rationale = ""
        if disagrees:
            n_disagreements += 1
        if disagrees and not disagreement_rationale.strip():
            warnings.append(
                f"{dimension}: disagrees with deterministic prior but missing disagreement_rationale"
            )
            if confidence == "high":
                confidence = "medium"

        consistency[dimension] = {
            "score": consistency_score,
            "llm_level": level,
            "population_reference_level": reference_level,
            "ordinal_distance": distance,
            "population_reference_score": reference.get("population_reference_score"),
            "evidence_coverage": reference.get("evidence_coverage"),
            "disagrees_with_deterministic_prior": disagrees,
            "soft_advisory_band": (
                risk_soft_band if dimension == "risk_anticipation" else None
            ),
            "overharsh_vs_soft_advisory_band": (
                overharsh_vs_soft_band if dimension == "risk_anticipation" else False
            ),
            "note": (
                "Audit only: disagreement with the deterministic prior is allowed when "
                "broader package evidence is cited. risk_anticipation soft_advisory_band "
                "is informational; LLM levels are never rewritten post hoc."
            ),
        }
        if not valid_support:
            confidence = "low"
            warnings.append(f"{dimension}: no valid population supporting evidence")
        if level and assessment.strip() and valid_support:
            n_valid += 1
        validated[dimension] = {
            "level": level,
            "assessment": assessment,
            "supporting_evidence": valid_support,
            "confidence": confidence,
            "disagrees_with_deterministic_prior": disagrees,
            "disagreement_rationale": disagreement_rationale if disagrees else "",
        }

    finite_scores = [
        cell["score"] for cell in consistency.values()
        if isinstance(cell.get("score"), (int, float))
    ]
    return {
        "capability_assessment": validated,
        "capability_validation": {
            "n_capabilities": len(CAPABILITY_DIMENSIONS),
            "n_valid_capabilities": n_valid,
            "n_supporting_evidence": n_evidence,
            "n_valid_supporting_evidence": n_valid_evidence,
            "n_disagreements_with_prior": n_disagreements,
            "n_risk_anticipation_overharsh_vs_soft_band": n_risk_overharsh_vs_soft_band,
            "invalid_evidence_references": sorted(set(invalid_evidence_references)),
            "evidence_value_mismatches": sorted(set(value_mismatches)),
            "warnings": warnings,
        },
        "llm_auditability": {
            "diagnosis_evidence_consistency_by_capability": consistency,
            "mean_diagnosis_evidence_consistency": (
                float(np.mean(finite_scores)) if finite_scores else None
            ),
            "consistency_definition": (
                "1 - ordinal_distance/4 between the LLM level and deterministic "
                "population-reference prior; disagreement is allowed and audited, not rejected"
            ),
        },
    }


def _capability_level_phrase(level: str) -> str:
    return {
        "critical_weakness": "critical weakness",
        "major_weakness": "major weakness",
        "mixed": "mixed performance",
        "generally_capable": "generally capable performance",
        "strong": "strong performance",
    }.get(level, level)


def rebuild_experiment_result_summary(
    planner_name: Optional[str],
    capability_assessment: Mapping[str, Any],
) -> str:
    """Build a short literal summary from validated capability levels for panel c."""
    planner = str(planner_name or "Planner").strip() or "Planner"
    buckets: Dict[str, List[str]] = {level: [] for level in CAPABILITY_LEVELS}
    for dim in CAPABILITY_DIMENSIONS:
        entry = capability_assessment.get(dim) if isinstance(capability_assessment, Mapping) else None
        entry = entry if isinstance(entry, Mapping) else {}
        level = entry.get("level")
        if level in buckets:
            buckets[level].append(CAPABILITY_DIMENSION_DISPLAY_NAMES.get(dim, dim))

    parts: List[str] = []
    for level in CAPABILITY_LEVELS:
        names = buckets.get(level) or []
        if not names:
            continue
        phrase = _capability_level_phrase(level)
        if len(names) == 1:
            parts.append(f"{phrase} in {names[0].lower()}")
        elif len(names) == 2:
            parts.append(f"{phrase} in {names[0].lower()} and {names[1].lower()}")
        else:
            parts.append(
                f"{phrase} in {', '.join(n.lower() for n in names[:-1])}, and {names[-1].lower()}"
            )
    if not parts:
        return f"{planner} capability profile could not be summarized from validated levels."
    if len(parts) == 1:
        body = parts[0]
    elif len(parts) == 2:
        body = f"{parts[0]}, and {parts[1]}"
    else:
        body = f"{', '.join(parts[:-1])}, and {parts[-1]}"
    return f"{planner} exhibits {body}."


def _render_llm_interpretable_markdown(bundle: Dict[str, Any]) -> str:
    analysis = bundle.get("failure_analysis") or {}
    guidance = bundle.get("improvement_guidance") or {}
    lines = ["# LLM-Enhanced Interpretable AV Evaluation", ""]
    lines.append(
        "The LLM analyzes deterministic rollout evidence, forms explicitly qualified failure hypotheses, and proposes planner improvements. "
        "Its interpretations and recommendations remain auditable against the referenced cases and metrics."
    )

    def _as_lines(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return "; ".join(str(v) for v in value) or "n/a"
        return str(value) if value not in (None, "") else "n/a"

    # Task 1: experiment-result summary (renamed from executive_assessment; keep fallback).
    summary_block = analysis.get("experiment_result_summary") or analysis.get("executive_assessment") or {}
    lines.extend(["", "## Planner diagnosis (experiment-result summary)", ""])
    if isinstance(summary_block, str):
        lines.append(f"- **Overall assessment:** {summary_block}")
    elif summary_block:
        lines.append(f"- **Overall assessment:** {summary_block.get('summary', 'n/a')}")
        lines.append(f"- **Planner characterization:** {summary_block.get('planner_characterization', 'n/a')}")
        if summary_block.get("key_strengths"):
            lines.append(f"- **Key strengths:** {_as_lines(summary_block.get('key_strengths'))}")
        if summary_block.get("key_weaknesses"):
            lines.append(f"- **Key weaknesses:** {_as_lines(summary_block.get('key_weaknesses'))}")
        lines.append(f"- **Most critical limitation:** {summary_block.get('most_critical_limitation', 'n/a')}")
        lines.append(f"- **Confidence:** {summary_block.get('confidence', 'n/a')}")

    # Capability assessment: rubric-constrained qualitative level per dimension, grounded in
    # deterministically validated supporting metrics (see _validate_capability_assessments).
    capability_assessment = analysis.get("capability_assessment") or {}
    if capability_assessment:
        lines.extend(["", "## Capability assessment", ""])
        for dim in CAPABILITY_DIMENSIONS:
            entry = capability_assessment.get(dim)
            if not isinstance(entry, dict):
                continue
            display_name = CAPABILITY_DIMENSION_DISPLAY_NAMES.get(dim, dim)
            lines.append(f"### {display_name}")
            lines.append("")
            level = entry.get("level", entry.get("capability_level", "n/a"))
            level_display = str(level).replace("_", " ") if level else "n/a"
            lines.append(f"- **Capability level:** {level_display}")
            lines.append(f"- **Assessment:** {entry.get('assessment', 'n/a') or 'n/a'}")
            supporting_metrics = entry.get("supporting_evidence") or entry.get("supporting_metrics") or []
            if supporting_metrics:
                lines.append("- **Supporting population evidence:**")
                for metric_item in supporting_metrics:
                    case_id = metric_item.get("case_id")
                    metric_key = metric_item.get("evidence_id", metric_item.get("metric", "?"))
                    value = metric_item.get("value")
                    unit = metric_item.get("unit")
                    if isinstance(value, bool):
                        value_str = "true" if value else "false"
                    elif isinstance(value, float):
                        value_str = f"{value:.2f}"
                    else:
                        value_str = str(value)
                    unit_str = f" {unit}" if unit else ""
                    prefix = f"{case_id}: " if case_id else ""
                    lines.append(f"  - {prefix}{metric_key} = {value_str}{unit_str}")
            else:
                lines.append("- **Supporting population evidence:** none")
            lines.append(f"- **Confidence:** {entry.get('confidence', 'n/a')}")
            lines.append("")

    # Task 2: cross-scene failure patterns, stratified into observed evidence vs diagnostic inference.
    patterns = analysis.get("failure_patterns") or []
    if patterns:
        lines.extend(["", "## Cross-scene failure patterns", ""])
        for pattern in patterns:
            lines.append(f"### {pattern.get('pattern_id', '?')} — {pattern.get('name', 'Unnamed pattern')}")
            lines.append("")
            observed = pattern.get("observed_evidence")
            if observed is None:
                observed = pattern.get("observation")
            lines.append(f"- **Observed evidence:** {_as_lines(observed)}")
            diagnostic = pattern.get("diagnostic_inference") or pattern.get("planner_limitation_hypothesis")
            lines.append(f"- **Diagnostic inference:** {_as_lines(diagnostic)}")
            lines.append(f"- **Supporting cases:** {', '.join(pattern.get('supporting_case_ids') or []) or 'none'}")
            lines.append(f"- **ODD conditions:** {', '.join(pattern.get('odd_conditions') or []) or 'not isolated'}")
            lines.append(f"- **Alternative explanations:** {_as_lines(pattern.get('alternative_explanations')) if pattern.get('alternative_explanations') else 'none stated'}")
            lines.append(f"- **Confidence:** {pattern.get('confidence', 'n/a')}")
            lines.append("")

    # Task 3: ODD analysis.
    odd = analysis.get("odd_weaknesses") or []
    if odd:
        lines.extend(["## LLM-identified ODD weaknesses", ""])
        for item in odd:
            observed = item.get("observed_evidence") or item.get("observed_risk")
            lines.append(
                f"- **{item.get('odd_cell', 'unknown')}:** {_as_lines(observed)} "
                f"Diagnostic inference: {item.get('diagnostic_inference', 'n/a')}. "
                f"Support/uncertainty: {item.get('support_and_uncertainty', 'n/a')}"
            )

    # Task 4: mechanism & causal-hypothesis analysis (explicit risk -> response -> outcome chains).
    causal = analysis.get("causal_hypotheses") or []
    if causal:
        lines.extend(["", "## Mechanism & causal hypotheses (unverified)", ""])
        for hyp in causal:
            lines.append(f"### {hyp.get('hypothesis_id', '?')} — {hyp.get('name', 'Unnamed hypothesis')}")
            lines.append("")
            lines.append(f"- **Linked patterns:** {', '.join(hyp.get('linked_pattern_ids') or []) or 'none'}")
            lines.append(f"- **Candidate causal chain:** {hyp.get('candidate_causal_chain', 'n/a')}")
            lines.append(f"- **Risk-onset evidence:** {_as_lines(hyp.get('risk_onset_evidence'))}")
            lines.append(f"- **Response-behavior evidence:** {_as_lines(hyp.get('response_behavior_evidence'))}")
            lines.append(f"- **Outcome evidence:** {_as_lines(hyp.get('outcome_evidence'))}")
            lines.append(f"- **Mechanism hypothesis:** {hyp.get('mechanism_hypothesis', 'n/a')}")
            lines.append(f"- **Alternative explanations:** {_as_lines(hyp.get('alternative_explanations')) if hyp.get('alternative_explanations') else 'none stated'}")
            lines.append(f"- **Discriminating intervention:** {hyp.get('discriminating_intervention', 'n/a')}")
            lines.append(f"- **Confidence:** {hyp.get('confidence', 'n/a')}")
            lines.append("")

    # Representative-case analyses stratified into the five epistemic categories.
    cases = analysis.get("representative_case_analyses") or []
    if cases:
        lines.extend(["## Representative case analyses", ""])
        for case in cases:
            lines.append(f"### {case.get('case_id', 'case')}")
            lines.append("")
            observed = case.get("observed_evidence") or case.get("evidence_used")
            lines.append(f"- **Observed evidence:** {_as_lines(observed)}")
            diagnostic = case.get("diagnostic_inference") or case.get("failure_interpretation")
            lines.append(f"- **Diagnostic inference:** {_as_lines(diagnostic)}")
            causal_h = case.get("causal_hypothesis")
            lines.append(f"- **Causal hypothesis:** {_as_lines(causal_h)}")
            alt = case.get("alternative_explanation")
            lines.append(f"- **Alternative explanation:** {_as_lines(alt)}")
            improve = case.get("improvement_direction") or case.get("safer_counterfactual_behavior")
            lines.append(f"- **Improvement direction:** {_as_lines(improve)}")
            lines.append(f"- **Confidence:** {case.get('confidence', 'n/a')}")
            lines.append("")

    guidance_obj = guidance if isinstance(guidance, dict) else {}
    strategy = guidance_obj.get("improvement_strategy") or {}
    lines.extend(["", "## Planner improvement strategy", ""])
    if strategy:
        lines.append(f"- **Overall direction:** {strategy.get('overall_direction', 'n/a')}")
        lines.append(f"- **Highest-value next step:** {strategy.get('highest_value_next_step', 'n/a')}")
        lines.append(f"- **Reason:** {strategy.get('reason', 'n/a')}")

    recommendations = (
        guidance if isinstance(guidance, list)
        else guidance_obj.get("improvement_guidance")
        or guidance_obj.get("prioritized_guidance")
        or []
    )
    for rec in recommendations:
        addressed_mechanism = rec.get("observed_limitation", rec.get("addressed_failure_mechanism", "n/a"))
        addressed_ids = (rec.get("addressed_pattern_ids") or []) + (rec.get("addressed_causal_hypothesis_ids") or [])
        lines.extend(
            [
                "",
                f"### Priority {rec.get('priority', '?')}: {rec.get('title', 'Untitled recommendation')}",
                "",
                f"- **Addressed failure mechanism:** {addressed_mechanism}"
                + (f" ({', '.join(addressed_ids)})" if addressed_ids else ""),
                f"- **Target capability:** {rec.get('affected_capability', rec.get('target_capability', 'n/a'))}",
                f"- **Target component:** {rec.get('target_component_or_module', rec.get('target_component', 'n/a'))}",
                f"- **Evidence:** {', '.join(rec.get('supporting_evidence_ids') or rec.get('evidence_refs') or []) or 'not specified'}",
                f"- **Possible mechanism:** {rec.get('possible_mechanism', rec.get('rationale', 'n/a'))}",
                f"- **Recommended improvement:** {rec.get('recommended_improvement', 'n/a')}",
                f"- **Validation metric:** {rec.get('validation_metric', 'n/a')}",
                f"- **Validation scenario:** {rec.get('validation_scenario', 'n/a')}",
                "- **Acceptance criteria:** " + "; ".join(rec.get("acceptance_criteria") or ["n/a"]),
                "- **Regression risks:** " + "; ".join(rec.get("regression_risks") or ["n/a"]),
                f"- **Confidence:** {rec.get('confidence', 'n/a')}",
            ]
        )

    plan = guidance_obj.get("targeted_evaluation_plan") or []
    if plan:
        lines.extend(["", "## Targeted validation plan", ""])
        for item in plan:
            lines.append(
                f"- **{item.get('experiment', 'Experiment')}:** {item.get('purpose', 'n/a')} "
                f"Slice: {item.get('scenes_or_odd_slice', 'n/a')}; metrics: {', '.join(item.get('metrics') or [])}."
            )

    capability_validation = bundle.get("capability_validation") or {}
    if capability_validation:
        lines.extend(["", "## Capability assessment validation", ""])
        lines.append(
            f"- **Valid capability dimensions:** {capability_validation.get('n_valid_capabilities', 'n/a')}/"
            f"{capability_validation.get('n_capabilities', 'n/a')}"
        )
        lines.append(
            f"- **Valid supporting metrics:** {capability_validation.get('n_valid_supporting_metrics', 'n/a')}/"
            f"{capability_validation.get('n_supporting_metrics', 'n/a')}"
        )
        for key in (
            "invalid_case_references",
            "invalid_metric_references",
            "metric_value_mismatches",
            "unit_corrections",
            "warnings",
        ):
            items = capability_validation.get(key) or []
            if items:
                lines.append(f"- **{key.replace('_', ' ').capitalize()}:** {', '.join(str(i) for i in items)}")

    auditability = bundle.get("llm_auditability") or {}
    if auditability:
        lines.extend(["", "## LLM auditability statistics", ""])
        for key, label in (
            ("valid_case_reference_rate", "Valid case reference rate"),
            ("valid_metric_reference_rate", "Valid metric reference rate"),
            ("metric_value_match_rate", "Metric value match rate"),
            ("capability_evidence_coverage", "Capability evidence coverage"),
            ("capability_schema_completeness", "Capability schema completeness"),
        ):
            value = auditability.get(key)
            if isinstance(value, (int, float)):
                lines.append(f"- **{label}:** {value:.2%}")

    warnings = bundle.get("reference_warnings") or []
    if warnings:
        lines.extend(["", "## Reference validation warnings", ""])
        lines.extend(f"- {item}" for item in warnings)
    return "\n".join(lines)


def run_llm_interpretable_analysis(
    report: Dict[str, Any],
    cfg,
    out_dir: str,
) -> Optional[Dict[str, str]]:
    """Run two evidence-grounded LLM stages: diagnosis, then improvement guidance."""
    api_base = str(getattr(cfg, "eval_api_base", "") or "").strip()
    api_key = str(getattr(cfg, "eval_api_key", "") or "").strip()
    model = str(getattr(cfg, "eval_model", "gpt-4o") or "gpt-4o").strip()
    timeout = float(getattr(cfg, "eval_timeout", 600.0))
    temperature = float(getattr(cfg, "eval_llm_temperature", 0.25))
    max_tokens = int(getattr(cfg, "eval_llm_max_tokens", 6000))
    evidence = _build_llm_evidence_package(report, cfg)
    prompt_evidence = _evidence_package_for_prompt(evidence)

    Logger.log(f"[eval][llm] stage 1/2: failure analysis (model={model})")
    failure_analysis = _openai_compatible_json_call(
        api_base=api_base,
        api_key=api_key,
        model=model,
        system_prompt=_failure_analysis_system_prompt(),
        user_prompt=_failure_analysis_user_prompt(prompt_evidence),
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    Logger.log("[eval][llm] stage 2/2: planner improvement guidance")
    improvement_guidance = _openai_compatible_json_call(
        api_base=api_base,
        api_key=api_key,
        model=model,
        system_prompt=_improvement_guidance_system_prompt(),
        user_prompt=_improvement_guidance_user_prompt(prompt_evidence, failure_analysis),
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    valid_case_ids = [item["case_id"] for item in evidence.get("representative_risk_cases") or []]
    warnings = _collect_case_reference_warnings(failure_analysis, valid_case_ids)
    warnings.extend(_collect_case_reference_warnings(improvement_guidance, valid_case_ids))
    warnings = sorted(set(warnings))

    # Deterministically validate/normalize the LLM's capability_assessment against the evidence
    # package (never trust the LLM's own numbers) and overwrite the raw field with the validated,
    # backward-compatible version before it is saved or rendered.
    capability_result = _validate_population_capability_assessments(
        failure_analysis, evidence
    )
    if isinstance(failure_analysis, dict):
        failure_analysis["capability_assessment"] = capability_result["capability_assessment"]
    capability_validation = capability_result["capability_validation"]
    llm_auditability = capability_result["llm_auditability"]

    guidance_items = (
        improvement_guidance.get("improvement_guidance")
        if isinstance(improvement_guidance, dict)
        else improvement_guidance
    )
    if not isinstance(guidance_items, list):
        guidance_items = []

    bundle = {
        "schema_version": "2.0",
        "analysis_scope": "single-planner LLM-enhanced interpretable AV evaluation",
        "generated_unix_time": time.time(),
        "model": model,
        "planner_context": evidence.get("planner_context") or {},
        "failure_analysis": failure_analysis,
        "improvement_guidance": guidance_items,
        "reference_warnings": warnings,
        "capability_validation": capability_validation,
        "llm_auditability": llm_auditability,
        "evidence_case_index": [
            {"case_id": item.get("case_id"), **(item.get("scene_key") or {})}
            for item in evidence.get("representative_risk_cases") or []
        ],
    }

    mkdir(out_dir)
    json_path = os.path.join(out_dir, "llm_interpretable_analysis.json")
    md_path = os.path.join(out_dir, "llm_interpretable_analysis.md")
    guidance_path = os.path.join(out_dir, "planner_improvement_guidance.json")
    evidence_path = os.path.join(out_dir, "llm_evidence_package.json")
    full_report_path = os.path.join(out_dir, "interpretable_eval_full.md")
    auditability_path = os.path.join(out_dir, "llm_auditability.json")
    llm_markdown = _render_llm_interpretable_markdown(bundle)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(llm_markdown)
    with open(guidance_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "improvement_guidance": guidance_items,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    with open(auditability_path, "w", encoding="utf-8") as f:
        json.dump(
            {"capability_validation": capability_validation, "llm_auditability": llm_auditability},
            f,
            ensure_ascii=False,
            indent=2,
        )
    deterministic_md_path = os.path.join(out_dir, "interpretable_eval.md")
    deterministic_markdown = ""
    if os.path.isfile(deterministic_md_path):
        with open(deterministic_md_path, "r", encoding="utf-8") as f:
            deterministic_markdown = f.read().rstrip()
    with open(full_report_path, "w", encoding="utf-8") as f:
        if deterministic_markdown:
            f.write(deterministic_markdown + "\n\n---\n\n")
        f.write(llm_markdown)
    return {
        "json": json_path,
        "markdown": md_path,
        "improvement_guidance": guidance_path,
        "llm_auditability": auditability_path,
        "evidence_package": evidence_path,
        "full_report": full_report_path,
    }


# ---------------------------------------------------------------------------
# Per-sample planner metrics
# ---------------------------------------------------------------------------
def _compute_planner_metrics(
    *,
    ego_traj_pred: torch.Tensor,
    ego_lw: torch.Tensor,
    oth_traj: torch.Tensor,
    oth_lw: torch.Tensor,
    ego_traj_gt: Optional[torch.Tensor],
    veh_coll: Optional[np.ndarray],
    coll_t: Optional[np.ndarray],
    dt: float,
    near_miss_clearance_m: float = 1.0,
    brake_accel_threshold: float = -0.5,
    accel_into_conflict_threshold: float = 0.5,
    kinematics_upsample_dt: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute evidence-rich planner metrics for one rollout.

    Trajectories must be unnormalized in the global frame. ``min_dist_m`` is
    retained for backward compatibility and denotes center-to-center distance.
    ``min_clearance_proxy_m`` is a signed separating-axis (SAT) margin for the
    two oriented boxes. Positive values indicate separation on at least one SAT
    axis; non-positive values indicate overlap. It is not Euclidean box distance.
    """
    metrics: Dict[str, Any] = {
        "did_coll": False,
        "coll_step": None,
        "ttc_sec": None,
        "collision_partner_idx": None,
        "min_dist_m": None,
        "min_clearance_proxy_m": None,
        "min_box_separation_margin_m": None,
        "min_clearance_agent_idx": None,
        "min_clearance_step": None,
        "closest_agent_idx": None,
        "closest_step": None,
        "did_near_crash": False,
        "rel_speed_at_min_dist": None,
        "rel_heading_at_min_dist": None,
        "rel_long_gap_at_min_dist": None,
        "rel_lat_gap_at_min_dist": None,
        "relative_speed_at_collision_mps": None,
        "relative_heading_at_collision_rad": None,
        "relative_long_gap_at_collision_m": None,
        "relative_lat_gap_at_collision_m": None,
        "ego_speed_at_collision": None,
        "brake-before-collision": None,
        "accelerate-into-collision": None,
        "brake_onset_step": None,
        "brake_onset_time_sec": None,
        "brake_onset_ttc_sec": None,
        "mean_accel_pre_collision": None,
        "min_accel_pre_collision": None,
        "ego_path_len_m": None,
        "ego_avg_speed_mps": None,
        "ego_max_decel": None,
        "ego_max_accel": None,
        "ego_mean_abs_accel": None,
        "ego_jerk_rms": None,
        "ego_max_abs_jerk": None,
        "ego_lat_accel_max": None,
        "ego_lat_accel_rms": None,
        "ego_net_displacement_m": None,
        "ego_gt_net_displacement_m": None,
        "ego_path_efficiency": None,
        "ego_progress_ratio": None,
        "ego_ade_m": None,
        "ego_fde_m": None,
        "ego_valid_fraction": None,
        "other_valid_fraction": None,
        "has_other_agents": False,
    }

    ego_np = ego_traj_pred.detach().cpu().numpy().astype(np.float64)
    FT = int(ego_np.shape[0])
    if FT < 1:
        return metrics
    ego_valid = np.isfinite(ego_np[:, :4]).all(axis=1)
    metrics["ego_valid_fraction"] = float(np.mean(ego_valid)) if ego_valid.size else None

    ego_kin_np = ego_np
    kin_dt = float(dt)
    if kinematics_upsample_dt is not None:
        from agent.eval.trajectory_utils import upsample_trajectory_waypoints

        ego_kin_np, kin_dt = upsample_trajectory_waypoints(ego_np, float(dt), float(kinematics_upsample_dt))

    def finite_stat(values: np.ndarray, reducer) -> Optional[float]:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(reducer(arr)) if arr.size else None

    speed_per_step = np.full((FT,), np.nan, dtype=np.float64)
    accel = np.asarray([], dtype=np.float64)
    if FT >= 2:
        steps = np.diff(ego_np[:, :2], axis=0)
        valid_steps = ego_valid[1:] & ego_valid[:-1]
        seg_len = np.linalg.norm(steps, axis=1)
        seg_len[~valid_steps] = np.nan
        path_len = float(np.nansum(seg_len)) if np.any(np.isfinite(seg_len)) else None
        speed = seg_len / max(1e-6, float(dt))
        speed_per_step[1:] = speed
        if speed.size > 0 and np.isfinite(speed[0]):
            speed_per_step[0] = speed[0]
        elif np.any(np.isfinite(speed)):
            speed_per_step[0] = speed[np.flatnonzero(np.isfinite(speed))[0]]
        avg_speed = finite_stat(speed, np.mean)

        valid_xy_idx = np.flatnonzero(np.isfinite(ego_np[:, :2]).all(axis=1))
        if valid_xy_idx.size >= 2:
            start_idx, end_idx = int(valid_xy_idx[0]), int(valid_xy_idx[-1])
            net_disp = float(np.linalg.norm(ego_np[end_idx, :2] - ego_np[start_idx, :2]))
            metrics["ego_net_displacement_m"] = net_disp
            if path_len is not None and path_len > 1e-6:
                metrics["ego_path_efficiency"] = float(net_disp / path_len)

        kin_valid = np.isfinite(ego_kin_np[:, :4]).all(axis=1)
        if int(ego_kin_np.shape[0]) >= 2:
            kin_steps = np.diff(ego_kin_np[:, :2], axis=0)
            kin_valid_steps = kin_valid[1:] & kin_valid[:-1]
            kin_seg_len = np.linalg.norm(kin_steps, axis=1)
            kin_seg_len[~kin_valid_steps] = np.nan
            kin_speed = kin_seg_len / max(1e-6, kin_dt)
            if kin_speed.size >= 2:
                accel = np.diff(kin_speed) / max(1e-6, kin_dt)
                metrics["ego_max_accel"] = finite_stat(accel, np.max)
                metrics["ego_max_decel"] = finite_stat(accel, np.min)
                metrics["ego_mean_abs_accel"] = finite_stat(np.abs(accel), np.mean)
                if accel.size >= 2:
                    jerk = np.diff(accel) / max(1e-6, kin_dt)
                    metrics["ego_jerk_rms"] = finite_stat(np.square(jerk), lambda x: math.sqrt(float(np.mean(x))))
                    metrics["ego_max_abs_jerk"] = finite_stat(np.abs(jerk), np.max)

            kin_ft = int(ego_kin_np.shape[0])
            if kin_ft >= 3 and kin_speed.size >= 2:
                h = np.arctan2(ego_kin_np[:, 3], ego_kin_np[:, 2])
                hdot = np.diff(h) / max(1e-6, kin_dt)
                hdot = (hdot + np.pi) % (2.0 * np.pi) - np.pi
                if kin_speed.size == hdot.size:
                    a_lat = kin_speed * hdot
                    metrics["ego_lat_accel_max"] = finite_stat(np.abs(a_lat), np.max)
                    metrics["ego_lat_accel_rms"] = finite_stat(np.square(a_lat), lambda x: math.sqrt(float(np.mean(x))))

        metrics["ego_path_len_m"] = path_len
        metrics["ego_avg_speed_mps"] = avg_speed

        if speed.size >= 2:
            accel_coarse = np.diff(speed) / max(1e-6, float(dt))
            brake_candidates = np.flatnonzero(
                np.isfinite(accel_coarse) & (accel_coarse <= float(brake_accel_threshold))
            )
            if brake_candidates.size:
                brake_step = int(brake_candidates[0] + 2)
                metrics["brake_onset_step"] = brake_step
                metrics["brake_onset_time_sec"] = float(brake_step * dt)

    if ego_traj_gt is not None and ego_traj_gt.numel() > 0:
        gt_np = ego_traj_gt.detach().cpu().numpy().astype(np.float64)
        if gt_np.shape[0] == FT:
            valid = np.isfinite(ego_np[:, :2]).all(axis=1) & np.isfinite(gt_np[:, :2]).all(axis=1)
            if np.any(valid):
                disp_err = np.linalg.norm(ego_np[valid, :2] - gt_np[valid, :2], axis=1)
                metrics["ego_ade_m"] = finite_stat(disp_err, np.mean)
                last_valid = int(np.flatnonzero(valid)[-1])
                metrics["ego_fde_m"] = float(np.linalg.norm(ego_np[last_valid, :2] - gt_np[last_valid, :2]))
            gt_valid_idx = np.flatnonzero(np.isfinite(gt_np[:, :2]).all(axis=1))
            if gt_valid_idx.size >= 2:
                gt_net = float(np.linalg.norm(gt_np[gt_valid_idx[-1], :2] - gt_np[gt_valid_idx[0], :2]))
                metrics["ego_gt_net_displacement_m"] = gt_net
                pred_net = _as_finite_float(metrics.get("ego_net_displacement_m"))
                if pred_net is not None and gt_net > 1e-3:
                    metrics["ego_progress_ratio"] = float(pred_net / gt_net)

    if oth_traj.numel() > 0 and oth_traj.size(0) > 0:
        metrics["has_other_agents"] = True
        oth_np = oth_traj.detach().cpu().numpy().astype(np.float64)
        oth_valid = np.isfinite(oth_np[:, :, :4]).all(axis=-1)
        metrics["other_valid_fraction"] = float(np.mean(oth_valid)) if oth_valid.size else None
        d = np.linalg.norm(ego_np[None, :, :2] - oth_np[:, :, :2], axis=-1)
        valid_pair = oth_valid & ego_valid[None, :]
        d_valid = np.where(valid_pair & np.isfinite(d), d, np.inf)
        if np.any(np.isfinite(d_valid)):
            min_flat_idx = int(np.argmin(d_valid))
            min_agent, min_step = divmod(min_flat_idx, FT)
            min_dist = float(d_valid[min_agent, min_step])
            metrics["min_dist_m"] = min_dist
            metrics["closest_agent_idx"] = int(min_agent + 1)
            metrics["closest_step"] = int(min_step)

            ego_lw_np = ego_lw.detach().cpu().numpy().astype(np.float64).reshape(-1)
            oth_lw_np = oth_lw.detach().cpu().numpy().astype(np.float64)

            def sat_margin(a_state: np.ndarray, a_lw: np.ndarray, b_state: np.ndarray, b_lw: np.ndarray) -> float:
                a_h = np.asarray(a_state[2:4], dtype=np.float64)
                b_h = np.asarray(b_state[2:4], dtype=np.float64)
                a_norm, b_norm = np.linalg.norm(a_h), np.linalg.norm(b_h)
                if a_norm <= 1e-9 or b_norm <= 1e-9:
                    return float("nan")
                a_u, b_u = a_h / a_norm, b_h / b_norm
                a_v = np.array([-a_u[1], a_u[0]], dtype=np.float64)
                b_v = np.array([-b_u[1], b_u[0]], dtype=np.float64)
                delta = np.asarray(b_state[:2] - a_state[:2], dtype=np.float64)
                a_ext = np.asarray(a_lw[:2], dtype=np.float64) * 0.5
                b_ext = np.asarray(b_lw[:2], dtype=np.float64) * 0.5
                seps: List[float] = []
                for axis in (a_u, a_v, b_u, b_v):
                    ra = a_ext[0] * abs(float(np.dot(a_u, axis))) + a_ext[1] * abs(float(np.dot(a_v, axis)))
                    rb = b_ext[0] * abs(float(np.dot(b_u, axis))) + b_ext[1] * abs(float(np.dot(b_v, axis)))
                    seps.append(abs(float(np.dot(delta, axis))) - ra - rb)
                return float(max(seps))

            if ego_lw_np.size >= 2 and oth_lw_np.ndim == 2 and oth_lw_np.shape[1] >= 2:
                margin = np.full((oth_np.shape[0], FT), np.inf, dtype=np.float64)
                for aidx in range(oth_np.shape[0]):
                    for tidx in range(FT):
                        if valid_pair[aidx, tidx]:
                            margin[aidx, tidx] = sat_margin(
                                ego_np[tidx], ego_lw_np, oth_np[aidx, tidx], oth_lw_np[aidx]
                            )
                if np.any(np.isfinite(margin)):
                    margin_flat = int(np.argmin(margin))
                    margin_agent, margin_step = divmod(margin_flat, FT)
                    min_margin = float(margin[margin_agent, margin_step])
                    metrics["min_clearance_proxy_m"] = min_margin
                    metrics["min_box_separation_margin_m"] = min_margin
                    metrics["min_clearance_agent_idx"] = int(margin_agent + 1)
                    metrics["min_clearance_step"] = int(margin_step)

            ego_xy = ego_np[min_step, :2]
            oth_xy = oth_np[min_agent, min_step, :2]
            rel_xy = oth_xy - ego_xy
            ego_h = ego_np[min_step, 2:4]
            h_norm = np.linalg.norm(ego_h)
            if np.isfinite(h_norm) and h_norm > 1e-6:
                h_unit = ego_h / h_norm
                left = np.array([-h_unit[1], h_unit[0]], dtype=np.float64)
                metrics["rel_long_gap_at_min_dist"] = float(np.dot(rel_xy, h_unit))
                metrics["rel_lat_gap_at_min_dist"] = float(np.dot(rel_xy, left))
            if 1 <= min_step < FT and valid_pair[min_agent, min_step] and valid_pair[min_agent, min_step - 1]:
                ego_v = (ego_np[min_step, :2] - ego_np[min_step - 1, :2]) / max(1e-6, float(dt))
                oth_v = (oth_np[min_agent, min_step, :2] - oth_np[min_agent, min_step - 1, :2]) / max(1e-6, float(dt))
                metrics["rel_speed_at_min_dist"] = float(np.linalg.norm(ego_v - oth_v))
            ego_yaw = float(np.arctan2(ego_np[min_step, 3], ego_np[min_step, 2]))
            oth_yaw = float(np.arctan2(oth_np[min_agent, min_step, 3], oth_np[min_agent, min_step, 2]))
            metrics["rel_heading_at_min_dist"] = float((oth_yaw - ego_yaw + np.pi) % (2.0 * np.pi) - np.pi)

        if veh_coll is not None and coll_t is not None and len(veh_coll) > 0:
            coll_mask = np.asarray(veh_coll).astype(bool)
            metrics["did_coll"] = bool(np.any(coll_mask))
            if metrics["did_coll"]:
                coll_steps = np.asarray(coll_t)[coll_mask]
                coll_agents = np.flatnonzero(coll_mask)
                if coll_steps.size:
                    local_idx = int(np.argmin(coll_steps))
                    cs = int(coll_steps[local_idx])
                    partner = int(coll_agents[local_idx])
                    metrics["coll_step"] = cs
                    metrics["ttc_sec"] = float(cs * dt)
                    metrics["collision_partner_idx"] = partner + 1
                    if 0 <= cs < speed_per_step.shape[0] and np.isfinite(speed_per_step[cs]):
                        metrics["ego_speed_at_collision"] = float(speed_per_step[cs])

                    if 0 <= cs < FT and partner < oth_np.shape[0] and valid_pair[partner, cs]:
                        ego_xy = ego_np[cs, :2]
                        oth_xy = oth_np[partner, cs, :2]
                        rel_xy = oth_xy - ego_xy
                        ego_h = ego_np[cs, 2:4]
                        h_norm = np.linalg.norm(ego_h)
                        if np.isfinite(h_norm) and h_norm > 1e-6:
                            h_unit = ego_h / h_norm
                            left = np.array([-h_unit[1], h_unit[0]], dtype=np.float64)
                            metrics["relative_long_gap_at_collision_m"] = float(np.dot(rel_xy, h_unit))
                            metrics["relative_lat_gap_at_collision_m"] = float(np.dot(rel_xy, left))
                        ego_yaw = float(np.arctan2(ego_np[cs, 3], ego_np[cs, 2]))
                        oth_yaw = float(np.arctan2(oth_np[partner, cs, 3], oth_np[partner, cs, 2]))
                        metrics["relative_heading_at_collision_rad"] = float((oth_yaw - ego_yaw + np.pi) % (2.0 * np.pi) - np.pi)
                    if 1 <= cs < FT and partner < oth_np.shape[0] and valid_pair[partner, cs] and valid_pair[partner, cs - 1]:
                        ego_v = (ego_np[cs, :2] - ego_np[cs - 1, :2]) / max(1e-6, float(dt))
                        oth_v = (oth_np[partner, cs, :2] - oth_np[partner, cs - 1, :2]) / max(1e-6, float(dt))
                        metrics["relative_speed_at_collision_mps"] = float(np.linalg.norm(ego_v - oth_v))

                    if accel.size > 0:
                        right = min(accel.size, max(0, cs - 1))
                        left_idx = max(0, right - 4)
                        a_win = accel[left_idx:right]
                        a_win = a_win[np.isfinite(a_win)]
                        if a_win.size:
                            mean_a = float(np.mean(a_win))
                            metrics["mean_accel_pre_collision"] = mean_a
                            metrics["min_accel_pre_collision"] = float(np.min(a_win))
                            metrics["brake-before-collision"] = bool(np.any(a_win <= float(brake_accel_threshold)))
                            metrics["accelerate-into-collision"] = bool(mean_a >= float(accel_into_conflict_threshold))
                    brake_step = metrics.get("brake_onset_step")
                    if isinstance(brake_step, int):
                        metrics["brake_onset_ttc_sec"] = float((cs - brake_step) * dt)

        clearance = _as_finite_float(metrics.get("min_clearance_proxy_m"))
        metrics["did_near_crash"] = bool(
            not metrics["did_coll"]
            and clearance is not None
            and clearance <= float(near_miss_clearance_m)
        )

    for k, v in list(metrics.items()):
        if isinstance(v, (np.integer,)):
            metrics[k] = int(v)
        elif isinstance(v, (np.floating,)):
            metrics[k] = float(v) if math.isfinite(float(v)) else None
        elif isinstance(v, float) and not math.isfinite(v):
            metrics[k] = None
    return metrics

def _load_results_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                Logger.log(f"[warn] skip invalid JSONL line {line_no} in {path}: {e}")
    return records


def _run_eval_report(
    cfg,
    results_records: List[Dict[str, Any]],
    results_jsonl_path: str,
    extra_context: Optional[Dict[str, Any]] = None,
) -> None:
    """Write auditable deterministic artifacts, then optionally run two evidence-grounded LLM agents.

    The deterministic layer is always authoritative. When ``--eval_use_llm`` is set and credentials
    are available, a Failure Analysis Agent and a Planner Improvement Agent reason over the evidence
    package to produce cross-scene diagnosis and prioritized, testable improvement guidance.
    """
    if not results_records:
        Logger.log("[eval] no records; skip report.")
        return

    thresholds = InterpretabilityThresholds.from_cfg(cfg)
    first = results_records[0]
    ctx: Dict[str, Any] = {
        "planner": first.get("planner", str(getattr(cfg, "planner", "unknown"))),
        "planner_cfg": first.get("planner_cfg", str(getattr(cfg, "planner_cfg", ""))),
        "dataroot": str(getattr(cfg, "dataroot", "")),
        "version": str(getattr(cfg, "version", "")),
        "results_path": results_jsonl_path,
        "pipeline_version": EVALUATION_PIPELINE_VERSION,
        "evaluation_scope": "single planner",
    }
    if extra_context:
        ctx.update(extra_context)

    baseline_records: Optional[List[Dict[str, Any]]] = None
    baseline_path = str(getattr(cfg, "eval_baseline_results", "") or "").strip()
    if baseline_path:
        if os.path.isfile(baseline_path):
            baseline_records = _load_results_jsonl(baseline_path)
            ctx["baseline_results_path"] = baseline_path
            Logger.log(f"[eval] loaded {len(baseline_records)} baseline records from {baseline_path}")
        else:
            Logger.log(f"[warn] baseline results not found: {baseline_path}; paired comparison skipped.")

    eval_llm_only = bool(getattr(cfg, "eval_llm_only", False))
    eval_rebuild_report = bool(getattr(cfg, "eval_rebuild_report", False))
    report_path = os.path.join(str(cfg.out), "interpretable_eval.json")
    reuse_existing = False
    if eval_llm_only and (not eval_rebuild_report) and os.path.isfile(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            existing_report = json.load(f)
        if _deterministic_report_is_current(existing_report):
            report = existing_report
            reuse_existing = True
            Logger.log(
                f"[eval] eval_llm_only: loaded current deterministic report from {report_path} "
                f"(deterministic_evidence_version={DETERMINISTIC_EVIDENCE_VERSION})"
            )
        else:
            Logger.log(
                "[eval] eval_llm_only: existing deterministic report is stale "
                f"(need deterministic_evidence_version={DETERMINISTIC_EVIDENCE_VERSION}); "
                "rebuilding it from results.jsonl before LLM analysis."
            )
    elif eval_llm_only and eval_rebuild_report:
        Logger.log(
            "[eval] eval_rebuild_report: forcing rebuild of deterministic report "
            "(capability prior) from results.jsonl before LLM analysis."
        )
    elif eval_llm_only:
        Logger.log(
            "[eval] eval_llm_only: interpretable_eval.json missing; "
            "rebuilding deterministic report before LLM analysis."
        )

    if not reuse_existing:
        report = build_interpretable_evaluation(
            results_records,
            thresholds,
            baseline_records=baseline_records,
            context=ctx,
            cfg=cfg,
        )
        paths = write_interpretable_evaluation(report, str(cfg.out))
        Logger.log(
            "[eval] deterministic interpretable evaluation written: "
            + ", ".join(f"{name}={path}" for name, path in paths.items())
        )

    eval_use_llm_flag = bool(getattr(cfg, "eval_use_llm", False)) or eval_llm_only
    api_key = str(getattr(cfg, "eval_api_key", "") or "").strip()
    api_base = str(getattr(cfg, "eval_api_base", "") or "").strip()
    if not eval_use_llm_flag:
        Logger.log("[eval] LLM analysis disabled; deterministic interpretable report is complete.")
        return
    if not api_key:
        Logger.log(
            "[eval] LLM analysis requested but --eval_api_key is empty; "
            "deterministic report remains available."
        )
        return
    if not api_base:
        Logger.log(
            "[eval] LLM analysis requested but --eval_api_base is empty; "
            "deterministic report remains available."
        )
        return

    try:
        Logger.log(
            f"[eval] running two-stage LLM analysis (model={str(getattr(cfg, 'eval_model', 'gpt-4o'))}, "
            f"n_records={len(results_records)})"
        )
        llm_paths = run_llm_interpretable_analysis(report, cfg, str(cfg.out))
        if llm_paths:
            Logger.log(
                "[eval] LLM failure analysis and planner improvement guidance written: "
                + ", ".join(f"{name}={path}" for name, path in llm_paths.items())
            )
    except Exception as e:
        Logger.log(
            f"[warn] LLM-enhanced analysis failed ({e}); deterministic interpretable report remains available."
        )


def _ensure_single_dataset_stages_for_joint(
    run_record: Mapping[str, Any],
    cfg: Any,
    *,
    regenerate_upstream: bool,
) -> Dict[str, Any]:
    """Reuse valid single-dataset stages; optionally rebuild invalid upstream stages.

    Never reruns a valid rollout/evidence/LLM stage merely because the joint
    summary is missing. Rollout itself is not executed here — missing rollout
    is reported as failed so the caller can skip the pair.
    """
    from agent.eval.crashsim_joint_eval import (
        deterministic_report_is_valid,
        inspect_run_stages,
        llm_analysis_is_valid,
        mark_stage,
        valid_results_jsonl,
    )

    inspection = inspect_run_stages(
        run_record,
        expected_evidence_version=DETERMINISTIC_EVIDENCE_VERSION,
    )
    stages = dict(inspection.get("stages") or {})
    paths = run_record.get("paths") or {}
    run_dir = str(run_record.get("run_dir") or "")
    results_path = str(paths.get("results") or "")

    if not valid_results_jsonl(results_path):
        mark_stage(
            stages,
            "rollout",
            "failed",
            path=results_path or None,
            reason="results.jsonl missing/invalid; joint analysis will not run rollout",
        )
        inspection["stages"] = stages
        inspection["ready_for_joint"] = False
        return inspection

    mark_stage(
        stages,
        "rollout",
        "reused",
        path=results_path,
        reason="valid results.jsonl present",
    )

    det_path = str(paths.get("deterministic_evaluation") or "")
    report = None
    if os.path.isfile(det_path):
        with open(det_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    if deterministic_report_is_valid(
        report, expected_evidence_version=DETERMINISTIC_EVIDENCE_VERSION
    ):
        mark_stage(
            stages,
            "deterministic_evidence",
            "reused",
            path=det_path,
            reason="valid interpretable_eval.json present",
        )
    elif regenerate_upstream and valid_results_jsonl(results_path):
        try:
            records = _load_results_jsonl(results_path)
            thresholds = InterpretabilityThresholds.from_cfg(cfg)
            first = records[0] if records else {}
            context = {
                "planner": first.get("planner", run_record.get("planner")),
                "planner_cfg": first.get("planner_cfg", ""),
                "dataroot": str(getattr(cfg, "dataroot", "") or ""),
                "version": str(getattr(cfg, "version", "") or ""),
                "results_path": results_path,
                "pipeline_version": EVALUATION_PIPELINE_VERSION,
                "evaluation_scope": "single planner",
                "dataset_kind": run_record.get("dataset_kind"),
            }
            report = build_interpretable_evaluation(
                records, thresholds, context=context, cfg=cfg
            )
            write_interpretable_evaluation(report, run_dir)
            status = "regenerated" if os.path.isfile(det_path) else "generated"
            mark_stage(
                stages,
                "deterministic_evidence",
                status,
                path=det_path,
                reason="rebuilt interpretable_eval.json from results.jsonl",
            )
        except Exception as exc:
            mark_stage(
                stages,
                "deterministic_evidence",
                "failed",
                path=det_path or None,
                reason=f"failed to rebuild deterministic evidence: {exc}",
            )
            inspection["stages"] = stages
            inspection["ready_for_joint"] = False
            return inspection
    else:
        mark_stage(
            stages,
            "deterministic_evidence",
            "failed",
            path=det_path or None,
            reason=(
                "interpretable_eval.json missing/invalid; pass "
                "--joint_regenerate_upstream to rebuild from results.jsonl"
            ),
        )
        inspection["stages"] = stages
        inspection["ready_for_joint"] = False
        return inspection

    llm_path = str(paths.get("llm_analysis") or "")
    bundle = None
    if os.path.isfile(llm_path):
        with open(llm_path, "r", encoding="utf-8") as handle:
            bundle = json.load(handle)
    if llm_analysis_is_valid(bundle):
        mark_stage(
            stages,
            "single_dataset_llm",
            "reused",
            path=llm_path,
            reason="valid llm_interpretable_analysis.json present",
        )
    elif regenerate_upstream:
        api_key = str(getattr(cfg, "eval_api_key", "") or "").strip()
        api_base = str(getattr(cfg, "eval_api_base", "") or "").strip()
        if not api_key or not api_base:
            mark_stage(
                stages,
                "single_dataset_llm",
                "failed",
                path=llm_path or None,
                reason="LLM outputs invalid and API credentials unavailable for regeneration",
            )
            inspection["stages"] = stages
            inspection["ready_for_joint"] = False
            return inspection
        try:
            if report is None:
                with open(det_path, "r", encoding="utf-8") as handle:
                    report = json.load(handle)
            prev_out = getattr(cfg, "out", None)
            cfg.out = run_dir
            try:
                run_llm_interpretable_analysis(report, cfg, run_dir)
            finally:
                if prev_out is not None:
                    cfg.out = prev_out
            with open(llm_path, "r", encoding="utf-8") as handle:
                bundle = json.load(handle)
            if not llm_analysis_is_valid(bundle):
                raise RuntimeError("regenerated LLM bundle failed validation")
            mark_stage(
                stages,
                "single_dataset_llm",
                "regenerated",
                path=llm_path,
                reason="regenerated single-dataset LLM analysis",
            )
        except Exception as exc:
            mark_stage(
                stages,
                "single_dataset_llm",
                "failed",
                path=llm_path or None,
                reason=f"failed to regenerate LLM analysis: {exc}",
            )
            inspection["stages"] = stages
            inspection["ready_for_joint"] = False
            return inspection
    else:
        mark_stage(
            stages,
            "single_dataset_llm",
            "failed",
            path=llm_path or None,
            reason=(
                "llm_interpretable_analysis.json missing/invalid; pass "
                "--joint_regenerate_upstream to rebuild"
            ),
        )
        inspection["stages"] = stages
        inspection["ready_for_joint"] = False
        return inspection

    ready = all(
        (stages.get(name) or {}).get("status") in {"reused", "generated", "regenerated"}
        for name in ("rollout", "deterministic_evidence", "single_dataset_llm")
    )
    inspection["stages"] = stages
    inspection["ready_for_joint"] = bool(ready)
    return inspection


def _joint_cross_dataset_system_prompt() -> str:
    return (
        "You are an AV planner analyst. Produce ONE joint cross-dataset diagnosis "
        "for a single planner using paired nuScenes (naturalistic) and nuCrash "
        "(long-tail) evidence. Numerical deltas are already computed in the package — "
        "do not restate exact rates or Δ values in prose. "
        "Do not invent planner internals. "
        "Write synthesis and possible_mechanism directly (no meta phrases). "
        "Keep mechanisms cautious with may/could/is consistent with as candidate "
        "explanations of observed behavior. "
        "Distinguish limitation_scope among "
        "{general, amplified_on_long_tail, long_tail_specific, inconclusive}. "
        "Improvement guidance: emphasize ONE primary actionable recommendation "
        "linked to the diagnosed failure pattern — never capability slogans "
        "such as 'improve risk anticipation'. "
        "Do not claim the fix is already validated. Return a JSON object only."
    )


def _joint_cross_dataset_user_prompt(package: Mapping[str, Any]) -> str:
    return (
        "Paired cross-dataset evidence package follows. Use both datasets.\n"
        "Return JSON with keys: "
        "cross_dataset_diagnostic_synthesis, naturalistic_performance_summary, "
        "long_tail_performance_summary, observed_limitation, possible_mechanism, "
        "improvement_guidance, limitation_scope_confirmation "
        "(must match package.deterministic_cross_dataset_comparison.limitation_scope "
        "unless you briefly justify disagreement).\n\n"
        + json.dumps(package, ensure_ascii=False, indent=2)
    )


def _maybe_run_joint_llm(
    package: Mapping[str, Any],
    cfg: Any,
) -> Optional[Dict[str, Any]]:
    api_key = str(getattr(cfg, "eval_api_key", "") or "").strip()
    api_base = str(getattr(cfg, "eval_api_base", "") or "").strip()
    if not api_key or not api_base:
        return None
    model = str(getattr(cfg, "eval_model", "gpt-4o") or "gpt-4o").strip()
    timeout = float(getattr(cfg, "eval_timeout", 600.0))
    temperature = float(getattr(cfg, "eval_llm_temperature", 0.25))
    max_tokens = int(getattr(cfg, "eval_llm_max_tokens", 6000))
    Logger.log(f"[joint][llm] joint synthesis (model={model})")
    return _openai_compatible_json_call(
        api_base=api_base,
        api_key=api_key,
        model=model,
        system_prompt=_joint_cross_dataset_system_prompt(),
        user_prompt=_joint_cross_dataset_user_prompt(package),
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def run_joint_analysis(cfg: Any) -> Dict[str, Any]:
    """Pair runs, reuse/regenerate single-dataset stages, write joint analysis."""
    from agent.eval.crashsim_joint_eval import (
        build_deterministic_cross_dataset_comparison,
        build_joint_analysis,
        compact_single_dataset_for_joint,
        joint_summary_output_dir,
        mark_stage,
        pair_runs_by_planner_seed,
        save_joint_analysis,
        unavailable_joint_summary,
    )

    runs_dir = str(getattr(cfg, "runs_dir", "") or "").strip()
    if not runs_dir:
        throw_err("--joint requires --runs_dir")
    runs_dir = os.path.abspath(runs_dir)
    if not os.path.isdir(runs_dir):
        throw_err(f"--runs_dir not found: {runs_dir}")

    pairing = pair_runs_by_planner_seed(runs_dir)
    planner_filter = str(getattr(cfg, "joint_planner", "") or "").strip()
    seed_filter = getattr(cfg, "joint_seed", None)
    regenerate = bool(getattr(cfg, "joint_regenerate_upstream", False))
    use_llm = bool(getattr(cfg, "joint_use_llm", False))

    complete = list(pairing.get("complete_pairs") or [])
    incomplete = list(pairing.get("incomplete_pairs") or [])
    if planner_filter:
        complete = [p for p in complete if p.get("planner") == planner_filter]
        incomplete = [p for p in incomplete if p.get("planner") == planner_filter]
    if seed_filter is not None:
        complete = [p for p in complete if int(p.get("seed")) == int(seed_filter)]
        incomplete = [p for p in incomplete if int(p.get("seed")) == int(seed_filter)]

    Logger.log(
        f"[joint] complete_pairs={len(complete)} incomplete_pairs={len(incomplete)}"
    )
    for item in incomplete:
        Logger.log(f"[joint] incomplete: {item.get('reason')}")

    results: Dict[str, Any] = {
        "runs_dir": runs_dir,
        "complete_pairs": [],
        "incomplete_pairs": [
            {
                "planner": p.get("planner"),
                "seed": p.get("seed"),
                "reason": p.get("reason"),
                "missing_datasets": p.get("missing_datasets"),
            }
            for p in incomplete
        ],
        "summaries": [],
    }
    min_support = int(getattr(cfg, "eval_min_group_size", 5))

    for pair in complete:
        planner = str(pair["planner"])
        seed = int(pair["seed"])
        nu_run = pair["nuScenes"]
        crash_run = pair["nuCrash"]
        Logger.log(
            f"[joint] processing {planner} seed={seed} "
            f"({nu_run['run_id']} + {crash_run['run_id']})"
        )

        nu_stages = _ensure_single_dataset_stages_for_joint(
            nu_run, cfg, regenerate_upstream=regenerate
        )
        crash_stages = _ensure_single_dataset_stages_for_joint(
            crash_run, cfg, regenerate_upstream=regenerate
        )
        stage_status = {
            "nuScenes": nu_stages,
            "nuCrash": crash_stages,
            "joint_comparison": {"status": "skipped", "path": None, "reason": "pending"},
            "joint_synthesis": {"status": "skipped", "path": None, "reason": "pending"},
        }
        out_dir = joint_summary_output_dir(runs_dir, planner, seed)

        if not nu_stages.get("ready_for_joint") or not crash_stages.get("ready_for_joint"):
            reason = (
                "paired single-dataset outputs incomplete: "
                f"nuScenes_ready={nu_stages.get('ready_for_joint')} "
                f"nuCrash_ready={crash_stages.get('ready_for_joint')}"
            )
            mark_stage(stage_status, "joint_comparison", "skipped", reason=reason)
            mark_stage(stage_status, "joint_synthesis", "skipped", reason=reason)
            summary = unavailable_joint_summary(
                planner_name=planner,
                seed=seed,
                reason=reason,
                stage_status=stage_status,
                nuscenes_run=nu_run,
                nucrash_run=crash_run,
            )
            paths = save_joint_analysis(summary, out_dir)
            results["summaries"].append(
                {
                    "planner": planner,
                    "seed": seed,
                    "available": False,
                    "paths": paths,
                    "reason": reason,
                }
            )
            continue

        with open(nu_run["paths"]["deterministic_evaluation"], "r", encoding="utf-8") as f:
            nu_report = json.load(f)
        with open(crash_run["paths"]["deterministic_evaluation"], "r", encoding="utf-8") as f:
            crash_report = json.load(f)
        with open(nu_run["paths"]["llm_analysis"], "r", encoding="utf-8") as f:
            nu_llm = json.load(f)
        with open(crash_run["paths"]["llm_analysis"], "r", encoding="utf-8") as f:
            crash_llm = json.load(f)

        comparison = build_deterministic_cross_dataset_comparison(
            nuscenes_report=nu_report,
            nucrash_report=crash_report,
            nuscenes_llm=nu_llm,
            nucrash_llm=crash_llm,
            min_support=min_support,
        )
        comparison_path = os.path.join(out_dir, "cross_dataset_comparison.json")
        os.makedirs(out_dir, exist_ok=True)
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        mark_stage(
            stage_status,
            "joint_comparison",
            "generated",
            path=comparison_path,
            reason="deterministic cross-dataset comparison computed in code",
        )

        joint_llm = None
        if use_llm:
            prompt_package = {
                "planner_name": planner,
                "seed": seed,
                "source_runs": {
                    "nuScenes": nu_run.get("run_id"),
                    "nuCrash": crash_run.get("run_id"),
                },
                "deterministic_cross_dataset_comparison": comparison,
                "nuScenes_llm_summary": compact_single_dataset_for_joint(nu_llm),
                "nuCrash_llm_summary": compact_single_dataset_for_joint(crash_llm),
            }
            try:
                joint_llm = _maybe_run_joint_llm(prompt_package, cfg)
            except Exception as exc:
                Logger.log(f"[joint][warn] joint LLM failed ({exc}); using deterministic comparison")
                joint_llm = None

        summary = build_joint_analysis(
            planner_name=planner,
            seed=seed,
            nuscenes_run=nu_run,
            nucrash_run=crash_run,
            comparison=comparison,
            nuscenes_llm=nu_llm,
            nucrash_llm=crash_llm,
            stage_status=stage_status,
            joint_llm=joint_llm,
        )
        paths = save_joint_analysis(summary, out_dir)
        mark_stage(
            stage_status,
            "joint_synthesis",
            "generated",
            path=paths["json"],
            reason=(
                "joint analysis assembled from paired single-dataset outputs"
                + (" with LLM overlay" if joint_llm else " (deterministic)")
            ),
        )
        summary["stage_status"] = stage_status
        with open(paths["json"], "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        results["complete_pairs"].append(
            {
                "planner": planner,
                "seed": seed,
                "nuScenes": nu_run.get("run_id"),
                "nuCrash": crash_run.get("run_id"),
                "available": bool(summary.get("available")),
                "limitation_scope": summary.get("limitation_scope"),
                "paths": paths,
                "stage_status": stage_status,
            }
        )
        results["summaries"].append(
            {
                "planner": planner,
                "seed": seed,
                "available": bool(summary.get("available")),
                "paths": paths,
                "limitation_scope": summary.get("limitation_scope"),
            }
        )
        Logger.log(
            f"[joint] wrote {paths['json']} "
            f"(available={summary.get('available')}, scope={summary.get('limitation_scope')})"
        )

    index_path = os.path.join(runs_dir, "comparison", "joint", "index.json")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    results["index_path"] = index_path
    return results


@torch.no_grad()
def main():
    cfg, cfg_dict = parse_cfg()

    # output & logger
    mkdir(cfg.out)
    Logger.init(os.path.join(cfg.out, "crashsim_planner_log.txt"))
    safe_cfg_dict = dict(cfg_dict)
    if safe_cfg_dict.get("eval_api_key"):
        safe_cfg_dict["eval_api_key"] = "<redacted>"
    Logger.log("Args: " + str(safe_cfg_dict))

    if bool(getattr(cfg, "joint", False)):
        joint_results = run_joint_analysis(cfg)
        Logger.log(
            "[joint] done: "
            f"complete={len(joint_results.get('complete_pairs') or [])} "
            f"incomplete={len(joint_results.get('incomplete_pairs') or [])} "
            f"index={joint_results.get('index_path')}"
        )
        return

    if bool(getattr(cfg, "eval_llm_only", False)) and not bool(getattr(cfg, "eval_only", False)):
        throw_err("--eval_llm_only requires --eval_only (rollout is skipped).")

    if bool(getattr(cfg, "eval_only", False)):
        results_jsonl_path = str(getattr(cfg, "eval_results", "") or "").strip()
        if not results_jsonl_path:
            results_jsonl_path = os.path.join(cfg.out, "results.jsonl")
        if not os.path.isfile(results_jsonl_path):
            throw_err(f"--eval_only: results file not found: {results_jsonl_path}")
        results_records = _load_results_jsonl(results_jsonl_path)
        Logger.log(f"[eval_only] loaded {len(results_records)} records from {results_jsonl_path}")
        _run_eval_report(cfg, results_records, results_jsonl_path)
        return

    if not str(getattr(cfg, "dataroot", "") or "").strip():
        throw_err("--dataroot is required unless --eval_only is used.")

    device = get_device()
    Logger.log(f"Using device {device} ...")

    # Map env: singapore maps may use the flip_singapore convention; expose a switch and keep
    # dataset trajectories consistent with the chosen coordinate system.
    map_env = NuScenesMapEnv(
        os.path.abspath(cfg.dataroot),
        bounds=cfg.map_obs_bounds,
        L=cfg.map_obs_size_pix,
        W=cfg.map_obs_size_pix,
        layers=cfg.map_layers,
        device=device,
        flip_singapore=bool(cfg.map_flip_singapore),
        load_lanegraph=bool(cfg.load_lanegraph),
        lanegraph_res_meters=float(cfg.lanegraph_res_meters),
        pix_per_m=int(cfg.pix_per_m),
    )

    meta = load_nusc_crash_meta(cfg.dataroot, cfg.version)
    dataset = NuScenesCrashGraphDataset(
        meta,
        map_env,
        npast=int(cfg.past_len),
        nfuture=int(cfg.future_len),
        seq_interval=int(cfg.seq_interval),
        categories=cfg.agent_types,
        reduce_cats=bool(cfg.reduce_cats),
        dt=0.5,
    )
    Logger.log(f"Loaded crash dataset: scenes={len(meta.scene)}, subseq={len(dataset)}")

    loader = GraphDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=False,
        worker_init_fn=lambda _: np.random.seed(),
    )

    planner = create_planner(cfg, cfg.planner, dataset, map_env, device, ckpt=cfg.planner_ckpt)
    Logger.log(f"Planner: {cfg.planner} (cfg={getattr(cfg, 'planner_cfg', None)})")
    interpretability_thresholds = InterpretabilityThresholds.from_cfg(cfg)
    # All interpretable metrics (collision, TTC, response latency, jerk) use the
    # rollout coarse grid (dataset.dt, typically 0.5 s). Optional fine-grid
    # upsampling is available via evaluate_rollout(kinematics_upsample_dt=...) but
    # is not enabled by default for any planner.
    kinematics_upsample_dt = None

    viz_dir = os.path.join(cfg.out, "viz") if cfg.viz else None
    if viz_dir:
        mkdir(viz_dir)

    total = 0
    total_with_other = 0
    total_coll = 0

    results_jsonl_path = os.path.join(cfg.out, "results.jsonl")
    results_records: List[Dict[str, Any]] = []
    # Interpretable per-scene records produced by the new post-rollout evaluator.
    interp_scene_records: List[Dict[str, Any]] = []
    # Open the results file in write mode and stream records line-by-line so a
    # crash mid-evaluation still leaves a partial, parseable artifact behind.
    results_f = open(results_jsonl_path, "w", encoding="utf-8")

    for i, (scene_graph, map_idx) in enumerate(loader):
        if cfg.max_seqs is not None and int(cfg.max_seqs) > 0 and i >= int(cfg.max_seqs):
            break

        scene_graph = scene_graph.to(device)
        map_idx = map_idx.to(device)  # (B=1,)

        B = int(map_idx.size(0))
        assert B == 1
        NA = int(scene_graph.past_gt.size(0))
        if NA <= 0:
            continue

        # ego mask for batch=1: first node is ego
        ego_mask = torch.zeros((NA,), dtype=torch.bool, device=device)
        ego_mask[0] = True

        # others future (unnormalized) for planner observation
        normalizer = dataset.get_state_normalizer()
        att_normalizer = dataset.get_att_normalizer()
        init_state = normalizer.unnormalize(scene_graph.past_gt[:, -1, :])  # (NA,6)
        vehicle_atts = att_normalizer.unnormalize(scene_graph.lw)  # (NA,2)

        # assemble ego rollout
        if cfg.planner == "replay":
            ego_plan_fut = normalizer.unnormalize(scene_graph.future_gt[ego_mask, :, :4])  # (1,FT,4)
        elif cfg.planner == 'Lane-graph' or cfg.planner == 'IDM' or cfg.planner == 'PDM-Closed':
            # reset planner for this sequence
            planner.reset(init_state, vehicle_atts, scene_graph.batch, B, map_idx)

            plan_t = np.linspace(0.5, 0.5 * int(cfg.future_len), int(cfg.future_len))  # dt=0.5
            init_agt_ptr = scene_graph.ptr - torch.arange(B + 1, device=device)
            # (NA-1,FT,4) other agents futures in global frame, UNNORMALIZED
            other_fut = normalizer.unnormalize(scene_graph.future_gt[~ego_mask, :, :4]).detach().cpu().numpy()
            ego_plan_fut = planner.rollout(
                other_fut,
                plan_t,
                init_agt_ptr.detach().cpu().numpy(),
                plan_t,
                control_all=False,
            ).to(scene_graph.future_gt)  # (B,FT,4)
        elif cfg.planner == 'IL' or cfg.planner == 'IL-multi' or cfg.planner == 'Diffusion':
            # rollout returns UNNORMALIZED (B, FT, 4); normalize once below with other planners.
            ego_plan_fut = planner.rollout(scene_graph, map_idx, map_env, normalizer, ego_mask)
        else:
            throw_err(f"Unknown planner type: {cfg.planner}")

        # build predicted future: replace ego only (normalized)
        future_pred = scene_graph.future_gt[:, :, :4].clone()
        future_pred[ego_mask] = normalizer.normalize(ego_plan_fut.view(-1, ego_plan_fut.size(1), 4))[0]

        # collision check ego vs others (unnormalized)
        ego_traj = normalizer.unnormalize(future_pred[0])  # (FT,4)
        ego_lw = att_normalizer.unnormalize(scene_graph.lw[0])
        ego_traj_gt = normalizer.unnormalize(scene_graph.future_gt[0, :, :4])
        if NA > 1:
            total_with_other += 1
            oth_traj = normalizer.unnormalize(future_pred[1:])
            oth_lw = att_normalizer.unnormalize(scene_graph.lw[1:])
            veh_coll, coll_t = check_single_veh_coll_min_box_gap(ego_traj, ego_lw, oth_traj, oth_lw)
            did_coll = bool(np.any(veh_coll))
            total_coll += int(did_coll)
        else:
            did_coll = False
            coll_t = None
            veh_coll = None
            oth_traj = torch.empty((0, ego_traj.size(0), 4), dtype=ego_traj.dtype, device=ego_traj.device)
            oth_lw = torch.empty((0, 2), dtype=ego_lw.dtype, device=ego_lw.device)

        # ---- Per-sample planner metrics (TTC, ADE/FDE, smoothness, ...) ----
        adv_meta_info = _detect_adv_agent_meta(scene_graph, meta)
        scene_token_str = ""
        raw_scene_name = getattr(scene_graph, "scene_name", None)
        if isinstance(raw_scene_name, (list, tuple)) and len(raw_scene_name) == 1:
            raw_scene_name = raw_scene_name[0]
        if isinstance(raw_scene_name, str):
            scene_token_str = raw_scene_name
        scene_meta = meta.scene_by_token.get(scene_token_str, {}) if scene_token_str else {}
        odd_meta = _extract_scene_odd_meta(scene_token_str, meta) if scene_token_str else {}

        sample_metrics = _compute_planner_metrics(
            ego_traj_pred=ego_traj,
            ego_lw=ego_lw,
            oth_traj=oth_traj,
            oth_lw=oth_lw,
            ego_traj_gt=ego_traj_gt,
            veh_coll=veh_coll,
            coll_t=coll_t,
            dt=float(dataset.dt),
            near_miss_clearance_m=interpretability_thresholds.near_miss_clearance_m,
            brake_accel_threshold=interpretability_thresholds.brake_accel_threshold,
            accel_into_conflict_threshold=interpretability_thresholds.accel_into_conflict_threshold,
            kinematics_upsample_dt=kinematics_upsample_dt,
        )

        sidx_val = getattr(scene_graph, "sidx", None)
        if isinstance(sidx_val, torch.Tensor):
            try:
                sidx_val = int(sidx_val.view(-1)[0].item())
            except Exception:
                sidx_val = None
        elif isinstance(sidx_val, (list, tuple)):
            sidx_val = int(sidx_val[0]) if sidx_val else None

        record: Dict[str, Any] = {
            "scene_index": int(i),
            "scene_token": scene_token_str,
            "scene_name": scene_meta.get("name") if isinstance(scene_meta, dict) else None,
            "sidx": sidx_val,
            "planner": str(cfg.planner),
            "planner_cfg": str(getattr(cfg, "planner_cfg", "")),
            "NA": int(NA),
            "FT": int(ego_traj.size(0)),
            "dt": float(dataset.dt),
            "behavior_tag": adv_meta_info.get("behavior_tag"),
            "adv_token": adv_meta_info.get("adv_token"),
            "adv_idx": adv_meta_info.get("adv_idx"),
            **odd_meta,
            **sample_metrics,
        }
        _enrich_interpretability_record(record, interpretability_thresholds)

        # ---- Interpretable post-rollout evaluation (new deterministic module) ----
        # Complex metric computation lives in ``agent.eval.crashsim_post_rollout_eval`` so this
        # driver only assembles inputs, stores the returned scene record + trace,
        # and never recomputes metrics itself. Local import per modification-boundary
        # guidance (only invoked after the unified ``ego_plan_fut`` is available).
        try:
            from agent.eval.crashsim_post_rollout_eval import (
                evaluate_rollout,
                save_trace,
                PostRolloutThresholds,
            )

            # Run identity flows in via env vars set by run_interpretable_evaluation.py
            # (parse_cfg is not allowed to change, so we do not add CLI flags here).
            dataset_kind_str = str(
                getattr(cfg, "dataset_kind", None)
                or os.environ.get("CRASHSIM_DATASET_KIND", "")
                or ""
            ).strip()
            if not dataset_kind_str:
                _ver = str(getattr(cfg, "version", "") or "").lower()
                dataset_kind_str = "nuCrash" if "crash" in _ver else "nuScenes"

            _env_seed = os.environ.get("CRASHSIM_SEED")
            _env_repeat = os.environ.get("CRASHSIM_REPEAT_ID")
            scene_metadata = {
                "experiment_id": str(getattr(cfg, "experiment_id", None)
                                     or os.environ.get("CRASHSIM_EXPERIMENT_ID", "")
                                     or "") or None,
                "dataset_kind": dataset_kind_str,
                "dataset_version": str(getattr(cfg, "version", "") or "") or None,
                "scene_token": scene_token_str or None,
                "sample_token": (scene_meta.get("first_sample_token")
                                 if isinstance(scene_meta, dict) else None),
                "source_scene_token": (scene_meta.get("source_scene_token")
                                       if isinstance(scene_meta, dict) else None),
                "source_sample_token": (scene_meta.get("source_sample_token")
                                        if isinstance(scene_meta, dict) else None),
                "scene_index": int(i),
                "sidx": sidx_val,
                "crash_type_fine": (scene_meta.get("crash_type_fine")
                                    if isinstance(scene_meta, dict) else None),
                "crash_category_high": (scene_meta.get("crash_category_high")
                                        if isinstance(scene_meta, dict) else None),
                "behavior_tag": adv_meta_info.get("behavior_tag"),
            }
            planner_metadata = {
                "planner": str(cfg.planner),
                "actual_planner_cfg": str(getattr(cfg, "planner_cfg", "")),
                "seed": (int(_env_seed) if _env_seed not in (None, "") and str(_env_seed).lstrip("-").isdigit() else None),
                "repeat_id": (int(_env_repeat) if _env_repeat not in (None, "") and str(_env_repeat).lstrip("-").isdigit() else None),
            }
            post_thresholds = PostRolloutThresholds(
                near_miss_margin_threshold_m=interpretability_thresholds.near_miss_clearance_m,
                longitudinal_decel_threshold_mps2=interpretability_thresholds.brake_accel_threshold,
                severe_jerk_mps3=interpretability_thresholds.severe_jerk_mps3,
                severe_lat_accel_mps2=interpretability_thresholds.severe_lat_accel_mps2,
                severe_decel_mps2=interpretability_thresholds.severe_decel_mps2,
            )
            post_rollout_result = evaluate_rollout(
                ego_traj=ego_traj,
                ego_traj_gt=ego_traj_gt,
                ego_lw=ego_lw,
                other_traj=oth_traj,
                other_lw=oth_lw,
                collision_result={
                    "did_coll": bool(did_coll),
                    "veh_coll": veh_coll,
                    "coll_time": coll_t,
                },
                scene_metadata=scene_metadata,
                planner_metadata=planner_metadata,
                dt=float(dataset.dt),
                thresholds=post_thresholds,
                kinematics_upsample_dt=kinematics_upsample_dt,
            )
            interp_scene_record = post_rollout_result["scene_record"]

            trace_ref = {}
            trace_scene_id = scene_token_str or f"scene{int(i):06d}"
            trace_path = os.path.join(
                cfg.out, "traces", dataset_kind_str, str(cfg.planner),
                f"{trace_scene_id}_{(sidx_val if sidx_val is not None else 0)}.npz",
            )
            try:
                trace_ref = save_trace(post_rollout_result["trace_data"], trace_path)
            except Exception as trace_err:  # pragma: no cover - defensive I/O guard
                Logger.log(f"[warn] failed to save trace for scene {trace_scene_id}: {trace_err}")

            # Persist the trace reference on BOTH the legacy results record and the
            # interpretable scene record. The downstream comparison/plotting stage
            # reads interpretable_scene_records.jsonl, so trace_path must live there
            # for panels e/f to load per-scene temporal traces.
            trace_fields = {
                "trace_path": trace_ref.get("trace_path"),
                "trace_num_steps": trace_ref.get("trace_num_steps"),
                "trace_dt": trace_ref.get("trace_dt"),
            }
            interp_scene_record.update(trace_fields)

            # Merge interpretable fields (new schema) without clobbering legacy keys
            # that the untouched report code still relies on.
            for _k, _v in interp_scene_record.items():
                record.setdefault(_k, _v)
            record.update(trace_fields)
            if "data_quality_flags" not in record:
                record["data_quality_flags"] = post_rollout_result["data_quality_flags"]
            interp_scene_records.append(interp_scene_record)
        except Exception as post_err:  # pragma: no cover - never break the rollout loop
            Logger.log(f"[warn] post-rollout interpretable evaluation failed for scene {i}: {post_err}")

        results_records.append(record)
        results_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        results_f.flush()

        # viz
        if viz_dir:
            out_prefix = os.path.join(viz_dir, f"{i:06d}_planner_{cfg.planner}" + ("_coll" if did_coll else ""))
            # Similar to utils/scenario_gen.py::viz_optim_results: use fixed bounds + crop_t
            # for more stable and nicer framing.
            crop_t = None
            if cfg.viz_crop_t is not None:
                ct = int(cfg.viz_crop_t)
                if ct < 0:
                    crop_t = int(cfg.past_len) - 1
                else:
                    crop_t = ct
            viz_bounds = [float(x) for x in cfg.viz_bounds]
            adv_idx = adv_meta_info.get("adv_idx")
            car_colors = nutils.get_adv_coloring(NA, attack_agt=adv_idx, tgt_agt=0)

            # 1) PNG: draw trajectories (more intuitive)
            nutils.viz_scene_graph(
                scene_graph,
                map_idx,
                map_env,
                bidx=0,
                out_path=out_prefix,
                state_normalizer=normalizer,
                att_normalizer=att_normalizer,
                future_pred=future_pred,
                viz_traj=True,
                make_video=False,
                show_gt=True,
                show_gt_idx=None,
                viz_bounds=viz_bounds,
                crop_t=crop_t,
                center_viz=crop_t is None,
                car_colors=car_colors,
            )

            # 2) MP4: do not draw trajectories (cleaner)
            if bool(cfg.viz_video):
                nutils.viz_scene_graph(
                    scene_graph,
                    map_idx,
                    map_env,
                    bidx=0,
                    out_path=out_prefix + "_vid",
                    state_normalizer=normalizer,
                    att_normalizer=att_normalizer,
                    future_pred=future_pred,
                    viz_traj=False,
                    make_video=True,
                    show_gt=True,
                    show_gt_idx=None,
                    viz_bounds=viz_bounds,
                    crop_t=crop_t,
                    center_viz=crop_t is None,
                    car_colors=car_colors,
                )

        total += 1
        if (i + 1) % 20 == 0:
            Logger.log(f"processed={total}, with_other={total_with_other}, ego_coll={total_coll}")

    results_f.close()

    # ---- Single-planner interpretable aggregation + scene-record dump ----
    if interp_scene_records:
        try:
            from agent.eval.crashsim_post_rollout_eval import aggregate_single_planner

            interp_jsonl_path = os.path.join(cfg.out, "interpretable_scene_records.jsonl")
            with open(interp_jsonl_path, "w", encoding="utf-8") as isf:
                for rec in interp_scene_records:
                    isf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            summary = aggregate_single_planner(interp_scene_records)
            summary_path = os.path.join(cfg.out, "single_planner_summary.json")
            with open(summary_path, "w", encoding="utf-8") as sf:
                json.dump(summary, sf, ensure_ascii=False, indent=2)
            Logger.log(
                f"[post-rollout] wrote {len(interp_scene_records)} interpretable scene records "
                f"to {interp_jsonl_path}; summary at {summary_path}"
            )
        except Exception as agg_err:  # pragma: no cover - defensive
            Logger.log(f"[warn] single-planner aggregation failed: {agg_err}")

    Logger.log("Done =====================================")
    Logger.log(f"processed={total}")
    Logger.log(f"with_other={total_with_other}")
    if total_with_other > 0:
        Logger.log(f"ego_coll={total_coll}  rate={total_coll / float(total_with_other):.4f}")
    Logger.log(f"results saved to {results_jsonl_path} (records={len(results_records)})")
    Logger.log("==========================================")

    if bool(getattr(cfg, "eval_report", False)) and results_records:
        _run_eval_report(
            cfg,
            results_records,
            results_jsonl_path,
            extra_context={
                "n_scenes": len(meta.scene),
                "n_subseq": len(dataset),
                "max_seqs": int(cfg.max_seqs),
            },
        )


if __name__ == "__main__":
    main()

