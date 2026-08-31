import os
import argparse
import json
import tqdm
import torch
import numpy as np
import copy
import socket
import math
from typing import Optional, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.legend_handler import HandlerPatch

from torch_geometric.data import DataLoader as GraphDataLoader
from torch_geometric.data import Batch as GraphBatch

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.arcline_path_utils import discretize_lane

from utils.config import get_parser, add_base_args, parse_diffusion_mode
from utils.common import dict2obj, mkdir
from utils.torch import get_device, load_state
from utils.logger import Logger, throw_err
from utils.scenario_gen import detach_embed_info
from planners.planner import PlannerConfig

from datasets.map_env import NuScenesMapEnv
from datasets.nuscenes_dataset import NuScenesDataset
import datasets.nuscenes_utils as nutils
from models.traffic_model import TrafficModel
from utils.scenario_gen import determine_feasibility_nusc, detach_embed_info
from utils.init_optim import run_init_optim


def parse_args():
    parser = get_parser('Process NuScenes scene_graph: save visualization and GPT prompt text.')
    parser = add_base_args(parser)

    parser.add_argument("--out_dir", type=str, default="/path/to/out/viz_nusc_scene_graph/val", help="Output directory.")
    parser.add_argument("--map_size", type=int, default=720, help="Map crop size in pixels.")
    parser.add_argument("--pix_per_m", type=int, default=8, help="Map rasterization density.")
    parser.add_argument("--margin_m", type=float, default=5.0, help="Extra crop margin in meters.")
    parser.add_argument("--min_bound_m", type=float, default=60.0, help="Minimum half-size crop bound in meters.")
    parser.add_argument(
        "--use_rank_id",
        action="store_true",
        help="If set, use surrounding-agent distance rank id (rank_id) for labels/prompt; otherwise use original agent_idx (aidx).",
    )
    parser.set_defaults(use_rank_id=True)
    parser.add_argument(
        "--coords",
        type=str,
        choices=["global", "local", "frenet"],
        default="local",
        help="Coordinate frame used in prompt text for ego/surrounding current states and historical trajectories.",
    )
    parser.add_argument(
        "--no_lane_relation_debug",
        action="store_true",
        help="Disable [lane_relation] debug logs (default: enabled, written to lane_relation_debug.log).",
    )

    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (e.g., train, val, test).")
    parser.add_argument("--val_size", type=int, default=400, help="Validation set size.")
    parser.add_argument("--seq_interval", type=int, default=4, help="Sequence interval.")
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help="Shuffle data")
    parser.set_defaults(shuffle=False)
    parser.add_argument("--pretrained_ae_path", type=str, help="Path to pretrained autoencoder model.")
    parser.add_argument("--feasibility_check_sep", type=bool, default=True, help="Enable feasibility check separation.")
    parser.add_argument('--planner_cfg', type=str, default='default',
                        help='hyperparameter configuration to use for the planner (if relevant)')

    # determining feasibility
    parser.add_argument('--feasibility_thresh', type=float, default=10.0, help='Future samples for target must be within this many meters from another agent for the initialization scenario to be feasible.')
    parser.add_argument('--feasibility_time', type=int, default=4, help='For feasibility, only consider timesteps >= feasibility_time, i.e., do not try to crash at timestep 0.')
    parser.add_argument('--feasibility_vel', type=float, default=0.5, help='maximum velocity (delta position of one timestep) of sampled trajectory for an agent must be >= this thresh to be considered feasible')
    parser.add_argument('--feasibility_infront_min', type=float, default=0.0, help='threshold for how in-front-of the ego vehicle the attacker is (measured by cosine similarity).')

    parser.add_argument("--viz", type=bool, default=True, help="Enable visualization.")
    parser.add_argument("--save", type=bool, default=True, help="Enable saving results.")
    parser.add_argument("--num_iters", type=int, default=50, help="Number of iterations.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--init_loss_motion_prior_ext", type=float, default=0.01, help="Initial loss for motion prior extension.")
    parser.add_argument("--init_loss_match_ext", type=float, default=10.0, help="Initial loss for match extension.")
    parser.add_argument("--loss_coll_veh", type=float, default=20.0, help="Loss weight for vehicle collision.")
    parser.add_argument("--loss_coll_veh_plan", type=float, default=20.0, help="Loss weight for vehicle collision in planning.")
    parser.add_argument("--loss_coll_env", type=float, default=20.0, help="Loss weight for environment collision.")
    parser.add_argument("--loss_init_z", type=float, default=0.5, help="Initial loss for latent variable z.")
    parser.add_argument("--loss_init_z_atk", type=float, default=0.05, help="Initial loss for latent variable z in attack.")
    parser.add_argument("--loss_motion_prior", type=float, default=1.0, help="Loss weight for motion prior.")
    parser.add_argument("--loss_motion_prior_atk", type=float, default=0.005, help="Loss weight for motion prior in attack.")
    parser.add_argument("--loss_motion_prior_ext", type=float, default=0.0001, help="Loss weight for motion prior extension.")
    parser.add_argument("--loss_match_ext", type=float, default=10.0, help="Loss weight for match extension.")
    parser.add_argument("--loss_adv_crash", type=float, default=2.0, help="Loss weight for adversarial crash.")
    parser.add_argument("--test_sample_num", type=int, default=10, help="Number of test samples.")
    parser.add_argument("--test_sample_future_len", type=int, default=12, help="Future length for test samples.")
    parser.add_argument("--test_sample_viz_multi", type=bool, default=True, help="Enable multi-sample visualization for test.")
    parser.add_argument("--test_sample_viz_rollout", type=bool, default=False, help="Enable rollout visualization for test samples.")
    parser.add_argument("--test_sample_disp_err", type=bool, default=True, help="Display error for test samples.")
    parser.add_argument("--test_sample_coll_rate", type=bool, default=True, help="Display collision rate for test samples.")
    parser.add_argument("--test_recon_viz_multi", type=bool, default=False, help="Enable multi-sample visualization for reconstruction.")
    parser.add_argument("--test_recon_coll_rate", type=bool, default=False, help="Display collision rate for reconstruction.")
    parser.add_argument("--n_timesteps", type=int, default=20, help="Number of timesteps.")
    parser.add_argument("--dim", type=int, default=64, help="Model dimension.")
    parser.add_argument("--dim_mults", type=int, nargs='+', default=[1, 4, 8], help="Dimension multipliers.")
    parser.add_argument('--predict_epsilon', dest='predict_epsilon', action='store_false',
                    help="If given, frozen traffic model and train diffusion, instead train traffic model.")
    parser.set_defaults(predict_epsilon=True)

    args = parser.parse_args()
    config_dict = vars(args)
    parse_diffusion_mode(config_dict) #update dict
    
    # Config dict to object
    config = dict2obj(config_dict)
    
    return config, config_dict


def get_eval_style_crop(past, margin_m=5.0, min_bound_m=60.0, center_on_ego=True):
    ego_now = past[0, -1, :2]
    ego_first = past[0, 0, :2]

    cur_other = past[1:, -1, :2]
    valid_other = ~torch.isnan(cur_other).any(dim=1)
    ref_pts = [ego_now, ego_first]
    if torch.any(valid_other):
        valid_cur_other = cur_other[valid_other]
        valid_idx = torch.nonzero(valid_other, as_tuple=True)[0]
        dists = torch.linalg.norm(valid_cur_other - ego_now.view(1, 2), dim=1)
        nearest_local = torch.argmin(dists)
        nearest_agent = valid_idx[nearest_local] + 1

        nearest_past = past[nearest_agent, :, :2]
        nearest_valid_steps = torch.nonzero(~torch.isnan(nearest_past[:, 0]), as_tuple=True)[0]
        if nearest_valid_steps.numel() > 0:
            ref_pts.append(nearest_past[nearest_valid_steps[0]])
        else:
            ref_pts.append(past[nearest_agent, -1, :2])

    ref_stack = torch.stack(ref_pts, dim=0)
    # Keep evaluation-style bound selection, but center the crop on ego to
    # make visualization/prompt coordinate frame ego-centered.
    crop_pos = ego_now if center_on_ego else torch.mean(ref_stack, dim=0)
    bound_diffs = ref_stack - crop_pos.view(1, 2)
    bound_max = torch.amax(torch.abs(bound_diffs)).item() + margin_m
    bound_max = max(bound_max, min_bound_m)
    bounds = [-bound_max, -bound_max, bound_max, bound_max]
    return crop_pos, bounds


def draw_annotated(
    map_rend,
    crop_traj,
    crop_lw,
    sorted_other_rank,
    out_path=None,
    ped_crossing_idx=None,
    use_rank_id=False,
):
    out_h = int(map_rend.shape[1])
    out_w = int(map_rend.shape[2])
    dpi = 100
    fig = plt.figure(figsize=(out_w / dpi, out_h / dpi), dpi=dpi)
    nutils.render_map_observation(map_rend.cpu())

    # Re-render ped_crossing with dedicated base color for readability.
    if ped_crossing_idx is not None and 0 <= ped_crossing_idx < map_rend.shape[0]:
        ped_mask = map_rend[ped_crossing_idx].cpu().numpy().T
        ped_valid = ped_mask > 0.5
        if np.any(ped_valid):
            h, w = ped_mask.shape
            # neutral gray close to map base color (avoid green tint)
            ped_color = np.array([0.76, 0.76, 0.76], dtype=np.float32)

            base_rgba = np.zeros((h, w, 4), dtype=np.float32)
            base_rgba[..., :3] = ped_color
            # stronger cover to suppress underlying walkway tint
            base_rgba[..., 3] = ped_valid.astype(np.float32) * 0.88
            plt.imshow(base_rgba, origin="lower", zorder=2.2)

    na = crop_traj.size(0)
    ego_color = "limegreen"
    other_color = "royalblue"
    arrow_color = "black"
    car_edge_lw = 0.6
    traj_dot_size = 6
    current_dot_size = 12
    arrow_len = 30.0

    for aidx in range(na):
        if aidx != 0 and aidx not in sorted_other_rank:
            # Only render top-k surrounding agents kept in rank mapping.
            continue

        color = ego_color if aidx == 0 else other_color
        traj = crop_traj[aidx]
        valid = ~torch.isnan(traj).any(dim=1)
        traj_valid = traj[valid]
        if traj_valid.size(0) == 0:
            continue

        # Hide history trajectory rendering.
        # plt.plot(traj_valid[:, 0], traj_valid[:, 1], "-", color=color, linewidth=1.4, alpha=0.85, zorder=4)
        # if traj_valid.size(0) > 1:
        #     plt.scatter(traj_valid[:-1, 0], traj_valid[:-1, 1], c=color, s=traj_dot_size, alpha=0.85, zorder=5)
        # plt.scatter(traj_valid[-1:, 0], traj_valid[-1:, 1], c=color, s=current_dot_size, alpha=0.95, zorder=6)

        cur = traj_valid[-1]
        cur_l = crop_lw[aidx, 0]
        cur_w = crop_lw[aidx, 1]
        box = np.array([cur[0].item(), cur[1].item(), cur[2].item(), cur[3].item()])
        lw_pair = [cur_l.item(), cur_w.item()]
        corners = nutils.get_corners(box, lw_pair)
        heading = np.arctan2(box[3], box[2])
        arrow = np.array([box[:2], box[:2] + lw_pair[0] / 2.0 * np.array([np.cos(heading), np.sin(heading)])])

        plt.fill(
            corners[:, 0],
            corners[:, 1],
            color=color,
            edgecolor="k",
            alpha=0.75,
            linewidth=car_edge_lw,
            zorder=3,
        )
        plt.plot(arrow[:, 0], arrow[:, 1], color=arrow_color, alpha=0.75, zorder=3, linewidth=car_edge_lw)

        hnorm = torch.linalg.norm(cur[2:4]).item()
        if hnorm > 1e-6:
            hx = (cur[2] / hnorm).item()
            hy = (cur[3] / hnorm).item()
            plt.arrow(
                cur[0].item(),
                cur[1].item(),
                hx * arrow_len,
                hy * arrow_len,
                head_width=6.0,
                head_length=8.6,
                linewidth=1.1,
                color=arrow_color,
                alpha=0.8,
                length_includes_head=True,
                zorder=8,
            )

        if aidx != 0:
            rank_id = sorted_other_rank.get(aidx)
            if rank_id is None:
                continue
            label_id = rank_id if use_rank_id else aidx
            plt.text(
                cur[0].item(),
                cur[1].item(),
                str(label_id),
                color="white",
                fontsize=9,
                ha="center",
                va="center",
                fontweight="bold",
                zorder=9,
            )

    def _legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        return FancyArrowPatch(
            (xdescent, ydescent + 0.5 * height),
            (xdescent + width, ydescent + 0.5 * height),
            arrowstyle="-|>",
            mutation_scale=max(fontsize * 1.4, 10.0),
            color=arrow_color,
            lw=1.8,
        )

    legend_handles = [
        Patch(facecolor=ego_color, edgecolor="k", label="Ego vehicle"),
        Patch(facecolor=other_color, edgecolor="k", label="Surrounding vehicle"),
        FancyArrowPatch((0.0, 0.5), (1.0, 0.5), arrowstyle="-|>", mutation_scale=15, color=arrow_color, lw=1.8, label="Heading"),
    ]
    plt.legend(
        handles=legend_handles,
        handler_map={FancyArrowPatch: HandlerPatch(patch_func=_legend_arrow)},
        loc="upper left",
        fontsize=9,
        framealpha=0.95,
        facecolor="white",
    )

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, map_rend.shape[2])
    plt.ylim(0, map_rend.shape[1])
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    if out_path:
        plt.savefig(out_path, pad_inches=0)
    plt.close(fig)
    return img


def _fmt_float(v):
    if isinstance(v, torch.Tensor):
        v = v.item()
    if np.isnan(v):
        return "nan"
    return f"{float(v):.3f}"


def _pose_line(pose, coord_mode="local"):
    hx = pose[2].item() if isinstance(pose[2], torch.Tensor) else float(pose[2])
    hy = pose[3].item() if isinstance(pose[3], torch.Tensor) else float(pose[3])
    heading = np.arctan2(hy, hx)
    if coord_mode == "frenet":
        return (
            f"(s={_fmt_float(pose[0])} m, d={_fmt_float(pose[1])} m, "
            f"heading={_fmt_float(heading)})"
        )
    return (
        f"(x={_fmt_float(pose[0])} m, y={_fmt_float(pose[1])} m, "
        f"heading={_fmt_float(heading)})"
    )


def _frenet_current_state_line(pose_frenet, speed, pose_global):
    if isinstance(speed, torch.Tensor):
        speed = float(speed.item())
    else:
        speed = float(speed)
    if np.isnan(speed):
        v_long = float("nan")
        v_lat = float("nan")
    else:
        hx = pose_frenet[2].item() if isinstance(pose_frenet[2], torch.Tensor) else float(pose_frenet[2])
        hy = pose_frenet[3].item() if isinstance(pose_frenet[3], torch.Tensor) else float(pose_frenet[3])
        if np.isnan(hx) or np.isnan(hy):
            v_long = float("nan")
            v_lat = float("nan")
        else:
            v_long = speed * hx
            v_lat = speed * hy

    ghx = pose_global[2].item() if isinstance(pose_global[2], torch.Tensor) else float(pose_global[2])
    ghy = pose_global[3].item() if isinstance(pose_global[3], torch.Tensor) else float(pose_global[3])
    g_heading = np.arctan2(ghy, ghx)
    return (
        f"(s={_fmt_float(pose_frenet[0])} m, d={_fmt_float(pose_frenet[1])} m, "
        f"longitudinal velocity={_fmt_float(v_long)} m/s, lateral velocity={_fmt_float(v_lat)} m/s), "
        f"global heading={_fmt_float(g_heading)}"
    )


def _current_state_line(pose, pose_frenet, pose_global, speed, coord_mode="local"):
    # coord_mode controls how we represent the current state line.
    # - "frenet": use detailed Frenet representation with longitudinal/lateral velocity.
    # - "local"/"global": use (x, y, heading), plus scalar speed.
    if coord_mode == "frenet":
        return _frenet_current_state_line(pose_frenet, speed, pose_global)
    return f"{_pose_line(pose, coord_mode=coord_mode)}, speed={_fmt_float(speed)} m/s"


def _speed_line(speed):
    return f"{_fmt_float(speed)} m/s"


def _fmt_metric(v, unit=""):
    if v is None:
        return "nan"
    if isinstance(v, torch.Tensor):
        v = v.item()
    v = float(v)
    if np.isnan(v):
        return "nan"
    if np.isinf(v):
        return "inf"
    if unit:
        return f"{v:.3f} {unit}"
    return f"{v:.3f}"


def _fmt_ttc_text(ttc_s):
    if isinstance(ttc_s, torch.Tensor):
        ttc_s = ttc_s.item()
    v = float(ttc_s)
    if np.isnan(v):
        return "nan"
    if np.isinf(v):
        return "inf"
    return f"{v:.2f}s"


def _fmt_mdc_text(mdc_m):
    if isinstance(mdc_m, torch.Tensor):
        mdc_m = mdc_m.item()
    v = float(mdc_m)
    if np.isnan(v):
        return "nan"
    if np.isinf(v):
        return "inf"
    return f"{v:.2f} m"


def _compute_ttc_mdc(current_state, current_speed, agent_idx):
    """
    Estimate TTC and MDC between ego (index 0) and one surrounding agent
    under constant-velocity, point-mass assumption in ego-current frame.
    """
    ego_pose = current_state[0]
    ag_pose = current_state[agent_idx]
    if torch.isnan(ego_pose).any() or torch.isnan(ag_pose).any():
        return float("nan"), float("nan")

    ego_speed = current_speed[0].to(ego_pose.device) 
    ag_speed = current_speed[agent_idx].to(ego_pose.device) 
    if torch.isnan(ego_speed) or torch.isnan(ag_speed):
        return float("nan"), float("nan")

    ego_h = ego_pose[2:4]
    ag_h = ag_pose[2:4]
    ego_hn = torch.linalg.norm(ego_h).item()
    ag_hn = torch.linalg.norm(ag_h).item()
    if ego_hn < 1e-6 or ag_hn < 1e-6:
        return float("nan"), float("nan")

    ego_dir = ego_h / ego_hn
    ag_dir = ag_h / ag_hn
    ego_v = ego_dir * ego_speed
    ag_v = ag_dir * ag_speed

    rel_p = ag_pose[:2] - ego_pose[:2]
    rel_v = ag_v - ego_v
    rel_v_sq = float(torch.dot(rel_v, rel_v).item())

    if rel_v_sq < 1e-8:
        mdc = float(torch.linalg.norm(rel_p).item())
        return float("inf"), mdc

    # Time to closest approach (non-negative); used as TTC proxy.
    t_star = -float(torch.dot(rel_p, rel_v).item()) / rel_v_sq
    if t_star < 0.0:
        t_star = 0.0
    closest_rel = rel_p + rel_v * t_star
    mdc = float(torch.linalg.norm(closest_rel).item())

    closing_rate = -float(torch.dot(rel_p, rel_v).item())
    ttc = t_star if closing_rate > 0.0 else float("inf")
    return ttc, mdc


def _traj_xy(traj):
    if isinstance(traj, torch.Tensor):
        xy = traj[..., :2].detach().cpu().numpy()
    else:
        xy = np.asarray(traj)[..., :2]
    return xy.astype(np.float64)


def _compute_ttc_from_pred_future(pred_ego_future, pred_ag_future, dt=0.5):
    """
    Minimum time-to-collision between ego and one agent from predicted future xy.
    Uses closing-speed TTC along the prediction horizon (same logic as eval_adv_gen).
    """
    ego_xy = _traj_xy(pred_ego_future)
    ag_xy = _traj_xy(pred_ag_future)
    n = min(ego_xy.shape[0], ag_xy.shape[0])
    if n < 2:
        return float("nan")

    ego_xy = ego_xy[:n]
    ag_xy = ag_xy[:n]
    valid = np.isfinite(ego_xy).all(axis=1) & np.isfinite(ag_xy).all(axis=1)
    if int(valid.sum()) < 2:
        return float("nan")

    ego_xy = ego_xy[valid]
    ag_xy = ag_xy[valid]
    ct = ego_xy.shape[0]
    if ct < 2:
        return float("nan")

    ego_vel = np.diff(ego_xy, axis=0) / dt
    ag_vel = np.diff(ag_xy, axis=0) / dt
    rel_p = ag_xy[: ct - 1] - ego_xy[: ct - 1]
    rel_v = ag_vel - ego_vel
    dist = np.linalg.norm(rel_p, axis=-1)
    dot = np.sum(rel_p * rel_v, axis=-1)
    closing_speed = np.divide(-dot, dist, out=np.zeros_like(dist), where=dist > 1e-6)
    ttc_t = np.divide(
        dist,
        closing_speed,
        out=np.full_like(dist, np.inf),
        where=closing_speed > 1e-6,
    )
    ttc_t = np.where(dist <= 1e-6, 0.0, ttc_t)
    finite = ttc_t[np.isfinite(ttc_t)]
    if finite.size == 0:
        return float("inf")
    return float(np.min(finite))


def _compute_mdc_from_pred_future(pred_ego_future, pred_ag_future):
    """
    Minimum Distance to Collision (MDC): min over aligned timesteps of
    ||p_ego(t) - p_adv(t)|| (m). Same alignment as evaluation.compute_mdc_ego_adv.
    """
    ego_xy = _traj_xy(pred_ego_future)
    ag_xy = _traj_xy(pred_ag_future)
    n = min(ego_xy.shape[0], ag_xy.shape[0])
    if n <= 0:
        return float("nan")

    ego_xy = ego_xy[:n]
    ag_xy = ag_xy[:n]
    valid = np.isfinite(ego_xy).all(axis=1) & np.isfinite(ag_xy).all(axis=1)
    if int(valid.sum()) <= 0:
        return float("nan")

    dist = np.linalg.norm(ag_xy[valid] - ego_xy[valid], axis=-1)
    if dist.size == 0:
        return float("nan")
    return float(np.min(dist))


def _historical_positions_line(traj, decimals=2, coord_mode="local"):
    points = []
    for t in range(traj.size(0)):
        x = traj[t, 0]
        y = traj[t, 1]
        if isinstance(x, torch.Tensor):
            x = x.item()
        if isinstance(y, torch.Tensor):
            y = y.item()
        if np.isnan(x) or np.isnan(y):
            points.append("(nan, nan)")
        else:
            if coord_mode == "frenet":
                points.append(f"(s={float(x):.{decimals}f}, d={float(y):.{decimals}f})")
            else:
                points.append(f"({float(x):.{decimals}f}, {float(y):.{decimals}f})")
    return "[" + ", ".join(points) + "]"


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _build_centerline_arclength(xy):
    n = xy.shape[0]
    if n < 2:
        return None
    seg = xy[1:] - xy[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    if np.any(~np.isfinite(seg_len)):
        return None
    s = np.zeros((n,), dtype=np.float32)
    if n > 1:
        s[1:] = np.cumsum(seg_len, dtype=np.float32)
    return s


def _project_point_to_centerline_frenet(x, y, xy, s_along, s_ref=None, s_window_m=None):
    if xy.shape[0] < 2 or s_along is None:
        return None
    p = np.array([x, y], dtype=np.float32)
    use_window = s_ref is not None and s_window_m is not None and np.isfinite(s_ref)
    best_window = None
    best_global = None
    for i in range(xy.shape[0] - 1):
        a = xy[i]
        b = xy[i + 1]
        v = b - a
        seg_len = float(np.linalg.norm(v))
        if seg_len < 1e-6:
            continue
        t = float(np.dot(p - a, v) / (seg_len * seg_len))
        t = min(1.0, max(0.0, t))
        proj = a + t * v
        rel = p - proj
        d2 = float(np.dot(rel, rel))
        tangent = v / seg_len
        cross_z = float(tangent[0] * rel[1] - tangent[1] * rel[0])
        sign = 1.0 if cross_z >= 0.0 else -1.0
        cand = (
            d2,
            sign * float(np.sqrt(max(d2, 0.0))),
            float(s_along[i] + t * seg_len),
            float(np.arctan2(tangent[1], tangent[0])),
        )
        if best_global is None or cand[0] < best_global[0]:
            best_global = cand
        if use_window and abs(cand[2] - float(s_ref)) <= float(s_window_m):
            if best_window is None or cand[0] < best_window[0]:
                best_window = cand
    pick = best_window if best_window is not None else best_global
    if pick is None:
        return None
    _, best_d, best_s, best_h = pick
    if not np.isfinite(best_s) or not np.isfinite(best_d):
        return None
    return best_s, best_d, best_h


def build_frenet_history_states(past4):
    """
    Convert global history states into ego-frenet frame per timestamp.
    At each timestamp t, ego state at t is used as local reference.
    Returns:
      frenet_hist: [na, nt, 4] where columns are [delta_x, delta_y, hx, hy]
    """
    na, nt, _ = past4.shape
    past4_np = past4.detach().cpu().numpy()
    frenet_hist = np.full_like(past4_np, np.nan, dtype=np.float32)
    for t in range(nt):
        ego_t = past4_np[0, t]
        if np.isnan(ego_t).any():
            continue
        local_t = nutils.objects2frame(past4_np[:, t : t + 1, :], ego_t, toworld=False)[:, 0, :]
        frenet_hist[:, t, :] = local_t
    return torch.from_numpy(frenet_hist)


def build_lane_frenet_history_states(past, map_env=None, map_name=None):
    """
    Convert global history states into lane-frenet frame based on
    ego current lane centerline.
    If lane matching fails, fallback to ego-current local frame.
    Returns:
      frenet_hist: [na, nt, 4] where columns are [s, d, hx, hy]
    """
    na, nt, _ = past.shape
    past_np = past.detach().cpu().numpy()
    frenet_hist = np.full_like(past_np, np.nan, dtype=np.float32)
    ego_cur = past_np[0, -1]
    if map_env is None or map_name is None or np.isnan(ego_cur).any():
        for t in range(nt):
            ego_t = past_np[0, t]
            if np.isnan(ego_t).any():
                continue
            local_t = nutils.objects2frame(past_np[:, t : t + 1, :], ego_t, toworld=False)[:, 0, :]
            frenet_hist[:, t, :] = local_t
        return torch.from_numpy(frenet_hist)

    nmap = map_env.nusc_maps[map_name]
    cl_cache = {}
    ego_heading = _safe_heading(float(ego_cur[2]), float(ego_cur[3]))
    ego_lane, _, _ = _closest_lane_with_heading(
        nmap,
        float(ego_cur[0]),
        float(ego_cur[1]),
        ego_heading,
        cl_cache,
        search_radius_m=20.0,
        max_lane_dist_m=6.0,
    )
    if ego_lane is None:
        return torch.from_numpy(frenet_hist)

    xy = _build_connected_reference_centerline(nmap, ego_lane, cl_cache, res_m=1.0)
    if xy is None:
        centerline = _centerline_xy_and_heading(nmap, ego_lane, cl_cache, res_m=1.0)
        if centerline is None:
            return torch.from_numpy(frenet_hist)
        xy, _ = centerline
    s_along = _build_centerline_arclength(xy)
    if s_along is None:
        return torch.from_numpy(frenet_hist)

    for aidx in range(na):
        for t in range(nt):
            st = past_np[aidx, t]
            if np.isnan(st).any():
                continue
            fr = _project_point_to_centerline_frenet(float(st[0]), float(st[1]), xy, s_along)
            if fr is None:
                continue
            s_val, d_val, lane_h = fr
            obj_h = _safe_heading(float(st[2]), float(st[3]))
            if lane_h is None or obj_h is None:
                hd_x, hd_y = np.nan, np.nan
            else:
                dh = _wrap_to_pi(obj_h - lane_h)
                hd_x, hd_y = np.cos(dh), np.sin(dh)
            frenet_hist[aidx, t, 0] = s_val
            frenet_hist[aidx, t, 1] = d_val
            frenet_hist[aidx, t, 2] = hd_x
            frenet_hist[aidx, t, 3] = hd_y
    return torch.from_numpy(frenet_hist)


def classify_spatial_relation(delta_x, delta_y, delta_heading):
    """
    Classify surrounding spatial relation into:
      same-direction front/rear vehicle,
      opposing front/rear vehicle,
      crossing front/rear vehicle,
      left front/rear vehicle,
      right front/rear vehicle.
    """
    same_heading_th = np.deg2rad(35.0)
    opp_heading_th = np.deg2rad(145.0)
    lateral_dom_th = 3.0  # meters

    if np.isnan(delta_x) or np.isnan(delta_y) or np.isnan(delta_heading):
        return "spatial relation unknown"

    front_or_rear = "front vehicle" if delta_x >= 0.0 else "rear vehicle"
    abs_dh = abs(_wrap_to_pi(delta_heading))

    if abs_dh <= same_heading_th:
        if abs(delta_y) >= lateral_dom_th:
            if delta_y >= 0.0:
                return f"left {front_or_rear}"
            return f"right {front_or_rear}"
        return f"same-direction {front_or_rear}"

    if abs_dh >= opp_heading_th:
        return f"opposing {front_or_rear}"

    return f"crossing {front_or_rear}"


def _agent_drawable_in_crop(aidx, crop_traj, crop_lw, map_size=None, bounds_margin_px=40.0):
    """
    True when draw_annotated would render this agent (valid crop pose + box size).
    Agents outside the crop or with NaN objs2crop output are excluded from png/txt.
    """
    traj = crop_traj[aidx]
    valid = ~torch.isnan(traj).any(dim=1)
    if not valid.any():
        return False
    traj_valid = traj[valid]
    if traj_valid.size(0) == 0:
        return False
    cur = traj_valid[-1]
    if torch.isnan(cur[:4]).any():
        return False
    length = crop_lw[aidx, 0]
    width = crop_lw[aidx, 1]
    if torch.isnan(length) or torch.isnan(width):
        return False
    if map_size is not None:
        cx = float(cur[0].item())
        cy = float(cur[1].item())
        half = max(float(length.item()), float(width.item())) * 0.5 + float(bounds_margin_px)
        ms = float(map_size)
        if cx < -half or cy < -half or cx > ms + half or cy > ms + half:
            return False
    return True


def filter_sorted_other_rank_for_crop_viz(
    sorted_other_rank, crop_traj, crop_lw, map_size=None
):
    """Keep only surrounding agents that appear in the map crop visualization."""
    return {
        int(aidx): int(rank)
        for aidx, rank in sorted_other_rank.items()
        if _agent_drawable_in_crop(int(aidx), crop_traj, crop_lw, map_size=map_size)
    }


def build_surrounding_agent_rank(current_state, max_agents=10):
    """
    Rank surrounding agents by Euclidean distance to ego current position.
    Returns dict: agent_idx -> distance-rank (1-based).
    Rank is used for selecting/ordering top-k surrounding agents.
    Displayed Agent ID can be controlled by caller (original agent_idx vs rank_id).
    """
    na = current_state.size(0)
    if na <= 1:
        return {}

    ego_xy = current_state[0, :2]
    pairs = []
    for aidx in range(1, na):
        st = current_state[aidx]
        if torch.isnan(st).any():
            continue
        dist = torch.linalg.norm(st[:2] - ego_xy).item()
        if np.isnan(dist):
            continue
        pairs.append((dist, aidx))

    pairs.sort(key=lambda x: x[0])
    pairs = pairs[:max_agents]
    return {aidx: rid + 1 for rid, (_, aidx) in enumerate(pairs)}


def _centerline_xy_and_heading(nmap, lane_token, cache, res_m=1.0):
    if lane_token in cache:
        return cache[lane_token]

    arcline = nmap.arcline_path_3.get(lane_token, [])
    if len(arcline) == 0:
        cache[lane_token] = None
        return None

    try:
        pts = np.asarray(discretize_lane(arcline, res_m), dtype=np.float32)
    except Exception:
        cache[lane_token] = None
        return None

    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        cache[lane_token] = None
        return None

    xy = pts[:, :2]
    finite_mask = np.isfinite(xy).all(axis=1)
    xy = xy[finite_mask]
    if xy.shape[0] < 2:
        cache[lane_token] = None
        return None

    dxy = np.zeros_like(xy)
    dxy[:-1] = xy[1:] - xy[:-1]
    dxy[-1] = dxy[-2]
    seg_norm = np.linalg.norm(dxy, axis=1)
    bad = seg_norm < 1e-6
    if np.any(bad):
        dxy[bad] = np.array([1.0, 0.0], dtype=np.float32)
    heading = np.arctan2(dxy[:, 1], dxy[:, 0])
    out = (xy, heading)
    cache[lane_token] = out
    return out


def _polyline_length(xy):
    if xy is None or xy.shape[0] < 2:
        return 0.0
    seg = xy[1:] - xy[:-1]
    return float(np.sum(np.linalg.norm(seg, axis=1)))


def _trace_connected_lanes(
    nmap, start_token, cache, direction, max_hops=6, max_length_m=120.0, ref_heading=None
):
    """
    Greedily trace connected lanes along map connectivity.
    direction: "forward" uses outgoing, "backward" uses incoming.
    ref_heading: optional ego/agent heading to prefer the driving corridor at forks.
    """
    if direction not in {"forward", "backward"}:
        return []

    chain = []
    cur = start_token
    visited = {start_token}
    total_len = 0.0

    for _ in range(max_hops):
        conn = nmap.connectivity.get(cur, {})
        key = "outgoing" if direction == "forward" else "incoming"
        cands = [tok for tok in (conn.get(key, []) or []) if tok not in visited]
        if len(cands) == 0:
            break

        cur_info = _centerline_xy_and_heading(nmap, cur, cache, res_m=1.0)
        if cur_info is None:
            break
        cur_xy, cur_heading = cur_info
        if direction == "forward":
            anchor_xy = cur_xy[-1]
            anchor_h = float(cur_heading[-1])
        else:
            anchor_xy = cur_xy[0]
            anchor_h = float(cur_heading[0])

        best_tok = None
        best_score = float("inf")
        best_len = 0.0
        for tok in cands:
            info = _centerline_xy_and_heading(nmap, tok, cache, res_m=1.0)
            if info is None:
                continue
            xy, heading = info
            if direction == "forward":
                cand_xy = xy[0]
                cand_h = float(heading[0])
            else:
                cand_xy = xy[-1]
                cand_h = float(heading[-1])
            gap = float(np.linalg.norm(cand_xy - anchor_xy))
            dh = abs(_wrap_to_pi(cand_h - anchor_h))
            score = gap + 2.0 * dh
            if ref_heading is not None:
                score += 1.5 * abs(_wrap_to_pi(cand_h - ref_heading))
            if score < best_score:
                best_score = score
                best_tok = tok
                best_len = _polyline_length(xy)

        if best_tok is None:
            break

        chain.append(best_tok)
        visited.add(best_tok)
        cur = best_tok
        total_len += best_len
        if total_len >= max_length_m:
            break

    return chain


def _build_connected_reference_centerline(nmap, ego_lane, cache, res_m=1.0, ego_heading=None):
    """
    Build a longer reference centerline by stitching connected lanes:
    upstream lanes + ego lane + downstream lanes.
    """
    ego_info = _centerline_xy_and_heading(nmap, ego_lane, cache, res_m=res_m)
    if ego_info is None:
        return None

    backward = _trace_connected_lanes(
        nmap,
        ego_lane,
        cache,
        direction="backward",
        max_hops=4,
        max_length_m=60.0,
        ref_heading=ego_heading,
    )
    forward = _trace_connected_lanes(
        nmap,
        ego_lane,
        cache,
        direction="forward",
        max_hops=8,
        max_length_m=180.0,
        ref_heading=ego_heading,
    )
    ordered_tokens = list(reversed(backward)) + [ego_lane] + forward

    stitched = []
    for tok in ordered_tokens:
        info = _centerline_xy_and_heading(nmap, tok, cache, res_m=res_m)
        if info is None:
            continue
        xy, _ = info
        if len(stitched) == 0:
            stitched.append(xy.copy())
            continue
        last_pt = stitched[-1][-1]
        start_pt = xy[0]
        if float(np.linalg.norm(start_pt - last_pt)) < 2.5 and xy.shape[0] > 1:
            stitched.append(xy[1:].copy())
        else:
            stitched.append(xy.copy())

    if len(stitched) == 0:
        return None
    out_xy = np.concatenate(stitched, axis=0)
    if out_xy.shape[0] < 2:
        return None
    return out_xy


def _point_to_polyline_distance(x, y, polyline_xy):
    diff = polyline_xy - np.array([x, y], dtype=np.float32).reshape(1, 2)
    d2 = np.sum(diff * diff, axis=1)
    min_idx = int(np.argmin(d2))
    return float(np.sqrt(d2[min_idx])), min_idx


def _signed_lateral_to_lane_centerline(nmap, lane_token, x, y, cl_cache):
    """
    Signed lateral offset (m) from (x, y) to a lane centerline.
    Positive/negative follows the usual frenet left/right convention.
    """
    lane_info = _centerline_xy_and_heading(nmap, lane_token, cl_cache, res_m=1.0)
    if lane_info is None:
        return None, None
    xy, heading = lane_info
    if xy.shape[0] < 2:
        return None, None
    s_along = _build_centerline_arclength(xy)
    if s_along is None:
        return None, None
    proj = _project_point_to_centerline_frenet(float(x), float(y), xy, s_along)
    if proj is None:
        return None, None
    _, d_signed, lane_h = proj
    if not np.isfinite(d_signed):
        return None, None
    return float(abs(d_signed)), lane_h


def _closest_lane_distance_only(
    nmap,
    x,
    y,
    cl_cache,
    search_radius_m=12.0,
    max_lane_dist_m=6.0,
    connector_tokens=None,
    connector_penalty_m=2.0,
):
    """Snap to the geometrically nearest lane (no heading prior)."""
    return _closest_lane_with_heading(
        nmap,
        x,
        y,
        None,
        cl_cache,
        search_radius_m=search_radius_m,
        max_lane_dist_m=max_lane_dist_m,
        connector_tokens=connector_tokens,
        connector_penalty_m=connector_penalty_m,
    )


def _safe_heading(hx, hy):
    if not np.isfinite(hx) or not np.isfinite(hy):
        return None
    hn = np.hypot(hx, hy)
    if hn < 1e-6:
        return None
    return float(np.arctan2(hy, hx))


def _closest_lane_with_heading(
    nmap,
    x,
    y,
    obj_heading,
    cl_cache,
    search_radius_m=12.0,
    max_lane_dist_m=6.0,
    connector_tokens=None,
    connector_penalty_m=2.0,
):
    layers = ["lane", "lane_connector"]
    try:
        recs = nmap.get_records_in_radius(float(x), float(y), float(search_radius_m), layers)
    except Exception:
        recs = {k: [] for k in layers}

    cands = []
    for layer in layers:
        cands.extend(recs.get(layer, []))
    # keep order, remove duplicates
    cands = list(dict.fromkeys(cands))
    if len(cands) == 0:
        return None, float("inf"), None

    best_token = None
    best_score = float("inf")
    best_dist = float("inf")
    best_lane_heading = None

    for tok in cands:
        lane_info = _centerline_xy_and_heading(nmap, tok, cl_cache, res_m=1.0)
        if lane_info is None:
            continue
        xy, heading = lane_info
        dist, idx = _point_to_polyline_distance(x, y, xy)
        lane_h = float(heading[idx])

        score = dist
        if obj_heading is not None:
            # light heading prior to avoid snapping to wrong-direction lane.
            score += 0.8 * abs(_wrap_to_pi(lane_h - obj_heading))
        if connector_tokens is not None and tok in connector_tokens:
            score += float(connector_penalty_m)
        if score < best_score:
            best_score = score
            best_dist = dist
            best_token = tok
            best_lane_heading = lane_h

    if best_token is None or best_dist > max_lane_dist_m:
        return None, float("inf"), None
    return best_token, best_dist, best_lane_heading


def _nearest_lane_in_radius(
    nmap,
    x,
    y,
    obj_heading,
    cl_cache,
    search_radius_m=40.0,
    connector_tokens=None,
    connector_penalty_m=2.0,
    use_heading_prior=True,
):
    """
    Nearest lane/connector by geometry within radius, without max_lane_dist_m cutoff.
    Used by snap fallbacks so a candidate at 15 m is not discarded when max_dist=6 m.
    """
    layers = ["lane", "lane_connector"]
    try:
        recs = nmap.get_records_in_radius(
            float(x), float(y), float(search_radius_m), layers
        )
    except Exception:
        recs = {k: [] for k in layers}

    cands = []
    for layer in layers:
        cands.extend(recs.get(layer, []))
    cands = list(dict.fromkeys(cands))
    if len(cands) == 0:
        return None, float("inf"), None, 0

    best_token = None
    best_score = float("inf")
    best_dist = float("inf")
    best_lane_heading = None

    for tok in cands:
        lane_info = _centerline_xy_and_heading(nmap, tok, cl_cache, res_m=1.0)
        if lane_info is None:
            continue
        xy, heading = lane_info
        dist, idx = _point_to_polyline_distance(x, y, xy)
        lane_h = float(heading[idx])
        score = dist
        if use_heading_prior and obj_heading is not None:
            score += 0.5 * abs(_wrap_to_pi(lane_h - obj_heading))
        if connector_tokens is not None and tok in connector_tokens:
            score += float(connector_penalty_m)
        if score < best_score:
            best_score = score
            best_dist = dist
            best_token = tok
            best_lane_heading = lane_h

    if best_token is None:
        return None, float("inf"), None, len(cands)
    return best_token, best_dist, best_lane_heading, len(cands)


def _closest_lane_best_effort(
    nmap,
    x,
    y,
    obj_heading,
    cl_cache,
    connector_tokens=None,
    connector_penalty_m=2.5,
    search_radius_m=40.0,
    geometric_cap_m=25.0,
):
    """Return the geometrically closest lane within *geometric_cap_m* (no strict 6 m cut)."""
    layers = ["lane", "lane_connector"]
    try:
        recs = nmap.get_records_in_radius(
            float(x), float(y), float(search_radius_m), layers
        )
    except Exception:
        recs = {k: [] for k in layers}

    cands = []
    for layer in layers:
        cands.extend(recs.get(layer, []))
    cands = list(dict.fromkeys(cands))
    if len(cands) == 0:
        return None, float("inf"), None

    best_token = None
    best_score = float("inf")
    best_dist = float("inf")
    best_lane_heading = None

    for tok in cands:
        lane_info = _centerline_xy_and_heading(nmap, tok, cl_cache, res_m=1.0)
        if lane_info is None:
            continue
        xy, heading = lane_info
        dist, idx = _point_to_polyline_distance(x, y, xy)
        lane_h = float(heading[idx])
        score = dist
        if obj_heading is not None:
            score += 0.5 * abs(_wrap_to_pi(lane_h - obj_heading))
        if connector_tokens is not None and tok in connector_tokens:
            score += float(connector_penalty_m)
        if score < best_score:
            best_score = score
            best_dist = dist
            best_token = tok
            best_lane_heading = lane_h

    if best_token is None or best_dist > geometric_cap_m:
        return None, float("inf"), None
    return best_token, best_dist, best_lane_heading


def _lane_token_layers_on_point(nmap, x, y, connector_tokens=None):
    """Direct map point query fallback when radius search finds nothing."""
    try:
        info = nmap.layers_on_point(
            float(x), float(y), layer_names=["lane", "lane_connector"]
        )
    except Exception:
        return None, float("inf"), None
    if not isinstance(info, dict):
        return None, float("inf"), None
    tok = info.get("lane") or info.get("lane_connector")
    if not tok:
        return None, float("inf"), None
    if connector_tokens is not None and tok in connector_tokens:
        # prefer explicit lane layer when both exist
        tok = info.get("lane") or tok
    return tok, 0.0, None


def _ego_frame_lateral_m(ego_x, ego_y, ego_heading, ax, ay):
    """Signed lateral offset of (ax, ay) in ego frame; matches objects2frame."""
    if ego_heading is None:
        return None
    dx = float(ax) - float(ego_x)
    dy = float(ay) - float(ego_y)
    c = np.cos(ego_heading)
    s = np.sin(ego_heading)
    return float(-s * dx + c * dy)


def _min_lane_centerline_separation_m(nmap, lane_tok_a, lane_tok_b, cl_cache):
    """Minimum distance (m) between two lane/connector centerline polylines."""
    if lane_tok_a is None or lane_tok_b is None:
        return None
    if lane_tok_a == lane_tok_b:
        return 0.0
    a_info = _centerline_xy_and_heading(nmap, lane_tok_a, cl_cache, res_m=1.0)
    b_info = _centerline_xy_and_heading(nmap, lane_tok_b, cl_cache, res_m=1.0)
    if a_info is None or b_info is None:
        return None
    return _polyline_min_distance(a_info[0], b_info[0])


def _polyline_min_distance(poly_a, poly_b):
    # small centerline sets; direct pairwise distance is fine here.
    delta = poly_a[:, None, :] - poly_b[None, :, :]
    d2 = np.sum(delta * delta, axis=2)
    return float(np.sqrt(np.min(d2)))


def _gather_road_dividers(nmap, cx, cy, radius_m=80.0):
    """
    Collect shapely LineString geometries for `road_divider` records near (cx, cy).
    Per nuScenes schema, road_divider separates lanes flowing in OPPOSING
    directions on a physical road. This includes both physical medians and
    painted center lines; a line-crossing test alone cannot distinguish them,
    so road_divider is combined with a drivable-area gap test downstream.
    """
    try:
        recs = nmap.get_records_in_radius(float(cx), float(cy), float(radius_m), ["road_divider"])
    except Exception:
        return []
    out = []
    for tok in recs.get("road_divider", []):
        try:
            rec = nmap.get("road_divider", tok)
            line = nmap.extract_line(rec["line_token"])
        except Exception:
            continue
        if line is None or line.is_empty:
            continue
        out.append(line)
    return out


def _segment_crosses_any_divider(p1, p2, dividers):
    """
    Whether the straight line segment from p1 to p2 crosses any of the
    pre-extracted road_divider line geometries.
    """
    if not dividers:
        return False
    try:
        from shapely.geometry import LineString as ShapelyLine
    except Exception:
        return False
    try:
        seg = ShapelyLine([(float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))])
    except Exception:
        return False
    if seg.is_empty or seg.length < 1e-6:
        return False
    for line in dividers:
        try:
            if seg.intersects(line):
                return True
        except Exception:
            continue
    return False


def _point_is_drivable(nmap, x, y):
    """Return True if (x, y) is inside any drivable_area polygon."""
    try:
        info = nmap.layers_on_point(float(x), float(y), layer_names=["drivable_area"])
    except Exception:
        return True  # fail-open: avoid false median detections on API errors.
    if not isinstance(info, dict):
        return True
    return bool(info.get("drivable_area", ""))


def _segment_has_nondrivable_gap(nmap, p1, p2, samples=11, min_run=4):
    """
    Sample points along segment p1->p2 and return True if a consecutive run of
    `min_run` interior samples falls outside any drivable_area polygon, i.e.
    there is a non-drivable gap (a physical median / curb / grass strip).
    Short runs are ignored because lane polygons are not gap-filling at the
    painted center line of a two-way road without median: a thin centerline
    marking can briefly leave drivable_area but quickly re-enters on the other
    lane. Use a higher min_run so opposing-adjacent pairs on undivided roads
    are not misclassified as separated by a median.
    """
    if samples < 3:
        samples = 3
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
        return False
    run = 0
    # use interior samples only (skip endpoints which are on the lanes themselves)
    for i in range(1, samples):
        t = i / samples
        sx = x1 + t * (x2 - x1)
        sy = y1 + t * (y2 - y1)
        if _point_is_drivable(nmap, sx, sy):
            run = 0
        else:
            run += 1
            if run >= min_run:
                return True
    return False


def _pair_in_different_road_blocks(nmap, p1, p2):
    """Different road_block tokens → separated by intersection arm / divided road."""
    t1 = _road_block_token_at(nmap, p1[0], p1[1])
    t2 = _road_block_token_at(nmap, p2[0], p2[1])
    return bool(t1 and t2 and t1 != t2)


def _pair_separated_by_physical_median(
    nmap,
    p1,
    p2,
    lateral_sep_m=None,
    road_dividers=None,
    ego_frame_lateral_m=None,
    close_lateral_th=3.5,
):
    """
    True when ego and agent are separated by a physical median / non-drivable strip.

    Uses ego-frame lateral for the 'close enough to skip' test — not map gap=0.
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
        return False
    seg_len = float(np.hypot(x2 - x1, y2 - y1))
    if seg_len < 1e-3:
        return False

    if _pair_in_different_road_blocks(nmap, p1, p2):
        return True

    close_lat = ego_frame_lateral_m
    if close_lat is None or not np.isfinite(close_lat):
        close_lat = lateral_sep_m
    if close_lat is not None and np.isfinite(close_lat) and close_lat <= float(close_lateral_th):
        return False

    samples = 13
    interior = max(1, samples - 2)
    min_run = max(4, int(np.ceil(0.42 * interior * min(1.0, seg_len / 14.0))))
    # Wide physical medians (divided roads): lower run threshold on longer chords.
    if seg_len > 12.0 and close_lat is not None and close_lat > 5.0:
        min_run = max(3, min_run - 1)

    if road_dividers and _segment_crosses_any_divider(p1, p2, road_dividers):
        return _segment_has_nondrivable_gap(nmap, p1, p2, samples=samples, min_run=2)

    return _segment_has_nondrivable_gap(nmap, p1, p2, samples=samples, min_run=min_run)


def _resolve_map_idx(map_env, map_name):
    try:
        return map_env.map_list.index(map_name)
    except ValueError:
        return 0


def _pair_separated_by_nondrivable_raster(map_env, map_idx, ego_xy, agent_xy):
    """
    True when the straight segment ego->agent crosses non-drivable area on the
    merged drivable raster (nusc_raster channel 0), matching
    determine_feasibility_nusc / check_line_layer.
    """
    x1, y1 = float(ego_xy[0]), float(ego_xy[1])
    x2, y2 = float(agent_xy[0]), float(agent_xy[1])
    if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
        return False
    if float(np.hypot(x2 - x1, y2 - y1)) < 1e-3:
        return False

    device = map_env.nusc_raster.device
    start = torch.tensor([[x1, y1]], dtype=torch.float32, device=device)
    end = torch.tensor([[x2, y2]], dtype=torch.float32, device=device)
    if isinstance(map_idx, torch.Tensor):
        mapix = map_idx.reshape(-1)[:1].to(device=device, dtype=torch.long)
    else:
        mapix = torch.tensor([int(map_idx)], dtype=torch.long, device=device)

    intersect = nutils.check_line_layer(
        map_env.nusc_raster[:, 0], map_env.nusc_dx, start, end, mapix
    )
    return bool(intersect[0].item())


def _road_block_token_at(nmap, x, y):
    """Return road_block token containing the point (x, y), or None."""
    try:
        info = nmap.layers_on_point(float(x), float(y), layer_names=["road_block"])
    except Exception:
        return None
    if not isinstance(info, dict):
        return None
    tok = info.get("road_block", "")
    return tok if tok else None


def _build_same_lane_family(nmap, ego_lane_token, cache=None, ego_heading=None):
    """
    Build tokens treated as "same lane" with ego:
    - ego lane itself
    - direct predecessor / successor lanes
    - plus one extra hop through lane_connector to cover
      lane -> connector -> lane transitions.
    - and a longer forward/backward greedy chain to cover
      farther same-lane agents in front/rear of ego.
    """
    lane_tokens = {rec["token"] for rec in getattr(nmap, "lane", [])}
    connector_tokens = {rec["token"] for rec in getattr(nmap, "lane_connector", [])}

    token_layer = {}
    for tok in lane_tokens:
        token_layer[tok] = "lane"
    for tok in connector_tokens:
        token_layer[tok] = "lane_connector"

    def _neighbors(tok):
        conn = nmap.connectivity.get(tok, {})
        inc = conn.get("incoming", []) or []
        out = conn.get("outgoing", []) or []
        return list(inc) + list(out)

    same_tokens = {ego_lane_token}
    direct = [tok for tok in _neighbors(ego_lane_token) if tok in token_layer]
    same_tokens.update(direct)

    # If a direct neighbor is a connector, include the lane on its other side.
    for tok in direct:
        if token_layer.get(tok) != "lane_connector":
            continue
        hop2 = [ntok for ntok in _neighbors(tok) if token_layer.get(ntok) == "lane"]
        same_tokens.update(hop2)

    # Extend same-lane family along connected driving direction.
    # This fixes far-front same-lane agents being misclassified as non-adjacent.
    if cache is not None:
        try:
            forward = _trace_connected_lanes(
                nmap,
                ego_lane_token,
                cache,
                direction="forward",
                max_hops=10,
                max_length_m=220.0,
                ref_heading=ego_heading,
            )
            backward = _trace_connected_lanes(
                nmap,
                ego_lane_token,
                cache,
                direction="backward",
                max_hops=5,
                max_length_m=80.0,
                ref_heading=ego_heading,
            )
            same_tokens.update(forward)
            same_tokens.update(backward)
        except Exception:
            pass

    return same_tokens


LANE_RELATION_LOG_PATH = None


def _lane_relation_debug_enabled():
    """Toggle with env LANE_RELATION_DEBUG=0 or --no_lane_relation_debug."""
    return os.environ.get("LANE_RELATION_DEBUG", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def set_lane_relation_log_path(path):
    """Optional dedicated log file (e.g. out_dir/lane_relation_debug.log)."""
    global LANE_RELATION_LOG_PATH
    LANE_RELATION_LOG_PATH = path


def _lane_relation_log(log_tag, msg):
    if not _lane_relation_debug_enabled():
        return
    tag = f" [{log_tag}]" if log_tag else ""
    line = f"[lane_relation]{tag} {msg}"
    if LANE_RELATION_LOG_PATH:
        try:
            with open(LANE_RELATION_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
    if getattr(Logger, "logger", None) is not None:
        Logger.log(line)
    else:
        print(line, flush=True)


def _lane_relation_fmt_metrics(metrics):
    if not metrics:
        return "{}"
    parts = [f"{k}={metrics[k]:.3f}" for k in sorted(metrics.keys())]
    return "{" + ", ".join(parts) + "}"


def _resolve_connector_to_driving_lane(
    nmap,
    connector_tok,
    x,
    y,
    ref_heading,
    cl_cache,
    connector_tokens,
):
    """
    When snap lands on a lane_connector, pick the geometrically nearest
    adjacent *lane* token so centerline stitching / lateral metrics use
    the driving corridor instead of a short intersection arm.
    """
    lane_tokens = {rec["token"] for rec in getattr(nmap, "lane", [])}
    conn = nmap.connectivity.get(connector_tok, {})
    neigh = list(conn.get("incoming", []) or []) + list(conn.get("outgoing", []) or [])
    cands = [
        tok
        for tok in neigh
        if tok in lane_tokens and tok not in connector_tokens
    ]
    if len(cands) == 0:
        return None

    best_tok = None
    best_score = float("inf")
    for tok in cands:
        lane_info = _centerline_xy_and_heading(nmap, tok, cl_cache, res_m=1.0)
        if lane_info is None:
            continue
        xy, heading = lane_info
        dist, idx = _point_to_polyline_distance(x, y, xy)
        lane_h = float(heading[idx])
        score = dist
        if ref_heading is not None:
            score += 0.6 * abs(_wrap_to_pi(lane_h - ref_heading))
        if score < best_score:
            best_score = score
            best_tok = tok
    return best_tok


def _snap_vehicle_lane(
    nmap,
    x,
    y,
    obj_heading,
    cl_cache,
    connector_tokens,
    *,
    is_ego=False,
):
    """
    Robust lane snap with fallbacks. Ego uses a slightly wider search so we
    do not abort the whole relation pass when heading prior is ambiguous.
    """
    if is_ego:
        attempts = [
            (20.0, 6.0, 2.5, True),
            (25.0, 8.0, 2.5, True),
            (30.0, 10.0, 2.5, True),
            (30.0, 12.0, 2.5, False),
            (40.0, 18.0, 3.0, False),
        ]
        search_radius = 50.0
        on_drivable = _point_is_drivable(nmap, x, y)
        geometric_cap = 40.0 if on_drivable else 28.0
    else:
        attempts = [
            (20.0, 7.0, 2.0, False),
            (25.0, 10.0, 2.0, False),
            (35.0, 14.0, 2.0, False),
        ]
        search_radius = 35.0
        on_drivable = True
        geometric_cap = 16.0

    best = (None, float("inf"), None)
    for radius_m, max_dist_m, conn_pen, use_heading in attempts:
        h = obj_heading if use_heading else None
        tok, dist, lane_h = _closest_lane_with_heading(
            nmap,
            x,
            y,
            h,
            cl_cache,
            search_radius_m=radius_m,
            max_lane_dist_m=max_dist_m,
            connector_tokens=connector_tokens,
            connector_penalty_m=conn_pen,
        )
        if tok is None:
            continue
        if dist < best[1]:
            best = (tok, dist, lane_h)

    tok, dist, lane_h = best
    if tok is None:
        # Uncapped nearest in wide radius (fixes strict max_dist rejecting 15 m lanes).
        for use_h in (True, False):
            tok_u, dist_u, lane_h_u, _ = _nearest_lane_in_radius(
                nmap,
                x,
                y,
                obj_heading,
                cl_cache,
                search_radius_m=search_radius,
                connector_tokens=connector_tokens,
                connector_penalty_m=3.0 if is_ego else 2.0,
                use_heading_prior=use_h,
            )
            if tok_u is not None and dist_u < (dist if tok is not None else float("inf")):
                tok, dist, lane_h = tok_u, dist_u, lane_h_u
            if tok is not None and dist <= geometric_cap:
                break

    if tok is None or (tok is not None and dist > geometric_cap):
        tok_be, dist_be, lane_h_be = _closest_lane_best_effort(
            nmap,
            x,
            y,
            obj_heading,
            cl_cache,
            connector_tokens=connector_tokens,
            connector_penalty_m=3.0 if is_ego else 2.0,
            search_radius_m=search_radius,
            geometric_cap_m=geometric_cap,
        )
        if tok_be is not None and (
            tok is None or dist_be < dist
        ):
            tok, dist, lane_h = tok_be, dist_be, lane_h_be

    if tok is None:
        tok, dist, _ = _lane_token_layers_on_point(nmap, x, y, connector_tokens)
        if tok is not None:
            info = _centerline_xy_and_heading(nmap, tok, cl_cache, res_m=1.0)
            if info is not None:
                xy, heading = info
                dist_pt, idx = _point_to_polyline_distance(x, y, xy)
                dist = dist_pt
                lane_h = float(heading[idx])

    if tok is not None and dist > geometric_cap:
        tok = None
        dist = float("inf")
        lane_h = None

    if tok is None:
        return None, float("inf"), None

    if is_ego and tok in connector_tokens:
        resolved = _resolve_connector_to_driving_lane(
            nmap, tok, x, y, obj_heading, cl_cache, connector_tokens
        )
        if resolved is not None:
            info = _centerline_xy_and_heading(nmap, resolved, cl_cache, res_m=1.0)
            if info is not None:
                xy, heading = info
                _, idx = _point_to_polyline_distance(x, y, xy)
                lane_h = float(heading[idx])
            tok = resolved
    return tok, dist, lane_h


def _unified_lateral_distance(lat_metrics, ego_is_connector=False):
    """Combine lateral metrics; drop map outliers that dominate on connectors."""
    if not lat_metrics:
        return float("inf")
    m = dict(lat_metrics)
    local_keys = ("ego_frame_y", "state_local_y")
    local_vals = [m[k] for k in local_keys if k in m]
    local_min = min(local_vals) if local_vals else None

    if ego_is_connector and "agent_on_ego_lane" in m:
        del m["agent_on_ego_lane"]
    elif "agent_on_ego_lane" in m and local_min is not None:
        if m["agent_on_ego_lane"] > 1.35 * local_min + 1.5:
            del m["agent_on_ego_lane"]

    # Shared connector / duplicate snap can report gap=0 while the vehicle sits
    # on a parallel lane (large offset to ego centerline).
    if "lane_centerline_gap" in m and m["lane_centerline_gap"] < 0.15:
        on_ego = m.get("agent_on_ego_lane")
        if on_ego is not None and on_ego > 2.2:
            del m["lane_centerline_gap"]
        elif local_min is not None and local_min > 2.2:
            del m["lane_centerline_gap"]

    if not m:
        return float("inf")
    return float(min(m.values()))


def _local_lateral_min(lat_metrics):
    local_vals = [
        lat_metrics[k]
        for k in ("ego_frame_y", "state_local_y")
        if k in lat_metrics
    ]
    return float(min(local_vals)) if local_vals else None


def _map_lateral_trustworthy(lat_metrics, ego_lane_dist):
    """False when ego snapped far from its lane — agent_on_ego_lane is unreliable."""
    if ego_lane_dist is None or not np.isfinite(ego_lane_dist):
        return True
    if float(ego_lane_dist) > 4.0:
        return False
    on_ego = lat_metrics.get("agent_on_ego_lane")
    local_min = _local_lateral_min(lat_metrics)
    if on_ego is None or local_min is None:
        return ego_lane_dist <= 4.0
    return on_ego <= max(3.5, 1.4 * local_min + 1.0)


def _same_lane_lateral_ref(lat_metrics, a_lane, ego_lane, ego_lane_dist=None):
    """
    Lateral metric for *same-lane* checks.
    Uses ego-frame |y| when map projection is untrustworthy (poor ego snap or
    inflated agent_on_ego_lane on long stitched refs).
    """
    if not lat_metrics:
        return float("inf")
    on_ego = lat_metrics.get("agent_on_ego_lane")
    # Same lane token: centerline distance is authoritative even if ego snap is poor.
    if (
        a_lane is not None
        and ego_lane is not None
        and a_lane == ego_lane
        and on_ego is not None
        and np.isfinite(on_ego)
    ):
        return float(on_ego)
    local_min = _local_lateral_min(lat_metrics)
    map_ok = _map_lateral_trustworthy(lat_metrics, ego_lane_dist)

    if map_ok and on_ego is not None and np.isfinite(on_ego):
        return float(on_ego)
    if local_min is not None:
        if on_ego is None or not map_ok or on_ego > max(3.5, 1.4 * local_min + 1.0):
            return float(local_min)
    if on_ego is not None and np.isfinite(on_ego):
        return float(on_ego)
    gap = lat_metrics.get("lane_centerline_gap")
    if (
        gap is not None
        and np.isfinite(gap)
        and a_lane is not None
        and ego_lane is not None
        and a_lane != ego_lane
    ):
        return float(gap)
    if gap is not None and np.isfinite(gap):
        return float(gap)
    return float("inf")


def _topology_indicates_same_lane(
    a_lane,
    ego_lane,
    same_lane_tokens,
    lat_metrics,
    ego_lane_dist=None,
    *,
    on_ego_th=2.2,
    gap_th=1.2,
):
    """Topology alone is insufficient when map lateral separation is large."""
    if a_lane is None or a_lane not in same_lane_tokens:
        return False
    on_ego = lat_metrics.get("agent_on_ego_lane")
    gap = lat_metrics.get("lane_centerline_gap")
    # Shared lane token: trust centerline distance, not ego-frame |y| inflated by poor ego snap.
    if a_lane == ego_lane:
        if on_ego is not None and np.isfinite(on_ego):
            return on_ego <= on_ego_th
        local_min = _local_lateral_min(lat_metrics)
        return local_min is not None and local_min <= on_ego_th
    if not _map_lateral_trustworthy(lat_metrics, ego_lane_dist):
        on_ego = _local_lateral_min(lat_metrics)
    if on_ego is not None and np.isfinite(on_ego) and on_ego > on_ego_th:
        return False
    if gap is not None and np.isfinite(gap) and gap > gap_th:
        return False
    if gap is not None and np.isfinite(gap) and gap <= gap_th:
        return True
    return on_ego is not None and np.isfinite(on_ego) and on_ego <= on_ego_th


def _adjacency_lateral_ref(
    lat_metrics,
    proj_trusted,
    ego_ref_len_m,
    ego_is_connector=False,
):
    """
    Lateral distance for adjacency / median checks: prefer lane gap and ego-frame
    metrics; ignore inflated stitched / ego-lane projections on long references.
    """
    if not lat_metrics:
        return float("inf")
    m = {}
    long_ref = (
        ego_ref_len_m is not None
        and np.isfinite(ego_ref_len_m)
        and float(ego_ref_len_m) > 80.0
    )
    local_vals = [
        lat_metrics[k]
        for k in ("ego_frame_y", "state_local_y")
        if k in lat_metrics
    ]
    local_min = min(local_vals) if local_vals else None

    if "lane_centerline_gap" in lat_metrics:
        gap = lat_metrics["lane_centerline_gap"]
        if not (
            gap < 0.15
            and local_min is not None
            and local_min > 2.5
        ):
            m["lane_centerline_gap"] = gap
    if "ego_on_agent_lane" in lat_metrics:
        eoa = lat_metrics["ego_on_agent_lane"]
        if local_min is None or eoa <= 1.35 * local_min + 1.5:
            m["ego_on_agent_lane"] = eoa
    for k in ("ego_frame_y", "state_local_y"):
        if k in lat_metrics:
            m[k] = lat_metrics[k]

    if not long_ref and "agent_on_ego_lane" in lat_metrics and not ego_is_connector:
        aoe = lat_metrics["agent_on_ego_lane"]
        if local_min is None or aoe <= 1.35 * local_min + 1.5:
            m["agent_on_ego_lane"] = aoe

    if (
        proj_trusted
        and "stitched_proj_d" in lat_metrics
        and (not long_ref or local_min is None)
    ):
        m["stitched_proj_d"] = lat_metrics["stitched_proj_d"]
    elif (
        proj_trusted
        and "stitched_proj_d" in lat_metrics
        and local_min is not None
    ):
        spd = lat_metrics["stitched_proj_d"]
        if spd <= 1.35 * local_min + 1.5:
            m["stitched_proj_d"] = spd

    if not m:
        return float("inf")
    return float(min(m.values()))


def _heading_delta_rad(a_heading, ego_heading, ref_a_h, ref_ego_h):
    if a_heading is not None and ego_heading is not None:
        return abs(_wrap_to_pi(a_heading - ego_heading))
    if ref_a_h is not None and ref_ego_h is not None:
        return abs(_wrap_to_pi(ref_a_h - ref_ego_h))
    return None


def _build_lane_relation_local_fallback(
    na,
    current_state_global,
    state_local,
    log_tag,
):
    """Map-free fallback when ego lane snap fails."""
    relation = {aidx: "non-adjacent lane" for aidx in range(1, na)}
    ego_heading = _safe_heading(
        float(current_state_global[0, 2].item()),
        float(current_state_global[0, 3].item()),
    )
    same_lane_lat = 2.5
    adjacent_lat = 6.0
    opposing_lat = 8.0
    # Far *behind* with |dy| under ~1 lane width → parallel lane beside ego, not same lane.
    # Do not apply to dx>0: same-direction leaders ahead (large dx, small |dy|) are same lane.
    parallel_adjacent_dy_th = 1.15
    parallel_adjacent_behind_dx_th = 20.0
    same_dir_th = np.deg2rad(60.0)
    opp_dir_th = np.deg2rad(120.0)

    _lane_relation_log(log_tag, "fallback: ego-frame-only lane relation (no map snap)")

    for aidx in range(1, na):
        ag = current_state_global[aidx]
        if torch.isnan(ag).any():
            continue
        if state_local is None or aidx >= state_local.size(0):
            continue
        dx = float(state_local[aidx, 0].item())
        dy = float(state_local[aidx, 1].item())
        if not (np.isfinite(dx) and np.isfinite(dy)):
            continue
        if np.hypot(dx, dy) > 80.0:
            continue
        lat = abs(dy)
        a_heading = _safe_heading(float(ag[2].item()), float(ag[3].item()))
        dh = _heading_delta_rad(a_heading, ego_heading, None, None)

        if lat <= same_lane_lat and dh is not None and dh <= same_dir_th:
            if (
                dx < -parallel_adjacent_behind_dx_th
                and lat < parallel_adjacent_dy_th
            ):
                relation[aidx] = "same-direction adjacent lane"
            else:
                relation[aidx] = "same lane"
        elif lat <= adjacent_lat and dh is not None and dh <= same_dir_th:
            relation[aidx] = "same-direction adjacent lane"
        elif lat <= opposing_lat and dh is not None and dh >= opp_dir_th:
            relation[aidx] = "opposing adjacent lane"
        # else: non-adjacent (incl. perpendicular crossing with dh in 70–110°)
        _lane_relation_log(
            log_tag,
            f"agent {aidx}: fallback local=({dx:.1f},{dy:.1f}) lat={lat:.2f} "
            f"dh_deg={np.rad2deg(dh):.1f} -> {relation[aidx]}"
            if dh is not None
            else f"agent {aidx}: fallback local=({dx:.1f},{dy:.1f}) lat={lat:.2f} -> {relation[aidx]}",
        )

    _lane_relation_log(log_tag, f"summary: {relation}")
    return relation


def _project_agent_onto_ego_reference(
    ax,
    ay,
    dist_xy,
    ego_ref_xy,
    ego_ref_s,
    ego_s_on_ref,
    s_hint,
    *,
    nearby_agent_dist_th,
    nearby_s_window_m,
    far_agent_s_window_m,
    long_reference_m,
):
    """
    Project agent onto ego stitched centerline. Long stitched references
    (intersection connectors) must use longitudinal windows; otherwise global
    nearest-segment search picks a parallel road branch and inflates |d|.
    """
    if ego_ref_s is None:
        return None

    use_long_ref = (
        long_reference_m is not None
        and np.isfinite(long_reference_m)
        and float(long_reference_m) > 80.0
    )
    has_hint = s_hint is not None and np.isfinite(s_hint)

    if has_hint:
        s_window = (
            nearby_s_window_m if dist_xy <= nearby_agent_dist_th else far_agent_s_window_m
        )
        proj = _project_point_to_centerline_frenet(
            ax, ay, ego_ref_xy, ego_ref_s, s_ref=s_hint, s_window_m=s_window
        )
        if proj is not None:
            return proj
        if not use_long_ref:
            return _project_point_to_centerline_frenet(ax, ay, ego_ref_xy, ego_ref_s)

    if use_long_ref:
        return None

    if dist_xy <= nearby_agent_dist_th:
        return _project_point_to_centerline_frenet(ax, ay, ego_ref_xy, ego_ref_s)

    proj = _project_point_to_centerline_frenet(
        ax, ay, ego_ref_xy, ego_ref_s, s_ref=s_hint, s_window_m=far_agent_s_window_m
    )
    proj_near = _project_point_to_centerline_frenet(ax, ay, ego_ref_xy, ego_ref_s)
    if proj_near is not None:
        if proj is None:
            return proj_near
        _, d_win, _ = proj
        _, d_near, _ = proj_near
        if abs(d_near) < abs(d_win):
            return proj_near
    return proj


def build_lane_relation_by_agent(
    map_env,
    map_name,
    crop_pos,
    bounds,
    current_state_global,
    current_state_local,
    log_tag=None,
    map_idx=None,
):
    """
    Build lane relation for each surrounding agent (index > 0):
      - same lane
      - same-direction adjacent lane
      - opposing adjacent lane
      - non-adjacent lane

    Methodology (uses nuScenes Map layers instead of a single hardcoded gap):
    - Ego current lane is snapped via heading-aware nearest-lane search.
    - Per-agent classification uses BOTH map topology and geometry:
      * Same lane: agent's snapped lane is in ego's connected lane family
        (predecessor / successor chain + lane_connector hop, traced with ego
        heading at forks), OR unified lateral offset (min of longitudinal-aware
        stitched projection, ego lane, ego offset on agent lane, ego-frame |y|)
        within about one lane width with a matching heading (70° for motion).
      * Adjacency by lateral offset: min of ego-lane projection, ego-frame |y|,
        reciprocal ego-on-agent-lane offset (when agent snap is reliable),
        and stitched centerline (unbounded for agents within ~25 m; windowed
        for farther agents). Pairs separated by a physical median are forced
        non-adjacent; painted markings between adjacent lanes are ignored when
        lateral separation is within ~one lane width.
      * Heading gray zone (roughly 70°–110°): still treated as adjacent when
        laterally close; split at 90° into same-direction vs opposing adjacent.
      * Lateral distance combines projection onto ego's stitched centerline,
        ego's immediate lane, and ego-frame |y| (no single hardcoded 4.5 m).
      * Agent lane snap uses geometric distance only so opposing traffic is
        not pulled onto ego's lane by a heading prior.
      * Same-direction vs opposing is decided by the wrapped heading delta
        between the agent's lane heading (or agent heading as fallback) and
        the lane heading at the projection point on ego's centerline.
      * If the ego->agent segment crosses non-drivable on the merged
        drivable raster (same as filter_scene feasibility), force
        non-adjacent lane.
    """
    del crop_pos, bounds  # kept in signature for future extensions.

    if log_tag is None:
        log_tag = map_name

    _lane_relation_log(log_tag, "---- begin lane relation ----")

    na = current_state_global.size(0)
    if na <= 1:
        _lane_relation_log(log_tag, "skip: na<=1, no surrounding agents")
        return {}

    nmap = map_env.nusc_maps[map_name]
    if map_idx is None:
        map_idx = _resolve_map_idx(map_env, map_name)
    cl_cache = {}
    connector_tokens = {rec["token"] for rec in getattr(nmap, "lane_connector", [])}
    relation = {aidx: "non-adjacent lane" for aidx in range(1, na)}

    ego = current_state_global[0]
    if torch.isnan(ego).any():
        _lane_relation_log(log_tag, "abort: ego state contains NaN")
        return relation

    ego_x = float(ego[0].item())
    ego_y = float(ego[1].item())
    ego_heading = _safe_heading(float(ego[2].item()), float(ego[3].item()))
    ego_lane, ego_lane_dist, ego_lane_h = _snap_vehicle_lane(
        nmap,
        ego_x,
        ego_y,
        ego_heading,
        cl_cache,
        connector_tokens,
        is_ego=True,
    )
    state_local = current_state_local
    if state_local is None:
        try:
            state_global_np = current_state_global.detach().cpu().numpy()
            state_local = torch.from_numpy(
                nutils.objects2frame(
                    state_global_np[:, np.newaxis, :],
                    state_global_np[0],
                    toworld=False,
                )[:, 0, :]
            )
        except Exception:
            state_local = None

    if ego_lane is None:
        ego_h_deg = (
            f"{np.rad2deg(ego_heading):.1f}"
            if ego_heading is not None
            else "None"
        )
        _tok_d, _near_dist, _, _n_cand = _nearest_lane_in_radius(
            nmap,
            ego_x,
            ego_y,
            ego_heading,
            cl_cache,
            search_radius_m=50.0,
            connector_tokens=connector_tokens,
            use_heading_prior=False,
        )
        _drv = _point_is_drivable(nmap, ego_x, ego_y)
        _lane_relation_log(
            log_tag,
            f"warn: ego lane snap failed at ({ego_x:.2f},{ego_y:.2f}) "
            f"heading_deg={ego_h_deg} drivable={_drv} n_lane_cands={_n_cand} "
            f"nearest_uncapped_dist={_near_dist:.2f}m cap=40; local-frame fallback",
        )
        return _build_lane_relation_local_fallback(
            na, current_state_global, state_local, log_tag
        )

    ego_centerline = _centerline_xy_and_heading(nmap, ego_lane, cl_cache, res_m=1.0)
    if ego_centerline is None:
        _lane_relation_log(
            log_tag,
            f"warn: no centerline for ego_lane={ego_lane[:8]}...; local-frame fallback",
        )
        return _build_lane_relation_local_fallback(
            na, current_state_global, state_local, log_tag
        )
    ego_xy, _ = ego_centerline
    same_lane_tokens = _build_same_lane_family(
        nmap, ego_lane, cache=cl_cache, ego_heading=ego_heading
    )
    ego_ref_xy = _build_connected_reference_centerline(
        nmap, ego_lane, cl_cache, res_m=1.0, ego_heading=ego_heading
    )
    if ego_ref_xy is None:
        ego_ref_xy = ego_xy
    ego_ref_s = _build_centerline_arclength(ego_ref_xy)
    ego_s_on_ref = None
    if ego_ref_s is not None:
        ego_proj_ref = _project_point_to_centerline_frenet(
            ego_x, ego_y, ego_ref_xy, ego_ref_s
        )
        if ego_proj_ref is not None:
            ego_s_on_ref = ego_proj_ref[0]

    road_dividers = _gather_road_dividers(nmap, ego_x, ego_y, radius_m=80.0)

    too_far_from_ego_th = 80.0
    nearby_agent_dist_th = 25.0
    nearby_s_window_m = 40.0
    far_agent_s_window_m = 70.0
    # Lateral offsets are measured against ego's stitched centerline frenet d.
    # nuScenes lane width ~3.5 m: center within ~one lane width of ego corridor.
    same_lane_lateral_th = 2.5
    same_lane_map_gap_th = 1.5
    # Agents up to ~one extra lane away (~6-7.5 m) are still considered adjacent
    # provided no physical median separates them.
    adjacent_lateral_th = 6.0
    opposing_adjacent_lateral_th = 8.0
    same_lane_heading_th = np.deg2rad(45.0)
    same_dir_th = np.deg2rad(60.0)
    opp_dir_th = np.deg2rad(120.0)
    ambiguous_split_th = np.deg2rad(90.0)

    ego_is_connector = ego_lane in connector_tokens
    ego_ref_len_m = _polyline_length(ego_ref_xy)
    _lane_relation_log(
        log_tag,
        f"ego snap lane={ego_lane[:8]}... dist={ego_lane_dist:.2f} "
        f"connector={ego_is_connector} "
        f"same_lane_family={len(same_lane_tokens)} ref_len_m={ego_ref_len_m:.1f} "
        f"road_dividers={len(road_dividers)} "
        f"state_local={'ok' if state_local is not None else 'None'}",
    )

    for aidx in range(1, na):
        ag = current_state_global[aidx]
        if torch.isnan(ag).any():
            _lane_relation_log(log_tag, f"agent {aidx}: skip NaN state")
            continue

        ax = float(ag[0].item())
        ay = float(ag[1].item())

        if _pair_separated_by_nondrivable_raster(
            map_env, map_idx, (ego_x, ego_y), (ax, ay)
        ):
            relation[aidx] = "non-adjacent lane"
            _lane_relation_log(
                log_tag,
                f"agent {aidx}: non-adjacent (nondrivable_raster) "
                f"ego=({ego_x:.1f},{ego_y:.1f}) agent=({ax:.1f},{ay:.1f})",
            )
            continue

        # far-away agents are considered non-adjacent by definition.
        if state_local is not None and aidx < state_local.size(0):
            dx_loc = float(state_local[aidx, 0].item())
            dy_loc = float(state_local[aidx, 1].item())
            if np.isfinite(dx_loc) and np.isfinite(dy_loc):
                if np.hypot(dx_loc, dy_loc) > too_far_from_ego_th:
                    _lane_relation_log(
                        log_tag,
                        f"agent {aidx}: non-adjacent (too_far) dist_loc="
                        f"{np.hypot(dx_loc, dy_loc):.1f}m > {too_far_from_ego_th}m",
                    )
                    continue

        dist_xy = float(np.hypot(ax - ego_x, ay - ego_y))
        a_heading = _safe_heading(float(ag[2].item()), float(ag[3].item()))
        # Distance-only snap avoids pulling opposing traffic onto ego's lane.
        a_lane, a_lane_dist, a_lane_h = _snap_vehicle_lane(
            nmap,
            ax,
            ay,
            a_heading,
            cl_cache,
            connector_tokens,
            is_ego=False,
        )
        if a_lane_h is None and a_heading is not None:
            a_lane_h = a_heading

        # Project onto ego stitched centerline (long refs need s_hint windows).
        s_hint = None
        if state_local is not None and aidx < state_local.size(0):
            dx_loc_hint = float(state_local[aidx, 0].item())
            if ego_s_on_ref is not None and np.isfinite(dx_loc_hint):
                s_hint = float(ego_s_on_ref) + dx_loc_hint
        proj = _project_agent_onto_ego_reference(
            ax,
            ay,
            dist_xy,
            ego_ref_xy,
            ego_ref_s,
            ego_s_on_ref,
            s_hint,
            nearby_agent_dist_th=nearby_agent_dist_th,
            nearby_s_window_m=nearby_s_window_m,
            far_agent_s_window_m=far_agent_s_window_m,
            long_reference_m=ego_ref_len_m,
        )

        # Unified lateral offset: min of ego-frame |y|, lane-centerline gap,
        # stitched projection (outlier-rejected), ego-lane projection, reciprocal.
        d_lat = float("inf")
        ref_lane_h = None
        lat_metrics = {}
        d_proj_raw = None
        proj_trusted = None

        ego_frame_lat = _ego_frame_lateral_m(ego_x, ego_y, ego_heading, ax, ay)
        if ego_frame_lat is not None and np.isfinite(ego_frame_lat):
            lat_metrics["ego_frame_y"] = abs(ego_frame_lat)

        if state_local is not None and aidx < state_local.size(0):
            dy_loc = float(state_local[aidx, 1].item())
            if np.isfinite(dy_loc):
                lat_metrics["state_local_y"] = abs(dy_loc)

        if a_lane is not None:
            lane_gap = _min_lane_centerline_separation_m(nmap, ego_lane, a_lane, cl_cache)
            if lane_gap is not None and np.isfinite(lane_gap):
                lat_metrics["lane_centerline_gap"] = lane_gap

        if proj is not None:
            _, d_proj, ref_lane_h = proj
            if np.isfinite(d_proj):
                d_abs = abs(d_proj)
                d_proj_raw = d_abs
                trust_proj = True
                if lat_metrics:
                    local_ref = min(lat_metrics.values())
                    if d_abs > max(adjacent_lateral_th, 1.35 * local_ref + 1.0):
                        trust_proj = False
                proj_trusted = trust_proj
                if trust_proj:
                    lat_metrics["stitched_proj_d"] = d_abs

        d_ego_lane, ref_lane_h_ego = _signed_lateral_to_lane_centerline(
            nmap, ego_lane, ax, ay, cl_cache
        )
        if d_ego_lane is not None and np.isfinite(d_ego_lane):
            lat_metrics["agent_on_ego_lane"] = d_ego_lane
            if ref_lane_h is None:
                ref_lane_h = ref_lane_h_ego

        if a_lane is not None:
            d_ego_on_agent_lane, ref_lane_h_agent = _signed_lateral_to_lane_centerline(
                nmap, a_lane, ego_x, ego_y, cl_cache
            )
            if d_ego_on_agent_lane is not None and np.isfinite(d_ego_on_agent_lane):
                lat_metrics["ego_on_agent_lane"] = d_ego_on_agent_lane
                if ref_lane_h is None:
                    ref_lane_h = ref_lane_h_agent

        d_lat = _unified_lateral_distance(lat_metrics, ego_is_connector=ego_is_connector)

        a_is_connector = a_lane in connector_tokens if a_lane is not None else False
        if state_local is not None and aidx < state_local.size(0):
            _dx_log = float(state_local[aidx, 0].item())
            _dy_log = float(state_local[aidx, 1].item())
            local_str = f"({_dx_log:.1f},{_dy_log:.1f})"
        else:
            local_str = "None"
        a_lane_str = (a_lane[:8] + "...") if a_lane else "None"
        a_dist_str = f"{a_lane_dist:.2f}" if np.isfinite(a_lane_dist) else "inf"
        d_lat_str = f"{d_lat:.3f}" if np.isfinite(d_lat) else "inf"
        _lane_relation_log(
            log_tag,
            f"agent {aidx}: dist_xy={dist_xy:.1f}m local={local_str} "
            f"a_lane={a_lane_str} connector={a_is_connector} a_lane_dist={a_dist_str} "
            f"lat={_lane_relation_fmt_metrics(lat_metrics)} d_lat={d_lat_str} "
            f"d_proj_raw={d_proj_raw} proj_trusted={proj_trusted}",
        )

        d_same = _same_lane_lateral_ref(
            lat_metrics, a_lane, ego_lane, ego_lane_dist=ego_lane_dist
        )

        # 1) Same-lane via topology + confirmed small map/lateral separation.
        if _topology_indicates_same_lane(
            a_lane,
            ego_lane,
            same_lane_tokens,
            lat_metrics,
            ego_lane_dist=ego_lane_dist,
            on_ego_th=same_lane_lateral_th,
            gap_th=same_lane_map_gap_th,
        ):
            relation[aidx] = "same lane"
            _lane_relation_log(
                log_tag,
                f"agent {aidx}: -> same lane (topology) d_same={d_same:.3f}",
            )
            continue
        if a_lane is not None and a_lane in same_lane_tokens:
            _lane_relation_log(
                log_tag,
                f"agent {aidx}: topology family but lateral rejects same-lane "
                f"d_same={d_same:.3f} gap={lat_metrics.get('lane_centerline_gap')}",
            )

        # 1b) Same-lane via geometry: small offset to ego centerline + similar heading.
        ref_a_h_same = a_lane_h if a_lane_h is not None else a_heading
        heading_ok_same = False
        if a_heading is not None and ego_heading is not None:
            heading_ok_same = abs(_wrap_to_pi(a_heading - ego_heading)) <= same_dir_th
        elif (
            ref_lane_h is not None
            and ref_a_h_same is not None
            and np.isfinite(d_same)
        ):
            heading_ok_same = (
                abs(_wrap_to_pi(ref_a_h_same - ref_lane_h)) <= same_lane_heading_th
            )
        if np.isfinite(d_same) and d_same <= same_lane_lateral_th and heading_ok_same:
            relation[aidx] = "same lane"
            _lane_relation_log(
                log_tag,
                f"agent {aidx}: -> same lane (geometry) d_same={d_same:.3f} "
                f"heading_ok={heading_ok_same}",
            )
            continue
        if np.isfinite(d_same) and d_same <= same_lane_lateral_th and not heading_ok_same:
            _lane_relation_log(
                log_tag,
                f"agent {aidx}: same-lane geometry rejected (heading) d_same={d_same:.3f}",
            )

        # 2) Adjacency by lateral offset (prefer ego-frame / lane-gap over long-ref map noise).
        d_ref = _adjacency_lateral_ref(
            lat_metrics,
            proj_trusted,
            ego_ref_len_m,
            ego_is_connector=ego_is_connector,
        )
        if not np.isfinite(d_ref):
            d_ref = d_lat
        if not np.isfinite(d_ref) and a_lane is not None:
            d_ref = _min_lane_centerline_separation_m(nmap, ego_lane, a_lane, cl_cache)

        ref_ego_h_lane = ref_lane_h if ref_lane_h is not None else (
            ego_lane_h if ego_lane_h is not None else ego_heading
        )
        ref_a_h_lane = a_lane_h if a_lane_h is not None else a_heading
        dh = _heading_delta_rad(a_heading, ego_heading, ref_a_h_lane, ref_ego_h_lane)

        d_adj_th = adjacent_lateral_th
        if dh is not None and dh >= opp_dir_th:
            d_adj_th = opposing_adjacent_lateral_th

        local_lat = _local_lateral_min(lat_metrics)
        on_ego_lat = lat_metrics.get("agent_on_ego_lane")
        if local_lat is not None and local_lat > d_adj_th:
            map_says_close = (
                on_ego_lat is not None
                and np.isfinite(on_ego_lat)
                and on_ego_lat <= d_adj_th
            )
            d_ref_pre = _adjacency_lateral_ref(
                lat_metrics,
                proj_trusted,
                ego_ref_len_m,
                ego_is_connector=ego_is_connector,
            )
            ref_says_close = (
                d_ref_pre is not None
                and np.isfinite(d_ref_pre)
                and d_ref_pre <= d_adj_th
            )
            if not map_says_close and not ref_says_close:
                _lane_relation_log(
                    log_tag,
                    f"agent {aidx}: non-adjacent (local_lat_too_large) "
                    f"local_lat={local_lat:.3f} th={d_adj_th} "
                    f"on_ego={on_ego_lat} d_ref_pre={d_ref_pre}",
                )
                continue

        if not np.isfinite(d_ref) or d_ref > d_adj_th:
            reason = "d_ref_nan" if not np.isfinite(d_ref) else "d_ref_too_large"
            _lane_relation_log(
                log_tag,
                f"agent {aidx}: non-adjacent ({reason}) d_ref={d_ref} "
                f"th={d_adj_th}",
            )
            continue

        # Physical median / road_block split (incl. divided highway with hard barrier).
        if _pair_separated_by_physical_median(
            nmap,
            (ego_x, ego_y),
            (ax, ay),
            lateral_sep_m=d_ref,
            road_dividers=road_dividers,
            ego_frame_lateral_m=local_lat,
            close_lateral_th=3.5,
        ):
            _lane_relation_log(
                log_tag,
                f"agent {aidx}: non-adjacent (physical_median) d_ref={d_ref:.3f} "
                f"local_lat={local_lat}",
            )
            continue

        # 4) Direction: prefer vehicle motion headings; lane headings are fallback.
        if dh is None:
            relation[aidx] = "same-direction adjacent lane"
            _lane_relation_log(
                log_tag,
                f"agent {aidx}: -> same-direction adjacent (no heading) d_ref={d_ref:.3f}",
            )
            continue

        if dh <= same_dir_th:
            relation[aidx] = "same-direction adjacent lane"
        elif dh >= opp_dir_th:
            relation[aidx] = "opposing adjacent lane"
        else:
            # ~Perpendicular (70°–110°): different road arm / crossing, not adjacent.
            relation[aidx] = "non-adjacent lane"
            _lane_relation_log(
                log_tag,
                f"agent {aidx}: non-adjacent (heading_gray_zone) d_ref={d_ref:.3f} "
                f"dh_deg={np.rad2deg(dh):.1f}",
            )
            continue
        _lane_relation_log(
            log_tag,
            f"agent {aidx}: -> {relation[aidx]} d_ref={d_ref:.3f} "
            f"dh_deg={np.rad2deg(dh):.1f}",
        )

    _lane_relation_log(log_tag, f"summary: {relation}")
    return relation


def _extract_agent_tokens(scene_graph):
    """
    Return the list of per-agent-row identifiers attached by NuScenesDataset.
    Index 0 is ego (placeholder 'ego'); other rows are nuScenes instance_token strings.

    The dataset stores this as a JSON-encoded string on the Graph so that
    torch_geometric's batching cannot mangle the structure. After batching,
    the attribute may appear as the raw string, or wrapped in one or more
    lists (batch aggregation, depending on PyG version / batch_size).
    """
    raw = getattr(scene_graph, "agent_tokens", None)
    if raw is None:
        return None
    # fully unwrap any list wrappers introduced by PyG batching until we
    # reach the JSON string payload
    while isinstance(raw, (list, tuple)) and len(raw) > 0:
        raw = raw[0]
    if not isinstance(raw, str):
        return None
    try:
        tokens = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(tokens, list):
        return None
    return [str(t) if t is not None else None for t in tokens]


def _get_scene_name_and_sidx(scene_graph, sample_idx):
    """
    Retrieve the scene name and sample index (sidx) for a given sample index.
    Includes validation and error handling for unsupported formats.
    """
    idx_info = scene_graph.seq_map[sample_idx]
    if isinstance(idx_info, (list, tuple)) and len(idx_info) >= 2:
        scene_name = idx_info[0]
        sidx = idx_info[1]
        if isinstance(scene_name, str):
            return scene_name, sidx
    raise ValueError(f"Unsupported seq_map format for extracting scene metadata: {idx_info}")


def _get_sample_token(scene_graph, nusc_obj, sample_idx):
    """
    Retrieve the sample token for a given dataset and sample index.
    Includes additional checks for validity and returns detailed information.
    """
    scene_name, sidx = _get_scene_name_and_sidx(scene_graph, sample_idx)
    scene_recs = [
        rec for rec in nusc_obj.sample
        if nusc_obj.get("scene", rec["scene_token"])["name"] == scene_name
    ]
    scene_recs = sorted(scene_recs, key=lambda x: x["timestamp"])

    cur_tidx = sidx + scene_graph.npast - 1
    if cur_tidx < 0 or cur_tidx >= len(scene_recs):
        return {
            "scene_name": scene_name,
            "sample_token": None,
            "error": "Index out of bounds",
        }

    sample_token = scene_recs[cur_tidx]["token"]
    return {
        "scene_name": scene_name,
        "sample_token": sample_token,
        "timestamp": scene_recs[cur_tidx]["timestamp"],
    }


def _collect_map_and_traffic_info(map_env, map_name, crop_pos, bounds):
    nmap = map_env.nusc_maps[map_name]
    cx, cy = float(crop_pos[0].item()), float(crop_pos[1].item())
    radius = float(max(abs(bounds[0]), abs(bounds[1]), abs(bounds[2]), abs(bounds[3])))

    layers = ["lane", "lane_connector", "road_segment", "road_block", "ped_crossing", "walkway", "traffic_light"]
    try:
        recs = nmap.get_records_in_radius(cx, cy, radius, layers)
    except Exception:
        recs = {k: [] for k in layers}

    def _token_polygon_xy(layer_name, token):
        try:
            rec = nmap.get(layer_name, token)
        except Exception:
            return None
        if "polygon_token" in rec and rec["polygon_token"]:
            try:
                poly = nmap.extract_polygon(rec["polygon_token"])
                coords = np.array(poly.exterior.coords, dtype=np.float32)
                if coords.shape[0] >= 3:
                    return coords[:, :2]
            except Exception:
                return None
        return None

    def _token_center_xy(layer_name, token):
        try:
            rec = nmap.get(layer_name, token)
        except Exception:
            return None

        if "polygon_token" in rec and rec["polygon_token"]:
            try:
                poly = nmap.extract_polygon(rec["polygon_token"])
                return float(poly.centroid.x), float(poly.centroid.y)
            except Exception:
                pass
        if "line_token" in rec and rec["line_token"]:
            try:
                line = nmap.extract_line(rec["line_token"])
                coords = np.array(line.coords)
                if coords.shape[0] > 0:
                    return float(np.mean(coords[:, 0])), float(np.mean(coords[:, 1]))
            except Exception:
                pass
        if "node_tokens" in rec and rec["node_tokens"]:
            pts = []
            for ntok in rec["node_tokens"]:
                try:
                    nrec = nmap.get("node", ntok)
                    pts.append([nrec["x"], nrec["y"]])
                except Exception:
                    continue
            if len(pts) > 0:
                pts = np.array(pts)
                return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        if "x" in rec and "y" in rec:
            return float(rec["x"]), float(rec["y"])
        return None

    ped_crossings = []
    for tok in recs.get("ped_crossing", []):
        cxy = _token_center_xy("ped_crossing", tok)
        pxy = _token_polygon_xy("ped_crossing", tok)
        if cxy is not None:
            ped_crossings.append({"token": tok, "center_xy_global": cxy, "polygon_xy_global": pxy})

    traffic_lights = []
    for tok in recs.get("traffic_light", []):
        cxy = _token_center_xy("traffic_light", tok)
        color = "unknown"
        try:
            trec = nmap.get("traffic_light", tok)
            for k in ["color", "status", "traffic_light_color", "signal_color", "state"]:
                if k in trec and trec[k] is not None and str(trec[k]) != "":
                    color = str(trec[k])
                    break
        except Exception:
            pass
        traffic_lights.append({"token": tok, "center_xy_global": cxy, "color": color})

    return {
        "crop_center": (cx, cy),
        "crop_bounds": bounds,
        "map_layers": list(map_env.layer_names),
        "nearby_counts": {k: len(recs.get(k, [])) for k in layers},
        "ped_crossings": ped_crossings,
        "traffic_lights": traffic_lights,
    }


def convert_states_to_ego_current_frame(past, current_state):
    """
    Convert global-map-frame states into ego-current frame.
    In this local frame, ego current state is near (x=0, y=0, hx=1, hy=0).
    """
    ego_center = current_state[0].detach().cpu().numpy()
    past_np = past.detach().cpu().numpy()
    cur_np = current_state.detach().cpu().numpy()

    past_local = nutils.objects2frame(past_np, ego_center, toworld=False)
    cur_local = nutils.objects2frame(cur_np[:, np.newaxis, :], ego_center, toworld=False)[:, 0, :]
    return torch.from_numpy(past_local), torch.from_numpy(cur_local)


def convert_xy_to_ego_current_frame(xy_global, ego_current_state):
    obj = np.array([[xy_global[0], xy_global[1], 1.0, 0.0]], dtype=np.float32)
    obj_local = nutils.objects2frame(
        obj[:, np.newaxis, :],
        ego_current_state.detach().cpu().numpy(),
        toworld=False,
    )[0, 0]
    return float(obj_local[0]), float(obj_local[1])


def build_sample_prompt_text(
    map_name,
    split,
    sample_idx,
    past,
    current_state,
    current_state_global,
    current_speed,
    sorted_other_rank,
    map_info,
    past_frenet,
    current_state_frenet,
    lane_relation_by_agent,
    use_rank_id=False,
    coord_mode="local",
    pred_ego_future=None,
    pred_sur_future=None,
):
    lines = []
    na = past.size(0)
    n_other = max(na - 1, 0)

    if split == "test":
        coord_desc = "ego-centric" if coord_mode == "local" else coord_mode
        lines.append(f"Current State Coordinate System: {coord_desc}")
        lines.append("")

    lines.append("EGO:")
    ego_state_str = _current_state_line(
        current_state[0],
        current_state_frenet[0],
        current_state_global[0],
        current_speed[0],
        coord_mode=coord_mode,
    )
    lines.append(f"  Current state: {ego_state_str}")
    lines.append("")

    n_listed = len(sorted_other_rank)
    lines.append(
        "SURROUNDING AGENTS "
        f"({n_listed} visible in map crop, ranked by distance to ego current position):"
    )
    if n_listed == 0:
        lines.append("  none")
    else:
        ordered = sorted([(rank, aidx) for aidx, rank in sorted_other_rank.items()], key=lambda x: x[0])
        for _rank_id, aidx in ordered:
            display_id = _rank_id if use_rank_id else aidx
            state_str = _current_state_line(
                current_state[aidx],
                current_state_frenet[aidx],
                current_state_global[aidx],
                current_speed[aidx],
                coord_mode=coord_mode,
            )
            lines.append(f"  Agent {display_id}:")
            lines.append(f"    Current state: {state_str}")
            # ttc_s, mdc_m = _compute_ttc_mdc(current_state, current_speed, aidx)
            # lines.append(
            #     f"    Interaction metrics with ego: Time-to-Collision={_fmt_metric(ttc_s, 's')}, "
            #     f"Minimum Distance to Collision={_fmt_metric(mdc_m, 'm')}"
            # )
            ego_s = float(current_state_frenet[0, 0].item())
            ego_d = float(current_state_frenet[0, 1].item())
            dx = float(current_state_frenet[aidx, 0].item()) - ego_s
            dy = float(current_state_frenet[aidx, 1].item()) - ego_d
            dh = _wrap_to_pi(
                np.arctan2(
                    float(current_state_frenet[aidx, 3].item()),
                    float(current_state_frenet[aidx, 2].item()),
                )
            )
            spatial_relation = classify_spatial_relation(dx, dy, dh)
            # Euclidean distance to ego.
            dist_to_ego = float(np.hypot(dx, dy))
            if np.isnan(dist_to_ego):
                dist_str = ""
            else:
                dist_str = f", distance to ego: {dist_to_ego:.1f} m"
            lines.append(f"    Spatial relation with ego: {spatial_relation}{dist_str}")
            lane_relation = lane_relation_by_agent.get(aidx, "non-adjacent lane")
            lines.append(f"    Lane relation with respect to ego lane: {lane_relation}")
            # if pred_ego_future is not None and pred_sur_future is not None and aidx >= 1:
            #     sur_idx = aidx - 1
            #     if sur_idx < pred_sur_future.size(0):
            #         pred_ag = pred_sur_future[sur_idx]
            #         ttc_s = _compute_ttc_from_pred_future(pred_ego_future, pred_ag)
            #         mdc_m = _compute_mdc_from_pred_future(pred_ego_future, pred_ag)
            #         lines.append(
            #             f"    Estimated time-to-collision (TTC) with ego: "
            #             f"{_fmt_ttc_text(ttc_s)}"
            #         )
            #         lines.append(
            #             f"    Estimated minimum distance to collision (MDC) with ego: "
            #             f"{_fmt_mdc_text(mdc_m)}"
            #         )
            # lines.append("")

    # lines.append("TRAFFIC LIGHTS:")
    # if len(map_info["traffic_lights"]) > 0:
    #     nearest_tl = min(
    #         map_info["traffic_lights"],
    #         key=lambda x: (x["center_xy_local"][0] ** 2 + x["center_xy_local"][1] ** 2),
    #     )
    #     tx, ty = nearest_tl["center_xy_local"]
    #     lines.append(
    #         "  There is at least one traffic light in the current area; "
    #         f"the nearest one is around (x={tx:.3f} m, y={ty:.3f} m). "
    #         f"Traffic light color is {nearest_tl['color']}."
    #     )
    #     if nearest_tl["color"] == "unknown":
    #         lines.append("  Note: NuScenes map is static, so real-time traffic-light color may be unavailable.")
    # else:
    #     lines.append("  There is no traffic light in the current area.")

    return "\n".join(lines).rstrip() + "\n"


def save_sample_prompt_text(
    map_name,
    split,
    past,
    current_state,
    current_state_global,
    current_speed,
    sorted_other_rank,
    map_info,
    past_frenet,
    current_state_frenet,
    lane_relation_by_agent,
    txt_path=None,
    sample_idx=None,
    use_rank_id=False,
    coord_mode="local",
    pred_ego_future=None,
    pred_sur_future=None,
):
    text = build_sample_prompt_text(
        map_name,
        split,
        sample_idx,
        past,
        current_state,
        current_state_global,
        current_speed,
        sorted_other_rank,
        map_info,
        past_frenet,
        current_state_frenet,
        lane_relation_by_agent,
        use_rank_id=use_rank_id,
        coord_mode=coord_mode,
        pred_ego_future=pred_ego_future,
        pred_sur_future=pred_sur_future,
    )
    if txt_path is not None:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


# NuScenes history sampling (2 Hz); wheelbase matches agent/rag/crash_data_processor.WHEELBASE
NUSC_HISTORY_DT = 0.5
WHEELBASE = 2.8


def compute_kinematics(
    v: np.ndarray,
    psi: np.ndarray,
    dt: float = NUSC_HISTORY_DT,
    wheelbase: float = WHEELBASE,
    acc_measured: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute acceleration and steering angle from speed + heading sequences.

    Returns (acceleration, steering_angle_deg) arrays matching input length.
    """
    if acc_measured is not None and len(acc_measured) == len(v):
        acc = np.asarray(acc_measured, dtype=float).copy()
        bad = ~np.isfinite(acc)
        if bad.any():
            acc[bad] = np.gradient(v, dt)[bad]
    else:
        acc = np.gradient(v, dt)

    psi_cont = np.unwrap(np.asarray(psi, dtype=float))
    yaw_rate = np.gradient(psi_cont, dt)

    v_safe = np.maximum(np.abs(v), 1.5)
    low_speed_mask = np.abs(v) < 1.5
    steer_rad = np.arctan(yaw_rate * wheelbase / v_safe)
    steer_rad[low_speed_mask] = 0.0
    steer_deg = np.degrees(steer_rad)

    return acc, steer_deg


def _v_psi_from_xyh(xyh: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Longitudinal speed and heading from (x, y, hx, hy) for use with compute_kinematics."""
    x = np.asarray(xyh[:, 0], dtype=np.float64)
    y = np.asarray(xyh[:, 1], dtype=np.float64)
    hx = np.asarray(xyh[:, 2], dtype=np.float64)
    hy = np.asarray(xyh[:, 3], dtype=np.float64)
    psi = np.arctan2(hy, hx)
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    v = vx * np.cos(psi) + vy * np.sin(psi)
    return v, psi


def compute_control_sequence(
    past_local,
    sorted_other_rank,
    dt: float = NUSC_HISTORY_DT,
    wheelbase: float = WHEELBASE,
):
    """
    Per-timestep [acc_mps2, steer_deg] from ego-centric (x,y,hx,hy), same row order as save_trajectory_json.
    """
    if isinstance(past_local, torch.Tensor):
        pl = past_local.detach().cpu().numpy()
    else:
        pl = np.asarray(past_local, dtype=np.float64)
    if pl.ndim != 3 or pl.shape[2] < 4:
        raise ValueError(f"past_local must be (N, T, >=4), got {pl.shape}")

    def _control_row_for_agent(aidx: int):
        xyh = pl[aidx]
        tlen = xyh.shape[0]
        row = []
        if tlen == 0:
            return row
        valid = (
            np.isfinite(xyh[:, 0])
            & np.isfinite(xyh[:, 1])
            & np.isfinite(xyh[:, 2])
            & np.isfinite(xyh[:, 3])
        )
        if valid.all():
            v, psi = _v_psi_from_xyh(xyh, dt)
            acc, steer_deg = compute_kinematics(v, psi, dt=dt, wheelbase=wheelbase)
            for t in range(tlen):
                row.append([float(acc[t]), float(steer_deg[t])])
        else:
            v, psi = _v_psi_from_xyh(xyh, dt)
            acc, steer_deg = compute_kinematics(v, psi, dt=dt, wheelbase=wheelbase)
            for t in range(tlen):
                if valid[t] and np.isfinite(acc[t]) and np.isfinite(steer_deg[t]):
                    row.append([float(acc[t]), float(steer_deg[t])])
                else:
                    row.append([None, None])
        return row

    rows = [_control_row_for_agent(0)]
    ordered = sorted([(rank, aidx) for aidx, rank in sorted_other_rank.items()], key=lambda x: x[0])
    for _rank_id, aidx in ordered:
        rows.append(_control_row_for_agent(int(aidx)))
    return rows

def compute_history_speed_mean(past_full, sorted_other_rank):
    # history_speed_mean only averages over time dimension T, output shape is [N].
    # Row order is aligned with history_trajectory/history_control: ego first, then ranked agents.
    speed_abs = past_full[:, :, 4].abs()

    def _agent_time_mean(aidx: int):
        vals = speed_abs[aidx]
        valid = torch.isfinite(vals)
        if valid.any():
            return float(vals[valid].mean().item())
        return 0.0

    history_speed_mean = [_agent_time_mean(0)]
    ordered = sorted([(rank, aidx) for aidx, rank in sorted_other_rank.items()], key=lambda x: x[0])
    for _rank_id, aidx in ordered:
        history_speed_mean.append(_agent_time_mean(int(aidx)))
    return history_speed_mean


def _agent_length_width_entry(agent_lw, aidx: int):
    length = agent_lw[aidx, 0]
    width = agent_lw[aidx, 1]
    if isinstance(length, torch.Tensor):
        length = length.item()
    if isinstance(width, torch.Tensor):
        width = width.item()
    fl, fw = float(length), float(width)
    if np.isnan(fl) or np.isnan(fw):
        return {"length": None, "width": None}
    return {"length": fl, "width": fw}


def build_agent_length_width_from_mapping(agent_index_mapping, agent_lw):
    """List [N] of {length, width} aligned with history row order (rank 0, 1, ...)."""
    ordered_ranks = sorted(agent_index_mapping.keys(), key=lambda k: int(k))
    return [_agent_length_width_entry(agent_lw, agent_index_mapping[r]) for r in ordered_ranks]


def save_trajectory_json(
    past_global,
    past_local,
    past_control_sequence,
    history_speed_mean,
    current_state_global,
    ego_future_traj,
    sur_future_traj,
    sorted_other_rank,
    agent_lw,
    json_path,
):
    agent_index_mapping = {}

    def _xy_vec(x, y):
        if isinstance(x, torch.Tensor):
            x = x.item()
        if isinstance(y, torch.Tensor):
            y = y.item()
        fx, fy = float(x), float(y)
        if np.isnan(fx) or np.isnan(fy):
            return [None, None]
        return [fx, fy]

    def _heading_from_hvec(hx, hy):
        if isinstance(hx, torch.Tensor):
            hx = hx.item()
        if isinstance(hy, torch.Tensor):
            hy = hy.item()
        fhx, fhy = float(hx), float(hy)
        if np.isnan(fhx) or np.isnan(fhy):
            return None
        return float(np.arctan2(fhy, fhx))

    def _wrap_to_pi_local(angle):
        return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi

    def _history_diff_rows(past_xyh):
        """
        Build per-agent history diffs aligned with time axis T.
        Output shape: [N, T], each entry is {"delta_s": float|None, "delta_heading": float|None}.
        - delta_s uses Euclidean displacement between consecutive positions.
        - delta_heading uses wrapped heading difference between consecutive heading angles.
        """
        if isinstance(past_xyh, torch.Tensor):
            arr = past_xyh.detach().cpu().numpy()
        else:
            arr = np.asarray(past_xyh)
        if arr.ndim != 3 or arr.shape[2] < 4:
            raise ValueError(f"past must be (N, T, >=4), got {arr.shape}")

        N, T_local, _ = arr.shape
        rows = []
        for aidx in range(N):
            x = arr[aidx, :, 0].astype(np.float64, copy=False)
            y = arr[aidx, :, 1].astype(np.float64, copy=False)
            hx = arr[aidx, :, 2].astype(np.float64, copy=False)
            hy = arr[aidx, :, 3].astype(np.float64, copy=False)
            heading = np.arctan2(hy, hx)

            row = [{"delta_s": None, "delta_heading": None}]
            for t in range(1, T_local):
                if not (
                    np.isfinite(x[t - 1])
                    and np.isfinite(y[t - 1])
                    and np.isfinite(x[t])
                    and np.isfinite(y[t])
                    and np.isfinite(heading[t - 1])
                    and np.isfinite(heading[t])
                ):
                    row.append({"delta_s": None, "delta_heading": None})
                    continue
                dx = float(x[t] - x[t - 1])
                dy = float(y[t] - y[t - 1])
                delta_s = float(np.hypot(dx, dy))
                delta_heading = _wrap_to_pi_local(float(heading[t] - heading[t - 1]))
                row.append(
                    {
                        "delta_s": round(delta_s, 4),
                        "delta_heading": round(delta_heading, 4),
                    }
                )
            # Match the convention used in crash_data_processor: controls are defined per-step,
            # and first step should not be null. Here we copy step-1 into step-0 when possible.
            if T_local >= 2:
                row[0] = dict(row[1])
            rows.append(row)
        return rows

    T = past_global.size(1)
    history_global_ntd = []
    history_local_ntd = []
    history_diff_ntd = []

    ego_row_global = []
    ego_row_local = []
    for t in range(T):
        ego_row_global.append(_xy_vec(past_global[0, t, 0], past_global[0, t, 1]))
        ego_row_local.append(_xy_vec(past_local[0, t, 0], past_local[0, t, 1]))
    history_global_ntd.append(ego_row_global)
    history_local_ntd.append(ego_row_local)
    history_diff_ntd.append(_history_diff_rows(past_local[:1, :, :4])[0])
    agent_index_mapping["0"] = 0

    ordered = sorted([(rank, aidx) for aidx, rank in sorted_other_rank.items()], key=lambda x: x[0])
    for _rank_id, aidx in ordered:
        row_global = []
        row_local = []
        for t in range(T):
            row_global.append(_xy_vec(past_global[aidx, t, 0], past_global[aidx, t, 1]))
            row_local.append(_xy_vec(past_local[aidx, t, 0], past_local[aidx, t, 1]))
        history_global_ntd.append(row_global)
        history_local_ntd.append(row_local)
        history_diff_ntd.append(_history_diff_rows(past_local[aidx : aidx + 1, :, :4])[0])
        agent_index_mapping[str(int(_rank_id))] = int(aidx)

    ego_current_state = {
        "x": _xy_vec(current_state_global[0, 0], current_state_global[0, 1])[0],
        "y": _xy_vec(current_state_global[0, 0], current_state_global[0, 1])[1],
        "heading": _heading_from_hvec(current_state_global[0, 2], current_state_global[0, 3]),
    }

    ego_future_traj_list = []
    for t in range(ego_future_traj.size(0)):
        ego_future_traj_list.append(_xy_vec(ego_future_traj[t, 0], ego_future_traj[t, 1]))

    sur_future_traj_list = []
    for n in range(sur_future_traj.size(0)):
        sur_future_traj_list.append([])
        for t in range(sur_future_traj.size(1)):
            sur_future_traj_list[n].append(_xy_vec(sur_future_traj[n, t, 0], sur_future_traj[n, t, 1]))

    agent_length_width = build_agent_length_width_from_mapping(agent_index_mapping, agent_lw)

    payload = {
        "history_trajectory_global": history_global_ntd,
        "history_trajectory_local": history_local_ntd,
        "history_diff": history_diff_ntd,
        "history_control_local": past_control_sequence,
        "history_speed_mean": history_speed_mean,
        "ego_current_state": ego_current_state,
        "predicted_ego_future_trajectory_local": ego_future_traj_list,
        "predicted_surrounding_future_trajectory_local": sur_future_traj_list,
        "agent_length_width": agent_length_width,
        "agent_index_mapping": agent_index_mapping,
        "note": (
            "history_trajectory_global/local are nested list [N,T,D] with D=2 (x,y); row index 0 is ego, "
            "surrounding agents are ordered by current distance rank. agent_index_mapping keys are rank_id "
            "(0 for ego, 1..K for selected surrounding agents). Invalid steps are [null,null]. "
            "history_diff is nested list [N,T] aligned with history_trajectory_local row/time order; each step is "
            '{"delta_s": float|null, "delta_heading": float|null} where delta_s is Euclidean displacement between '
            "consecutive local-frame positions and delta_heading is wrapped heading difference (radians) between "
            "consecutive local-frame heading vectors (hx, hy). The first step (t=0) is always nulls. "
            "history_control_local: [N,T] aligned rows, each step [acc_mps2, steer_deg] from compute_kinematics("
            "v,psi) with v,psi from np.gradient on x,y projected on heading (NuScenes history dt=0.5 s). "
            "history_speed_mean is a 1D list [N] and each value is mean(|speed|) over time dimension T "
            "for that row's agent. Row order is aligned with history_trajectory/history_control: ego first, "
            "then selected surrounding agents ordered by distance rank. "
            "agent_length_width is a list [N] aligned with history_trajectory row order (ego first, then "
            "selected surrounding agents by distance rank); each entry is "
            '{"length": float|null, "width": float|null} in meters (unnormalized nuScenes box size). '
            "ego_current_state is in global frame and heading is in radians. "
            "predicted_ego_future_trajectory_local is a list of [x, y] positions for future trajectory steps."
        ),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def visualize_sample(
    ego_future_traj,
    sur_future_traj,
    pred_ego_future_traj,
    pred_sur_future_traj,
    scene_graph,
    map_idx,
    state_norm,
    att_norm,
    map_env,
    nusc_obj,
    sample_idx,
    out_dir,
    map_size,
    margin_m,
    min_bound_m,
    split,
    use_rank_id=False,
    text_use_global_coords="local",
    local_rank="0",
):
    map_idx_t = torch.tensor([map_idx], dtype=torch.long).to(ego_future_traj.device)

    past_full = state_norm.unnormalize(scene_graph.past.clone())
    past = past_full[:, :, :4]

    lw = att_norm.unnormalize(scene_graph.lw.clone())

    if past.size(1) < 4:
        raise ValueError(f"past_len={past.size(1)} < 4, cannot build 4-point history.")

    current_state = past[:, -1, :4]
    current_speed = past_full[:, -1, 4]
    past_local, current_state_local = convert_states_to_ego_current_frame(past, current_state)
    ego_future_local = convert_states_to_ego_current_frame(ego_future_traj.unsqueeze(0), current_state)[0][0]
    sur_future_local = convert_states_to_ego_current_frame(sur_future_traj, current_state)[0]
    # print("sur_future_local:", sur_future_local.shape, sur_future_local)
    past_frenet = build_frenet_history_states(past)
    current_state_frenet = past_frenet[:, -1, :]
    sorted_other_rank = build_surrounding_agent_rank(current_state, max_agents=10)

    crop_pos, bounds = get_eval_style_crop(past, margin_m=margin_m, min_bound_m=min_bound_m)
    ego_h = current_state[0, 2:4]
    ego_h_norm = torch.linalg.norm(ego_h).item()
    if ego_h_norm > 1e-6:
        # Map rendering uses crop length axis as horizontal (x) after transpose.
        # Rotate ego heading by -90 deg for crop frame, so ego heading appears vertical up.
        crop_h = torch.tensor([[ego_h[1].item(), -ego_h[0].item()]], dtype=torch.float32).to(ego_future_traj.device)
    else:
        crop_h = torch.tensor([[1.0, 0.0]], dtype=torch.float32).to(ego_future_traj.device)
    crop_kin = torch.cat([crop_pos.view(1, 2), crop_h], dim=1)

    map_rend = map_env.get_map_crop_pos(
        crop_kin,
        map_idx_t,
        bounds=bounds,
        L=map_size,
        W=map_size,
    )[0]

    na, nt, _ = past.size()
    crop_traj, crop_lw = map_env.objs2crop(
        crop_kin[0],
        past.reshape(na * nt, 4),
        lw,
        None,
        bounds=bounds,
        L=map_size,
        W=map_size,
    )
    crop_traj = crop_traj.reshape(na, nt, 4)

    sorted_other_rank = filter_sorted_other_rank_for_crop_viz(
        sorted_other_rank, crop_traj, crop_lw, map_size=map_size
    )

    map_name = map_env.map_list[int(map_idx)]
    # scene_name, sample_token = _get_sample_token(scene_graph, nusc_obj, sample_idx)
    lane_relation_by_agent = build_lane_relation_by_agent(
        map_env=map_env,
        map_name=map_name,
        crop_pos=crop_pos,
        bounds=bounds,
        current_state_global=current_state,
        current_state_local=current_state_local,
        log_tag=f"sample={sample_idx} map={map_name}",
        map_idx=map_idx,
    )
    map_info = _collect_map_and_traffic_info(map_env, map_name, crop_pos, bounds)
    map_info["crop_center"] = convert_xy_to_ego_current_frame(map_info["crop_center"], current_state[0])
    for i in range(len(map_info["ped_crossings"])):
        cxy = map_info["ped_crossings"][i]["center_xy_global"]
        map_info["ped_crossings"][i]["center_xy_local"] = convert_xy_to_ego_current_frame(cxy, current_state[0])
    for i in range(len(map_info["traffic_lights"])):
        cxy = map_info["traffic_lights"][i]["center_xy_global"]
        if cxy is None:
            map_info["traffic_lights"][i]["center_xy_local"] = (float("inf"), float("inf"))
        else:
            map_info["traffic_lights"][i]["center_xy_local"] = convert_xy_to_ego_current_frame(cxy, current_state[0])

    sample_dir = os.path.join(out_dir, f"{local_rank}_{sample_idx:06d}")
    os.makedirs(sample_dir, exist_ok=True)
    if use_rank_id:
        rank_id_to_aidx = {int(rid): int(aidx) for aidx, rid in sorted_other_rank.items()}
        aidx_to_rank_id = {int(aidx): int(rid) for aidx, rid in sorted_other_rank.items()}
        agent_tokens_list = _extract_agent_tokens(scene_graph)
        rank_id_to_agent_token = {}
        aidx_to_agent_token = {}
        if agent_tokens_list is not None:
            for aidx, rid in sorted_other_rank.items():
                aidx_i = int(aidx)
                tok = agent_tokens_list[aidx_i] if 0 <= aidx_i < len(agent_tokens_list) else None
                rank_id_to_agent_token[int(rid)] = tok
                aidx_to_agent_token[aidx_i] = tok
        mapping_payload = {
            "sample_idx": int(sample_idx),
            "rank_id_to_agent_idx": dict(sorted(rank_id_to_aidx.items(), key=lambda kv: kv[0])),
            "agent_idx_to_rank_id": dict(sorted(aidx_to_rank_id.items(), key=lambda kv: kv[0])),
            "rank_id_to_agent_token": dict(sorted(rank_id_to_agent_token.items(), key=lambda kv: kv[0])),
            "agent_idx_to_agent_token": dict(sorted(aidx_to_agent_token.items(), key=lambda kv: kv[0])),
            "note": (
                "rank_id is 1-based distance rank among selected surrounding agents (ego excluded). "
                "agent_token is the nuScenes instance_token uniquely identifying an agent across the scene; "
                "ego vehicle has no instance_token and is represented as 'ego' at agent_idx=0."
            ),
        }
        map_path = os.path.join(sample_dir, "rank_id_mapping.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(mapping_payload, f, ensure_ascii=False, indent=2)
    safe_map_name = map_name.replace("/", "_").replace(" ", "_")
    prefix = os.path.join(sample_dir, f"{safe_map_name}_{split}_{local_rank}_{sample_idx:06d}")
    img_path = prefix + ".png"
    txt_path = prefix + ".txt"
    trajectory_json_path = prefix + "_trajectory.json"

    img= draw_annotated(
        map_rend,
        crop_traj,
        crop_lw,
        sorted_other_rank,
        out_path=img_path,
        ped_crossing_idx=map_env.layer_map.get("ped_crossing"),
        use_rank_id=use_rank_id,
    )
    if text_use_global_coords == "global":
        text_past = past
        text_current_state = current_state
    elif text_use_global_coords == "frenet":
        text_past = past_frenet
        text_current_state = current_state_frenet
    else:
        text_past = past_local
        text_current_state = current_state_local
    text = save_sample_prompt_text(
        map_name,
        split,
        text_past,
        text_current_state,
        current_state,
        current_speed,
        sorted_other_rank,
        map_info,
        past_frenet,
        current_state_frenet,
        lane_relation_by_agent,
        txt_path=txt_path,
        use_rank_id=use_rank_id,
        coord_mode=text_use_global_coords,
        pred_ego_future=pred_ego_future_traj,
        pred_sur_future=pred_sur_future_traj,
    )
    past_control_sequence = compute_control_sequence(past_local, sorted_other_rank)
    history_speed_mean = compute_history_speed_mean(past_full, sorted_other_rank)

    save_trajectory_json(
        past,
        past_local,
        past_control_sequence,
        history_speed_mean,
        current_state,
        ego_future_local,
        sur_future_local,
        sorted_other_rank,
        lw,
        trajectory_json_path,
    )

    print(f"[{sample_idx}] saved: {img_path}")
    print(f"[{sample_idx}] saved: {txt_path}")
    print(f"[{sample_idx}] saved: {trajectory_json_path}")


@torch.inference_mode()
def gen_future(
    model,
    scene_graph,
    map_idx,
    map_env,
    device,
):
    NA = scene_graph.past.size(0)
    ego_mask = torch.zeros((NA), dtype=torch.bool)
    ego_mask[0] = True
    
    # # use model rollout
    embed_info_attached = model.embed(scene_graph, map_idx, map_env)
    prior_z = embed_info_attached['prior_out'][0] #(NA, z)
    post_z = embed_info_attached['posterior_out'][0] #(NA, z)
    target_signal = post_z.unsqueeze(2)
    target_encoding = model.encoder(x_start=target_signal, cond=prior_z)
    del post_z
    
    model.reset_var(zero_var=True)
    target_recon = model(cond=prior_z, noise=target_encoding, step=1).trajectories
    del target_encoding

    traj_recon = model.decode_embedding(target_recon.squeeze(2), embed_info_attached, scene_graph, map_idx, map_env)['future_pred']
    del target_recon
    del embed_info_attached
    del prior_z

    traj = model.get_normalizer().unnormalize(traj_recon)
    del traj_recon

    pred_ego_future_traj = traj[0, :, :4].detach().to(device)
    pred_sur_future_traj = traj[1:, :, :4].detach().to(device)
    del traj

    torch.cuda.empty_cache()

    ## use gt
    gt_future = scene_graph.future_gt[:, :, :4] #(NA, HT, 4)
    gt_future[scene_graph.future_vis == 0.0] = 0.0
    ego_future_traj = model.get_normalizer().unnormalize(gt_future[ego_mask]).squeeze(0).detach().to(device) #(HT, 4)
    sur_future_traj = model.get_normalizer().unnormalize(gt_future[~ego_mask]).detach().to(device) #(NA-1, HT, 4)
    # print("ego_future_traj (GT):", ego_future_traj.shape, ego_future_traj)

    return ego_future_traj, sur_future_traj, pred_ego_future_traj, pred_sur_future_traj


def gen_image_text(
    scene_graph,
    map_idx,
    state_norm,
    att_norm,
    map_env,
    map_size,
    margin_m,
    min_bound_m,
    split,
    use_rank_id=False,
    text_use_global_coords="local",
    log_tag=None,
):
    map_idx_t = torch.tensor([map_idx], dtype=torch.long)

    past_full = state_norm.unnormalize(scene_graph.past.clone())
    past = past_full[:, :, :4]
    lw = att_norm.unnormalize(scene_graph.lw.clone())

    if past.size(1) < 4:
        raise ValueError(f"past_len={past.size(1)} < 4, cannot build 4-point history.")
    past = past[:, -4:, :4]

    current_state = past[:, -1, :4]
    current_speed = past_full[:, -1, 4]
    past_local, current_state_local = convert_states_to_ego_current_frame(past, current_state)
    past_frenet = build_frenet_history_states(past)
    current_state_frenet = past_frenet[:, -1, :]
    sorted_other_rank = build_surrounding_agent_rank(current_state, max_agents=10)

    crop_pos, bounds = get_eval_style_crop(past, margin_m=margin_m, min_bound_m=min_bound_m)
    ego_h = current_state[0, 2:4]
    ego_h_norm = torch.linalg.norm(ego_h).item()
    if ego_h_norm > 1e-6:
        # Map rendering uses crop length axis as horizontal (x) after transpose.
        # Rotate ego heading by -90 deg for crop frame, so ego heading appears vertical up.
        crop_h = torch.tensor([[ego_h[1].item(), -ego_h[0].item()]], dtype=torch.float32)
    else:
        crop_h = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    crop_kin = torch.cat([crop_pos.view(1, 2), crop_h], dim=1)

    map_rend = map_env.get_map_crop_pos(
        crop_kin,
        map_idx_t,
        bounds=bounds,
        L=map_size,
        W=map_size,
    )[0]

    na, nt, _ = past.size()
    crop_traj, crop_lw = map_env.objs2crop(
        crop_kin[0],
        past.reshape(na * nt, 4),
        lw,
        None,
        bounds=bounds,
        L=map_size,
        W=map_size,
    )
    crop_traj = crop_traj.reshape(na, nt, 4)

    sorted_other_rank = filter_sorted_other_rank_for_crop_viz(
        sorted_other_rank, crop_traj, crop_lw, map_size=map_size
    )

    map_name = map_env.map_list[int(map_idx)]
    if log_tag is None:
        sidx = getattr(scene_graph, "sidx", None)
        log_tag = f"gen_image_text sidx={sidx} map={map_name}" if sidx is not None else f"map={map_name}"
    lane_relation_by_agent = build_lane_relation_by_agent(
        map_env=map_env,
        map_name=map_name,
        crop_pos=crop_pos,
        bounds=bounds,
        current_state_global=current_state,
        current_state_local=current_state_local,
        log_tag=log_tag,
        map_idx=map_idx,
    )
    map_info = _collect_map_and_traffic_info(map_env, map_name, crop_pos, bounds)
    map_info["crop_center"] = convert_xy_to_ego_current_frame(map_info["crop_center"], current_state[0])
    for i in range(len(map_info["ped_crossings"])):
        cxy = map_info["ped_crossings"][i]["center_xy_global"]
        map_info["ped_crossings"][i]["center_xy_local"] = convert_xy_to_ego_current_frame(cxy, current_state[0])
    for i in range(len(map_info["traffic_lights"])):
        cxy = map_info["traffic_lights"][i]["center_xy_global"]
        if cxy is None:
            map_info["traffic_lights"][i]["center_xy_local"] = (float("inf"), float("inf"))
        else:
            map_info["traffic_lights"][i]["center_xy_local"] = convert_xy_to_ego_current_frame(cxy, current_state[0])

    img = draw_annotated(
        map_rend,
        crop_traj,
        crop_lw,
        sorted_other_rank,
        ped_crossing_idx=map_env.layer_map.get("ped_crossing"),
        use_rank_id=use_rank_id,
    )
    if text_use_global_coords == "global":
        text_past = past
        text_current_state = current_state
    elif text_use_global_coords == "frenet":
        text_past = past_frenet
        text_current_state = current_state_frenet
    else:
        text_past = past_local
        text_current_state = current_state_local
    text = save_sample_prompt_text(
        map_name,
        split,
        text_past,
        text_current_state,
        current_state,
        current_speed,
        sorted_other_rank,
        map_info,
        past_frenet,
        current_state_frenet,
        lane_relation_by_agent,
        use_rank_id=use_rank_id,
        coord_mode=text_use_global_coords,
    )

    print("text", text)
    print("img", img)
    
    return img, text


def _posterior_interaction_keep_mask(
    current_state_local,
    current_speed,
    *,
    max_range_m=60.0,
    behind_x_m=1.0,
    max_behind_dist_m=35.0,
    moving_away_rel_speed_mps=1.0,
    horizon_s=6.0,
    reaction_time_s=1.0,
    max_decel_mps2=6.0,
    lane_like_half_width_m=2.0,
    max_lateral_far_m=25.0,
    min_heading_diff_rad=math.pi / 4,
    debug=True,
):
    """
    R1-R5 posterior interaction rules (ego-local frame: +x forward, +y left).

    Mirrors agent/scene_agent._filter_adversarial_vehicle_ids_posterior without R0.
    Returns (NA-1,) bool mask for agents indexed 1..NA-1 in current_state_local.
    """
    na = int(current_state_local.size(0))
    if na <= 1:
        return torch.zeros(0, dtype=torch.bool)

    keep = torch.ones(na - 1, dtype=torch.bool)
    ego_heading = 0.0

    ego_speed = float(current_speed[0].item()) if not torch.isnan(current_speed[0]) else 0.0
    if (not math.isfinite(ego_speed)) or ego_speed < 0.0:
        ego_speed = 0.0

    reaction_s_f = max(0.0, float(reaction_time_s))
    amax = float(max_decel_mps2)
    if (not math.isfinite(amax)) or amax <= 1e-3:
        amax = 6.0
    break_s_f = ego_speed / amax
    if (not math.isfinite(break_s_f)) or break_s_f < 0.0:
        break_s_f = 0.0
    horizon_s_f = max(0.0, float(horizon_s) - reaction_s_f - break_s_f)
    ego_brake_dist = (ego_speed * ego_speed) / (2.0 * amax)
    ego_react_dist = ego_speed * reaction_s_f
    ego_horizon_dist = ego_speed * horizon_s_f
    base_dynamic_range = 10.0 + ego_react_dist + ego_horizon_dist + ego_brake_dist
    if not math.isfinite(base_dynamic_range):
        base_dynamic_range = 50.0
    base_dynamic_range = float(np.clip(base_dynamic_range, 15.0, float(max_range_m)))

    behind_dist_m = max(float(max_behind_dist_m), float(behind_x_m) + 1.0)
    band = float(min_heading_diff_rad)

    def _dbg(msg):
        if debug:
            Logger.log(f"[posterior_filter] {msg}")

    for aidx in range(1, na):
        st = current_state_local[aidx]
        if torch.isnan(st).any():
            keep[aidx - 1] = False
            _dbg(f"filtered aidx={aidx} (non-finite state)")
            continue

        lx = float(st[0].item())
        ly = float(st[1].item())
        hx = float(st[2].item())
        hy = float(st[3].item())
        dist = math.hypot(lx, ly)
        if (not math.isfinite(dist)) or dist < 1e-6:
            keep[aidx - 1] = False
            _dbg(f"filtered aidx={aidx} (non-finite/zero dist), lx={lx}, ly={ly}")
            continue

        agent_speed = float(current_speed[aidx].item()) if not torch.isnan(current_speed[aidx]) else 0.0
        if (not math.isfinite(agent_speed)) or agent_speed < 0.0:
            agent_speed = 0.0

        agent_h = math.atan2(hy, hx) if math.hypot(hx, hy) > 1e-6 else None
        rvx = agent_speed * math.cos(agent_h) if agent_h is not None else None
        rvy = agent_speed * math.sin(agent_h) if agent_h is not None else None

        rel_radial = None
        if rvx is not None and rvy is not None:
            rel_radial = (lx * rvx + ly * rvy) / dist

        # R1: dynamic range gate
        dyn_range = base_dynamic_range + agent_speed * float(horizon_s)
        if not math.isfinite(dyn_range):
            dyn_range = float(max_range_m)
        dyn_range = float(np.clip(dyn_range, 30.0, float(max_range_m)))
        if dist > dyn_range:
            keep[aidx - 1] = False
            _dbg(
                f"R1 filtered aidx={aidx} dist={dist:.2f}m > dyn_range={dyn_range:.2f}m "
                f"(ego_v={ego_speed:.2f}, agent_v={agent_speed:.2f}, horizon={horizon_s_f:.2f})"
            )
            continue

        is_behind = lx < -float(behind_x_m)
        same_lane = abs(ly) <= float(lane_like_half_width_m)

        # R2: same-lane follower behind ego
        if is_behind and same_lane:
            keep[aidx - 1] = False
            _dbg(
                f"R2 filtered aidx={aidx} (behind+same_lane), lx={lx:.2f}, ly={ly:.2f}, "
                f"lane_half_w={float(lane_like_half_width_m):.2f}"
            )
            continue

        # R3: far behind ego
        if lx < -behind_dist_m:
            keep[aidx - 1] = False
            _dbg(
                f"R3 filtered aidx={aidx} (far_behind), lx={lx:.2f}, "
                f"max_behind_dist={behind_dist_m:.2f}"
            )
            continue

        # R4: behind + moving away
        is_moving_away = rel_radial is not None and rel_radial >= float(moving_away_rel_speed_mps)
        if is_behind and is_moving_away:
            keep[aidx - 1] = False
            _dbg(
                f"R4 filtered aidx={aidx} (behind+away), lx={lx:.2f}, ly={ly:.2f}, "
                f"rel_radial={rel_radial:.2f}"
            )
            continue

        # R5: lateral far / same-dir away >=45deg / opposite-dir radially away >=45deg
        lateral_far = abs(ly) > float(max_lateral_far_m)

        agent_heading_rad = agent_h
        heading_diff_rad = None
        if agent_heading_rad is not None:
            heading_diff_rad = abs(_wrap_to_pi(agent_heading_rad - ego_heading))

        same_dir = heading_diff_rad is not None and heading_diff_rad <= band
        opp_dir = heading_diff_rad is not None and heading_diff_rad >= (math.pi - band)
        is_radially_away = rel_radial is not None and rel_radial >= float(moving_away_rel_speed_mps)

        vel_angle_rad = None
        if rvx is not None and rvy is not None and math.hypot(rvx, rvy) > 1e-3:
            vel_angle_rad = math.atan2(rvy, rvx)

        same_dir_away_45 = False
        if same_dir and is_radially_away and vel_angle_rad is not None:
            vel_vs_ego = abs(_wrap_to_pi(vel_angle_rad - ego_heading))
            same_dir_away_45 = vel_vs_ego >= band

        opp_dir_away_45 = bool(opp_dir and is_radially_away)

        if lateral_far or same_dir_away_45 or opp_dir_away_45:
            reasons = []
            if lateral_far:
                reasons.append("lateral_far")
            if same_dir_away_45:
                reasons.append("same_dir_away_45")
            if opp_dir_away_45:
                reasons.append("opp_dir_away_45")
            keep[aidx - 1] = False
            _dbg(
                f"R5 filtered aidx={aidx} ({'+'.join(reasons)}), ly={ly:.2f}, "
                f"rel_radial={rel_radial if rel_radial is not None else float('nan'):.2f}, "
                f"heading_diff={math.degrees(heading_diff_rad) if heading_diff_rad is not None else float('nan'):.1f}deg, "
                f"vel_vs_ego={math.degrees(abs(_wrap_to_pi(vel_angle_rad - ego_heading))) if vel_angle_rad is not None else float('nan'):.1f}deg, "
                f"band={math.degrees(band):.1f}deg, max_lateral_far={float(max_lateral_far_m):.2f}"
            )

    return keep


def filter_scene(scene_graph, map_idx, map_env, i, traffic_model, data_loader, cfg):
    batch_scene_graph = []
    batch_map_idx = []
    batch_total_NA = 0
    strive_loss_weights = {
        'coll_veh' : cfg.loss_coll_veh,
        'coll_veh_plan' : cfg.loss_coll_veh_plan,
        'coll_env' : cfg.loss_coll_env,
        'motion_prior' : cfg.loss_motion_prior,
        'motion_prior_atk' : cfg.loss_motion_prior_atk,
        'init_z' : cfg.loss_init_z,
        'init_z_atk': cfg.loss_init_z_atk,
        'motion_prior_ext' : cfg.loss_motion_prior_ext,
        'match_ext' : cfg.loss_match_ext,
        'adv_crash' : cfg.loss_adv_crash,
        'init_match_ext' : cfg.init_loss_match_ext,
        'init_motion_prior_ext' : cfg.init_loss_motion_prior_ext
    }

    is_last_batch = i == (len(data_loader)-1)

    # First sample prior to get possible futures
    with torch.no_grad():
        sample_pred = traffic_model.sample(scene_graph, map_idx, map_env, 20, include_mean=True)

    # determine if this sequence is feasible for scenario generation
    feasible, feasible_time, feasible_dist = determine_feasibility_nusc(sample_pred['future_pred'],
                                                                        traffic_model.get_normalizer(),
                                                                        cfg.feasibility_thresh,
                                                                        cfg.feasibility_time,
                                                                        0.0,
                                                                        feasibility_infront_min=cfg.feasibility_infront_min,
                                                                        check_non_drivable_separation=cfg.feasibility_check_sep,
                                                                        map_env=map_env,
                                                                        map_idx=map_idx)

    # Only consider agents shown in visualization (top-k by distance to ego).
    if feasible is not None:
        state_norm = traffic_model.get_normalizer()
        past_full = state_norm.unnormalize(scene_graph.past.clone())
        past = past_full[:, :, :4]
        current_state = past[:, -1, :4]
        current_speed = past_full[:, -1, 4]
        sorted_other_rank = build_surrounding_agent_rank(current_state, max_agents=10)
        topk_mask = torch.zeros(feasible.size(0), dtype=torch.bool, device=feasible.device)
        for aidx in sorted_other_rank:
            topk_mask[aidx - 1] = True
        n_feas_before = int(torch.sum(feasible).item())
        feasible = torch.logical_and(feasible, topk_mask)
        n_feas_after = int(torch.sum(feasible).item())
        if n_feas_before > n_feas_after:
            Logger.log(
                'Filtered %d feasible agent(s) outside visualization top-%d'
                % (n_feas_before - n_feas_after, len(sorted_other_rank))
            )

        # R1-R5 posterior interaction rules (same defaults as scene_agent).
        past4 = past[:, -4:, :4]
        current_state_global = past4[:, -1, :4]
        _, current_state_local = convert_states_to_ego_current_frame(past4, current_state_global)
        posterior_keep = _posterior_interaction_keep_mask(
            current_state_local,
            current_speed,
            debug=True,
        ).to(feasible.device)
        n_post_before = int(torch.sum(feasible).item())
        feasible = torch.logical_and(feasible, posterior_keep)
        n_post_after = int(torch.sum(feasible).item())
        if n_post_before > n_post_after:
            Logger.log(
                'Posterior filter (R1-R5) removed %d feasible agent(s)'
                % (n_post_before - n_post_after)
            )

    # make sure some sample of the ego went over the velocity thresh
    #   so with some confidence it will be an interesting scenario
    ego_samps = traffic_model.get_normalizer().unnormalize(sample_pred['future_pred'][0].detach()) # NS x FT x 4
    ego_vels = torch.norm(ego_samps[:, 1:, :2] - ego_samps[:, :-1, :2], dim=-1) # NS x FT-1
    max_vel = torch.max(ego_vels).cpu().item()
    if max_vel < cfg.feasibility_vel:
        Logger.log('Ego samples not moving more than velocity threshold, skipping...')
        if not is_last_batch:
            return True

    if feasible is None:
        Logger.log('Only ego vehicle in scene, skipping...')
        if not is_last_batch:
            return True
    elif torch.sum(feasible).item() == 0:
        Logger.log('Infeasible, no vehicles near ego after top-k/posterior filter, skipping...')
        if not is_last_batch:
            return True

    is_feas = False
    if feasible is not None and torch.sum(feasible).item() > 0:
        is_feas = True

        # print which vehicle is the best candidate (for info/debug purposes)
        feasible_dist[~feasible] = float('inf')
        temp_attack_agt = torch.min(feasible_dist, dim=0)[1] + 1
        print('Heuristic attack agt is %d' % (temp_attack_agt))
        print('Heuristic attack time is %d' % (feasible_time[temp_attack_agt-1]))  

    # This is a feasible seed, add it to the batch
    if is_feas:
        Logger.log('Feasible. Adding to batch...')
        batch_scene_graph += scene_graph.to_data_list()
        batch_map_idx.append(map_idx)
        batch_total_NA += scene_graph.future_gt.size(0)
        Logger.log('Current batch NA: %d' % (batch_total_NA))

    if batch_total_NA < cfg.batch_size and not is_last_batch:
        # collect more before performing optim
        return True
    else:
        if len(batch_scene_graph) == 0:
            # this is the last seq in dataset, and we have no other seqs queueued
            return True
        else:
            # create the batch
            scene_graph = GraphBatch.from_data_list(batch_scene_graph)
            map_idx = torch.cat(batch_map_idx, dim=0)

    B = map_idx.size(0)
    NA = scene_graph.past.size(0)
    ego_inds = scene_graph.ptr[:-1]
    ego_mask = torch.zeros((NA), dtype=torch.bool)
    ego_mask[ego_inds] = True   

    #
    # Initialize optimization
    #
    # embed past and map to get inputs to decoder used during optim
    with torch.no_grad():
        embed_info_attached = traffic_model.embed(scene_graph, map_idx, map_env)
    # need to detach all the encoder outputs from current comp graph to be used in optimization
    embed_info = detach_embed_info(embed_info_attached)
    init_future_pred = init_traj = z_init = init_coll_env = None

    ## basic planner: hardcode planner
    from planners.hardcode_goalcond_nusc import HardcodeNuscPlanner, CONFIG_DICT
    assert(cfg.planner_cfg in CONFIG_DICT)
    Logger.log('Using planner config:')
    Logger.log(CONFIG_DICT[cfg.planner_cfg])
    basic_planner = HardcodeNuscPlanner(map_env, PlannerConfig(**CONFIG_DICT[cfg.planner_cfg]))

    # start from GT scene future (reconstructed with motion model)
    z_init = embed_info_attached['posterior_out'][0].detach()
    init_traj = scene_graph.future_gt[:, :, :4].clone().detach()
    Logger.log('Running initialization optimization...')

    # run initial optimization to closely fit nuscenes scene
    z_init, init_fit_traj, _ = run_init_optim(z_init, init_traj, scene_graph.future_vis, 0.1, strive_loss_weights, traffic_model,
                                            scene_graph, map_env, map_idx, 75, embed_info, embed_info['prior_out'])

    # basic planner: hardcode planner, replace ego with planner rollout
    # reset planner
    all_init_state = traffic_model.get_normalizer().unnormalize(scene_graph.past_gt[:, -1, :])
    all_init_veh_att = traffic_model.get_att_normalizer().unnormalize(scene_graph.lw)
    basic_planner.reset(all_init_state, all_init_veh_att, scene_graph.batch, B, map_idx)

    # rollout
    init_non_ego = traffic_model.normalizer.unnormalize(init_fit_traj[~ego_mask]).cpu().numpy()
    plan_t = np.linspace(traffic_model.dt, traffic_model.dt*traffic_model.FT, traffic_model.FT)
    init_agt_ptr = scene_graph.ptr - torch.arange(B+1, device=scene_graph.ptr.device)
    planner_init = basic_planner.rollout(init_non_ego, plan_t, init_agt_ptr.cpu().numpy(), plan_t,
                                    control_all=False).to(scene_graph.future_gt)
    planner_init = traffic_model.get_normalizer().normalize(planner_init)
    # replace init traj ego's with planner traj
    init_traj[ego_mask] = planner_init

    # and optim a bit more, now to match the planner traj
    Logger.log('Fine-tune init with planner rollout...')
    lr = 0.05
    z_init, init_fit_traj, _ = run_init_optim(z_init, init_traj, scene_graph.future_vis, lr, strive_loss_weights, traffic_model,
                                                scene_graph, map_env, map_idx, 100, embed_info, embed_info['prior_out'])

    # check if planner collides with scene trajectories already. if so, not worth continuing
    from losses.adv_gen_nusc import check_single_veh_coll
    bvalid = []
    for b in range(B):
        init_hardcode_coll, _ = check_single_veh_coll(traffic_model.get_normalizer().unnormalize(init_fit_traj[scene_graph.ptr[b]]),
                                                        traffic_model.get_att_normalizer().unnormalize(scene_graph.lw[scene_graph.ptr[b]]),
                                                        traffic_model.get_normalizer().unnormalize(init_fit_traj[(scene_graph.ptr[b]+1):scene_graph.ptr[b+1]]),
                                                        traffic_model.get_att_normalizer().unnormalize(scene_graph.lw[(scene_graph.ptr[b]+1):scene_graph.ptr[b+1]])
                                                        )
        bvalid.append(np.sum(init_hardcode_coll) == 0)

    bvalid = np.array(bvalid, dtype=np.bool_)
    if np.sum(bvalid) < B:
        Logger.log('Planner already caused collision after init, removing from batch...')
        if np.sum(bvalid) == 0:
            Logger.log('No valid sequences left in batch! Skipping...')
            return True

    return False


def main():
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    node_name = socket.gethostname()

    cfg, cfg_dict = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)

    if getattr(cfg, "no_lane_relation_debug", False):
        os.environ["LANE_RELATION_DEBUG"] = "0"
    else:
        os.environ["LANE_RELATION_DEBUG"] = "1"

    # Logging: rank 0 -> process_log.txt; each rank -> process_log_rank{R}.txt
    log_path = os.path.join(cfg.out_dir, "process_log.txt")
    rank_log_path = os.path.join(cfg.out_dir, f"process_log_rank{global_rank}.txt")
    print(
        f"[rank {global_rank}/{world_size} local={local_rank} host={node_name}] "
        f"out_dir={cfg.out_dir}",
        flush=True,
    )

    if global_rank == 0:
        print(f"[rank 0] main log: {log_path}", flush=True)
        Logger.init(log_path)
        if _lane_relation_debug_enabled():
            lr_log = os.path.join(cfg.out_dir, "lane_relation_debug.log")
            set_lane_relation_log_path(lr_log)
            with open(lr_log, "w", encoding="utf-8") as f:
                f.write("# lane_relation debug log\n")
            Logger.log(f"Lane relation debug: ON -> {lr_log}")
        else:
            Logger.log(
                "Lane relation debug: OFF (--no_lane_relation_debug); "
                "no lane_relation_debug.log"
            )
        Logger.log("Args: " + str(cfg_dict))
    else:
        Logger.init(rank_log_path)
        Logger.log(
            f"rank {global_rank}/{world_size} local_rank={local_rank} host={node_name}"
        )
        if _lane_relation_debug_enabled():
            lr_log = os.path.join(cfg.out_dir, "lane_relation_debug.log")
            set_lane_relation_log_path(lr_log)

    if global_rank == 0 and not _lane_relation_debug_enabled():
        print(
            "[rank 0] lane_relation_debug disabled (--no_lane_relation_debug)",
            flush=True,
        )
    elif _lane_relation_debug_enabled():
        print(
            f"[rank {global_rank}] lane_relation_debug: "
            f"{os.path.join(cfg.out_dir, 'lane_relation_debug.log')}",
            flush=True,
        )

    # device setup
    device = get_device()
    Logger.log('Using device %s...' % (str(device)))

    data_path = os.path.join(cfg.data_dir, cfg.data_version)
    map_env = NuScenesMapEnv(
        data_path,
        bounds=cfg.map_obs_bounds,
        L=cfg.map_obs_size_pix,
        W=cfg.map_obs_size_pix,
        layers=cfg.map_layers,
        device=device,
        load_lanegraph=True,
        lanegraph_res_meters=1.0
    )

    nusc_obj = NuScenes(version=f"v1.0-{cfg.data_version}", dataroot=data_path, verbose=False)
    test_dataset = NuScenesDataset(
        data_path,
        map_env,
        version=cfg.data_version,
        split=cfg.split,
        categories=cfg.agent_types,
        npast=cfg.past_len,
        nfuture=cfg.future_len,
        seq_interval=cfg.seq_interval,
        nusc=nusc_obj,
        randomize_val=cfg.val_size in (200, 400),
        val_size=cfg.val_size,
        reduce_cats=cfg.reduce_cats
    )

    # create loaders   
    test_sampler = DistributedSampler(test_dataset, shuffle=cfg.shuffle) 
    test_loader = GraphDataLoader(test_dataset,
                            batch_size=1, # will collect batches on the fly after determining feasibility
                            sampler=test_sampler,
                            num_workers=cfg.num_workers,
                            pin_memory=False,
                            worker_init_fn=lambda _: np.random.seed()) # get around numpy RNG seed bug
    # create model
    ae = TrafficModel(cfg.past_len, cfg.future_len, cfg.map_obs_size_pix, len(test_dataset.categories),
                        map_feat_size=cfg.map_feat_size,
                        past_feat_size=cfg.past_feat_size,
                        future_feat_size=cfg.future_feat_size,
                        latent_size=cfg.latent_size,
                        output_bicycle=cfg.model_output_bicycle,
                        conv_channel_in=map_env.num_layers,
                        conv_kernel_list=cfg.conv_kernel_list,
                        conv_stride_list=cfg.conv_stride_list,
                        conv_filter_list=cfg.conv_filter_list
                        ).to(device)
    
    # load model weights
    if cfg.pretrained_ae_path is not None:
        ckpt_epoch, _ = load_state(cfg.pretrained_ae_path, ae, map_location=device)
        Logger.log('Loaded checkpoint from epoch %d...' % (ckpt_epoch))
    else:
        throw_err('Must pass in model weights to do scenario generation!')

    # so can unnormalize as needed
    ae.set_normalizer(test_dataset.get_state_normalizer())
    ae.set_att_normalizer(test_dataset.get_att_normalizer())
    if cfg.model_output_bicycle:
        from datasets.utils import NUSC_BIKE_PARAMS
        ae.set_bicycle_params(NUSC_BIKE_PARAMS)

    from diffuser import TemporalUnet, TrafficDiffusion
    unet = TemporalUnet(cfg.horizon, cfg.transition_dim, cfg.cond_dim,
                    dim=cfg.dim,
                    dim_mults=cfg.dim_mults,
                    attention=True)
    diffuser = TrafficDiffusion(copy.deepcopy(ae), unet, cfg.horizon, cfg.observation_dim, cfg.action_dim, n_timesteps=cfg.n_timesteps,
                                loss_type=cfg.loss_type, clip_denoised=False, predict_epsilon=cfg.predict_epsilon,
                                action_weight=1.0, loss_discount=1.0, loss_weights=None, use_ddim=True).to(device)

    if cfg.ckpt is not None:
        ckpt_epoch, _ = diffuser.load_state(cfg.ckpt, map_location=device)
        Logger.log('Loaded model checkpoint from epoch %d...' % (ckpt_epoch))
    else:
        throw_err('Must pass in model weights to evaluate a trained model!')

    model = DDP(diffuser, device_ids=[local_rank]).module

    model.eval()

    traffic_model = None
    traffic_model = copy.deepcopy(ae).to(device)
    traffic_model = DDP(traffic_model, device_ids=[local_rank]).module

    del ae

    pbar = tqdm.tqdm(test_loader)
    # for idx in range(start, end):
    for i, data in enumerate(pbar):
        scene_graph, map_idx = data
        scene_graph = scene_graph.to(device)
        map_idx = map_idx.to(device)
        if filter_scene(scene_graph, map_idx, map_env, i, traffic_model, test_loader, cfg):
            continue
        ego_future, sur_future, pred_ego_future, pred_sur_future = gen_future(model, scene_graph, map_idx, map_env, device)
        visualize_sample(
            ego_future_traj=ego_future,
            sur_future_traj=sur_future,
            pred_ego_future_traj=pred_ego_future,
            pred_sur_future_traj=pred_sur_future,
            scene_graph=scene_graph,
            map_idx=map_idx,
            map_env=map_env,
            state_norm=test_dataset.get_state_normalizer(),
            att_norm=test_dataset.get_att_normalizer(),
            nusc_obj=nusc_obj,
            sample_idx=i,
            out_dir=cfg.out_dir,
            map_size=cfg.map_size,
            margin_m=cfg.margin_m,
            min_bound_m=cfg.min_bound_m,
            split=cfg.split,
            use_rank_id=cfg.use_rank_id,
            text_use_global_coords=cfg.coords,
            local_rank=local_rank,
        )


if __name__ == "__main__":
    main()
