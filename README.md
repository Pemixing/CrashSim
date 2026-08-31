# Human Behavior-Informed Crash Scenario Generation with Real-World Crash Priors for Autonomous Vehicle Safety Evaluation

CrashSim is a framework for safety-critical autonomous driving scenarios. It connects **nuScenes** real-world driving data, **LLM/VLM multi-agent reasoning**, **real-world crash-prior RAG retrieval**, and **planner rollout evaluation** to synthesize, build, and evaluate collision-oriented safety-critical scenarios (the **nuScenes-Crash** dataset).

## Key Features

- **Crash database construction**: Builds a hierarchical vector retrieval database (crash profiles, mode / variant / sample vector stores) from SHRP2 NDS reconstructed trajectory data, providing real-world crash priors for trajectory generation.
- **Scene understanding and adversarial vehicle identification**: Uses a VLM on BEV images and structured scene text to identify potential adversarial vehicles and behavior patterns.
- **Adversarial trajectory generation**: Combines a real-world crash knowledge base with RAG to produce anchor guidance for downstream modules such as diffusion models.
- **nuScenes-Crash dataset construction**: Converts adversarial scenario generation outputs back into the standard nuScenes v1.0 metadata format.
- **Planner rollout and interpretable evaluation**: Rolls out ego planners on nuScenes-Crash (surrounding vehicles follow future trajectories recorded in the dataset) and outputs deterministic evidence plus optional LLM diagnostic reports.

## `out/` Directory

| Path | Description |
|------|-------------|
| `out/processing_nusc/` | Scene-graph processing outputs from `process_nusc_scene_graph.py`; each sample includes a BEV visualization image (`.png`) and structured scene text (`.txt`) |
| `out/crash_rag/` | Crash RAG database built from SHRP2 NDS reconstructed trajectories, including `crash_profiles.json` and three-layer vector stores (mode / variant / sample) |
| `out/models/` | Pretrained model weights: `traffic_model.pth` is a traffic-prior VAE from [STRIVE](https://github.com/nv-tlabs/STRIVE); `diffusion_model.pth` is a trained diffusion model for safety-critical scenario generation |
| `out/nusc_crash_val/` | nuScenes-Crash dataset (standard nuScenes v1.0 metadata format and maps) |

## Data Flow Overview

```
nuScenes raw data
    │
    ▼
process_nusc_scene_graph.py  →  BEV images + scene text
    │
    ▼
vlm_agent (SceneAgent)       →  adversarial vehicle identification + scene understanding
    │
    ▼
llm_agent (TrajectoryAgent)  →  adversarial anchor trajectories (RAG-enhanced)
    │
    ▼
adv_gen / scenario generation  →  scene_*.json
    │
    ▼
build_nusc_crash.py          →  nuScenes-Crash metadata
    │
    ▼
crashsim_planner.py          →  planner rollout + interpretable evaluation
```

## Related Repositories

We acknowledge all the open-source contributors for the following projects to make this work possible:

1. [STRIVE](https://github.com/nv-tlabs/STRIVE) — Generating Useful Accident-Prone Driving Scenarios via a Learned Traffic Prior (CVPR 2022)
2. [BirdsEyeTrajectoryReconstructionSHRP2NDS](https://github.com/Yiru-Jiao/BirdsEyeTrajectoryReconstructionSHRP2NDS) — Bird's eye view trajectory reconstruction of naturalistic crashes and near-crashes in the SHRP2 NDS

## Citations

Please cite our paper if you find this repository useful:

```bibtex
@article{peng2026,
  title={Human Behavior-Informed Crash Scenario Generation with Real-World Crash Priors for Autonomous Vehicle Safety Evaluation},
  author={Peng, Mingxing et al.},
  journal={arXiv preprint arXiv:xxx},
  year={2026}
}
```
