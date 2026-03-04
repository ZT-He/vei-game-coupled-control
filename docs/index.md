---
title: "VEI Stackelberg Game + Hierarchical Planning"
---

# Stackelberg Game Modeling and Decision-Making for Vehicle–E-Scooter Interaction (VEI)

This project demonstrates a reproducible **vehicle–e-scooter interaction (VEI)** simulation framework built around a **discrete-time Stackelberg game** and a set of collision-avoidance planners spanning:
- **Rule-based vs optimization-based**
- **Longitudinal-only (1D) vs coupled longitudinal–lateral (2D)**

> Paper: *Stackelberg Game Modeling and Decision-Making for Vehicle-E-Scooter Interaction* (ITSC 2026 submission)

---

## Highlights
- **Game-based VEI**: ego vehicle as leader, e-scooter as type-conditioned follower (go / yield).
- **Bounded perception**: detection gate to mimic late/partial awareness.
- **Four planners**: Rule1D, Rule2D, MPC1D, MPC2D.
- **MPC2D**: hierarchical “contract selection + contract-constrained execution” with safety overrides.

---

## Framework Overview

![VEI framework pipeline (Fig. 3)](assets/figures/fig3_framework.png)

The workflow follows a three-stage loop: scenario identification → game-based simulation modeling → quantitative + qualitative evaluation.

---

## Interaction Model (Stackelberg)

At each step, the ego selects an action; the rider responds via a one-step best response under a latent rider type (e.g., Aggressive vs Conservative/Normal). The leader can be belief-aware, and replans in receding horizon.

---

## Planners

### Rule-based baselines
- **Rule1D**: finite-state logic (cruise / brake) using predicted TTC as a safety gate.
- **Rule2D**: adds a lightweight minimum-risk lateral reference selector before committing to braking.

### Optimization-based planners
- **MPC1D**: belief-weighted Stackelberg evaluation over candidate accelerations (steering fixed).
- **MPC2D (proposed)**: two-layer hierarchy:
  1) **Strategic contract selection** (KEEP / YIELD / DETOUR_LEFT / DETOUR_RIGHT)
  2) **Tactical execution** under hard corridor + speed-envelope constraints  
  3) **Safety overrides** if execution becomes infeasible

---

## Scenarios

### 1) Intersection crossing
![Intersection layout](assets/figures/fig2a_intersection.png)

### 2) Straight-road lane change (cut-in)
![Straight-road layout](assets/figures/fig2b_straightroad.png)

---

## Results (summary)

| Scenario | Controller | Collision rate (%) | Safety score |
|---|---:|---:|---:|
| Intersection | Rule1D | 8.6 | 0.914 |
|  | Rule2D | 5.6 | 0.944 |
|  | MPC1D | 2.4 | 0.976 |
|  | **MPC2D** | **2.0** | **0.980** |
| Straight road | Rule1D | 3.2 | 0.967 |
|  | Rule2D | 1.2 | 0.984 |
|  | **MPC1D** | **0.0** | **1.000** |
|  | **MPC2D** | **0.0** | **1.000** |

---

## Simulation Videos (placeholders)

### Intersection — controller comparison
<video controls width="100%">
  <source src="assets/videos/intersection_placeholder.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Straight-road — controller comparison
<video controls width="100%">
  <source src="assets/videos/straightroad_placeholder.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

> **Note:** Replace the placeholder MP4 files in `assets/videos/`.

---

## Qualitative Example Figure (placeholder)

![Qualitative comparison (Fig. 6)](assets/figures/fig6_qualitative_placeholder.png)

---

## Citation

If you use this framework, please cite the paper:

```bibtex
@inproceedings{He2026VEIStackelberg,
  title={Stackelberg Game Modeling and Decision-Making for Vehicle-E-Scooter Interaction},
  author={...},
  booktitle={IEEE International Conference on Intelligent Transportation Systems (ITSC)},
  year={2026}
}
