---
title: "VEI Stackelberg Game + Hierarchical Planning"
---

# Stackelberg Game Modeling and Decision-Making for Vehicle–E-Scooter Interaction (VEI)

This project demonstrates a reproducible **vehicle–e-scooter interaction (VEI)** simulation framework built around a **discrete-time Stackelberg game** and a set of collision-avoidance planners spanning:
- **Rule-based vs Optimization-based**
- **Longitudinal-only (1D) vs Coupled longitudinal–lateral (2D)**

> Paper: *Stackelberg Game Modeling and Decision-Making for Vehicle-E-Scooter Interaction* (in Proc. 2026 International Conference on Intelligent Transportation Systems (ITSC 2026), accepted, Naples, Italy, September 2026.)

---

## Highlights
- **Game-based VEI**: ego vehicle as leader, e-scooter as type-conditioned follower.
- **Bounded perception**: detection gate to mimic late/partial awareness.
- **Four planners**: Rule1D, Rule2D, MPC1D, MPC2D.
- **MPC2D**: hierarchical “contract selection + contract-constrained execution” with safety overrides.

---

## Framework Overview

![VEI framework pipeline (Fig. 3)](assets/figures/VEI-game_framework.png)

The workflow follows a three-stage loop: scenario identification → game-based simulation modeling → quantitative + qualitative evaluation.

---

## Interaction Model (Stackelberg)

At each step, the ego selects an action; the rider responds via a one-step best response under a latent rider type (e.g., Aggressive vs Normal). The leader can be belief-aware, and replans in a receding horizon.

---

## Planners

### Rule-based baselines
- **Rule1D**: finite-state logic (cruise/brake) using predicted TTC as a safety gate.
- **Rule2D**: adds a lightweight minimum-risk lateral reference selector before committing to braking.

### Optimization-based planners
- **MPC1D**: belief-weighted Stackelberg evaluation over candidate accelerations.
- **MPC2D (proposed)**: two-layer hierarchy:
    1. **Strategic contract selection** (KEEP / YIELD / DETOUR\_LEFT / DETOUR\_RIGHT)
    2. **Tactical execution** under hard corridor + speed-envelope constraints
    3. **Safety overrides** if execution becomes infeasible

---

## Scenarios

### 1) E-scooter intersection crossing
![Intersection layout](assets/figures/VEI-game_configuration_a.png)

### 2) E-scooter straight-road lane change (cut-in)
![Straight-road layout](assets/figures/VEI-game_configuration_b.png)

---

## Results (summary)

| Scenario | Controller | Collision rate (%) | Safety score |
|---|---:|---:|---:|
| Intersection | Rule1D | 11.6 | 0.884 |
|  | Rule2D | 5.8 | 0.942 |
|  | MPC1D | 3.2 | 0.968 |
|  | **MPC2D** | **0.4** | **0.996** |
| Straight road | Rule1D | 10.4 | 0.872 |
|  | Rule2D | 4.8 | 0.909 |
|  | MPC1D | 2.4 | 0.950 |
|  | **MPC2D** | **1.2** | **0.954** |


---


## Qualitative Example Figure (placeholder)

![Qualitative comparison (Fig. 6)](assets/figures/inter_qual_scrns.png)

---

## Simulation Videos (placeholders)

### Intersection — controller comparison
<video controls width="100%">
  <source src="assets/videos/rule1d_simulation_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<video controls width="100%">
  <source src="assets/videos/mpc1d_simulation_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<video controls width="100%">
  <source src="assets/videos/rule2d_simulation_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<video controls width="100%">
  <source src="assets/videos/mpc2d_simulation_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Straight-road — controller comparison
<video controls width="100%">
  <source src="assets/videos/straightroad_placeholder.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

> **Note:** Replace the placeholder MP4 files in `assets/videos/`.


---
