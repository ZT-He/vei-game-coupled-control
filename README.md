# VEI Game-Coupled Control

Game-theoretic coupled control for Vehicle–VRU (Vulnerable Road User) interaction.

A Stackelberg leader-follower framework where the ego vehicle acts as leader and the VRU (pedestrian, e-scooter, motorcyclist) acts as follower. Four ego controllers are provided — two rule-based and two MPC variants — evaluated across intersection and straight-road cut-in scenarios.

## Project overview

```
interaction/        Game state, dynamics, Stackelberg solver, payoff, belief, oracle
                    controllers/  rule1d · rule2d · mpc1d · mpc2d
motion/             Bicycle model (vehicle), point-mass Newton (VRU), social force
perception/         FOV detection, V2X awareness (Markov link quality), observation model
traffic_env/        Road geometry and lane definitions
evaluation/         Collision detection (OBB), TTC/safety/efficiency/comfort metrics
qual_sim/           Qualitative single-scenario simulation with Matplotlib rendering
quant_sim/          Batch quantitative evaluation → Excel summary / details / collisions
```

## Installation

Python 3.9+ recommended.

```bash
pip install -r requirements.txt
```

## Running scenarios

All scripts must be run from the **repo root** so that relative package imports resolve correctly.

### Qualitative simulation (single scenario, visual)

```bash
# Intersection — default controller (mpc2d), aggressive VRU, no V2X
python -m qual_sim.sim_vei_intersection --ego mpc2d --rider Aggressive --render

# Straight-road cut-in — rule-based 1D controller, conservative VRU
python -m qual_sim.sim_vei_straight_road --ego rule1d --rider Conservative --render
```

Available `--ego` values: `rule1d`, `rule2d`, `mpc1d`, `mpc2d`  
Available `--rider` values: `Aggressive`, `Conservative`  
Available `--profile` values (V2X link quality): `no`, `bad`, `good`, `perfect`

### Quantitative batch evaluation (all controllers, Excel output)

```bash
# Intersection — sweep ego speed 10–20 m/s, both rider types
python -m quant_sim.test_quant_intersection --speed-min 10 --speed-max 20 --speed-count 10

# Straight-road cut-in
python -m quant_sim.test_quant_straight_road --speed-min 10 --speed-max 20 --speed-count 10
```

Results are saved under `quant_sim/quant_sim_results/<timestamp>/` as three Excel files:
`*_summary_*.xlsx`, `*_details_*.xlsx`, `*_collisions_*.xlsx`.

## Controllers

| Key      | Description |
|----------|-------------|
| `rule1d` | 1-D FSM: detect → predict → brake (longitudinal only) |
| `rule2d` | 2-D FSM: adds occupancy-map lane-change planning |
| `mpc1d`  | 1-D Stackelberg MPC: enumerates constant accelerations, leader-follower game |
| `mpc2d`  | Hierarchical 2-D MPC: strategic corridor selection + tactical trajectory optimisation |

## GitHub Pages demo

The project website is served from [`docs/`](docs/index.md).
