# Smart Traffic Signal Control System

Computer vision + reinforcement learning traffic signal control project.

## Runtime Modes

The project now supports two inference backends:

1. `dqn` (default)
2. `qtable` (fallback/reference)

Default tuned DQN model path:

- `rl/dqn_model_tuned_best.pt`

## Quick Start

Run the full pipeline in mock mode (no camera/serial hardware):

```bash
python main.py --mock
```

Run full pipeline with explicit DQN model:

```bash
python main.py --model-type dqn --model rl/dqn_model_tuned_best.pt --mock
```

Run controller module in mock mode with DQN:

```bash
python integration/controller.py --model-type dqn --model rl/dqn_model_tuned_best.pt --mock
```

Run with Q-table fallback:

```bash
python main.py --model-type qtable --model rl/q_table_best.pkl --mock
```

## Environment Variables

You can configure runtime behavior using these environment variables:

- `SMART_TRAFFIC_MODEL_TYPE` -> `dqn` or `qtable` (default: `dqn`)
- `SMART_TRAFFIC_DQN_MODEL_PATH` -> path to DQN model (default: `rl/dqn_model_tuned_best.pt`)
- `SMART_TRAFFIC_MODEL_PATH` -> path to Q-table model (default: `rl/q_table.pkl`)
- `SMART_TRAFFIC_SERIAL_PORT` -> serial port (default: `COM5`)
- `SMART_TRAFFIC_SERIAL_BAUD` -> serial baudrate (default: `115200`)
- `SMART_TRAFFIC_SERIAL_TIMEOUT` -> serial timeout in seconds (default: `1.0`)
- `SMART_TRAFFIC_LOG_LEVEL` -> log level (default: `INFO`)

## Benchmark / Guardrail

Run policy benchmark manually:

```bash
python rl/evaluate.py --model rl/dqn_model_tuned_best.pt --model-type dqn --episodes 80 --steps 120 --arrival-min 0 --arrival-max 1 --min-green-steps 3 --yellow-steps 0 --output-csv rl/benchmark_dqn_tuned_best.csv
```

Run test suite (includes DQN performance guardrail test):

```bash
pytest -q
```

## Training Recipes

### 1) Tabular RL Ablation

Run environment/control ablation for Q-table training, then retrain top candidates:

```bash
python rl/ablation_sweep.py --short-episodes 900 --long-episodes 3500 --eval-episodes-short 30 --eval-episodes-long 80 --eval-steps 120 --top-k 3 --sweep-csv rl/ablation_sweep_results.csv --final-csv rl/ablation_final_results.csv --best-model-output rl/q_table_best_ablation.pkl --models-dir rl/ablation_models
```

Benchmark the selected tabular model:

```bash
python rl/evaluate.py --model rl/q_table_best_ablation.pkl --episodes 80 --steps 120 --arrival-min 0 --arrival-max 1 --min-green-steps 3 --yellow-steps 0 --output-csv rl/benchmark_best_ablation_model.csv
```

### 2) Focused DQN Tuning

Run focused hyperparameter tuning and 3-seed final validation:

```bash
python rl/tune_dqn.py --sweep-episodes 600 --final-episodes 1200 --steps 120 --eval-episodes 80 --arrival-min 0 --arrival-max 1 --service-capacity 5 --min-green-steps 3 --yellow-steps 0 --base-seed 42 --sweep-csv rl/dqn_tuning_sweep.csv --final-csv rl/dqn_tuning_final.csv --best-model-output rl/dqn_model_tuned_best.pt --models-dir rl/dqn_tuned_models
```

### 3) Multi-Seed DQN Evaluation

Train three seeds manually:

```bash
python rl/train_dqn.py --episodes 1200 --steps 120 --arrival-min 0 --arrival-max 1 --service-capacity 5 --min-green-steps 3 --yellow-steps 0 --seed 42 --output rl/dqn_model_seed_42.pt
python rl/train_dqn.py --episodes 1200 --steps 120 --arrival-min 0 --arrival-max 1 --service-capacity 5 --min-green-steps 3 --yellow-steps 0 --seed 43 --output rl/dqn_model_seed_43.pt
python rl/train_dqn.py --episodes 1200 --steps 120 --arrival-min 0 --arrival-max 1 --service-capacity 5 --min-green-steps 3 --yellow-steps 0 --seed 44 --output rl/dqn_model_seed_44.pt
```

Evaluate each seed:

```bash
python rl/evaluate.py --model rl/dqn_model_seed_42.pt --model-type dqn --episodes 80 --steps 120 --arrival-min 0 --arrival-max 1 --min-green-steps 3 --yellow-steps 0 --output-csv rl/benchmark_dqn_seed_42.csv
python rl/evaluate.py --model rl/dqn_model_seed_43.pt --model-type dqn --episodes 80 --steps 120 --arrival-min 0 --arrival-max 1 --min-green-steps 3 --yellow-steps 0 --output-csv rl/benchmark_dqn_seed_43.csv
python rl/evaluate.py --model rl/dqn_model_seed_44.pt --model-type dqn --episodes 80 --steps 120 --arrival-min 0 --arrival-max 1 --min-green-steps 3 --yellow-steps 0 --output-csv rl/benchmark_dqn_seed_44.csv
```
