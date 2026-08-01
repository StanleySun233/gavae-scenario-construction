# GAVAE Scenario Construction

Official code for the main GAVAE experiment and scenario construction workflow for autonomous ship digital testing.

## Demo

<video src="https://github.com/user-attachments/assets/20da4ba8-c5a5-4978-945a-78606d851b3c" controls width="100%"></video>

## Data Layout

Training data is stored under `data`:

- `data/route1/train.npy`
- `data/route2/train.npy`

Experiment outputs are written under `data/runs`:

- `data/runs/train`
- `data/runs/test`
- `data/runs/generate`
- `data/runs/scenario`

Core parameters are in `config.py`. The command line only selects the action and route.

## 1. Train

```bash
python inference.py train --route 1
python inference.py train --route 2
```

Each route writes its checkpoint, generated arrays, denoised arrays, metrics, and summary to `data/runs/train/route1` or `data/runs/train/route2`.

## 2. Test

```bash
python inference.py test --route 1
python inference.py test --route 2
```

The test step evaluates the denoised training output for each route and writes metrics to `data/runs/test`.

## 3. Generate

```bash
python inference.py generate --route 1
python inference.py generate --route 2
python scenario.py
```

The generate step loads the route checkpoint from `data/runs/train` and writes generated and denoised trajectory arrays to `data/runs/generate`. `scenario.py` reads the denoised `.npy` files, then writes scenario summaries and pre-encounter, encounter, and post-encounter clips to `data/runs/scenario`. It also writes 50 selected images for each encounter type to `scenarios/crossing`, `scenarios/headon`, and `scenarios/overtaking`.
