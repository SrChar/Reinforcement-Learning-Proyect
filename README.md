# Museum Heist - Reinforcement Learning

Museum Heist is a stochastic-policy reinforcement learning project set in a gridworld museum. A surveillance agent learns a tabular Softmax policy with REINFORCE to decide which room to monitor, while a thief plans shortest paths to steal a painting and escape while avoiding recently watched rooms.

## Project Structure

- `museum_heist.py`: environment, topologies, thief behavior, policy-gradient agent, training loop, evaluation, model persistence, and visualization.
- `variables.py`: main configuration file for mode, topology, training settings, rendering, rewards, and model paths.
- `models/`: trained models saved as JSON files.
- `archivos/`: auxiliary course/project resources.

## Requirements

The project uses only the Python standard library.

Recommended version:

```powershell
python --version
```

Python 3.10 or newer is recommended.

## Usage

Edit `variables.py`, then run:

```powershell
python museum_heist.py
```

The project is intentionally configured through `variables.py` instead of command-line arguments.

Available modes:

- `train`: train a new model and optionally save it.
- `play`: load a trained model and show episodes.
- `evaluate`: evaluate a trained model without updating it.
- `topologies`: list available museum layouts.
- `models`: list saved model files.

Available topologies:

- `open4x4`
- `gallery5x5`
- `maze6x6`

## Model

The guard learns a probability distribution over museum rooms:

```text
pi(a) = exp(theta(a) / tau) / sum_b exp(theta(b) / tau)
```

Training uses REINFORCE. Evaluation reports capture rate, detection rate, escape rate, average episode length, and average return.

## Visualization

When `RENDER=True`, the simulation displays the museum grid, thief position, active camera, painting room, start room, and learned room-selection heatmap.

## Saved Models

Training saves JSON models in `models/` when `SAVE_MODEL=True`. If `SAVE_PATH=None`, the filename is generated from the timestamp, topology, `TAU`, and `BETA`.

Example:

```text
models/20260619_090950_maze6x6_tau0.6_beta4.0.json
```

## Validation

Run a syntax check before submitting:

```powershell
python -m py_compile variables.py museum_heist.py
```
