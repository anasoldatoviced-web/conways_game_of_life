# Conway's Game of Life

A Python implementation of Conway's Game of Life for the TAOCP2 Final Project.
PSL University — Bachelor in AI — Semester 2, 2026.

## Requirements

Python 3.9+ and the following packages:

```bash
pip install numpy scipy matplotlib
```

Tkinter is included with most Python distributions. If it is missing:

```bash
# Ubuntu / Debian
sudo apt-get install python3-tk

# macOS (Homebrew)
brew install python-tk
```

## How to run

### Interactive simulation (main program)

```bash
python main.py
```

### Run the benchmark (generates `benchmark_results.png`)

```bash
python benchmark.py
```

### Run the test suite

```bash
python tests.py
```

---

## Controls

| Input | Action |
|---|---|
| Left-click / drag | Paint live cells |
| Right-click / drag | Erase cells |
| Space | Play / Pause |
| → (Right arrow) | Step one generation (while paused) |
| C | Clear the grid |
| F | Fit view to current pattern |
| Scroll wheel | Zoom in / out |

---

## Project structure

```
conways_game_of_life/
│
├── main.py            # Entry point — Tkinter GUI (MVC architecture)
├── engine_sparse.py   # Sparse-set engine  O(k) per generation
├── engine_dense.py    # Dense NumPy engine O(W×H) per generation
├── file_io.py         # Load/save .cells, .rle, .txt patterns
├── benchmark.py       # Benchmark comparing sparse vs dense engines
├── tests.py           # Unit tests for engines, file I/O, and rules
│
└── patterns/          # Example starting configurations (.cells format)
    ├── glider.cells           — Classic diagonal spaceship
    ├── glider_gun.cells       — Gosper glider gun (period 30)
    ├── pulsar.cells           — Period-3 oscillator
    ├── pentadecathlon.cells   — Period-15 oscillator
    ├── lwss.cells             — Lightweight spaceship
    ├── blinker.cells          — Simplest period-2 oscillator
    ├── toad.cells             — Period-2 oscillator
    ├── block.cells            — Most common still life
    ├── beehive.cells          — Second most common still life
    └── r_pentomino.cells      — Methuselah (~1100 gens to stabilise)
```

---

## Reproducing a benchmark result

To reproduce the density-crossover experiment (Test 1 in the report):

```bash
python benchmark.py
```

This runs four benchmark tests and saves `benchmark_results.png` in the
current directory. Expected runtime: ~1–2 minutes.

---

## Loading a specific pattern from the command line

```python
from file_io import load_pattern
from engine_sparse import SparseEngine

cells = load_pattern('patterns/glider.cells')
engine = SparseEngine(cells)
for _ in range(10):
    engine.step()
print(f'Population after 10 steps: {engine.population()}')
```

---

## Architecture (MVC)

The program follows a Model–View–Controller separation:

- **Model** — `SparseEngine` stores the set of live cells and applies B3/S23 rules.
- **View** — `Renderer` draws cells onto the Tkinter canvas (dirty-rectangle optimisation).
- **Controller** — `GameController` handles user input, drives the animation loop via `after()`.

The `DenseEngine` is used only for the benchmark; the GUI always runs the sparse engine.

---

## Data structure discussion

| Representation | Memory | Cost per generation | Best for |
|---|---|---|---|
| Dense 2D array (NumPy) | O(W × H) | O(W × H) | Small dense grids, fixed size |
| Sparse set (Python `set`) | O(k) | O(k) | Large grids, sparse patterns |
| Hashlife (quad-tree) | O(unique subgrids) | Sub-linear (exponential skip) | Periodic / explosive long runs |

where k = number of live cells, W × H = grid dimensions.

The sparse engine uses a `defaultdict(int)` neighbour-counting trick:
for each live cell, increment a counter for each of its 8 neighbours.
The next generation is the set of cells with count = 3 (birth) plus live
cells with count = 2 or 3 (survival). This is O(k) per generation.

The dense engine uses `scipy.signal.convolve2d` with a 3×3 kernel of ones
to count neighbours in a single vectorised C-level operation.

Hashlife is discussed conceptually in the report but not fully implemented,
as a correct implementation requires canonical quad-tree memoisation and
is beyond the scope of a 6-week project.
