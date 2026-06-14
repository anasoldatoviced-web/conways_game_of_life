"""
benchmark.py — Empirical comparison of the sparse-set and dense-array engines.

Test cases chosen to stress the representations differently
(following the professor's review):

  1. Small dense pattern  (50×50 random soup, 30% density)
       → dense array should win (low overhead, vectorised)
  2. Large sparse pattern (single glider on a 1000×1000 grid)
       → sparse set should win (only 5 cells ever touched)
  3. Explosive long-running pattern (R-pentomino, ~1100 generations)
       → fair test: population grows, then stabilises
  4. Periodic oscillator (pulsar, period 3)
       → Hashlife would win; we show sparse vs dense

Metrics: generations per second, peak memory (MB), time to stabilise.
"""

import time
import tracemalloc
import random

import matplotlib
matplotlib.use('Agg')          # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from engine_sparse import SparseEngine
from engine_dense  import DenseEngine


# ======================================================================
# Helper: run N generations and return elapsed time + peak memory
# ======================================================================

def _bench_sparse(live_cells, n_gen):
    engine = SparseEngine(live_cells)
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(n_gen):
        engine.step()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1e6   # seconds, MB


def _bench_dense(live_cells, grid_w, grid_h, n_gen):
    engine = DenseEngine(width=grid_w, height=grid_h)
    engine.set_from_sparse(live_cells)
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(n_gen):
        engine.step()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1e6


# ======================================================================
# Pattern builders
# ======================================================================

def _random_soup(width, height, density, seed=42):
    """Random soup of approximately (density * width * height) live cells."""
    rng = random.Random(seed)
    return {(x, y) for x in range(width) for y in range(height)
            if rng.random() < density}


def _glider(offset_x=0, offset_y=0):
    """Classic glider (5 cells)."""
    return {(1+offset_x, 0+offset_y), (2+offset_x, 1+offset_y),
            (0+offset_x, 2+offset_y), (1+offset_x, 2+offset_y),
            (2+offset_x, 2+offset_y)}


def _r_pentomino(offset_x=500, offset_y=500):
    """R-pentomino: small seed that explodes for ~1100 generations."""
    return {(1+offset_x, 0+offset_y), (2+offset_x, 0+offset_y),
            (0+offset_x, 1+offset_y), (1+offset_x, 1+offset_y),
            (1+offset_x, 2+offset_y)}


def _pulsar(offset_x=0, offset_y=0):
    """Pulsar: period-3 oscillator (48 cells)."""
    coords = [
        (2,0),(3,0),(4,0),(8,0),(9,0),(10,0),
        (0,2),(5,2),(7,2),(12,2),
        (0,3),(5,3),(7,3),(12,3),
        (0,4),(5,4),(7,4),(12,4),
        (2,5),(3,5),(4,5),(8,5),(9,5),(10,5),
        (2,7),(3,7),(4,7),(8,7),(9,7),(10,7),
        (0,8),(5,8),(7,8),(12,8),
        (0,9),(5,9),(7,9),(12,9),
        (0,10),(5,10),(7,10),(12,10),
        (2,12),(3,12),(4,12),(8,12),(9,12),(10,12),
    ]
    return {(x + offset_x, y + offset_y) for x, y in coords}


# ======================================================================
# Individual benchmark tests
# ======================================================================

def bench_density_crossover(n_gen=50, sizes=(50, 100, 200)):
    """
    Test 1: at what density does sparse stop being faster than dense?
    Run both engines over a range of densities for a fixed grid size.
    Returns a dict of results for plotting.
    """
    print('\n=== Test 1: Density crossover ===')
    densities = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
    grid_size = 100
    results = {'densities': densities, 'sparse_times': [], 'dense_times': []}

    for d in densities:
        soup = _random_soup(grid_size, grid_size, d)
        st, _ = _bench_sparse(soup, n_gen)
        dt, _ = _bench_dense(soup, grid_size, grid_size, n_gen)
        results['sparse_times'].append(st)
        results['dense_times'].append(dt)
        print(f'  density={d:.0%}  k={len(soup):5d}  '
              f'sparse={st:.3f}s  dense={dt:.3f}s  '
              f'winner={"sparse" if st < dt else "dense "}')

    return results


def bench_sparse_pattern(n_gen=200):
    """
    Test 2: large grid, tiny pattern — sparse should dominate.
    """
    print('\n=== Test 2: Sparse pattern (glider on 1000×1000) ===')
    glider = _glider(offset_x=500, offset_y=500)
    grid_w, grid_h = 1000, 1000

    st, sm = _bench_sparse(glider, n_gen)
    dt, dm = _bench_dense(glider, grid_w, grid_h, n_gen)

    print(f'  sparse: {st:.3f}s  {sm:.2f}MB')
    print(f'  dense:  {dt:.3f}s  {dm:.2f}MB')
    return {'sparse': (st, sm), 'dense': (dt, dm)}


def bench_explosive(n_gen=500):
    """
    Test 3: R-pentomino on a large grid — population grows significantly.
    Track population over time for both engines.
    """
    print('\n=== Test 3: Explosive pattern (R-pentomino, 500 generations) ===')
    seed = _r_pentomino(500, 500)
    grid_w, grid_h = 1100, 1100

    # Sparse: track population
    sp_engine = SparseEngine(seed)
    sp_pops = []
    t0 = time.perf_counter()
    for _ in range(n_gen):
        sp_engine.step()
        sp_pops.append(sp_engine.population())
    sp_time = time.perf_counter() - t0

    # Dense: track population
    dn_engine = DenseEngine(width=grid_w, height=grid_h)
    dn_engine.set_from_sparse(seed)
    dn_pops = []
    t0 = time.perf_counter()
    for _ in range(n_gen):
        dn_engine.step()
        dn_pops.append(dn_engine.population())
    dn_time = time.perf_counter() - t0

    print(f'  sparse: {sp_time:.3f}s  final pop={sp_pops[-1]}')
    print(f'  dense:  {dn_time:.3f}s  final pop={dn_pops[-1]}')
    return {'sparse_pops': sp_pops, 'dense_pops': dn_pops,
            'sparse_time': sp_time, 'dense_time': dn_time}


def bench_oscillator(n_gen=300):
    """
    Test 4: periodic oscillator (pulsar) — both engines, compare speed.
    """
    print('\n=== Test 4: Oscillator (pulsar, period 3) ===')
    pulsar = _pulsar(50, 50)
    grid_w, grid_h = 200, 200

    st, sm = _bench_sparse(pulsar, n_gen)
    dt, dm = _bench_dense(pulsar, grid_w, grid_h, n_gen)

    print(f'  sparse: {st:.3f}s  {sm:.2f}MB')
    print(f'  dense:  {dt:.3f}s  {dm:.2f}MB')
    return {'sparse': (st, sm), 'dense': (dt, dm)}


# ======================================================================
# Plot all results
# ======================================================================

def plot_results(crossover, sparse_test, explosive, oscillator, out_path='benchmark_results.png'):
    """Produce a 2×2 figure summarising all four benchmark tests."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Conway\'s Game of Life — Sparse vs Dense Engine Benchmark',
                 fontsize=14, fontweight='bold')

    SPARSE_COLOR = '#2196F3'   # blue
    DENSE_COLOR  = '#F44336'   # red

    # --- Plot 1: Density crossover ---
    ax = axes[0, 0]
    dens = [d * 100 for d in crossover['densities']]
    ax.plot(dens, crossover['sparse_times'], 'o-', color=SPARSE_COLOR,
            label='Sparse set', linewidth=2)
    ax.plot(dens, crossover['dense_times'],  's-', color=DENSE_COLOR,
            label='Dense array', linewidth=2)
    ax.set_xlabel('Grid density (%)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Test 1: Time vs density (100×100, 50 generations)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Sparse-friendly pattern ---
    ax = axes[0, 1]
    labels  = ['Sparse set', 'Dense array']
    times   = [sparse_test['sparse'][0], sparse_test['dense'][0]]
    memories= [sparse_test['sparse'][1], sparse_test['dense'][1]]
    x = np.arange(2)
    bars = ax.bar(x, times, color=[SPARSE_COLOR, DENSE_COLOR], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Time (s)')
    ax.set_title('Test 2: Glider on 1000×1000 (200 generations)')
    for bar, mem in zip(bars, memories):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{mem:.1f} MB', ha='center', va='bottom', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Plot 3: Population over time (R-pentomino) ---
    ax = axes[1, 0]
    gens = list(range(1, len(explosive['sparse_pops']) + 1))
    ax.plot(gens, explosive['sparse_pops'], color=SPARSE_COLOR,
            label=f'Sparse ({explosive["sparse_time"]:.2f}s)', linewidth=1.5)
    ax.plot(gens, explosive['dense_pops'],  color=DENSE_COLOR,
            label=f'Dense  ({explosive["dense_time"]:.2f}s)',  linewidth=1.5, linestyle='--')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Live cells')
    ax.set_title('Test 3: R-pentomino population growth (500 generations)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Oscillator ---
    ax = axes[1, 1]
    categories = ['Sparse set\n(time)', 'Dense array\n(time)',
                  'Sparse set\n(memory)', 'Dense array\n(memory)']
    vals_t = [oscillator['sparse'][0], oscillator['dense'][0]]
    vals_m = [oscillator['sparse'][1], oscillator['dense'][1]]

    ax2 = ax.twinx()
    b1 = ax.bar([0, 1], vals_t,  color=[SPARSE_COLOR, DENSE_COLOR], alpha=0.7, width=0.35)
    b2 = ax2.bar([0.4, 1.4], vals_m, color=[SPARSE_COLOR, DENSE_COLOR], alpha=0.4, width=0.35, hatch='//')
    ax.set_xticks([0.2, 1.2])
    ax.set_xticklabels(['Sparse', 'Dense'])
    ax.set_ylabel('Time (s)', color='black')
    ax2.set_ylabel('Peak memory (MB)', color='grey')
    ax.set_title('Test 4: Pulsar oscillator (300 generations)')

    solid_patch = mpatches.Patch(color='steelblue', alpha=0.85, label='Time (s)')
    hatched_patch = mpatches.Patch(facecolor='steelblue', alpha=0.4, hatch='//', label='Memory (MB)')
    ax.legend(handles=[solid_patch, hatched_patch], fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\nBenchmark figure saved to: {out_path}')
    return out_path


# ======================================================================
# Entry point
# ======================================================================

def run_all(out_path='benchmark_results.png'):
    """Run all benchmark tests and produce the summary figure."""
    print('Running benchmarks — this may take a minute...')
    crossover = bench_density_crossover()
    sparse_test = bench_sparse_pattern()
    explosive = bench_explosive()
    oscillator = bench_oscillator()
    plot_results(crossover, sparse_test, explosive, oscillator, out_path)
    print('\nDone.')


if __name__ == '__main__':
    run_all()
