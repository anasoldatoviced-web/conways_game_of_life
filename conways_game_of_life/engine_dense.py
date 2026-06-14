"""
engine_dense.py — Dense 2D-array Conway's Game of Life engine.

The grid is stored as a NumPy boolean array of shape (height, width).
Every cell is stored, alive or dead, so memory is O(W × H) and each
generation costs O(W × H) regardless of how many cells are alive.

The neighbour count is computed with scipy.signal.convolve2d using a
3×3 kernel of ones (then subtracting the cell itself).  This runs in
vectorised C-level loops and is very fast for dense or medium-sized grids.

Boundary modes
--------------
'fill'  — cells outside the boundary are treated as dead (default).
'wrap'  — toroidal; cells wrap around the edges.
"""

import numpy as np
from scipy.signal import convolve2d


_KERNEL = np.ones((3, 3), dtype=np.int8)   # 3×3 neighbourhood kernel


class DenseEngine:
    """Conway's Game of Life using a dense NumPy array representation."""

    def __init__(self, width=200, height=200, boundary='fill'):
        """
        Parameters
        ----------
        width, height : int   — grid dimensions in cells.
        boundary      : str   — 'fill' (dead border) or 'wrap' (toroidal).
        """
        self.width = width
        self.height = height
        self.boundary = boundary          # 'fill' or 'wrap'
        self.grid = np.zeros((height, width), dtype=np.bool_)
        self.generation = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def step(self):
        """Advance the simulation by one generation."""
        # Count live neighbours for every cell using 2-D convolution.
        # convolve2d sums the 3×3 neighbourhood; subtract self to get neighbours only.
        g = self.grid.astype(np.int8)
        mode = 'wrap' if self.boundary == 'wrap' else 'fill'
        neighbour_count = convolve2d(g, _KERNEL, mode='same', boundary=mode) - g

        # Apply B3/S23 rules in one vectorised step:
        born     = (~self.grid) & (neighbour_count == 3)
        survives = self.grid    & ((neighbour_count == 2) | (neighbour_count == 3))
        self.grid = born | survives
        self.generation += 1
        return self.grid

    # ------------------------------------------------------------------
    # Interop with sparse representation
    # ------------------------------------------------------------------

    def set_from_sparse(self, live_cells):
        """
        Load a sparse set of (x, y) live-cell coordinates into the dense grid.
        Cells outside [0, width) × [0, height) are silently ignored.
        """
        self.grid[:] = False
        self.generation = 0
        for (x, y) in live_cells:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = True

    def to_sparse(self):
        """Export the current grid as a set of (x, y) live-cell coordinates."""
        ys, xs = np.where(self.grid)
        return set(zip(xs.tolist(), ys.tolist()))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_cell(self, x, y, alive=True):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = alive

    def toggle(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = not self.grid[y, x]

    def clear(self):
        self.grid[:] = False
        self.generation = 0

    def population(self):
        return int(np.sum(self.grid))

    def is_alive(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return bool(self.grid[y, x])
        return False
