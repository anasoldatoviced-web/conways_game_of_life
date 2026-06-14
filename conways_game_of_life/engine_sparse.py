"""
engine_sparse.py — Sparse-set Conway's Game of Life engine.

The grid is stored as a Python set of (x, y) tuples of live cells.
Only live cells and their neighbours are ever examined, so the cost
per generation is O(k) where k is the number of live cells.

Update rule (B3/S23):
  - A dead cell with exactly 3 live neighbours becomes alive (Birth).
  - A live cell with 2 or 3 live neighbours survives (Survival).
  - Every other cell dies or stays dead.

Neighbour-counting trick (the key algorithm):
  For each live cell, increment a counter for each of its 8 neighbours.
  After iterating over all live cells, every cell that appears in the
  counter with count == 3 is born, and every live cell with count == 2
  or 3 survives. This avoids scanning the whole grid.
"""

from collections import defaultdict


class SparseEngine:
    """Conway's Game of Life using a sparse set representation."""

    def __init__(self, live_cells=None):
        """
        Initialise the engine.

        Parameters
        ----------
        live_cells : iterable of (int, int), optional
            Initial set of live cell coordinates.
        """
        self.live_cells = set(live_cells) if live_cells else set()
        self.generation = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def step(self):
        """Advance the simulation by one generation and return the new live set."""
        # neighbour_count[cell] = number of live neighbours of that cell
        neighbour_count = defaultdict(int)

        for (x, y) in self.live_cells:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue          # skip the cell itself
                    neighbour_count[(x + dx, y + dy)] += 1

        next_gen = set()
        for cell, count in neighbour_count.items():
            if count == 3:                # birth: dead cell with 3 live neighbours
                next_gen.add(cell)
            elif count == 2 and cell in self.live_cells:   # survival
                next_gen.add(cell)

        self.live_cells = next_gen
        self.generation += 1
        return self.live_cells

    # ------------------------------------------------------------------
    # Ruleset support (for oral: "change to a different ruleset")
    # ------------------------------------------------------------------

    def step_custom(self, birth=(3,), survival=(2, 3)):
        """
        Advance one generation with a custom B/S ruleset.

        Parameters
        ----------
        birth    : tuple of ints — neighbour counts that birth a dead cell.
        survival : tuple of ints — neighbour counts that keep a live cell alive.
        """
        neighbour_count = defaultdict(int)
        for (x, y) in self.live_cells:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbour_count[(x + dx, y + dy)] += 1

        next_gen = set()
        for cell, count in neighbour_count.items():
            if count in birth:
                next_gen.add(cell)
            elif count in survival and cell in self.live_cells:
                next_gen.add(cell)

        self.live_cells = next_gen
        self.generation += 1
        return self.live_cells

    # ------------------------------------------------------------------
    # Toroidal (wrap-around) boundary — mentioned in professor's review
    # ------------------------------------------------------------------

    def step_toroidal(self, width, height):
        """
        Advance one generation on a toroidal grid of given width × height.
        Cells that walk off one edge reappear on the opposite edge.
        """
        neighbour_count = defaultdict(int)
        for (x, y) in self.live_cells:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = (x + dx) % width
                    ny = (y + dy) % height
                    neighbour_count[(nx, ny)] += 1

        next_gen = set()
        for cell, count in neighbour_count.items():
            if count == 3:
                next_gen.add(cell)
            elif count == 2 and cell in self.live_cells:
                next_gen.add(cell)

        self.live_cells = next_gen
        self.generation += 1
        return self.live_cells

    # ------------------------------------------------------------------
    # Cycle detection (professor's review suggestion)
    # ------------------------------------------------------------------

    def fingerprint(self):
        """Return a hashable fingerprint of the current state (for cycle detection)."""
        return frozenset(self.live_cells)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_cells(self, live_cells):
        """Replace the live cell set and reset the generation counter."""
        self.live_cells = set(live_cells)
        self.generation = 0

    def clear(self):
        """Kill all cells."""
        self.live_cells = set()
        self.generation = 0

    def toggle(self, x, y):
        """Toggle the state of cell (x, y)."""
        if (x, y) in self.live_cells:
            self.live_cells.discard((x, y))
        else:
            self.live_cells.add((x, y))

    def population(self):
        """Return the number of live cells."""
        return len(self.live_cells)

    def bounding_box(self):
        """
        Return (min_x, min_y, max_x, max_y) of all live cells,
        or (0, 0, 0, 0) if the grid is empty.
        """
        if not self.live_cells:
            return (0, 0, 0, 0)
        xs = [x for x, _ in self.live_cells]
        ys = [y for _, y in self.live_cells]
        return (min(xs), min(ys), max(xs), max(ys))
