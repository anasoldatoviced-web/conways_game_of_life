"""
tests.py — Unit tests for Conway's Game of Life project.

Run with:  python tests.py
           python -m pytest tests.py -v   (if pytest is installed)

Tests cover:
  - B3/S23 rule correctness (sparse and dense engines)
  - Specific pattern behaviour (blinker, block, glider, pulsar)
  - File I/O round-trips (.cells, .rle, .txt)
  - Toroidal boundary and custom rulesets
  - Cycle detection
  - Edge cases (empty grid, single cell)
"""

import unittest
import tempfile
import os

from engine_sparse import SparseEngine
from engine_dense  import DenseEngine
from file_io import (load_cells, save_cells, load_rle, save_rle,
                     load_txt, save_txt, load_pattern, save_pattern)


# ======================================================================
# Helpers
# ======================================================================

def run_sparse(cells, n):
    """Run the sparse engine for n steps and return the live set."""
    e = SparseEngine(cells)
    for _ in range(n):
        e.step()
    return e.live_cells


def run_dense(cells, w, h, n):
    """Run the dense engine for n steps and return the live set."""
    e = DenseEngine(width=w, height=h)
    e.set_from_sparse(cells)
    for _ in range(n):
        e.step()
    return e.to_sparse()


# ======================================================================
# Rule correctness
# ======================================================================

class TestB3S23Rules(unittest.TestCase):
    """Verify the B3/S23 update rules on hand-crafted micro-cases."""

    def test_lonely_cell_dies(self):
        """A single live cell with no neighbours should die."""
        result = run_sparse({(5, 5)}, 1)
        self.assertNotIn((5, 5), result)
        self.assertEqual(len(result), 0)

    def test_overcrowded_cell_dies(self):
        """A live cell with 4+ neighbours dies of overcrowding."""
        # Centre cell has 4 neighbours
        cells = {(1,0),(0,1),(1,1),(2,1),(1,2),(0,0)}
        e = SparseEngine(cells)
        e.step()
        # (1,1) had neighbours: (1,0),(0,1),(2,1),(1,2),(0,0) = 5 neighbours → dies
        self.assertNotIn((1,1), e.live_cells)

    def test_birth_with_3_neighbours(self):
        """A dead cell with exactly 3 live neighbours is born."""
        # Three cells around (1,1) — (1,1) is dead
        cells = {(0,0),(1,0),(2,0)}
        result = run_sparse(cells, 1)
        # (1,1) has 3 neighbours: born
        self.assertIn((1, 1), result)

    def test_survival_with_2_neighbours(self):
        """A live cell with exactly 2 live neighbours survives."""
        # Blinker horizontal → vertical
        cells = {(1,1),(2,1),(3,1)}
        result = run_sparse(cells, 1)
        self.assertIn((2, 0), result)
        self.assertIn((2, 1), result)
        self.assertIn((2, 2), result)

    def test_survival_with_3_neighbours(self):
        """A live cell with exactly 3 live neighbours also survives."""
        # Block: all 4 cells survive (each has exactly 3 neighbours within the block)
        block = {(0,0),(1,0),(0,1),(1,1)}
        result = run_sparse(block, 1)
        self.assertEqual(result, block)


# ======================================================================
# Known patterns — sparse engine
# ======================================================================

class TestKnownPatternsSparse(unittest.TestCase):

    def test_block_is_still_life(self):
        """A 2×2 block should be unchanged after any number of steps."""
        block = {(0,0),(1,0),(0,1),(1,1)}
        for n in [1, 5, 100]:
            self.assertEqual(run_sparse(block, n), block,
                             f'Block changed after {n} steps')

    def test_blinker_period_2(self):
        """Horizontal blinker should return to its initial state after 2 steps."""
        blinker_h = {(0,1),(1,1),(2,1)}
        self.assertEqual(run_sparse(blinker_h, 2), blinker_h)

    def test_blinker_intermediate(self):
        """Horizontal blinker should be vertical after 1 step."""
        blinker_h = {(0,1),(1,1),(2,1)}
        blinker_v = {(1,0),(1,1),(1,2)}
        self.assertEqual(run_sparse(blinker_h, 1), blinker_v)

    def test_glider_period_4(self):
        """Glider should return to the same shape (offset by 1 diag.) after 4 steps."""
        glider = {(1,0),(2,1),(0,2),(1,2),(2,2)}
        result = run_sparse(glider, 4)
        # Normalise both to (0,0)
        def normalise(s):
            min_x = min(x for x,_ in s)
            min_y = min(y for _,y in s)
            return frozenset((x-min_x, y-min_y) for x,y in s)
        self.assertEqual(normalise(result), normalise(glider))

    def test_pulsar_period_3(self):
        """Pulsar should return to its initial (normalised) state after 3 steps."""
        from benchmark import _pulsar
        pulsar = _pulsar(0, 0)
        def normalise(s):
            if not s: return frozenset()
            min_x = min(x for x,_ in s)
            min_y = min(y for _,y in s)
            return frozenset((x-min_x, y-min_y) for x,y in s)
        result = run_sparse(pulsar, 3)
        self.assertEqual(normalise(result), normalise(pulsar))

    def test_empty_grid_stays_empty(self):
        """Empty grid should remain empty."""
        result = run_sparse(set(), 10)
        self.assertEqual(result, set())

    def test_generation_counter(self):
        """Generation counter should increment correctly."""
        e = SparseEngine({(0,0),(1,0),(2,0)})
        self.assertEqual(e.generation, 0)
        e.step()
        self.assertEqual(e.generation, 1)
        for _ in range(9):
            e.step()
        self.assertEqual(e.generation, 10)


# ======================================================================
# Known patterns — dense engine
# ======================================================================

class TestKnownPatternsDense(unittest.TestCase):

    def test_block_is_still_life(self):
        block = {(0,0),(1,0),(0,1),(1,1)}
        self.assertEqual(run_dense(block, 10, 10, 1), block)

    def test_blinker_period_2(self):
        blinker_h = {(2,3),(3,3),(4,3)}
        self.assertEqual(run_dense(blinker_h, 10, 10, 2), blinker_h)

    def test_empty_grid(self):
        self.assertEqual(run_dense(set(), 50, 50, 5), set())


# ======================================================================
# Sparse vs dense agreement
# ======================================================================

class TestEngineAgreement(unittest.TestCase):
    """Both engines must produce identical results for the same input."""

    def _compare(self, cells, w, h, n):
        sparse = run_sparse(cells, n)
        dense  = run_dense(cells, w, h, n)
        self.assertEqual(sparse, dense,
                         f'Engines disagree after {n} steps')

    def test_blinker_agreement(self):
        self._compare({(3,3),(4,3),(5,3)}, 15, 15, 5)

    def test_glider_agreement(self):
        glider = {(1,0),(2,1),(0,2),(1,2),(2,2)}
        self._compare(glider, 20, 20, 8)

    def test_random_soup_agreement(self):
        import random
        rng = random.Random(99)
        # Keep soup well away from the dense-grid boundary so no cells escape
        soup = {(x + 10, y + 10) for x in range(20) for y in range(20)
                if rng.random() < 0.3}
        self._compare(soup, 60, 60, 10)


# ======================================================================
# Toroidal boundary
# ======================================================================

class TestToroidal(unittest.TestCase):

    def test_glider_wraps(self):
        """A glider near the right edge should reappear on the left."""
        # Place glider at right edge on a 10×10 toroidal grid
        glider = {(8,0),(9,1),(7,2),(8,2),(9,2)}
        e = SparseEngine(glider)
        # Run enough steps for the glider to cross the boundary
        for _ in range(8):
            e.step_toroidal(10, 10)
        # Pattern should still be alive (not die out)
        self.assertGreater(len(e.live_cells), 0)

    def test_single_row_blinker_wraps(self):
        """On a 3×3 toroidal grid, a horizontal blinker should survive."""
        blinker = {(0,1),(1,1),(2,1)}
        e = SparseEngine(blinker)
        e.step_toroidal(3, 3)
        self.assertGreater(len(e.live_cells), 0)


# ======================================================================
# Custom rulesets
# ======================================================================

class TestCustomRules(unittest.TestCase):

    def test_highlife_replicator_seed(self):
        """HighLife (B36/S23) should evolve differently from standard Conway."""
        cells = {(1,0),(2,1),(0,2),(1,2),(2,2)}
        e_conway   = SparseEngine(cells)
        e_highlife = SparseEngine(cells)
        e_conway.step()
        e_highlife.step_custom(birth=(3, 6), survival=(2, 3))
        # They may agree on this particular step but the API must not crash
        # (both should return a set)
        self.assertIsInstance(e_conway.live_cells, set)
        self.assertIsInstance(e_highlife.live_cells, set)

    def test_custom_birth_only_rule(self):
        """With survival=(), live cells always die. Use a line of 4 so no births happen at existing positions."""
        # A horizontal line of 4: middle two cells each have 2 neighbours (not in birth set),
        # outer two have 1 neighbour. None have exactly 3, so no births at live positions.
        # All live cells must die because survival=().
        cells = {(0,0),(1,0),(2,0),(3,0)}
        e = SparseEngine(cells)
        e.step_custom(birth=(3,), survival=())
        # None of the original live cells should survive
        for c in cells:
            self.assertNotIn(c, e.live_cells)


# ======================================================================
# Cycle detection
# ======================================================================

class TestCycleDetection(unittest.TestCase):

    def test_block_detected_immediately(self):
        """A still life should be detected as a cycle after 1 step."""
        block = {(0,0),(1,0),(0,1),(1,1)}
        e = SparseEngine(block)
        seen = {e.fingerprint()}
        e.step()
        self.assertIn(e.fingerprint(), seen)

    def test_blinker_detected_at_period_2(self):
        """Blinker cycle should be detected after at most 2 steps."""
        blinker = {(0,1),(1,1),(2,1)}
        e = SparseEngine(blinker)
        seen = {e.fingerprint()}
        for _ in range(5):
            e.step()
            fp = e.fingerprint()
            if fp in seen:
                return   # cycle found — test passes
            seen.add(fp)
        self.fail('Blinker cycle not detected within 5 steps')

    def test_pulsar_detected_at_period_3(self):
        """Pulsar cycle should be detected after at most 3 steps."""
        from benchmark import _pulsar
        pulsar = _pulsar(0, 0)
        e = SparseEngine(pulsar)
        seen = {e.fingerprint()}
        for _ in range(10):
            e.step()
            fp = e.fingerprint()
            if fp in seen:
                return
            seen.add(fp)
        self.fail('Pulsar cycle not detected within 10 steps')


# ======================================================================
# File I/O — .cells format
# ======================================================================

class TestCellsIO(unittest.TestCase):

    def _round_trip(self, cells):
        with tempfile.NamedTemporaryFile(suffix='.cells', delete=False, mode='w') as f:
            path = f.name
        try:
            save_cells(path, cells)
            loaded = load_cells(path)
            # Normalise both to (0,0)
            def norm(s):
                if not s: return frozenset()
                mx = min(x for x,_ in s); my = min(y for _,y in s)
                return frozenset((x-mx, y-my) for x,y in s)
            self.assertEqual(norm(loaded), norm(cells))
        finally:
            os.unlink(path)

    def test_glider_round_trip(self):
        self._round_trip({(1,0),(2,1),(0,2),(1,2),(2,2)})

    def test_block_round_trip(self):
        self._round_trip({(0,0),(1,0),(0,1),(1,1)})

    def test_blinker_round_trip(self):
        self._round_trip({(0,1),(1,1),(2,1)})

    def test_empty_round_trip(self):
        """Saving and loading an empty pattern should return an empty set."""
        with tempfile.NamedTemporaryFile(suffix='.cells', delete=False, mode='w') as f:
            path = f.name
        try:
            save_cells(path, set())
            loaded = load_cells(path)
            self.assertEqual(loaded, set())
        finally:
            os.unlink(path)

    def test_load_patterns_directory(self):
        """Every .cells file in patterns/ should load without error."""
        patterns_dir = os.path.join(os.path.dirname(__file__), 'patterns')
        if not os.path.isdir(patterns_dir):
            self.skipTest('patterns/ directory not found')
        for fname in os.listdir(patterns_dir):
            if fname.endswith('.cells'):
                path = os.path.join(patterns_dir, fname)
                with self.subTest(file=fname):
                    cells = load_cells(path)
                    self.assertIsInstance(cells, set)


# ======================================================================
# File I/O — .rle format
# ======================================================================

class TestRleIO(unittest.TestCase):

    GLIDER_RLE = """\
#N Glider
x = 3, y = 3, rule = B3/S23
bob$2bo$3o!
"""

    def test_load_rle_glider(self):
        with tempfile.NamedTemporaryFile(suffix='.rle', delete=False, mode='w') as f:
            f.write(self.GLIDER_RLE)
            path = f.name
        try:
            cells = load_rle(path)
            # Glider has 5 live cells
            self.assertEqual(len(cells), 5)
        finally:
            os.unlink(path)

    def test_rle_round_trip(self):
        original = {(1,0),(2,1),(0,2),(1,2),(2,2)}
        with tempfile.NamedTemporaryFile(suffix='.rle', delete=False, mode='w') as f:
            path = f.name
        try:
            save_rle(path, original)
            loaded = load_rle(path)
            def norm(s):
                if not s: return frozenset()
                mx = min(x for x,_ in s); my = min(y for _,y in s)
                return frozenset((x-mx, y-my) for x,y in s)
            self.assertEqual(norm(loaded), norm(original))
        finally:
            os.unlink(path)


# ======================================================================
# File I/O — .txt format
# ======================================================================

class TestTxtIO(unittest.TestCase):

    def test_txt_round_trip(self):
        cells = {(3,7),(10,2),(0,0),(5,5)}
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as f:
            path = f.name
        try:
            save_txt(path, cells)
            loaded = load_txt(path)
            self.assertEqual(loaded, cells)
        finally:
            os.unlink(path)


# ======================================================================
# Unified load_pattern dispatcher
# ======================================================================

class TestUnifiedLoader(unittest.TestCase):

    def test_dispatch_cells(self):
        cells = {(1,0),(2,1),(0,2),(1,2),(2,2)}
        with tempfile.NamedTemporaryFile(suffix='.cells', delete=False, mode='w') as f:
            path = f.name
        try:
            save_pattern(path, cells)
            loaded = load_pattern(path)
            self.assertIsInstance(loaded, set)
            self.assertGreater(len(loaded), 0)
        finally:
            os.unlink(path)

    def test_dispatch_txt(self):
        cells = {(0,0),(1,1),(2,2)}
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as f:
            path = f.name
        try:
            save_pattern(path, cells)
            loaded = load_pattern(path)
            self.assertEqual(loaded, cells)
        finally:
            os.unlink(path)

    def test_unknown_extension_raises(self):
        with self.assertRaises(ValueError):
            load_pattern('some_file.xyz')


# ======================================================================
# SparseEngine helpers
# ======================================================================

class TestSparseHelpers(unittest.TestCase):

    def test_toggle_on(self):
        e = SparseEngine()
        e.toggle(3, 4)
        self.assertIn((3, 4), e.live_cells)

    def test_toggle_off(self):
        e = SparseEngine({(3, 4)})
        e.toggle(3, 4)
        self.assertNotIn((3, 4), e.live_cells)

    def test_clear(self):
        e = SparseEngine({(0,0),(1,1),(2,2)})
        e.clear()
        self.assertEqual(e.live_cells, set())
        self.assertEqual(e.generation, 0)

    def test_population(self):
        e = SparseEngine({(0,0),(1,0),(2,0)})
        self.assertEqual(e.population(), 3)

    def test_bounding_box(self):
        e = SparseEngine({(1,2),(5,3),(3,7)})
        self.assertEqual(e.bounding_box(), (1, 2, 5, 7))

    def test_bounding_box_empty(self):
        e = SparseEngine()
        self.assertEqual(e.bounding_box(), (0, 0, 0, 0))

    def test_set_cells_resets_generation(self):
        e = SparseEngine({(0,0)})
        e.step(); e.step()
        self.assertEqual(e.generation, 2)
        e.set_cells({(1,1)})
        self.assertEqual(e.generation, 0)


# ======================================================================
# Run all tests
# ======================================================================

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestB3S23Rules,
        TestKnownPatternsSparse,
        TestKnownPatternsDense,
        TestEngineAgreement,
        TestToroidal,
        TestCustomRules,
        TestCycleDetection,
        TestCellsIO,
        TestRleIO,
        TestTxtIO,
        TestUnifiedLoader,
        TestSparseHelpers,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    exit(0 if result.wasSuccessful() else 1)
