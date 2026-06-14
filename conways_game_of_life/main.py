"""
main.py — Conway's Game of Life — Main application entry point.

Architecture (MVC):
  Model      → SparseEngine  (engine_sparse.py)   — grid state & rules
  View       → Renderer      (this file)           — canvas drawing
  Controller → GameController (this file)          — user input & main loop

The GUI is built with Tkinter. The canvas redraws only changed cells
each frame (dirty-rectangle optimisation) to maintain smooth animation
at ≥ 10 fps on a 500×500 grid.

Controls
--------
  Mouse left-click / drag  : toggle / paint cells
  Mouse right-click / drag : erase cells
  Space                    : play / pause
  Right arrow              : single step (when paused)
  C                        : clear grid
  F                        : fit view to pattern
  Scroll wheel             : zoom in / out
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys

from engine_sparse import SparseEngine
from file_io import load_pattern, save_pattern


# ======================================================================
# Constants & palette
# ======================================================================

BG_COLOR      = '#0d1117'    # dark background
GRID_COLOR    = '#21262d'    # faint grid lines
CELL_COLOR    = '#58a6ff'    # live cell fill (blue-white)
CELL_OUTLINE  = '#79c0ff'    # live cell border
UI_BG         = '#161b22'    # sidebar background
UI_FG         = '#c9d1d9'    # sidebar text
ACCENT        = '#58a6ff'    # button accent
BTN_BG        = '#21262d'
BTN_ACTIVE    = '#30363d'

DEFAULT_CELL_SIZE = 10       # pixels per cell at zoom=1
MIN_CELL_SIZE     = 2
MAX_CELL_SIZE     = 40
PATTERNS_DIR      = os.path.join(os.path.dirname(__file__), 'patterns')


# ======================================================================
# Renderer — draws the grid onto a Tkinter Canvas
# ======================================================================

class Renderer:
    """
    Responsible only for drawing.  Knows about the canvas, cell size,
    and viewport offset; knows nothing about rules or user input.
    """

    def __init__(self, canvas, cell_size=DEFAULT_CELL_SIZE):
        self.canvas    = canvas
        self.cell_size = cell_size
        self.offset_x  = 0      # viewport offset in cells
        self.offset_y  = 0
        self._cell_ids = {}      # (gx, gy) → canvas rectangle id
        self._prev_live = set()

    # ------------------------------------------------------------------ drawing

    def redraw(self, live_cells):
        """Redraw only cells that have changed since the last frame."""
        live_cells = set(live_cells)
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        cs = self.cell_size

        # Visible range in grid coordinates
        vis_x0 = self.offset_x - 1
        vis_y0 = self.offset_y - 1
        vis_x1 = self.offset_x + canvas_w // cs + 2
        vis_y1 = self.offset_y + canvas_h // cs + 2

        # Cells that appeared (need to be drawn)
        born = live_cells - self._prev_live
        # Cells that died (need to be erased)
        died = self._prev_live - live_cells

        for cell in died:
            cid = self._cell_ids.pop(cell, None)
            if cid is not None:
                self.canvas.delete(cid)

        for (gx, gy) in born:
            if vis_x0 <= gx <= vis_x1 and vis_y0 <= gy <= vis_y1:
                self._draw_cell(gx, gy)

        self._prev_live = live_cells

    def full_redraw(self, live_cells):
        """Clear the canvas and redraw everything from scratch."""
        self.canvas.delete('all')
        self._cell_ids.clear()
        self._draw_grid()
        live_cells = set(live_cells)
        for (gx, gy) in live_cells:
            self._draw_cell(gx, gy)
        self._prev_live = live_cells

    def _draw_cell(self, gx, gy):
        cs = self.cell_size
        x0 = (gx - self.offset_x) * cs
        y0 = (gy - self.offset_y) * cs
        cid = self.canvas.create_rectangle(
            x0, y0, x0 + cs, y0 + cs,
            fill=CELL_COLOR, outline=CELL_OUTLINE if cs >= 4 else CELL_COLOR,
            width=1
        )
        self._cell_ids[(gx, gy)] = cid

    def _draw_grid(self):
        """Draw faint grid lines when cells are large enough to see them."""
        if self.cell_size < 6:
            return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        cs = self.cell_size
        # vertical lines
        x = (-(self.offset_x % 1)) * cs
        while x < canvas_w:
            self.canvas.create_line(x, 0, x, canvas_h, fill=GRID_COLOR, tags='grid')
            x += cs
        # horizontal lines
        y = (-(self.offset_y % 1)) * cs
        while y < canvas_h:
            self.canvas.create_line(0, y, canvas_w, y, fill=GRID_COLOR, tags='grid')
            y += cs

    # ------------------------------------------------------------------ coordinate helpers

    def canvas_to_grid(self, cx, cy):
        """Convert canvas pixel coordinates to grid cell coordinates."""
        cs = self.cell_size
        return (cx // cs + self.offset_x, cy // cs + self.offset_y)

    def set_cell_size(self, new_size):
        self.cell_size = max(MIN_CELL_SIZE, min(MAX_CELL_SIZE, new_size))

    def center_on(self, gx, gy, canvas_w, canvas_h):
        """Pan the viewport so (gx, gy) is in the centre."""
        cs = self.cell_size
        self.offset_x = gx - canvas_w // (2 * cs)
        self.offset_y = gy - canvas_h // (2 * cs)

    def fit_to_pattern(self, live_cells, canvas_w, canvas_h):
        """Zoom & pan so the whole pattern fits in the viewport."""
        if not live_cells:
            return
        xs = [x for x, _ in live_cells]
        ys = [y for _, y in live_cells]
        pat_w = max(xs) - min(xs) + 1
        pat_h = max(ys) - min(ys) + 1
        new_cs = max(MIN_CELL_SIZE,
                     min(MAX_CELL_SIZE,
                         min(canvas_w // (pat_w + 4),
                             canvas_h // (pat_h + 4))))
        self.cell_size = new_cs
        cx = (max(xs) + min(xs)) // 2
        cy = (max(ys) + min(ys)) // 2
        self.center_on(cx, cy, canvas_w, canvas_h)


# ======================================================================
# GameController — handles user events and the simulation loop
# ======================================================================

class GameController:
    """
    Sits between the model (SparseEngine) and view (Renderer + Tkinter widgets).
    Handles all user input and drives the animation loop via Tkinter's after().
    """

    def __init__(self, root):
        self.root    = root
        self.engine  = SparseEngine()
        self.running = False
        self.speed   = 10          # generations per second
        self._paint_mode = True    # True=paint, False=erase
        self._drag_cell  = None    # last toggled cell during a drag

        self._build_ui()
        self._bind_events()
        self._schedule_next_frame()

    # ------------------------------------------------------------------ UI construction

    def _build_ui(self):
        self.root.title('Conway\'s Game of Life')
        self.root.configure(bg=UI_BG)
        self.root.geometry('1100x700')
        self.root.minsize(800, 500)

        # ---- Left sidebar ----
        sidebar = tk.Frame(self.root, bg=UI_BG, width=200, padx=10, pady=10)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text='Conway\'s\nGame of Life',
                 bg=UI_BG, fg=ACCENT,
                 font=('Helvetica', 14, 'bold')).pack(pady=(0, 16))

        # Play / pause button
        self.btn_play = self._make_button(sidebar, '▶  Play', self.toggle_play)
        self.btn_play.pack(fill=tk.X, pady=2)

        # Step button
        self._make_button(sidebar, '⏭  Step', self.step_once).pack(fill=tk.X, pady=2)

        # Clear button
        self._make_button(sidebar, '✕  Clear', self.clear).pack(fill=tk.X, pady=2)

        # Fit button
        self._make_button(sidebar, '⊞  Fit view', self.fit_view).pack(fill=tk.X, pady=2)

        tk.Label(sidebar, text='Speed (gen/s)', bg=UI_BG, fg=UI_FG,
                 font=('Helvetica', 9)).pack(pady=(12, 2))
        self.speed_var = tk.IntVar(value=self.speed)
        speed_slider = ttk.Scale(sidebar, from_=1, to=60,
                                 variable=self.speed_var,
                                 orient=tk.HORIZONTAL,
                                 command=self._on_speed_change)
        speed_slider.pack(fill=tk.X)
        self.speed_label = tk.Label(sidebar, text=f'{self.speed} gen/s',
                                    bg=UI_BG, fg=UI_FG, font=('Helvetica', 9))
        self.speed_label.pack()

        ttk.Separator(sidebar, orient='horizontal').pack(fill=tk.X, pady=12)

        # Load / Save
        self._make_button(sidebar, '📂  Load pattern', self.load_pattern).pack(fill=tk.X, pady=2)
        self._make_button(sidebar, '💾  Save pattern', self.save_pattern).pack(fill=tk.X, pady=2)

        ttk.Separator(sidebar, orient='horizontal').pack(fill=tk.X, pady=12)

        # Built-in patterns
        tk.Label(sidebar, text='Example patterns', bg=UI_BG, fg=UI_FG,
                 font=('Helvetica', 9, 'bold')).pack(pady=(0, 4))
        self._pattern_buttons(sidebar)

        ttk.Separator(sidebar, orient='horizontal').pack(fill=tk.X, pady=12)

        # Status
        self.lbl_gen = tk.Label(sidebar, text='Generation: 0',
                                bg=UI_BG, fg=UI_FG, font=('Helvetica', 9))
        self.lbl_gen.pack()
        self.lbl_pop = tk.Label(sidebar, text='Population: 0',
                                bg=UI_BG, fg=UI_FG, font=('Helvetica', 9))
        self.lbl_pop.pack()

        # ---- Canvas ----
        canvas_frame = tk.Frame(self.root, bg=BG_COLOR)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg=BG_COLOR,
                                highlightthickness=0, cursor='crosshair')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.renderer = Renderer(self.canvas, cell_size=DEFAULT_CELL_SIZE)

        # Keyboard shortcut hint bar
        hint = tk.Label(self.root,
                        text='Space: play/pause   →: step   C: clear   '
                             'F: fit   Scroll: zoom   RMB: erase',
                        bg=UI_BG, fg='#6e7681', font=('Helvetica', 8))
        hint.pack(side=tk.BOTTOM, fill=tk.X, pady=2)

    def _make_button(self, parent, text, command):
        return tk.Button(parent, text=text, command=command,
                         bg=BTN_BG, fg=UI_FG, activebackground=BTN_ACTIVE,
                         activeforeground='white', relief=tk.FLAT,
                         font=('Helvetica', 10), padx=6, pady=4,
                         cursor='hand2', bd=0)

    def _pattern_buttons(self, parent):
        """List all .cells files in the patterns/ directory as buttons."""
        if not os.path.isdir(PATTERNS_DIR):
            return
        for fname in sorted(os.listdir(PATTERNS_DIR)):
            if fname.endswith('.cells') or fname.endswith('.rle'):
                name = os.path.splitext(fname)[0].replace('_', ' ').title()
                path = os.path.join(PATTERNS_DIR, fname)
                btn = self._make_button(parent, f'  {name}',
                                        lambda p=path: self._load_from_path(p))
                btn.pack(fill=tk.X, pady=1)

    # ------------------------------------------------------------------ event binding

    def _bind_events(self):
        c = self.canvas
        c.bind('<Button-1>',        self._on_mouse_down)
        c.bind('<B1-Motion>',       self._on_mouse_drag)
        c.bind('<Button-3>',        self._on_rmouse_down)
        c.bind('<B3-Motion>',       self._on_rmouse_drag)
        c.bind('<MouseWheel>',      self._on_scroll)          # Windows / macOS
        c.bind('<Button-4>',        self._on_scroll)          # Linux scroll up
        c.bind('<Button-5>',        self._on_scroll)          # Linux scroll down
        c.bind('<Configure>',       self._on_resize)

        self.root.bind('<space>',       lambda e: self.toggle_play())
        self.root.bind('<Right>',       lambda e: self.step_once())
        self.root.bind('<c>',           lambda e: self.clear())
        self.root.bind('<C>',           lambda e: self.clear())
        self.root.bind('<f>',           lambda e: self.fit_view())
        self.root.bind('<F>',           lambda e: self.fit_view())

    # ------------------------------------------------------------------ mouse handlers

    def _on_mouse_down(self, event):
        gx, gy = self.renderer.canvas_to_grid(event.x, event.y)
        self._paint_mode = (gx, gy) not in self.engine.live_cells
        self.engine.toggle(gx, gy)
        self._drag_cell = (gx, gy)
        self._refresh_canvas()

    def _on_mouse_drag(self, event):
        gx, gy = self.renderer.canvas_to_grid(event.x, event.y)
        if (gx, gy) == self._drag_cell:
            return
        self._drag_cell = (gx, gy)
        if self._paint_mode:
            self.engine.live_cells.add((gx, gy))
        else:
            self.engine.live_cells.discard((gx, gy))
        self._refresh_canvas()

    def _on_rmouse_down(self, event):
        gx, gy = self.renderer.canvas_to_grid(event.x, event.y)
        self.engine.live_cells.discard((gx, gy))
        self._drag_cell = (gx, gy)
        self._refresh_canvas()

    def _on_rmouse_drag(self, event):
        gx, gy = self.renderer.canvas_to_grid(event.x, event.y)
        if (gx, gy) == self._drag_cell:
            return
        self._drag_cell = (gx, gy)
        self.engine.live_cells.discard((gx, gy))
        self._refresh_canvas()

    def _on_scroll(self, event):
        # Determine direction
        if event.num == 4 or event.delta > 0:
            delta = 1
        else:
            delta = -1
        # Zoom centred on mouse position
        gx, gy = self.renderer.canvas_to_grid(event.x, event.y)
        self.renderer.set_cell_size(self.renderer.cell_size + delta * 2)
        # Recalculate offset so the cell under the cursor stays put
        cs = self.renderer.cell_size
        self.renderer.offset_x = gx - event.x // cs
        self.renderer.offset_y = gy - event.y // cs
        self._refresh_canvas(full=True)

    def _on_resize(self, event):
        self._refresh_canvas(full=True)

    def _on_speed_change(self, val):
        self.speed = max(1, int(float(val)))
        self.speed_label.config(text=f'{self.speed} gen/s')

    # ------------------------------------------------------------------ simulation loop

    def _schedule_next_frame(self):
        """Schedule the next animation tick via Tkinter's after()."""
        interval = max(16, 1000 // self.speed)   # ms between frames
        self.root.after(interval, self._tick)

    def _tick(self):
        if self.running:
            self.engine.step()
            self._refresh_canvas()
            self._update_status()
        self._schedule_next_frame()

    def _refresh_canvas(self, full=False):
        if full:
            self.renderer.full_redraw(self.engine.live_cells)
        else:
            self.renderer.redraw(self.engine.live_cells)
        self._update_status()

    def _update_status(self):
        self.lbl_gen.config(text=f'Generation: {self.engine.generation}')
        self.lbl_pop.config(text=f'Population: {self.engine.population()}')

    # ------------------------------------------------------------------ controls

    def toggle_play(self):
        self.running = not self.running
        self.btn_play.config(text='⏸  Pause' if self.running else '▶  Play')

    def step_once(self):
        if self.running:
            return
        self.engine.step()
        self._refresh_canvas()

    def clear(self):
        was_running = self.running
        self.running = False
        self.btn_play.config(text='▶  Play')
        self.engine.clear()
        self._refresh_canvas(full=True)

    def fit_view(self):
        if not self.engine.live_cells:
            return
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self.renderer.fit_to_pattern(self.engine.live_cells, w, h)
        self._refresh_canvas(full=True)

    # ------------------------------------------------------------------ file I/O

    def load_pattern(self):
        path = filedialog.askopenfilename(
            title='Load pattern',
            filetypes=[('Pattern files', '*.cells *.rle *.txt'),
                       ('All files', '*.*')]
        )
        if path:
            self._load_from_path(path)

    def _load_from_path(self, path):
        try:
            cells = load_pattern(path)
        except Exception as e:
            messagebox.showerror('Error loading file', str(e))
            return
        # Centre pattern in viewport
        if cells:
            xs = [x for x, _ in cells]
            ys = [y for _, y in cells]
            cx, cy = (max(xs) + min(xs)) // 2, (max(ys) + min(ys)) // 2
        else:
            cx, cy = 0, 0
        was_running = self.running
        self.running = False
        self.btn_play.config(text='▶  Play')
        self.engine.set_cells(cells)
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self.renderer.fit_to_pattern(cells, w, h)
        self._refresh_canvas(full=True)
        if was_running:
            self.running = True
            self.btn_play.config(text='⏸  Pause')

    def save_pattern(self):
        path = filedialog.asksaveasfilename(
            title='Save pattern',
            defaultextension='.cells',
            filetypes=[('Plaintext cells', '*.cells'),
                       ('Run-length encoded', '*.rle'),
                       ('Coordinate list', '*.txt')]
        )
        if path:
            try:
                save_pattern(path, self.engine.live_cells)
                messagebox.showinfo('Saved', f'Pattern saved to:\n{path}')
            except Exception as e:
                messagebox.showerror('Error saving file', str(e))


# ======================================================================
# Entry point
# ======================================================================

def main():
    root = tk.Tk()
    # Style ttk widgets to match dark theme
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('TScale', background=UI_BG, troughcolor=BTN_BG,
                    slidercolor=ACCENT)
    style.configure('TSeparator', background='#30363d')

    app = GameController(root)
    root.mainloop()


if __name__ == '__main__':
    main()
