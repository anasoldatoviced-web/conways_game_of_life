"""
file_io.py — Load and save Game of Life patterns.

Supported formats
-----------------
.cells  (Plaintext)
    Lines starting with '!' are comments.
    '.' = dead cell, 'O' (letter O) = live cell.
    Each line is one row; rows are read top-to-bottom.

.rle    (Run-Length Encoded)
    The standard compact format used by most GoL collections.
    Header line: x = <width>, y = <height>[, rule = B3/S23]
    Body: runs of 'b' (dead) and 'o' (alive), '$' = end of row, '!' = end.

.txt    (Simple coordinate list, our own save format)
    One "x y" pair per line.  Easy to write and parse.
"""

import re
import os


# ======================================================================
# .cells (Plaintext) format
# ======================================================================

def load_cells(path):
    """
    Load a .cells file and return a set of (x, y) live-cell coordinates
    with the pattern anchored at (0, 0) top-left.
    """
    live = set()
    row = 0
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('!'):    # comment line
                continue
            for col, ch in enumerate(line):
                if ch == 'O' or ch == 'o' or ch == '*':
                    live.add((col, row))
            row += 1
    return live


def save_cells(path, live_cells, comment='Conway\'s Game of Life pattern'):
    """
    Save a set of (x, y) live-cell coordinates as a .cells file.
    The pattern is normalised so its top-left corner is at (0, 0).
    """
    if not live_cells:
        with open(path, 'w') as f:
            f.write(f'!{comment}\n')
        return

    min_x = min(x for x, _ in live_cells)
    min_y = min(y for _, y in live_cells)
    max_x = max(x for x, _ in live_cells)
    max_y = max(y for _, y in live_cells)

    # Normalise to (0,0)
    normalised = {(x - min_x, y - min_y) for x, y in live_cells}

    with open(path, 'w') as f:
        f.write(f'!{comment}\n')
        for row in range(max_y - min_y + 1):
            line = ''
            for col in range(max_x - min_x + 1):
                line += 'O' if (col, row) in normalised else '.'
            f.write(line.rstrip('.') + '\n')   # trim trailing dead cells


# ======================================================================
# .rle (Run-Length Encoded) format
# ======================================================================

def load_rle(path):
    """
    Load a .rle file and return a set of (x, y) live-cell coordinates.
    """
    live = set()
    header_found = False
    body_lines = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):        # comment
                continue
            if not header_found:
                # Header: x = N, y = M [, rule = ...]
                if line.lower().startswith('x'):
                    header_found = True
                    continue                # skip header line, we don't need dimensions
            else:
                body_lines.append(line)

    body = ''.join(body_lines)
    # Strip everything after '!'
    if '!' in body:
        body = body[:body.index('!')]

    x, y = 0, 0
    i = 0
    while i < len(body):
        # Read optional run-count
        run_str = ''
        while i < len(body) and body[i].isdigit():
            run_str += body[i]
            i += 1
        run = int(run_str) if run_str else 1

        if i >= len(body):
            break

        ch = body[i]
        i += 1

        if ch == 'b':                   # dead cells
            x += run
        elif ch == 'o':                 # live cells
            for _ in range(run):
                live.add((x, y))
                x += 1
        elif ch == '$':                 # end of row
            y += run
            x = 0
        # any other character: ignore

    return live


def save_rle(path, live_cells, comment=''):
    """
    Save a set of (x, y) live-cell coordinates as a .rle file.
    """
    if not live_cells:
        with open(path, 'w') as f:
            if comment:
                f.write(f'# {comment}\n')
            f.write('x = 0, y = 0, rule = B3/S23\n!\n')
        return

    min_x = min(x for x, _ in live_cells)
    min_y = min(y for _, y in live_cells)
    max_x = max(x for x, _ in live_cells)
    max_y = max(y for _, y in live_cells)
    normalised = {(x - min_x, y - min_y) for x, y in live_cells}

    width  = max_x - min_x + 1
    height = max_y - min_y + 1

    with open(path, 'w') as f:
        if comment:
            f.write(f'# {comment}\n')
        f.write(f'x = {width}, y = {height}, rule = B3/S23\n')

        body = ''
        for row in range(height):
            row_chars = []
            for col in range(width):
                row_chars.append('o' if (col, row) in normalised else 'b')
            # RLE encode the row
            body += _rle_encode_row(row_chars)
            if row < height - 1:
                body += '$'

        body += '!'
        # Wrap at 70 characters
        for k in range(0, len(body), 70):
            f.write(body[k:k+70] + '\n')


def _rle_encode_row(chars):
    """RLE-encode a list of 'b'/'o' characters into a run-length string."""
    if not chars:
        return ''
    result = ''
    count = 1
    for i in range(1, len(chars)):
        if chars[i] == chars[i - 1]:
            count += 1
        else:
            result += ('' if count == 1 else str(count)) + chars[i - 1]
            count = 1
    result += ('' if count == 1 else str(count)) + chars[-1]
    # Strip trailing dead cells
    result = result.rstrip('b').rstrip('0123456789')
    return result


# ======================================================================
# .txt (simple coordinate list) — our own save format
# ======================================================================

def load_txt(path):
    """Load a simple 'x y' coordinate file."""
    live = set()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                live.add((int(parts[0]), int(parts[1])))
    return live


def save_txt(path, live_cells, comment=''):
    """Save live cells as a simple 'x y' coordinate file."""
    with open(path, 'w') as f:
        if comment:
            f.write(f'# {comment}\n')
        for (x, y) in sorted(live_cells):
            f.write(f'{x} {y}\n')


# ======================================================================
# Unified loader (dispatch by extension)
# ======================================================================

def load_pattern(path):
    """
    Load a pattern from a file, dispatching on extension.
    Returns a set of (x, y) live-cell coordinates.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.cells':
        return load_cells(path)
    elif ext == '.rle':
        return load_rle(path)
    elif ext == '.txt':
        return load_txt(path)
    else:
        raise ValueError(f'Unknown pattern file extension: {ext!r}')


def save_pattern(path, live_cells, comment=''):
    """
    Save a pattern to a file, dispatching on extension.
    Supports .cells, .rle, .txt.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.cells':
        save_cells(path, live_cells, comment)
    elif ext == '.rle':
        save_rle(path, live_cells, comment)
    elif ext == '.txt':
        save_txt(path, live_cells, comment)
    else:
        raise ValueError(f'Unknown pattern file extension: {ext!r}')
