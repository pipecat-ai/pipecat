import asyncio
import tkinter as tk
from typing import Optional

from loguru import logger

from ojin.profiling_utils import FPSTracker


def create_fps_overlay(tk_root: tk.Tk, x: int = 8, y: int = 8, width: int = 160, height: int = 60) -> tk.Canvas:
    """Create and place a small FPS overlay Canvas on the given Tk root.

    Returns the created Canvas. The caller owns placement coordinates.
    """
    # Use the root's background color; empty string is invalid for Tk colors
    try:
        root_bg = tk_root.cget("bg")
    except Exception:
        root_bg = "#000000"
    canvas = tk.Canvas(
        tk_root,
        width=width,
        height=height,
        highlightthickness=0,
        bg=root_bg
    )
    canvas.place(x=x, y=y)
    return canvas
    # Raise the widget in the stacking order (not canvas items)
    # tk_root.tk.call('raise', canvas._w)
    # # Also schedule a raise after pending geometry changes
    # tk_root.after(0, lambda: tk_root.tk.call('raise', canvas._w))


async def _draw_fps_overlay(canvas: tk.Canvas, fps_tracker: FPSTracker, width: int, height: int) -> None:
    # Background panel
    canvas.delete("all")
    canvas.create_rectangle(0, 0, width, height, fill="#111111", outline="#333333")

    history = fps_tracker.get_fps_history()
    partial_history = fps_tracker.get_partial_fps_history()
    current_partial = fps_tracker.partial_average_fps
    current = fps_tracker.average_fps
    fps_tracker.register_history()    
    fps_tracker.reset_partial_average()

    # Grid/reference lines (30 and 60 fps)
    ref_lines = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
    max_v = max(35.0, max(history) if history else 0.0, max(partial_history) if partial_history else 0.0, 1.0)
    for val in ref_lines:
        y = height - int((val / max_v) * (height - 14)) - 12
        color = "#333333" if val != 60.0 else "#2a2a2a"
        canvas.create_line(2, y, width - 2, y, fill=color)
        canvas.create_text(width - 20, y - 1, text=f"{int(val)}", anchor="ne", fill="#555555", font=("TkDefaultFont", 7))

    # Plot FPS history as a line
    left_pad, right_pad = 2, 2
    top_pad, bot_pad = 12, 2
    w = width - left_pad - right_pad
    h = height - top_pad - bot_pad
    n = len(history)
    if n >= 2 and w > 1:
        step = n / float(w)
        points = []
        for i in range(w):
            idx = int(i * step)
            if idx >= n:
                idx = n - 1
            v = history[idx]
            y = top_pad + (h - int((v / max_v) * h))
            x = left_pad + i
            points.append((x, y))
        if len(points) >= 2:
            flat = [coord for pt in points for coord in pt]
            canvas.create_line(*flat, fill="#00d18f")

    # Plot partial FPS history as a second line (different color)
    n_partial = len(partial_history)
    if n_partial >= 2 and w > 1:
        step_p = n_partial / float(w)
        points_p = []
        for i in range(w):
            idx = int(i * step_p)
            if idx >= n_partial:
                idx = n_partial - 1
            v = partial_history[idx]
            y = top_pad + (h - int((v / max_v) * h))
            x = left_pad + i
            points_p.append((x, y))
        if len(points_p) >= 2:
            flat_p = [coord for pt in points_p for coord in pt]
            canvas.create_line(*flat_p, fill="#ffb347")

    # Current FPS text
    canvas.create_text(6, 4, anchor="nw", fill="#00ff88", font=("TkDefaultFont", 9, "bold"), text=f"{current:.1f} total fps")
    canvas.create_text(6, 24, anchor="nw", fill="#ffb347", font=("TkDefaultFont", 9, "bold"), text=f"{current_partial:.1f} partial fps")
    #canvas.lift()


async def update_tk_root(tk_root: tk.Tk, interval_ms: int = 10) -> None:
    """Periodically process Tk events and render the FPS overlay.

    - tk_root: the Tk root to update
    - canvas: the overlay canvas created by create_fps_overlay. If None, only tk updates are performed.
    - interval_ms: sleep between UI updates

    Designed to be run as an asyncio task.
    """
    # Determine the canvas size once
    while True:
        try:
            tk_root.update_idletasks()
            tk_root.update()
            await asyncio.sleep(interval_ms / 1000.0)
        except tk.TclError:
            break  # Window was closed
        except Exception as e:
            logger.error(f"Error updating Tkinter: {e}")
            break

async def update_tk_with_fps_periodically(tk_root: tk.Tk, fps_tracker: FPSTracker, canvas: Optional[tk.Canvas], interval_ms: int = 10) -> None:
    """Periodically process Tk events and render the FPS overlay.

    - tk_root: the Tk root to update
    - canvas: the overlay canvas created by create_fps_overlay. If None, only tk updates are performed.
    - interval_ms: sleep between UI updates

    Designed to be run as an asyncio task.
    """
    # Determine the canvas size once
    width = int(canvas["width"]) if canvas is not None else 0
    height = int(canvas["height"]) if canvas is not None else 0

    while True:
        try:
            tk_root.update_idletasks()
            tk_root.update()
            if canvas is not None:
                try:
                    # Throttle overlay redraws to reduce Canvas churn
                    skip = getattr(canvas, "_fps_draw_skip", 0)
                    skip = (skip + 1) % 3  # draw every 3rd loop (~30ms at 10ms interval)
                    setattr(canvas, "_fps_draw_skip", skip)
                    if skip == 0:
                        await _draw_fps_overlay(canvas, fps_tracker, width, height)
                except Exception as e:
                    # Avoid breaking the UI loop due to overlay
                    pass
                # Keep overlay on top
                try:
                    tk_root.tk.call('raise', canvas._w)
                except Exception:
                    pass
            await asyncio.sleep(interval_ms / 1000.0)
        except tk.TclError:
            break  # Window was closed
        except Exception as e:
            logger.error(f"Error updating Tkinter: {e}")
            break


def start_tk_updater(tk_root: tk.Tk, interval_ms: int = 10) -> asyncio.Task:
    """Start the periodic Tk updater + FPS renderer as a background task."""
    return asyncio.create_task(update_tk_root(tk_root, interval_ms))

def start_tk_fps_udpater(tk_root: tk.Tk, canvas: Optional[tk.Canvas], fps_tracker: FPSTracker, interval_ms: int = 10) -> asyncio.Task:
    """Start the periodic Tk updater + FPS renderer as a background task."""
    return asyncio.create_task(update_tk_with_fps_periodically(tk_root, canvas, fps_tracker, interval_ms))
