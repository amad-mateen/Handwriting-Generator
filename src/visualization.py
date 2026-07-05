import os
import re
import numpy as np
import matplotlib
# Use non-interactive Agg backend to prevent memory leaks and threading errors on servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def sanitize_filename(filename):
    """
    Sanitizes a string to make it safe for use as a filename on both Windows and Linux.
    Replaces spaces and invalid characters with underscores.
    """
    filename = filename.replace(' ', '_')
    # Invalid characters for Windows: < > : " / \ | ? *
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    if not filename:
        filename = "default_output"
    return filename

def plot_stroke(stroke, save_name=None):
    """
    Plots the static cumulative coordinates of the handwriting strokes and saves it to a file.
    """
    f, ax = plt.subplots()
    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])
    
    size_x = x.max() - x.min() + 1.0
    size_y = y.max() - y.min() + 1.0
    f.set_size_inches(5.0 * size_x / size_y, 5.0)
    
    cuts = np.where(stroke[:, 0] == 1)[0]
    start = 0
    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value], "k-", linewidth=3)
        start = cut_value + 1
        
    ax.axis("off")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name:
        try:
            plt.savefig(save_name, bbox_inches="tight", pad_inches=0.5)
            print(f"Saved plot to {save_name}")
        except Exception as e:
            print(f"Error saving image to {save_name}: {e}")

    plt.close(f)

def animate_stroke_one_by_one(stroke, save_name=None):
    """
    Generates a stroke-by-stroke animation (GIF) simulating natural writing speed and flow.
    """
    # Convert offsets (dx, dy) to absolute positions (x, y)
    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])
    pos = np.stack([x, y], axis=1)

    # Split drawing into independent segments delimited by end-of-stroke (1) indicators
    cuts = np.where(stroke[:, 0] == 1)[0]
    segments = []
    start = 0
    for cut in cuts:
        segments.append(pos[start:cut])
        start = cut + 1
    if start < len(pos):
        segments.append(pos[start:])

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')

    # Set coordinate limits based on max stroke sizes
    ax.set_xlim(pos[:, 0].min() - 10, pos[:, 0].max() + 10)
    ax.set_ylim(pos[:, 1].min() - 10, pos[:, 1].max() + 10)

    # Initialize empty line drawings
    lines = [ax.plot([], [], 'k-', linewidth=1.0)[0] for _ in segments]

    # Pre-calculate frames timing for smooth playback transition
    segment_lengths = [len(seg) for seg in segments]
    start_frames = np.cumsum([0] + segment_lengths[:-1])
    end_frames = np.cumsum(segment_lengths)
    total_frames = sum(segment_lengths)

    # Easing utility simulating human acceleration/drawing
    def ease_out(t):
        return 1 - (1 - t) ** 2

    def update(frame):
        for i, (start_f, end_f) in enumerate(zip(start_frames, end_frames)):
            if frame < start_f:
                lines[i].set_data([], [])
            elif frame >= end_f:
                lines[i].set_data(segments[i][:, 0], segments[i][:, 1])
            else:
                idx_float = frame - start_f
                length = len(segments[i])
                t = idx_float / (end_f - start_f)
                eased = ease_out(t)
                idx = max(1, int(eased * length))
                lines[i].set_data(segments[i][:idx, 0], segments[i][:idx, 1])
        return lines

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=30,
        blit=True
    )

    if save_name:
        try:
            ani.save(save_name, writer='pillow')
            print(f"Saved animation to {save_name}")
        except Exception as e:
            print(f"Error saving animation to {save_name}: {e}")

    plt.close(fig)
    return ani
