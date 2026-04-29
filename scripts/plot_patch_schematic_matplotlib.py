#!/usr/bin/env python3
"""Render a calorimeter patch schematic with matplotlib."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path as MplPath

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.geometry import get_geometry


EDGE_COLOR = "#2b2b2b"
GRID_COLOR = "#5b5b5b"
PATCH_COLOR = "#e76f51"
PATCH_SHADE = "#d24d33"
PATCH_INNER = "#c65a3d"
PATCH_BOTTOM = "#b84a2f"
EDGE_WIDTH_SCALE = 1.22
GRID_WIDTH_SCALE = 1.28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a calorimeter patch schematic.")
    parser.add_argument(
        "--geometry",
        type=str,
        default="CCD3_REBINNED_45X25X9",
        help="Geometry name from src.data.geometry.",
    )
    parser.add_argument("--patch-z", type=int, default=5, help="Patch depth in longitudinal bins.")
    parser.add_argument("--patch-phi", type=int, default=5, help="Patch width in azimuthal bins.")
    parser.add_argument("--patch-r", type=int, default=3, help="Patch width in radial bins.")
    parser.add_argument("--draw-z", type=int, default=None, help="Displayed longitudinal bins for the cylinder grid.")
    parser.add_argument("--draw-phi", type=int, default=None, help="Displayed azimuthal bins for the cylinder grid.")
    parser.add_argument("--draw-r", type=int, default=None, help="Displayed radial bins for the cylinder grid.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/patch_schematic.png"),
        help="Output image path. The script also writes an SVG sibling.",
    )
    return parser.parse_args()


def ellipse_arc(cx: float, cy: float, a: float, b: float, t1: float, t2: float, n: int = 240) -> np.ndarray:
    ts = np.linspace(t1, t2, n)
    return np.column_stack([cx + a * np.cos(ts), cy + b * np.sin(ts)])


def scale_linewidth(color: str, lw: float) -> float:
    return lw * (GRID_WIDTH_SCALE if color == GRID_COLOR else EDGE_WIDTH_SCALE)


def plot_line(ax, xs, ys, *, color: str, lw: float, **kwargs):
    ax.plot(xs, ys, color=color, lw=scale_linewidth(color, lw), **kwargs)


def add_path_patch(ax, points: np.ndarray, facecolor: str, edgecolor: str, lw: float = 1.0, alpha: float = 1.0):
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1) + [MplPath.CLOSEPOLY]
    closed = np.vstack([points, points[0]])
    scaled_lw = 0.0 if edgecolor == "none" else scale_linewidth(edgecolor, lw)
    patch = PathPatch(MplPath(closed, codes), facecolor=facecolor, edgecolor=edgecolor, lw=scaled_lw, alpha=alpha)
    ax.add_patch(patch)
    return patch


def draw_cylinder_outline(ax, cx: float, top_y: float, a: float, b: float, height: float):
    bottom_y = top_y - height
    top = ellipse_arc(cx, top_y, a, b, 0.0, 2 * math.pi, n=360)
    bottom_back = ellipse_arc(cx, bottom_y, a, b, 0.0, math.pi, n=180)
    bottom_front = ellipse_arc(cx, bottom_y, a, b, math.pi, 2 * math.pi, n=180)

    plot_line(ax, top[:, 0], top[:, 1], color=EDGE_COLOR, lw=1.2)
    plot_line(ax, bottom_back[:, 0], bottom_back[:, 1], color=GRID_COLOR, lw=0.8, alpha=0.5)
    plot_line(ax, bottom_front[:, 0], bottom_front[:, 1], color=EDGE_COLOR, lw=1.0)
    plot_line(ax, [cx - a, cx - a], [top_y, bottom_y], color=EDGE_COLOR, lw=1.2)
    plot_line(ax, [cx + a, cx + a], [top_y, bottom_y], color=EDGE_COLOR, lw=1.2)


def draw_cylinder_grid(ax, cx: float, top_y: float, a: float, b: float, height: float, n_phi: int, n_r: int, n_z: int):
    bottom_y = top_y - height

    for theta in np.linspace(math.pi, 2 * math.pi, n_phi + 1)[1:-1]:
        x = cx + a * math.cos(theta)
        y_top = top_y + b * math.sin(theta)
        y_bottom = bottom_y + b * math.sin(theta)
        plot_line(ax, [x, x], [y_top, y_bottom], color=GRID_COLOR, lw=0.7, alpha=0.8)

    for frac in np.arange(1, n_z) / n_z:
        y = top_y - frac * height
        front_arc = ellipse_arc(cx, y, a, b, math.pi, 2 * math.pi, n=180)
        plot_line(ax, front_arc[:, 0], front_arc[:, 1], color=GRID_COLOR, lw=0.7, alpha=0.75)

    for ring_frac in np.arange(1, n_r) / n_r:
        ring = ellipse_arc(cx, top_y, a * ring_frac, b * ring_frac, 0.0, 2 * math.pi, n=280)
        plot_line(ax, ring[:, 0], ring[:, 1], color=GRID_COLOR, lw=0.6, alpha=0.75)

    for theta in np.linspace(0.0, 2 * math.pi, n_phi, endpoint=False):
        x2 = cx + a * math.cos(theta)
        y2 = top_y + b * math.sin(theta)
        plot_line(ax, [cx, x2], [top_y, y2], color=GRID_COLOR, lw=0.55, alpha=0.7)


def draw_highlight_patch(
    ax,
    cx: float,
    top_y: float,
    a: float,
    b: float,
    height: float,
    phi_start: float,
    phi_end: float,
    r_inner: float,
    r_outer: float,
    z_frac: float,
    phi_bins: int,
    r_bins: int,
    z_bins: int,
):
    bottom_patch_y = top_y - z_frac * height

    top_outer = ellipse_arc(cx, top_y, a * r_outer, b * r_outer, phi_start, phi_end, n=120)
    top_inner = ellipse_arc(cx, top_y, a * r_inner, b * r_inner, phi_end, phi_start, n=120)
    bottom_outer = ellipse_arc(cx, bottom_patch_y, a * r_outer, b * r_outer, phi_start, phi_end, n=120)
    bottom_inner = ellipse_arc(cx, bottom_patch_y, a * r_inner, b * r_inner, phi_end, phi_start, n=120)

    # Visible volume faces.
    add_path_patch(ax, np.vstack([bottom_outer, bottom_inner]), PATCH_BOTTOM, EDGE_COLOR, lw=0.8, alpha=0.26)
    add_path_patch(ax, np.vstack([top_outer, top_inner]), PATCH_COLOR, EDGE_COLOR, lw=0.95, alpha=0.72)
    add_path_patch(ax, np.vstack([top_outer, bottom_outer[::-1]]), PATCH_SHADE, EDGE_COLOR, lw=0.9, alpha=0.40)
    add_path_patch(
        ax,
        np.vstack(
            [
                ellipse_arc(cx, top_y, a * r_inner, b * r_inner, phi_start, phi_end, n=100),
                ellipse_arc(cx, bottom_patch_y, a * r_inner, b * r_inner, phi_end, phi_start, n=100),
            ]
        ),
        PATCH_INNER,
        EDGE_COLOR,
        lw=0.85,
        alpha=0.28,
    )

    def face_polygon(theta: float, color: str, alpha: float):
        top_outer_pt = np.array([cx + a * r_outer * math.cos(theta), top_y + b * r_outer * math.sin(theta)])
        top_inner_pt = np.array([cx + a * r_inner * math.cos(theta), top_y + b * r_inner * math.sin(theta)])
        bottom_outer_pt = np.array(
            [cx + a * r_outer * math.cos(theta), bottom_patch_y + b * r_outer * math.sin(theta)]
        )
        bottom_inner_pt = np.array(
            [cx + a * r_inner * math.cos(theta), bottom_patch_y + b * r_inner * math.sin(theta)]
        )
        poly = np.vstack([top_inner_pt, top_outer_pt, bottom_outer_pt, bottom_inner_pt])
        ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor=EDGE_COLOR, lw=0.85, alpha=alpha))
        return top_inner_pt, top_outer_pt, bottom_outer_pt, bottom_inner_pt

    left_face = face_polygon(phi_start, PATCH_SHADE, 0.32)
    right_face = face_polygon(phi_end, PATCH_SHADE, 0.36)

    # Top and bottom voxel boundaries.
    radial_ts = np.arange(1, r_bins) / r_bins
    phi_thetas = np.linspace(phi_start, phi_end, phi_bins + 1)[1:-1]
    z_fracs = np.arange(1, z_bins) / z_bins

    for t in radial_ts:
        r = r_inner + (r_outer - r_inner) * t
        top_ring = ellipse_arc(cx, top_y, a * r, b * r, phi_start, phi_end, n=100)
        bottom_ring = ellipse_arc(cx, bottom_patch_y, a * r, b * r, phi_start, phi_end, n=100)
        plot_line(ax, top_ring[:, 0], top_ring[:, 1], color=GRID_COLOR, lw=0.52, alpha=0.85)
        plot_line(ax, bottom_ring[:, 0], bottom_ring[:, 1], color=GRID_COLOR, lw=0.40, alpha=0.42)

    for theta in phi_thetas:
        for y0, alpha, width in ((top_y, 0.85, 0.52), (bottom_patch_y, 0.42, 0.40)):
            p1 = np.array([cx + a * r_inner * math.cos(theta), y0 + b * r_inner * math.sin(theta)])
            p2 = np.array([cx + a * r_outer * math.cos(theta), y0 + b * r_outer * math.sin(theta)])
            plot_line(ax, [p1[0], p2[0]], [p1[1], p2[1]], color=GRID_COLOR, lw=width, alpha=alpha)

    # Outer/inner curved faces: z layers and phi columns.
    for frac in z_fracs:
        y = top_y - frac * z_frac * height
        outer_arc = ellipse_arc(cx, y, a * r_outer, b * r_outer, phi_start, phi_end, n=100)
        inner_arc = ellipse_arc(cx, y, a * r_inner, b * r_inner, phi_start, phi_end, n=100)
        plot_line(ax, outer_arc[:, 0], outer_arc[:, 1], color=GRID_COLOR, lw=0.52, alpha=0.72)
        plot_line(ax, inner_arc[:, 0], inner_arc[:, 1], color=GRID_COLOR, lw=0.42, alpha=0.48)

    for theta in phi_thetas:
        x_outer = cx + a * r_outer * math.cos(theta)
        y_outer_top = top_y + b * r_outer * math.sin(theta)
        y_outer_bottom = bottom_patch_y + b * r_outer * math.sin(theta)
        x_inner = cx + a * r_inner * math.cos(theta)
        y_inner_top = top_y + b * r_inner * math.sin(theta)
        y_inner_bottom = bottom_patch_y + b * r_inner * math.sin(theta)
        plot_line(ax, [x_outer, x_outer], [y_outer_top, y_outer_bottom], color=GRID_COLOR, lw=0.50, alpha=0.72)
        plot_line(ax, [x_inner, x_inner], [y_inner_top, y_inner_bottom], color=GRID_COLOR, lw=0.42, alpha=0.42)

    # Radial and z voxel boundaries on the phi side faces.
    for face in (left_face, right_face):
        top_inner_pt, top_outer_pt, bottom_outer_pt, bottom_inner_pt = face
        for t in radial_ts:
            top_pt = top_inner_pt + (top_outer_pt - top_inner_pt) * t
            bottom_pt = bottom_inner_pt + (bottom_outer_pt - bottom_inner_pt) * t
            plot_line(ax, [top_pt[0], bottom_pt[0]], [top_pt[1], bottom_pt[1]], color=GRID_COLOR, lw=0.46, alpha=0.52)
        for frac in z_fracs:
            left_pt = top_inner_pt + (bottom_inner_pt - top_inner_pt) * frac
            right_pt = top_outer_pt + (bottom_outer_pt - top_outer_pt) * frac
            plot_line(ax, [left_pt[0], right_pt[0]], [left_pt[1], right_pt[1]], color=GRID_COLOR, lw=0.46, alpha=0.52)


def compute_aligned_patch_bounds(args: argparse.Namespace, n_phi_draw: int, n_r_draw: int, n_z_draw: int):
    phi_bins = min(n_phi_draw - 1, max(1, int(args.patch_phi)))
    r_bins = min(n_r_draw - 1, max(1, int(args.patch_r)))
    z_bins = min(n_z_draw, max(1, int(args.patch_z)))

    phi_edges = np.linspace(math.pi, 2 * math.pi, n_phi_draw + 1)

    phi_start_idx = 1
    phi_end_idx = min(n_phi_draw, phi_start_idx + phi_bins)

    phi_start = float(phi_edges[phi_start_idx])
    phi_end = float(phi_edges[phi_end_idx])
    r_outer = 1.0
    r_inner = float((n_r_draw - r_bins) / n_r_draw)
    z_frac = float(z_bins / n_z_draw)
    return phi_start, phi_end, r_inner, r_outer, z_frac, phi_bins, r_bins, z_bins


def wedge_ring_sector(cx: float, cy: float, a_out: float, b_out: float, a_in: float, b_in: float, t1: float, t2: float):
    outer = ellipse_arc(cx, cy, a_out, b_out, t1, t2, n=80)
    inner = ellipse_arc(cx, cy, a_in, b_in, t2, t1, n=80)
    return np.vstack([outer, inner])


def draw_exploded_wedge(
    ax,
    cx: float,
    cy: float,
    scale: float,
    phi_bins: int,
    r_bins: int,
    z_bins: int,
    highlighted: bool = False,
):
    a_out = 0.92 * scale
    b_out = 0.28 * scale
    a_in = 0.52 * scale
    b_in = 0.15 * scale
    depth = 0.72 * scale
    center_theta = math.radians(270)
    span_theta = math.radians(140 * (2.0 / 3.0))
    theta1 = center_theta - span_theta / 2.0
    theta2 = center_theta + span_theta / 2.0

    outer_fill = PATCH_SHADE if highlighted else "#f9f9f9"
    left_fill = PATCH_INNER if highlighted else "#ededed"
    right_fill = PATCH_SHADE if highlighted else "#f1f1f1"
    inner_fill = PATCH_BOTTOM if highlighted else "#ffffff"
    top_fill = PATCH_COLOR if highlighted else "#f7f7f7"
    edge_color = EDGE_COLOR
    grid_color = GRID_COLOR
    outer_alpha = 0.24 if highlighted else 1.0
    left_alpha = 0.22 if highlighted else 1.0
    right_alpha = 0.26 if highlighted else 1.0
    top_alpha = 0.72 if highlighted else 1.0
    inner_alpha = 0.18 if highlighted else 0.95

    outer_top = ellipse_arc(cx, cy, a_out, b_out, theta1, theta2, n=100)
    outer_bottom = outer_top.copy()
    outer_bottom[:, 1] -= depth
    add_path_patch(ax, np.vstack([outer_top, outer_bottom[::-1]]), outer_fill, "none", lw=0.0, alpha=outer_alpha)

    def face(theta: float, color: str, alpha: float):
        top_outer = np.array([cx + a_out * math.cos(theta), cy + b_out * math.sin(theta)])
        top_inner = np.array([cx + a_in * math.cos(theta), cy + b_in * math.sin(theta)])
        bottom_outer = top_outer + np.array([0.0, -depth])
        bottom_inner = top_inner + np.array([0.0, -depth])
        poly = np.vstack([top_inner, top_outer, bottom_outer, bottom_inner])
        ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor="none", lw=0.0, alpha=alpha))
        return top_inner, top_outer, bottom_outer, bottom_inner

    left_face = face(theta1, left_fill, left_alpha)
    right_face = face(theta2, right_fill, right_alpha)

    inner_top = ellipse_arc(cx, cy, a_in, b_in, theta1, theta2, n=100)
    inner_bottom = inner_top.copy()
    inner_bottom[:, 1] -= depth
    add_path_patch(ax, np.vstack([inner_top, inner_bottom[::-1]]), inner_fill, "none", lw=0.0, alpha=inner_alpha)

    top_pts = wedge_ring_sector(cx, cy, a_out, b_out, a_in, b_in, theta1, theta2)
    add_path_patch(ax, top_pts, top_fill, "none", lw=0.0, alpha=top_alpha)

    bottom_outer = outer_bottom
    bottom_inner = inner_bottom

    # Explicit dark outlines so highlighted and unhighlighted patches share the same border style.
    plot_line(ax, outer_top[:, 0], outer_top[:, 1], color=edge_color, lw=0.9)
    plot_line(ax, bottom_outer[:, 0], bottom_outer[:, 1], color=edge_color, lw=0.8)
    plot_line(ax, inner_top[:, 0], inner_top[:, 1], color=edge_color, lw=0.8)
    plot_line(ax, bottom_inner[:, 0], bottom_inner[:, 1], color=edge_color, lw=0.72)

    radial_fracs = np.arange(1, r_bins) / r_bins
    phi_thetas = np.linspace(theta1, theta2, phi_bins + 1)[1:-1]
    for theta in (theta1, theta2):
        top_inner_pt = np.array([cx + a_in * math.cos(theta), cy + b_in * math.sin(theta)])
        top_outer_pt = np.array([cx + a_out * math.cos(theta), cy + b_out * math.sin(theta)])
        bottom_outer_pt = top_outer_pt + np.array([0.0, -depth])
        bottom_inner_pt = top_inner_pt + np.array([0.0, -depth])
        plot_line(ax, [top_inner_pt[0], top_outer_pt[0]], [top_inner_pt[1], top_outer_pt[1]], color=edge_color, lw=0.85)
        plot_line(ax, [bottom_inner_pt[0], bottom_outer_pt[0]], [bottom_inner_pt[1], bottom_outer_pt[1]], color=edge_color, lw=0.78)
        plot_line(ax, [top_outer_pt[0], bottom_outer_pt[0]], [top_outer_pt[1], bottom_outer_pt[1]], color=edge_color, lw=0.82)
        plot_line(ax, [top_inner_pt[0], bottom_inner_pt[0]], [top_inner_pt[1], bottom_inner_pt[1]], color=edge_color, lw=0.76)

    for frac in radial_fracs:
        ring_top = ellipse_arc(cx, cy, a_in + (a_out - a_in) * frac, b_in + (b_out - b_in) * frac, theta1, theta2, n=80)
        ring_bottom = ring_top.copy()
        ring_bottom[:, 1] -= depth
        plot_line(ax, ring_top[:, 0], ring_top[:, 1], color=grid_color, lw=0.5, alpha=0.75)
        plot_line(ax, ring_bottom[:, 0], ring_bottom[:, 1], color=grid_color, lw=0.42, alpha=0.5)
    for theta in phi_thetas:
        p1_top = np.array([cx + a_in * math.cos(theta), cy + b_in * math.sin(theta)])
        p2_top = np.array([cx + a_out * math.cos(theta), cy + b_out * math.sin(theta)])
        p1_bottom = p1_top + np.array([0.0, -depth])
        p2_bottom = p2_top + np.array([0.0, -depth])
        plot_line(ax, [p1_top[0], p2_top[0]], [p1_top[1], p2_top[1]], color=grid_color, lw=0.5, alpha=0.75)
        plot_line(ax, [p1_bottom[0], p2_bottom[0]], [p1_bottom[1], p2_bottom[1]], color=grid_color, lw=0.42, alpha=0.5)
        plot_line(ax, [p2_top[0], p2_bottom[0]], [p2_top[1], p2_bottom[1]], color=grid_color, lw=0.45, alpha=0.6)
        plot_line(ax, [p1_top[0], p1_bottom[0]], [p1_top[1], p1_bottom[1]], color=grid_color, lw=0.40, alpha=0.42)

    for frac in np.arange(1, z_bins) / z_bins:
        face_arc = ellipse_arc(cx, cy - depth * frac, a_out, b_out, theta1, theta2, n=80)
        plot_line(ax, face_arc[:, 0], face_arc[:, 1], color=grid_color, lw=0.45, alpha=0.55)
        inner_face_arc = ellipse_arc(cx, cy - depth * frac, a_in, b_in, theta1, theta2, n=80)
        plot_line(ax, inner_face_arc[:, 0], inner_face_arc[:, 1], color=grid_color, lw=0.38, alpha=0.4)

    for quad in (left_face, right_face):
        top_inner, top_outer, bottom_outer, bottom_inner = quad
        plot_line(ax, [top_inner[0], bottom_inner[0]], [top_inner[1], bottom_inner[1]], color=grid_color, lw=0.45, alpha=0.55)
        plot_line(ax, [top_outer[0], bottom_outer[0]], [top_outer[1], bottom_outer[1]], color=grid_color, lw=0.45, alpha=0.55)


def build_figure(args: argparse.Namespace):
    geometry = get_geometry(args.geometry)
    fig, ax = plt.subplots(figsize=(11.2, 5.0), dpi=220)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.06, top=0.98)
    ax.set_aspect("equal")
    ax.axis("off")

    cx = 6.9
    top_y = 3.35
    a = 1.35
    b = 0.42
    height = 4.15

    n_phi_draw = min(geometry.N_CELLS_PHI, args.draw_phi or 16)
    n_r_draw = min(geometry.N_CELLS_R, args.draw_r or 9)
    n_z_draw = min(geometry.N_CELLS_Z, args.draw_z or 9)

    draw_cylinder_outline(ax, cx, top_y, a, b, height)
    draw_cylinder_grid(ax, cx, top_y, a, b, height, n_phi_draw, n_r_draw, n_z_draw)

    phi_start, phi_end, r_inner, r_outer, z_frac, phi_bins, r_bins, z_bins = compute_aligned_patch_bounds(
        args, n_phi_draw, n_r_draw, n_z_draw
    )
    draw_highlight_patch(ax, cx, top_y, a, b, height, phi_start, phi_end, r_inner, r_outer, z_frac, phi_bins, r_bins, z_bins)

    draw_exploded_wedge(ax, 1.15, 1.02, 1.30, phi_bins, r_bins, z_bins, highlighted=True)
    draw_exploded_wedge(ax, 4.15, 1.02, 1.30, phi_bins, r_bins, z_bins)

    ax.set_xlim(-0.05, 8.95)
    ax.set_ylim(-1.42, 4.95)
    return fig


def main() -> int:
    args = parse_args()
    fig = build_figure(args)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, pad_inches=0.08, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), pad_inches=0.08, facecolor="white")
    plt.close(fig)

    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.svg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
