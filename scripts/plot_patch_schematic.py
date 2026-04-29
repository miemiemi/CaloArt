#!/usr/bin/env python3
"""Render a calorimeter patch schematic as an editable SVG.

This script intentionally uses only the Python standard library so it can run
in minimal environments. The output is a vector graphic suitable for papers or
slides and easy to polish further in Inkscape/Illustrator if needed.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.geometry import get_geometry


EDGE_COLOR = "#2b2b2b"
GRID_COLOR = "#5b5b5b"
PATCH_COLOR = "#e76f51"
PATCH_SHADE = "#d24d33"
SOFT_FILL = "#f7f7f7"
PATCH_INNER = "#c65a3d"
PATCH_BOTTOM = "#b84a2f"
EDGE_WIDTH_SCALE = 1.22
GRID_WIDTH_SCALE = 1.28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a calorimeter patch schematic to SVG.")
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
        default=Path("artifacts/patch_schematic.svg"),
        help="Requested output path. The actual rendered file is SVG.",
    )
    return parser.parse_args()


def ellipse_arc(cx: float, cy: float, a: float, b: float, t1: float, t2: float, n: int = 200):
    ts = [t1 + (t2 - t1) * i / (n - 1) for i in range(n)]
    return [(cx + a * math.cos(t), cy + b * math.sin(t)) for t in ts]


def fmt_points(points) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def closed_path(points) -> str:
    head = f"M {points[0][0]:.2f},{points[0][1]:.2f}"
    body = " ".join(f"L {x:.2f},{y:.2f}" for x, y in points[1:])
    return f"{head} {body} Z"


def polyline_path(points) -> str:
    head = f"M {points[0][0]:.2f},{points[0][1]:.2f}"
    body = " ".join(f"L {x:.2f},{y:.2f}" for x, y in points[1:])
    return f"{head} {body}"


def scale_linewidth(stroke: str, width: float) -> float:
    return width * (GRID_WIDTH_SCALE if stroke == GRID_COLOR else EDGE_WIDTH_SCALE)


def path(
    d: str,
    *,
    stroke: str,
    stroke_width: float,
    fill: str = "none",
    opacity: float = 1.0,
    stroke_opacity: float | None = None,
) -> str:
    stroke_opacity = opacity if stroke_opacity is None else stroke_opacity
    fill_opacity = opacity if fill != "none" else 1.0
    scaled_width = 0.0 if stroke == "none" else scale_linewidth(stroke, stroke_width)
    return (
        f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{scaled_width:.2f}" '
        f'fill-opacity="{fill_opacity:.3f}" stroke-opacity="{stroke_opacity:.3f}" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, *, color: str, width: float, opacity: float = 1.0) -> str:
    scaled_width = scale_linewidth(color, width)
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{scaled_width:.2f}" stroke-opacity="{opacity:.3f}" '
        'stroke-linecap="round"/>'
    )


def polygon(points, *, fill: str, stroke: str, stroke_width: float, opacity: float = 1.0) -> str:
    scaled_width = 0.0 if stroke == "none" else scale_linewidth(stroke, stroke_width)
    return (
        f'<polygon points="{fmt_points(points)}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{scaled_width:.2f}" fill-opacity="{opacity:.3f}" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
    )


def draw_cylinder(elements: list[str], cx: float, top_y: float, a: float, b: float, height: float, n_phi: int, n_r: int, n_z: int):
    bottom_y = top_y - height

    top = ellipse_arc(cx, top_y, a, b, 0.0, 2 * math.pi, 260)
    bottom_back = ellipse_arc(cx, bottom_y, a, b, 0.0, math.pi, 130)
    bottom_front = ellipse_arc(cx, bottom_y, a, b, math.pi, 2 * math.pi, 130)

    elements.append(path(polyline_path(top), stroke=EDGE_COLOR, stroke_width=1.4))
    elements.append(path(polyline_path(bottom_back), stroke=GRID_COLOR, stroke_width=0.8, stroke_opacity=0.5))
    elements.append(path(polyline_path(bottom_front), stroke=EDGE_COLOR, stroke_width=1.2))
    elements.append(line(cx - a, top_y, cx - a, bottom_y, color=EDGE_COLOR, width=1.4))
    elements.append(line(cx + a, top_y, cx + a, bottom_y, color=EDGE_COLOR, width=1.4))

    # Front-side phi columns.
    phi_angles = [math.pi + (math.pi * i / n_phi) for i in range(1, n_phi)]
    for theta in phi_angles:
        x = cx + a * math.cos(theta)
        y_top = top_y + b * math.sin(theta)
        y_bottom = bottom_y + b * math.sin(theta)
        elements.append(line(x, y_top, x, y_bottom, color=GRID_COLOR, width=0.8, opacity=0.82))

    # Longitudinal rows.
    for frac in [i / n_z for i in range(1, n_z)]:
        y = top_y - frac * height
        arc = ellipse_arc(cx, y, a, b, math.pi, 2 * math.pi, 130)
        elements.append(path(polyline_path(arc), stroke=GRID_COLOR, stroke_width=0.8, stroke_opacity=0.78))

    # Top-face rings.
    for ring_frac in [i / n_r for i in range(1, n_r)]:
        ring = ellipse_arc(cx, top_y, a * ring_frac, b * ring_frac, 0.0, 2 * math.pi, 220)
        elements.append(path(polyline_path(ring), stroke=GRID_COLOR, stroke_width=0.7, stroke_opacity=0.74))

    # Top-face phi boundaries.
    for i in range(n_phi):
        theta = 2 * math.pi * i / n_phi
        x2 = cx + a * math.cos(theta)
        y2 = top_y + b * math.sin(theta)
        elements.append(line(cx, top_y, x2, y2, color=GRID_COLOR, width=0.65, opacity=0.70))


def draw_highlight_patch(
    elements: list[str],
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
):
    bottom_patch_y = top_y - z_frac * height

    top_outer = ellipse_arc(cx, top_y, a * r_outer, b * r_outer, phi_start, phi_end, 100)
    top_inner = ellipse_arc(cx, top_y, a * r_inner, b * r_inner, phi_end, phi_start, 100)
    elements.append(path(closed_path(top_outer + top_inner), stroke=EDGE_COLOR, stroke_width=1.0, fill=PATCH_COLOR, opacity=0.72))

    side_top = ellipse_arc(cx, top_y, a * r_outer, b * r_outer, phi_start, phi_end, 90)
    side_bottom = ellipse_arc(cx, bottom_patch_y, a * r_outer, b * r_outer, phi_end, phi_start, 90)
    elements.append(path(closed_path(side_top + side_bottom), stroke=EDGE_COLOR, stroke_width=1.0, fill=PATCH_SHADE, opacity=0.46))

    for theta in (phi_start, phi_end):
        x_outer = cx + a * r_outer * math.cos(theta)
        x_inner = cx + a * r_inner * math.cos(theta)
        y_outer_top = top_y + b * r_outer * math.sin(theta)
        y_inner_top = top_y + b * r_inner * math.sin(theta)
        y_outer_bottom = bottom_patch_y + b * r_outer * math.sin(theta)
        elements.append(line(x_outer, y_outer_top, x_outer, y_outer_bottom, color=EDGE_COLOR, width=1.0))
        elements.append(line(x_inner, y_inner_top, x_outer, y_outer_top, color=EDGE_COLOR, width=0.85, opacity=0.88))


def compute_aligned_patch_bounds(args: argparse.Namespace, n_phi_draw: int, n_r_draw: int, n_z_draw: int):
    phi_bins = min(n_phi_draw - 1, max(1, int(args.patch_phi)))
    r_bins = min(n_r_draw - 1, max(1, int(args.patch_r)))
    z_bins = min(n_z_draw, max(1, int(args.patch_z)))

    phi_edges = [math.pi + math.pi * i / n_phi_draw for i in range(n_phi_draw + 1)]

    phi_start_idx = 1
    phi_end_idx = min(n_phi_draw, phi_start_idx + phi_bins)

    phi_start = phi_edges[phi_start_idx]
    phi_end = phi_edges[phi_end_idx]
    r_outer = 1.0
    r_inner = (n_r_draw - r_bins) / n_r_draw
    z_frac = z_bins / n_z_draw
    return phi_start, phi_end, r_inner, r_outer, z_frac, phi_bins, r_bins, z_bins


def draw_exploded_wedge(
    elements: list[str],
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
    left_fill = PATCH_INNER if highlighted else "#efefef"
    right_fill = PATCH_SHADE if highlighted else "#f2f2f2"
    inner_fill = PATCH_BOTTOM if highlighted else "#ffffff"
    top_fill = PATCH_COLOR if highlighted else SOFT_FILL
    edge_color = EDGE_COLOR
    grid_color = GRID_COLOR
    outer_opacity = 0.24 if highlighted else 1.0
    left_opacity = 0.22 if highlighted else 1.0
    right_opacity = 0.26 if highlighted else 1.0
    inner_opacity = 0.18 if highlighted else 0.95
    top_opacity = 0.72 if highlighted else 1.0

    outer_top = ellipse_arc(cx, cy, a_out, b_out, theta1, theta2, 100)
    outer_bottom = [(x, y - depth) for x, y in outer_top]
    elements.append(
        path(closed_path(outer_top + outer_bottom[::-1]), stroke="none", stroke_width=0.0, fill=outer_fill, opacity=outer_opacity)
    )

    def face(theta: float, face_fill: str, face_opacity: float):
        top_outer_pt = (cx + a_out * math.cos(theta), cy + b_out * math.sin(theta))
        top_inner_pt = (cx + a_in * math.cos(theta), cy + b_in * math.sin(theta))
        bottom_outer_pt = (top_outer_pt[0], top_outer_pt[1] - depth)
        bottom_inner_pt = (top_inner_pt[0], top_inner_pt[1] - depth)
        elements.append(
            polygon(
                [top_inner_pt, top_outer_pt, bottom_outer_pt, bottom_inner_pt],
                fill=face_fill,
                stroke="none",
                stroke_width=0.0,
                opacity=face_opacity,
            )
        )
        return top_inner_pt, top_outer_pt, bottom_outer_pt, bottom_inner_pt

    left_face = face(theta1, left_fill, left_opacity)
    right_face = face(theta2, right_fill, right_opacity)

    inner_top = ellipse_arc(cx, cy, a_in, b_in, theta1, theta2, 100)
    inner_bottom = [(x, y - depth) for x, y in inner_top]
    elements.append(path(closed_path(inner_top + inner_bottom[::-1]), stroke="none", stroke_width=0.0, fill=inner_fill, opacity=inner_opacity))

    top_outer = ellipse_arc(cx, cy, a_out, b_out, theta1, theta2, 80)
    top_inner = ellipse_arc(cx, cy, a_in, b_in, theta2, theta1, 80)
    elements.append(path(closed_path(top_outer + top_inner), stroke="none", stroke_width=0.0, fill=top_fill, opacity=top_opacity))

    bottom_outer = outer_bottom
    bottom_inner = inner_bottom

    # Explicit outlines.
    elements.append(path(polyline_path(outer_top), stroke=edge_color, stroke_width=0.9))
    elements.append(path(polyline_path(bottom_outer), stroke=edge_color, stroke_width=0.8))
    elements.append(path(polyline_path(inner_top), stroke=edge_color, stroke_width=0.8))
    elements.append(path(polyline_path(bottom_inner), stroke=edge_color, stroke_width=0.72))
    for theta in (theta1, theta2):
        top_inner_pt = (cx + a_in * math.cos(theta), cy + b_in * math.sin(theta))
        top_outer_pt = (cx + a_out * math.cos(theta), cy + b_out * math.sin(theta))
        bottom_outer_pt = (top_outer_pt[0], top_outer_pt[1] - depth)
        bottom_inner_pt = (top_inner_pt[0], top_inner_pt[1] - depth)
        elements.append(line(top_inner_pt[0], top_inner_pt[1], top_outer_pt[0], top_outer_pt[1], color=edge_color, width=0.85))
        elements.append(line(bottom_inner_pt[0], bottom_inner_pt[1], bottom_outer_pt[0], bottom_outer_pt[1], color=edge_color, width=0.78))
        elements.append(line(top_outer_pt[0], top_outer_pt[1], bottom_outer_pt[0], bottom_outer_pt[1], color=edge_color, width=0.82))
        elements.append(line(top_inner_pt[0], top_inner_pt[1], bottom_inner_pt[0], bottom_inner_pt[1], color=edge_color, width=0.76))

    radial_fracs = [i / r_bins for i in range(1, r_bins)]
    for frac in radial_fracs:
        ring_top = ellipse_arc(cx, cy, a_in + (a_out - a_in) * frac, b_in + (b_out - b_in) * frac, theta1, theta2, 70)
        ring_bottom = [(x, y - depth) for x, y in ring_top]
        elements.append(path(polyline_path(ring_top), stroke=grid_color, stroke_width=0.55, stroke_opacity=0.76))
        elements.append(path(polyline_path(ring_bottom), stroke=grid_color, stroke_width=0.42, stroke_opacity=0.50))
    for theta in [theta1 + (theta2 - theta1) * i / phi_bins for i in range(1, phi_bins)]:
        p1_top = (cx + a_in * math.cos(theta), cy + b_in * math.sin(theta))
        p2_top = (cx + a_out * math.cos(theta), cy + b_out * math.sin(theta))
        p1_bottom = (p1_top[0], p1_top[1] - depth)
        p2_bottom = (p2_top[0], p2_top[1] - depth)
        elements.append(line(p1_top[0], p1_top[1], p2_top[0], p2_top[1], color=grid_color, width=0.55, opacity=0.76))
        elements.append(line(p1_bottom[0], p1_bottom[1], p2_bottom[0], p2_bottom[1], color=grid_color, width=0.42, opacity=0.50))
        elements.append(line(p2_top[0], p2_top[1], p2_bottom[0], p2_bottom[1], color=grid_color, width=0.48, opacity=0.60))
        elements.append(line(p1_top[0], p1_top[1], p1_bottom[0], p1_bottom[1], color=grid_color, width=0.40, opacity=0.42))
    for frac in [i / z_bins for i in range(1, z_bins)]:
        face_arc = ellipse_arc(cx, cy - depth * frac, a_out, b_out, theta1, theta2, 70)
        elements.append(path(polyline_path(face_arc), stroke=grid_color, stroke_width=0.45, stroke_opacity=0.55))
        inner_face_arc = ellipse_arc(cx, cy - depth * frac, a_in, b_in, theta1, theta2, 70)
        elements.append(path(polyline_path(inner_face_arc), stroke=grid_color, stroke_width=0.38, stroke_opacity=0.40))
    for quad in (left_face, right_face):
        top_inner_pt, top_outer_pt, bottom_outer_pt, bottom_inner_pt = quad
        elements.append(line(top_inner_pt[0], top_inner_pt[1], bottom_inner_pt[0], bottom_inner_pt[1], color=grid_color, width=0.45, opacity=0.55))
        elements.append(line(top_outer_pt[0], top_outer_pt[1], bottom_outer_pt[0], bottom_outer_pt[1], color=grid_color, width=0.45, opacity=0.55))


def write_svg(args: argparse.Namespace) -> Path:
    geometry = get_geometry(args.geometry)

    width = 980
    height = 620
    view_min_y = -240
    view_height = 760
    elements: list[str] = []

    # Main cylinder.
    cx = 730.0
    top_y = 118.0
    a = 108.0
    b = 32.0
    body_height = 285.0

    n_phi_draw = min(geometry.N_CELLS_PHI, args.draw_phi or 16)
    n_r_draw = min(geometry.N_CELLS_R, args.draw_r or 9)
    n_z_draw = min(geometry.N_CELLS_Z, args.draw_z or 9)

    draw_cylinder(elements, cx, top_y, a, b, body_height, n_phi_draw, n_r_draw, n_z_draw)

    phi_start, phi_end, r_inner, r_outer, z_frac, phi_bins, r_bins, z_bins = compute_aligned_patch_bounds(
        args, n_phi_draw, n_r_draw, n_z_draw
    )
    draw_highlight_patch(elements, cx, top_y, a, b, body_height, phi_start, phi_end, r_inner, r_outer, z_frac)

    # Exploded wedges and arrow.
    draw_exploded_wedge(elements, 120.0, 282.0, 74.0, phi_bins, r_bins, z_bins, highlighted=True)
    draw_exploded_wedge(elements, 410.0, 282.0, 74.0, phi_bins, r_bins, z_bins)

    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 {view_min_y} {width} {view_height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            *elements,
            "</svg>",
        ]
    )

    requested_output = args.output.resolve()
    actual_output = requested_output if requested_output.suffix.lower() == ".svg" else requested_output.with_suffix(".svg")
    actual_output.parent.mkdir(parents=True, exist_ok=True)
    actual_output.write_text(svg, encoding="utf-8")
    return actual_output


def main() -> int:
    args = parse_args()
    output = write_svg(args)
    print(f"Wrote {output}")
    if args.output.suffix.lower() != ".svg":
        print(f"Note: wrote SVG instead of {args.output.name} because this helper is vector-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
