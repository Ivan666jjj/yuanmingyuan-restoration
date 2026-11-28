"""Trace the Garden Gate's carved panels from the 1786 engraving into relief height maps.

Source: Commons "圆明园铜版画册-2.jpg" (Palace Museum album scan of 《西洋楼二十景》 plate 4,
public domain, 4949x2922 px). Download it once and pass its path:

  python3 scripts/3d/huayuanmen_relief.py <album_2.jpg> scripts/3d/huayuanmen_relief

Needs numpy, scipy, scikit-image and Pillow (system Python, not Blender). Writes one 8-bit grey+alpha PNG per
region (grey = height, alpha = panel outline), a grey <name>.pocket.png where the ground is sunk
into the face, plus manifest.json (pixel box, relief depth, pocket depth and outline);
huayuanmen.py reads them.

The engraver shows carving as light shapes on a darker (shaded or hatched) ground; see the
two tracing modes below.
Symmetric regions are traced from their left half and mirrored, which also removes the scan's
0.5-0.7 degree tilt from the result.
"""

import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import approximate_polygon, find_contours
from skimage.morphology import skeletonize

AXIS_X = 2475  # gate axis in album_2 pixels

# name: pixel box (x0, y0, x1, y1) read off album_2.jpg, outline shape, mirror line (pixel x;
# the half left of it is traced; mirror_y likewise mirrors the upper half down), relief depth (m), tracing mode and its parameters.
#   shade: height follows the light level (modelled carving, e.g. the tympanum scrolls).
#   mask:  outline-drawn carving on a hatched ground; the carved shapes become rounded islands
#          (each with its own sunk edge), and the panels and frames around them are modelled in
#          huayuanmen.py.
REGIONS = {
    "tympanum": dict(box=(2299, 1422, 2651, 1598), shape=("semicircle", 2475, 1598, 176),
                     mirror=AXIS_X, depth=0.045, mode="shade", params=dict()),
    "door_medallion": dict(box=(2347, 1800, 2421, 1869), shape=("ellipse",), mirror=2384, depth=0.012, scale=6,
                           recess=0.014, mode="vector", source="huayuanmen_door_medallion.json",
                           decimate=0.12),   # glb under 10 MB
    # Left spandrel sprig, drawn as curves; the right spandrel mirrors it about the gate axis in
    # huayuanmen.py. The shape is the field inside the panel's inner rim (huayuanmen.py SPANDREL_*:
    # straight side 258.2 px and arc radius 262 px from the arch centre 2477.2/1587.3, top 1410.5).
    "spandrel": dict(box=(2209, 1400, 2300, 1572), mirror=None, depth=0.014, scale=4,
                     # The arc meets the straight side at y 1542.8, where the field ends in a point.
                     shape=("poly", [(2219.0, 1410.5), (2283.8, 1410.5)] +
                            [(2477.2 - math.sqrt(262 ** 2 - (1587.3 - y) ** 2), y) for y in range(1415, 1541, 5)] +
                            [(2219.0, 1587.3 - math.sqrt(262 ** 2 - 258.2 ** 2))]),
                     mode="vector", source="huayuanmen_spandrel.json"),
    # Attic cartouche, drawn as curves (huayuanmen_attic.json) and mirrored about the attic axis
    # (huayuanmen.py ATTIC_AXIS_PX; the cartouche's own mirror axis on album_2 is 2481.5).
    "attic": dict(box=(2380, 1118, 2584, 1276), shape=("rect",), mirror=2482, depth=0.035, scale=3,
                  mode="vector", source="huayuanmen_attic.json", decimate=0.12),   # glb under 10 MB
    # Crowning cartouche on the attic, traced on album_2 itself (huayuanmen_crown.json) about the
    # attic axis. It stands free against the sky, so its silhouette is written to the manifest and
    # huayuanmen.py cuts the crown's body to it.
    # Leaves on the console over each column, traced on album_2 itself (huayuanmen_console.json)
    # about the console's centre; huayuanmen.py sets the panel on the bell's front.
    "console": dict(box=(2098, 1386, 2174, 1440), shape=("rect",), mirror=2136, depth=0.012, scale=6,
                    mode="vector", source="huayuanmen_console.json", decimate=0.12),
    # Garland hanging from the cartouche's left wing down the attic field to the volute, drawn on
    # album_2 itself (huayuanmen_garland.json); huayuanmen.py sets it and its mirror image on the
    # field. The outline hugs the garland, so the panel carries no wide bare ground.
    "garland": dict(box=(2222, 1180, 2398, 1348), mirror=None, depth=0.035, scale=3,
                    shape=("poly", [(2372, 1180), (2396, 1186), (2392, 1212), (2384, 1240), (2370, 1262),
                                    (2356, 1282), (2340, 1302), (2322, 1320), (2304, 1334), (2285, 1346),
                                    (2228, 1346), (2224, 1334), (2248, 1326), (2270, 1316), (2288, 1299),
                                    (2300, 1280), (2310, 1260), (2316, 1240), (2321, 1222), (2334, 1205),
                                    (2355, 1186)]),
                    mode="vector", source="huayuanmen_garland.json", decimate=0.12),   # glb under 10 MB
    "crown": dict(box=(2384, 988, 2580, 1125), shape=("rect",), mirror=2482, depth=0.03, scale=3,
                  mode="vector", source="huayuanmen_crown.json", silhouette=True),
    # The door panels are drawn as curves too (one JSON each); they mirror left-right only. They
    # keep fewer triangles than the default (glb under 10 MB with the attic garlands).
    "door_top": dict(box=(2341, 1675, 2431, 1709), shape=("rect",), mirror=2386, depth=0.012, scale=6,
                     mode="vector", source="huayuanmen_door_top.json", decimate=0.16),
    "door_band": dict(box=(2340, 1761, 2430, 1788), shape=("rect",), mirror=2385, depth=0.012, scale=6,
                      mode="vector", source="huayuanmen_door_band.json", decimate=0.16),
    "door_low": dict(box=(2340, 1890, 2430, 1920), shape=("rect",), mirror=2385, depth=0.012, scale=6,
                     mode="vector", source="huayuanmen_door_low.json", decimate=0.16),
    # The wall ornament is drawn as curves (huayuanmen_ornament.json, see draw_vector) for its
    # upper-left quarter and mirrored about the ornament axes 3129 / 1778 (the best left-right and
    # top-bottom correlation of the engraving). The clipped corners keep the frame's corner
    # notches out.
    "wall_ornament": dict(box=(2985, 1682, 3273, 1874), mirror=3129, mirror_y=1778, depth=0.025, scale=3,
                          decimate=0.12,   # glb under 10 MB
                          shape=("poly", [(3013, 1682), (3245, 1682), (3273, 1704), (3273, 1852),
                                          (3245, 1874), (3013, 1874), (2985, 1852), (2985, 1704)]),
                          mode="vector", source="huayuanmen_ornament.json"),
}
STROKE_CONTRAST = 18     # grey levels a pixel sits below its 5 px closing to count as an engraved stroke


def shape_mask(region, h, w, up=1):
    x0, y0 = region["box"][:2]
    kind = region["shape"][0]
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    if kind == "rect":
        d.rectangle((0, 0, w - 1, h - 1), fill=255)
    elif kind == "ellipse":
        d.ellipse((0, 0, w - 1, h - 1), fill=255)
    elif kind == "poly":
        d.polygon([((x - x0) * up, (y - y0) * up) for x, y in region["shape"][1]], fill=255)
    elif kind == "semicircle":
        _, cx, cy, r = region["shape"]
        d.pieslice(tuple(v * up for v in (cx - r - x0, cy - r - y0, cx + r - x0, cy + r - y0)), 180, 360,
                   fill=255)
    else:
        raise ValueError(f"unknown shape {kind}")
    return np.asarray(img) > 0


def trace_shade(g, inner, floor=6, ceil=97):
    """Height 0..1 from the light level against the local ground (percentiles floor..ceil)."""
    band = ndi.gaussian_filter(g, 0.9) - ndi.gaussian_filter(g, 14)
    lo, hi = np.percentile(band[inner], (floor, ceil))
    h = np.clip((band - lo) / (hi - lo), 0, 1)
    return ndi.gaussian_filter(h, 0.6), np.ones(g.shape, bool)


def trace_mask(g, inner, up=1, min_px=40, radius_px=3.0, crease=0.35, carved_at=(), ground_at=(),
               carved_in=None, debug=None):
    """Carved shapes are blank paper outlined by engraved strokes, on a hatched ground.

    The outline strokes form closed loops. Every pocket they enclose is either the inside of a
    carved shape (blank paper) or ground caught between scrolls (hatched, so darker). A pocket
    is kept when its mean tone is above the Otsu threshold of the region; pockets the tone gets
    wrong were checked by eye against the engraving and are listed per region (carved_at,
    ground_at: one engraving point inside each). Where the carving is so dense and shaded that
    tone says little (carved_in, a mask), every pocket counts as carved unless listed as ground,
    and the engraved lines between them become cuts.

    The outline stroke belongs to the carving: a stroke within a stroke width of a carved
    pocket is part of the shape, so the silhouette is the outer side of the outline. The
    surface is rounded off toward the silhouette and cut down to `crease` along the centre line
    of every engraved line inside the shape (the lines between leaves and the veins), so the
    lines of the print become the cuts of the carving. Pixel sizes are in engraving pixels; up
    is the upsampling of g."""
    strokes = ((ndi.grey_closing(g, size=4 * up + 1) - g) > STROKE_CONTRAST) & inner
    closed = ndi.binary_closing(strokes, iterations=up) | strokes
    gy = ndi.gaussian_filter(g, (2.2 * up, 0.6 * up))
    paper_level = threshold_otsu(gy[inner & ~ndi.binary_dilation(strokes, iterations=up)])
    pockets, n = ndi.label(ndi.binary_fill_holes(closed) & ~closed)
    keep = np.zeros(n + 1, bool)
    if n:
        keep[1:] = np.asarray(ndi.mean(gy, pockets, range(1, n + 1))) > paper_level
    if carved_in is not None and n:
        keep[1:] |= np.asarray(ndi.mean(carved_in, pockets, range(1, n + 1))) > 0.5
    for seeds, value in ((carved_at, True), (ground_at, False)):
        for (r, c), point in seeds:
            if not pockets[r, c]:
                raise ValueError(f"seed {point} is not inside a pocket")
            keep[pockets[r, c]] = value
    keep[0] = False
    if debug is not None:
        debug.update(pockets=pockets, keep=keep, strokes=closed, g=g)
    carved = keep[pockets]
    dist = ndi.distance_transform_edt(~carved)
    # a stroke is part of the carving when a carved pocket lies within a stroke width of it
    m = carved | (closed & (dist <= 3 * up))
    lab, k = ndi.label(m)
    if k:
        m &= np.isin(lab, 1 + np.nonzero(ndi.sum(m, lab, range(1, k + 1)) >= min_px * up * up)[0])
    m = ndi.gaussian_filter(m.astype(float), 0.5 * up) > 0.5
    # cut every engraved line inside the carving (between lobes, and veins ending inside a leaf)
    # along the stroke's centre line; the outline itself is the silhouette, so lines within
    # a stroke width of it are left out
    seam = skeletonize(closed & m) & (ndi.distance_transform_edt(m) > 2 * up)
    lab, k = ndi.label(seam, structure=np.ones((3, 3)))
    if k:   # specks in the paper are not lines
        seam &= np.isin(lab, 1 + np.nonzero(ndi.sum(seam, lab, range(1, k + 1)) >= 4 * up)[0])

    h = dome(ndi.distance_transform_edt(m), radius_px * up) * (
        crease + (1 - crease) * dome(ndi.distance_transform_edt(~seam), radius_px * 0.5 * up))
    return ndi.gaussian_filter(h * m, 0.4 * up) * m, m


def dome(d, r):
    """Quarter-circle rise from 0 at d = 0 to 1 at d >= r."""
    t = np.clip(d / r, 0, 1)
    return np.sqrt(1 - (1 - t) ** 2)


def catmull_rom(points, closed=False, n=8):
    """Points on the uniform Catmull-Rom spline through points (the curve the drawing was made with)."""
    p = np.asarray(points, float)
    p = np.vstack([p[-1:], p, p[:2]]) if closed else np.vstack([p[:1], p, p[-1:]])
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    seg = [0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - e) * t ** 2 + (3 * b - a - 3 * c + e) * t ** 3)
           for a, b, c, e in zip(p, p[1:], p[2:], p[3:])]
    return np.vstack(seg + [p[-2:-1]])


def closed_outline(points, axis_x=None, n=8):
    """The drawn outline of a closed shape. A shape whose first and last points lie on the mirror
    axis is the left half of a symmetric form: it is drawn open, with no edge along the axis, and
    its end tangents come from the mirrored neighbours, so the mirrored halves join smoothly (crown
    boss and stem, self-check 2026-11-28: closing them drew a line on the axis that the mirror
    turned into a groove the engraving does not have)."""
    p = np.asarray(points, float)
    if axis_x is None or not np.allclose(p[[0, -1], 0], axis_x):
        c = catmull_rom(p, closed=True, n=n)
        return np.vstack([c, c[:1]])
    ghost = lambda q: [2 * axis_x - q[0], q[1]]
    c = catmull_rom(np.vstack([ghost(p[1]), p, ghost(p[-2])]), n=n)
    return np.vstack([c[n:len(c) - 1 - n], p[-1:]])


VECTOR_LINE = 1.5    # width of a drawn line, in pixels of the drawing's source scan
VECTOR_GAP = 2       # samples a line is thickened by while finding the ground (closes small gaps)
VECTOR_LEAF = 2.5    # album px from a shape's edge to its full height (a drawing of broad, swelling
                     # forms, as the attic cartouche, sets a larger "leaf")
VECTOR_GROOVE = 1.0  # album px from an inner line to the full height beside it
VECTOR_CREASE = 0.4  # height left at the bottom of a groove


def load_drawing(region):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), region["source"])) as f:
        return json.load(f)



def pocket_outline(region):
    """Album-px polygon of a drawing's sunk ground, or None.

    A drawing's "outline" (left half, from the axis round to the axis) bounds hatched ground that
    the engraving shows sunk below the plain face around it, as round the door medallion; the
    model cuts a pocket region["recess"] deep there and the carving rises from its floor.
    """
    if region["mode"] != "vector":
        return None
    d = load_drawing(region)
    if "outline" not in d:
        return None
    (mx, my), (ax, ay), s = d["axis"], d["album_axis"], d["album_scale"]
    o = catmull_rom(d["outline"])
    o = np.vstack([o, np.stack([2 * mx - o[::-1, 0], o[::-1, 1]], 1)])
    return np.stack([ax + (o[:, 0] - mx) * s, ay + (o[:, 1] - my) * s], 1)


def pocket_mask(region, h_px, w_px):
    """Samples of a region inside its pocket outline (all False without one)."""
    img = Image.new("L", (w_px, h_px), 0)
    poly = pocket_outline(region)
    if poly is not None:
        (x0, y0), up = region["box"][:2], region["scale"]
        ImageDraw.Draw(img).polygon([((x - x0) * up, (y - y0) * up) for x, y in poly], fill=255)
    return np.asarray(img) > 0


def raster_bands(bands, to_px, scale, w, h, ss):
    """Mask (h, w) of a drawing's bands: raised straps of a drawn width along a centre line
    (strapwork on a hatched ground), round-ended, drawn ss times supersampled. In draw_vector
    they block the ground like a closed outline and rise from their own edges. scale is samples
    per drawing px."""
    img = Image.new("L", (w * ss, h * ss), 0)
    draw = ImageDraw.Draw(img)
    for b in bands:
        c, r = to_px(catmull_rom(b["points"])) * ss, b["width"] * scale * ss / 2
        draw.line([tuple(q) for q in c], fill=255, width=max(1, round(2 * r)), joint="curve")
        for qx, qy in (c[0], c[-1]):   # round ends
            draw.ellipse((qx - r, qy - r, qx + r, qy + r), fill=255)
    return np.asarray(img.resize((w, h), Image.BOX)) > 127


def plinth_body(forms, r, mirrored):
    """The body a drawing's forms stand on, as the attic cartouche's mass under its scrolls: the
    forms closed up by a disc of radius r samples (so it hugs them and leaves the panel's corners
    plain), holes filled. draw_vector raises the open ground inside it to plinth_level, the forms
    rise from there, and only the marked pockets sink to the face. A mirrored drawing is closed
    across its axis."""
    r = int(round(r))
    m = np.pad(np.hstack([forms, forms[:, ::-1]]) if mirrored else forms, r + 1)
    disc = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r
    m = ndi.binary_fill_holes(ndi.binary_erosion(ndi.binary_dilation(m, disc), disc))
    return m[r + 1:-r - 1, r + 1:-r - 1][:, :forms.shape[1]]


def draw_vector(region, h_px, w_px, debug=None):
    """Height map and carved mask from hand-drawn curves (the left half, or the upper-left quarter
    when the region has mirror_y; the rest is mirrored).

    Every drawn line is an engraved edge. The ground is the open space reachable from the drawn
    part's outer (frame) sides without crossing a line (unless the drawing sets
    ground_from_edges false, for panels whose raised border runs out to the edge), plus the
    enclosed pockets marked ground_at (sunk scroll eyes); everything else is carved and rises to a
    rounded top, and the lines inside carved shapes become grooves (leaf veins, scroll turns).
    carved_at points must end up carved, so an outline gap that floods a form fails loudly.
    Beads are domes of the drawn radius; bands are straps of a drawn width.
    """
    d = load_drawing(region)
    up, (x0, y0) = region["scale"], region["box"][:2]
    (mx, my), (ax, ay), s = d["axis"], d["album_axis"], d["album_scale"]
    quarter = region.get("mirror_y") is not None
    mirrored = region["mirror"] is not None   # an unmirrored drawing covers the whole box
    if (mirrored and ax != region["mirror"]) or (quarter and ay != region["mirror_y"]):
        raise ValueError("drawing axes and region mirror lines disagree")
    hx, hy = (ax - x0) * up if mirrored else w_px, (ay - y0) * up if quarter else h_px

    def to_px(p):
        p = np.asarray(p, float).reshape(-1, 2)
        return np.stack([(ax + (p[:, 0] - mx) * s - x0) * up, (ay + (p[:, 1] - my) * s - y0) * up], 1)

    ss = 4    # drawn supersampled, then box-filtered down
    img = Image.new("L", (hx * ss, hy * ss), 0)
    draw = ImageDraw.Draw(img)
    width = max(1, round(d.get("line", VECTOR_LINE) * s * up * ss))   # finer engravings set their own
    curves = [closed_outline(p, mx if mirrored else None) for p in d.get("closed", {}).values()]
    curves += [catmull_rom(p) for p in d.get("lines", {}).values()]
    curves += [np.asarray(p, float) for p in d.get("polylines", {}).values()]  # straight edges
    for c in curves:
        draw.line([tuple(q) for q in to_px(c) * ss], fill=255, width=width, joint="curve")
    # Rings are engraved circles; beads are not lines but domes of radius r standing on whatever
    # lies under them, so the ground between neighbouring beads stays open as in the engraving.
    for (cx, cy), r in [(to_px((x, y))[0], r * s * up) for x, y, r in d.get("rings", [])]:
        draw.ellipse(tuple(v * ss for v in (cx - r, cy - r, cx + r, cy + r)), outline=255, width=width)
    lines = np.asarray(img.resize((hx, hy), Image.BOX)) > 64
    band = raster_bands(d.get("bands", {}).values(), to_px, s * up, hx, hy, ss)

    gap = d.get("gap", VECTOR_GAP)   # a drawing whose outlines all close can narrow it
    lab, _ = ndi.label(~ndi.binary_dilation(lines | band, iterations=gap))
    pockets = [lab[int(r), int(c)] for c, r in to_px(d.get("ground_at", []))]
    if 0 in pockets:
        raise ValueError(f"ground_at points lie on a drawn line: "
                         f"{[p for p, k in zip(d['ground_at'], pockets) if k == 0]}")
    edges = [lab[0], lab[:, 0]] + ([] if quarter else [lab[-1]]) + ([] if mirrored else [lab[:, -1]])
    seeds = np.setdiff1d(np.concatenate((edges if d.get("ground_from_edges", True) else []) + [pockets]), [0])
    yy, xx = np.mgrid[:hy, :hx] + 0.5
    bead = np.zeros((hy, hx))
    for x, y, r in d.get("beads", []):
        (cx, cy), r = to_px((x, y))[0], r * s * up
        bead = np.maximum(bead, np.sqrt(np.clip(1 - ((xx - cx) ** 2 + (yy - cy) ** 2) / r ** 2, 0, 1)))
    ground = ndi.binary_dilation(np.isin(lab, seeds), iterations=gap) & ~lines & ~band & (bead == 0)
    leaks = [p for p, (c, r) in zip(d.get("carved_at", []), to_px(d.get("carved_at", []))) if ground[int(r), int(c)]]
    if leaks:  # an outline gap let the ground flood into a carved form
        raise ValueError(f"carved_at points fell on ground: {leaks}")
    # A line within one line width of the ground is the form's edge (thin leaves are all edge);
    # further in, between two carved forms or inside one, it is a groove.
    groove = lines & (ndi.distance_transform_edt(~ground) > width / ss + 0.5)
    inside = plinth_body(~ground, d["plinth_close"] * up, mirrored) if "plinth_close" in d else np.zeros((hy, hx), bool)
    hollow = ndi.binary_dilation(np.isin(lab, np.setdiff1d(pockets, [0])), iterations=gap)
    plinth = inside & ground & ~hollow
    if debug is not None:
        debug.update(lines=lines, band=band, ground=ground & ~plinth, groove=groove, plinth=plinth)

    def whole(a):
        if mirrored:
            a = np.hstack([a, a[:, ::-1]])[:, :w_px]
        return np.vstack([a, a[::-1]])[:h_px] if quarter else a

    carved, groove, bead = whole(~ground), whole(groove), whole(bead)
    h = dome(ndi.distance_transform_edt(carved), d.get("leaf", VECTOR_LEAF) * up) * (
        VECTOR_CREASE + (1 - VECTOR_CREASE) * dome(ndi.distance_transform_edt(~groove), VECTOR_GROOVE * up))
    h = np.maximum(h, bead)
    if "plinth_close" in d:
        level, inside, plinth = d["plinth_level"], whole(inside), whole(plinth)
        h = np.where(plinth, level, np.where(inside, level + (1 - level) * h, h))
        carved |= plinth
    return ndi.gaussian_filter(h * carved, 0.4 * up) * carved, carved


def trace(album, region, debug=None):
    """Height map and outline of one region at region["scale"] samples per engraving pixel."""
    x0, y0, x1, y1 = region["box"]
    up = region.get("scale", 1)
    if region["mode"] == "vector":
        h_px, w_px = (y1 - y0) * up, (x1 - x0) * up
        h, carved = draw_vector(region, h_px, w_px, debug)
        shape = shape_mask(region, h_px, w_px, up) & (ndi.binary_dilation(carved, iterations=up) |
                                                      pocket_mask(region, h_px, w_px))
        return h * shape, shape
    pad = 20
    g = album[y0 - pad:y1 + pad, x0 - pad:x1 + pad]
    if up > 1:
        g = ndi.zoom(g, up, order=3)
    p, h_px, w_px = pad * up, (y1 - y0) * up, (x1 - x0) * up
    shape = shape_mask(region, h_px, w_px, up)
    inner = np.pad(shape, p)
    params = dict(region["params"])
    if region["mode"] == "shade":
        tracer = trace_shade
    else:
        tracer = trace_mask

        def local(points):
            return [((round((y - y0 + pad + 0.5) * up), round((x - x0 + pad + 0.5) * up)), (x, y)) for x, y in points]

        dense = None
        if "carved_in" in region:
            dense = np.pad(shape_mask(dict(box=region["box"], shape=("poly", region["carved_in"])), h_px, w_px, up), p)
        params.update(up=up, carved_at=local(region.get("carved_at", ())),
                      ground_at=local(region.get("ground_at", ())), carved_in=dense, debug=debug)
    h, carved = (a[p:-p, p:-p] for a in tracer(g, inner, **params))
    if region["mirror"] is not None:
        half = (region["mirror"] - x0) * up
        h, carved = (np.hstack([a[:, :half], a[:, :half][:, ::-1]])[:, :w_px] for a in (h, carved))
    if region.get("mirror_y") is not None:
        half = (region["mirror_y"] - y0) * up
        h, carved = (np.vstack([a[:half], a[:half][::-1]])[:h_px] for a in (h, carved))
    shape &= ndi.binary_dilation(carved, iterations=up)
    base = region.get("base", 0.0)
    h = base + (1 - base) * h
    return h * shape, shape


def silhouette(shape, region):
    """Album-px outline of a free-standing relief's body: its outline mask with the enclosed
    ground filled, traced along the mask's edge and simplified to within a sixth of a pixel."""
    filled = ndi.binary_fill_holes(shape)
    contour = max(find_contours(np.pad(filled, 1).astype(float), 0.5), key=len)
    (x0, y0), up = region["box"][:2], region["scale"]
    rc = approximate_polygon(contour, 0.5)[:-1] - 1
    return [[round(x0 + (c + 0.5) / up, 2), round(y0 + (r + 0.5) / up, 2)] for r, c in rc]


def main():
    src, out_dir = sys.argv[1:3]
    os.makedirs(out_dir, exist_ok=True)
    album = np.asarray(Image.open(src).convert("L")).astype(float)
    if album.shape != (2922, 4949):
        raise SystemExit(f"expected the 4949x2922 album scan, got {album.shape[::-1]}")
    manifest = {}
    for name, region in REGIONS.items():
        h, shape = trace(album, region)
        la = np.dstack([(h * 255).round(), shape * 255]).astype("uint8")
        Image.fromarray(la, "LA").save(os.path.join(out_dir, f"{name}.png"))
        manifest[name] = dict(box=region["box"], depth=region["depth"], scale=region.get("scale", 1))
        if "decimate" in region:   # share of triangles kept, where a large panel needs fewer
            manifest[name]["decimate"] = region["decimate"]
        if region.get("silhouette"):
            manifest[name]["silhouette"] = silhouette(shape, region)
        pocket = pocket_outline(region)
        if pocket is not None:
            Image.fromarray((pocket_mask(region, *h.shape) * 255).astype("uint8"), "L").save(
                os.path.join(out_dir, f"{name}.pocket.png"))
            manifest[name].update(recess=region["recess"], pocket=np.round(pocket, 2).tolist())
        print(f"RELIEF {name}: {h.shape[1]}x{h.shape[0]} px, raised {np.mean(h > 0.02):.0%}")
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
