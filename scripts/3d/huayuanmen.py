"""Parametric 3D reconstruction of the Garden Gate (花园门), Huanghuazhen garden,
Xiyang Lou, Changchun Yuan.

Basis: 《西洋楼二十景》 plate 4 "花園門北面" (1786, public domain), measured on the
Palace Museum album scan (Commons "圆明园铜版画册-2.jpg", 4949x2922 px).
Every elevation dimension below is an engraving pixel reading (PX_*) times one scale
factor K. K is NOT documented anywhere: it comes from the horizon-line estimate
(painter's eye at 1.55 m on the gate's ground plane), giving a 4.0 m gate width with a
plausible range of 3.5-4.7 m. Depths (projection, wall thickness) cannot be read from a
perspective engraving and are estimates (EST_*). The south face is not depicted and is
mirrored from the north face (inference). Materials are flat colours made here; the
engraving is monochrome, so all colours are inferred.

Coordinates: metres, Blender Z-up, +Y = north, +X = east, origin at the centre of the gate
threshold. The documented north face looks toward +Y.

Run (Blender 4.5 LTS):
  blender -b --factory-startup --python scripts/3d/huayuanmen.py -- --out <dir> [--samples 96] [--no-render]
Outputs in <dir>: huayuanmen.glb, huayuanmen-{hero,front,side,oblique}.webp, huayuanmen-thumb.webp
"""

import argparse
import json
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

# ---------------------------------------------------------------------------
# Measurements (album_2.jpg pixels; gate axis x=2475, threshold centre line y=2064)
# ---------------------------------------------------------------------------
K = 0.00471  # m per px at the gate plane; horizon-line estimate, see module docstring
AXIS_X, BASE_Y = 2475, 2064


def hx(px_x):
    """Horizontal engraving pixel -> metres from the gate axis (absolute)."""
    return abs(px_x - AXIS_X) * K


def vz(px_y):
    """Vertical engraving pixel -> metres above the threshold line."""
    return (BASE_Y - px_y) * K


# Horizontal stations, measured (left half, mirrored; symmetry check passed within 6.2 %).
X_OPEN = hx(2294)        # door opening half width (frame inner line)
X_FRAME = hx(2249)       # door-frame pilaster outer edge
X_PIER = hx(2182)        # pier face with recessed panel, outer edge
X_NARROW = hx(2154)      # narrow pilaster behind the column, outer edge
X_COL = hx(2129)         # column centre
R_COL = 50 * K / 2       # column shaft radius
X_BODY = hx(2056)        # gate body outer edge (side walls start here)
X_CORNICE = hx(2053)     # main cornice outer end
X_ATTIC_EAVE = hx(2272)  # attic flat eave half width
X_EAVE_ARCH = hx(2373)   # arched segment of the attic eave, half width
X_CORNER_BALL = hx(2326 + 7)  # corner ball centre (upper axis is shifted 7 px by scan tilt)
X_SCROLL = hx(2173)      # broken-pediment volute centre
X_LION = hx(2105)        # lion pedestal centre
X_BLOCK = hx(2290)       # small stone block centre at the door jamb
X_SPANDREL = (hx(2295), hx(2212))

# Vertical stations, measured.
Z_THRESHOLD = vz(2058)
Z_LEAF_TOP = vz(1621)
Z_TRANSOM_TOP = vz(1598)
Z_SPRING = vz(1595)
R_ARCH = 176.4 * K       # fitted semicircle radius
Z_ARCHIVOLT_TOP = vz(1396)
R_ARCHIVOLT = Z_ARCHIVOLT_TOP - Z_SPRING
Z_IMPOST = (vz(1683), vz(1640))
Z_COL_CAP = (vz(1535), vz(1490))
Z_ARCHITRAVE = (vz(1490), vz(1440))
Z_LEAF_BLOCK = (vz(1440), vz(1385))
Z_CORNICE = (vz(1380), vz(1337))
Z_CORNICE_CROWN = vz(1280)
Z_KEYSTONE = (vz(1353), vz(1310))
Z_SCROLL = vz(1267)
Z_ATTIC_EAVE = (vz(1170), vz(1138))
Z_EAVE_CROWN = vz(1100)
Z_CORNER_BALL_TOP = vz(1080)
R_CORNER_BALL = 32 * K / 2   # ball diameters 31 / 33 px
R_TOP_BALL = 38 * K / 2      # finial ball diameter 38 px
Z_CARTOUCHE = (vz(1100), vz(1013))
Z_CROWN = (vz(1013), vz(983))
Z_TOP = vz(950)
Z_WALL_TOP = vz(1556)
Z_COPING_BOTTOM = vz(1597)
# The lions stand in front of the gate, so their base sits lower in the picture than the
# threshold line; heights are taken from their own base line (y 2115).
H_LION_PEDESTAL = (2115 - 1985) * K
H_LION = (1985 - 1862) * K
Z_BLOCK_TOP = vz(1948)
WALL_PANEL_PITCH = 425 * K   # right-wall panel spacing (3112/3537/3962 px on the moulding)
X_FIRST_PANEL = hx(3125)  # first panel's frame centre (x 2964-3285 px)

# Estimates (not readable from the engraving).
EST_BODY_HALF_DEPTH = 0.45   # gate body projects 0.15 m beyond the wall faces
EST_WALL_HALF_THICK = 0.30
EST_WALL_LENGTH = 10.0       # wall runs off both edges of the engraving; length is illustrative
EST_ATTIC_HALF_DEPTH = 0.35
EST_COL_OFFSET = R_COL + 0.06  # column stands just in front of the pier face

GF = EST_BODY_HALF_DEPTH

# ---------------------------------------------------------------------------
# Materials: (base colour sRGB, roughness). Inferred: white marble for carved stone,
# pink-grey lime plaster for wall fields (the convention used for the other Xiyang Lou
# entries), bronze doors after the uncited "黄铜雕花门两扇" description.
# ---------------------------------------------------------------------------
MATERIALS = {
    "marble": ((0.84, 0.82, 0.77), 0.5),
    "plaster": ((0.80, 0.72, 0.69), 0.85),
    "bronze": ((0.47, 0.34, 0.18), 0.4),
    "paving": ((0.63, 0.62, 0.59), 0.9),
    "pebble": ((0.47, 0.45, 0.41), 0.95),
    "grass": ((0.24, 0.34, 0.15), 0.95),
}
METALLIC = {"bronze": 0.85}


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def make_materials():
    mats = {}
    for name, (rgb, rough) in MATERIALS.items():
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        bsdf = m.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*[srgb_to_linear(c) for c in rgb], 1.0)
        bsdf.inputs["Roughness"].default_value = rough
        bsdf.inputs["Metallic"].default_value = METALLIC.get(name, 0.0)
        mats[name] = m
    return mats


# ---------------------------------------------------------------------------
# Mesh helpers (bmesh only, no operators, so the build is fast and deterministic)
# ---------------------------------------------------------------------------
class Builder:
    def __init__(self, collection, mats):
        self.col = collection
        self.mats = mats
        self.count = 0
        self.section = "part"
        self.relief_meshes = {}

    def _add(self, mat, bm, smooth=False):
        self.count += 1
        if smooth:  # keep caps and creases crisp while curved surfaces shade smoothly
            for e in bm.edges:
                if len(e.link_faces) == 2 and e.calc_face_angle(0) > math.radians(40):
                    e.smooth = False
        me = bpy.data.meshes.new(f"{self.section}.{mat}.{self.count}")
        bm.to_mesh(me)
        bm.free()
        for p in me.polygons:
            p.use_smooth = smooth
        me.materials.append(self.mats[mat])
        obj = bpy.data.objects.new(me.name, me)
        self.col.objects.link(obj)
        return obj

    def box(self, mat, x0, x1, y0, y1, z0, z1):
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        m = Matrix.Translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)) @ Matrix.Diagonal(
            (abs(x1 - x0), abs(y1 - y0), abs(z1 - z0), 1))
        bmesh.ops.transform(bm, matrix=m, verts=bm.verts)
        return self._add(mat, bm)

    def rotated_box(self, mat, center, size, angle_y):
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        m = Matrix.Translation(center) @ Matrix.Rotation(angle_y, 4, "Y") @ Matrix.Diagonal((*size, 1))
        bmesh.ops.transform(bm, matrix=m, verts=bm.verts)
        return self._add(mat, bm)

    def cylinder(self, mat, center, radius, depth, axis="Z", segments=24, radius_top=None):
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments, radius1=radius,
                              radius2=radius if radius_top is None else radius_top, depth=depth)
        rot = {"Z": Matrix.Identity(4), "X": Matrix.Rotation(math.pi / 2, 4, "Y"),
               "Y": Matrix.Rotation(math.pi / 2, 4, "X")}[axis]
        bmesh.ops.transform(bm, matrix=Matrix.Translation(center) @ rot, verts=bm.verts)
        return self._add(mat, bm, smooth=True)

    def ellipsoid(self, mat, center, radii, segments=20):
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=segments // 2, radius=1.0)
        bmesh.ops.transform(bm, matrix=Matrix.Translation(center) @ Matrix.Diagonal((*radii, 1)),
                            verts=bm.verts)
        return self._add(mat, bm, smooth=True)

    def lathe(self, mat, center, profile, segments=32):
        """Revolve a closed (radius, z) profile (starting and ending on the axis) around Z."""
        bm = bmesh.new()
        rings = []
        for s in range(segments):
            a = 2 * math.pi * s / segments
            rings.append([bm.verts.new((center[0] + r * math.cos(a), center[1] + r * math.sin(a),
                                        center[2] + z)) for r, z in profile])
        for s in range(segments):
            r0, r1 = rings[s], rings[(s + 1) % segments]
            for i in range(len(profile) - 1):
                quad = [r0[i], r1[i], r1[i + 1], r0[i + 1]]
                if len({v.co.to_tuple(6) for v in quad}) >= 3:
                    bm.faces.new(_dedupe(quad))
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        return self._add(mat, bm, smooth=True)

    def _prism(self, mat, verts, offset):
        bm = bmesh.new()
        face = bm.faces.new([bm.verts.new(v) for v in verts])
        ext = bmesh.ops.extrude_face_region(bm, geom=[face])
        moved = [g for g in ext["geom"] if isinstance(g, bmesh.types.BMVert)]
        bmesh.ops.translate(bm, vec=offset, verts=moved)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        return self._add(mat, bm)

    def prism_xz(self, mat, pts, y0, y1):
        """Extrude a simple polygon drawn in the XZ (elevation) plane from y0 to y1."""
        return self._prism(mat, [(x, y0, z) for x, z in pts], (0, y1 - y0, 0))

    def prism_xy(self, mat, pts, z0, z1):
        """Extrude a simple polygon drawn in plan (XY) from z0 to z1."""
        return self._prism(mat, [(x, y, z0) for x, y in pts], (0, 0, z1 - z0))

    def prism_yz(self, mat, pts, x0, x1):
        """Extrude a simple polygon drawn in the YZ (section) plane from x0 to x1."""
        return self._prism(mat, [(x0, y, z) for y, z in pts], (x1 - x0, 0, 0))

    def cut_xz(self, obj, mat, pts, y0, y1):
        """Subtract a prism (simple polygon in XZ, extruded from y0 to y1) from obj's mesh."""
        cutter = self.prism_xz(mat, pts, y0, y1)
        mod = obj.modifiers.new("cut", "BOOLEAN")
        mod.operation, mod.object = "DIFFERENCE", cutter
        bpy.context.view_layer.update()
        me = bpy.data.meshes.new_from_object(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()))
        obj.modifiers.remove(mod)
        old, obj.data = obj.data, me
        me.name = old.name
        bpy.data.meshes.remove(old)
        bpy.data.meshes.remove(cutter.data)   # also removes the cutter object
        return obj

    def frame(self, mat, x0, x1, z0, z1, y0, y1, w):
        """Rectangular moulding frame from four bars that touch but never overlap."""
        xa, xb = sorted((x0, x1))
        self.box(mat, xa, xb, y0, y1, z0, z0 + w)
        self.box(mat, xa, xb, y0, y1, z1 - w, z1)
        self.box(mat, xa, xa + w, y0, y1, z0 + w, z1 - w)
        self.box(mat, xb - w, xb, y0, y1, z0 + w, z1 - w)

    def loft_xz(self, mat, rings):
        """Closed ring solid swept through (outline, y) rings. Outlines are closed XZ loops
        sampled at matching points; the last ring joins back to the first."""
        bm = bmesh.new()
        vs = [[bm.verts.new((x, y, z)) for x, z in path] for path, y in rings]
        n = len(vs[0])
        for a, b in zip(vs, vs[1:] + vs[:1]):
            for k in range(n):
                j = (k + 1) % n
                bm.faces.new((a[k], a[j], b[j], b[k]))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        return self._add(mat, bm, smooth=True)

    def tube(self, mat, points, radius, closed=False):
        """Round tube along a 3D polyline (scroll work, festoons, mouldings)."""
        cu = bpy.data.curves.new("tube", "CURVE")
        cu.dimensions = "3D"
        sp = cu.splines.new("POLY")
        sp.points.add(len(points) - 1)
        sp.use_cyclic_u = closed
        for i, p in enumerate(points):
            sp.points[i].co = (*p, 1.0)
        cu.bevel_depth = radius
        cu.bevel_resolution = 2
        cu.use_fill_caps = True
        tmp = bpy.data.objects.new("tube_tmp", cu)
        self.col.objects.link(tmp)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
        bpy.data.objects.remove(tmp, do_unlink=True)
        bpy.data.curves.remove(cu)
        bm = bmesh.new()
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
        return self._add(mat, bm, smooth=True)

    def relief(self, mat, name, px_x0, px_y0, y_face, s, mirror=False):
        """Place a traced relief panel (see huayuanmen_relief.py) on a face.

        The panel's top-left corner sits at engraving pixel (px_x0, px_y0); s=+1 faces north,
        -1 south (the south face repeats the north elevation as seen by its viewer); mirror
        flips the panel left-right. Instances share one mesh, so repeats cost nothing in the glb.
        """
        if name not in self.relief_meshes:
            self.relief_meshes[name] = relief_mesh(name, self.mats[mat])
        self.count += 1
        obj = bpy.data.objects.new(f"{self.section}.{mat}.relief.{self.count}", self.relief_meshes[name])
        obj.location = (-s * (px_x0 - AXIS_X) * K, y_face, (BASE_Y - px_y0) * K)
        obj.rotation_euler = (0, 0, 0 if s > 0 else math.pi)
        obj.scale = (-1 if mirror else 1, 1, 1)
        self.col.objects.link(obj)
        return obj



def _dedupe(verts):
    out = []
    for v in verts:
        if not out or (out[-1].co - v.co).length > 1e-7:
            out.append(v)
    if len(out) > 1 and (out[0].co - out[-1].co).length < 1e-7:
        out.pop()
    return out


def arc(cx, cz, r, a0, a1, n):
    return [(cx + r * math.cos(a0 + (a1 - a0) * i / n), cz + r * math.sin(a0 + (a1 - a0) * i / n))
            for i in range(n + 1)]


def spiral(cx, y, cz, r0, turns, n=48, flip=1):
    """Flat volute in the XZ plane, shrinking from r0 toward the centre."""
    pts = []
    for i in range(n + 1):
        t = i / n
        a = math.pi / 2 + flip * 2 * math.pi * turns * t
        r = r0 * (1 - 0.8 * t)
        pts.append((cx + flip * r * math.cos(a), y, cz + r * math.sin(a)))
    return pts


def segment_curve(half_w, z_end, z_crown, n=16):
    """Segmental arch through (±half_w, z_end) and (0, z_crown), sampled left to right."""
    rise = z_crown - z_end
    radius = (half_w ** 2 + rise ** 2) / (2 * rise)
    cz = z_crown - radius
    return [(x, cz + math.sqrt(radius ** 2 - x ** 2))
            for x in (-half_w + 2 * half_w * i / n for i in range(n + 1))]


# ---------------------------------------------------------------------------
# Traced relief panels
# ---------------------------------------------------------------------------
RELIEF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huayuanmen_relief")
RELIEF_LIFT = 0.003     # lowest carved level stands this far proud of its face (no shared faces)
RELIEF_SINK = 0.004     # the panel's outer edge dips this far into the face, so no gap shows
RELIEF_DECIMATE = 0.25  # share of triangles kept; flat ground and gentle slopes go first
RELIEF_SHARP = math.radians(50)   # faces meeting at more than this are a carved edge, not a curve


def relief_meta(name):
    with open(os.path.join(RELIEF_DIR, "manifest.json")) as f:
        return json.load(f)[name]


def load_relief(name):
    """Height (0..1) and outline mask arrays of a traced panel, rows top to bottom."""
    meta = relief_meta(name)
    px = load_relief_png(name)
    return px[..., 0], px[..., 3] > 0.5, meta


def load_relief_png(name):
    """RGBA pixels of one relief PNG, rows top to bottom."""
    img = bpy.data.images.load(os.path.join(RELIEF_DIR, f"{name}.png"))
    img.colorspace_settings.name = "Non-Color"
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1]
    bpy.data.images.remove(img)
    return px


def relief_mesh(name, material):
    """Height-field mesh of one traced panel in local space.

    Local axes: engraving columns run along -X (the north viewer's right is -X), rows along
    -Z, carving depth along +Y. One vertex per traced sample, then decimated.
    """
    height, inside, meta = load_relief(name)
    rows, cols = height.shape
    edge = inside & ~(np.roll(inside, 1, 0) & np.roll(inside, -1, 0) & np.roll(inside, 1, 1) &
                      np.roll(inside, -1, 1))
    edge[[0, -1], :] |= inside[[0, -1], :]
    edge[:, [0, -1]] |= inside[:, [0, -1]]
    # Uncarved ground inside the outline sinks into the face as well, so the carving rises straight
    # from the face instead of standing on a RELIEF_LIFT plinth one engraving pixel wide.
    sunk = -RELIEF_SINK
    if "recess" in meta:   # hatched ground sunk into the face: it lies on the pocket floor
        sunk = np.where(load_relief_png(f"{name}.pocket")[..., 0] > 0.5, -meta["recess"], -RELIEF_SINK)
    y = np.where(edge | (height <= 0), sunk, RELIEF_LIFT + meta["depth"] * height)
    index = -np.ones((rows, cols), dtype=np.int64)
    r, c = np.nonzero(inside)
    index[r, c] = np.arange(len(r))
    step = K / meta["scale"]   # traced at meta["scale"] samples per engraving pixel
    verts = np.stack([-c * step, y[r, c], -r * step], axis=1)
    q = inside[:-1, :-1] & inside[1:, :-1] & inside[1:, 1:] & inside[:-1, 1:]
    qr, qc = np.nonzero(q)
    faces = np.stack([index[qr, qc], index[qr + 1, qc], index[qr + 1, qc + 1], index[qr, qc + 1]], axis=1)
    me = bpy.data.meshes.new(f"relief.{name}.raw")
    me.from_pydata(verts.tolist(), [], faces.tolist())
    tmp = bpy.data.objects.new("relief_tmp", me)
    bpy.context.scene.collection.objects.link(tmp)
    mod = tmp.modifiers.new("decimate", "DECIMATE")
    mod.ratio = RELIEF_DECIMATE
    bpy.context.view_layer.update()
    out = bpy.data.meshes.new_from_object(tmp.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    bpy.data.objects.remove(tmp, do_unlink=True)
    bpy.data.meshes.remove(me)
    out.name = f"relief.{name}"
    out.shade_smooth()
    out.set_sharp_from_angle(angle=RELIEF_SHARP)   # keep carved walls crisp: no smoothing across a cliff
    out.materials.append(material)
    return out


# ---------------------------------------------------------------------------
# Building parts
# ---------------------------------------------------------------------------
def build_body(b):
    """Gate body: two piers and the arch spandrel, through the full depth."""
    top = Z_CORNICE[0]
    for sx in (-1, 1):
        b.prism_xz("plaster", [(sx * X_OPEN, 0), (sx * X_BODY, 0), (sx * X_BODY, top), (sx * X_OPEN, top)],
                   -GF, GF)
    head = [(-X_OPEN, top), (-X_OPEN, Z_SPRING)] + arc(0, Z_SPRING, R_ARCH, math.pi, 0, 32)[1:-1] + \
           [(X_OPEN, Z_SPRING), (X_OPEN, top)]
    b.prism_xz("plaster", head, -GF, GF)
    b.box("marble", -X_OPEN, X_OPEN, -GF, GF, 0, Z_THRESHOLD + 0.03)


DOOR_LEAF_HALF_T = 0.04
DOOR_FIELD = 0.006          # raised sub-panel fields stand this far proud of the leaf


def door_arc(px_cx, px_cy, r, px_x0, px_x1, upper, n=16):
    """Engraving-pixel points of a circular arc between two x positions (upper or lower half)."""
    sign = -1 if upper else 1
    return [(x, px_cy + sign * math.sqrt(r * r - (x - px_cx) ** 2))
            for x in (px_x0 + (px_x1 - px_x0) * i / n for i in range(n + 1))]


DOOR_RIM_W = 3              # raised rim round the arch-cut fields, engraving px (double line in the print)
DOOR_RIM_H = 0.012          # its height above the field, about the carved panels' own border


def door_fields(w=0):
    """Outlines (engraving px) of the upper and lower arch-cut fields of the left leaf, inset by w.

    Upper field: rectangle whose bottom is cut by the circle round the medallion; lower field: the
    same below. Each arch meets the sides where the circle crosses them.
    """
    upper = [(2338 + w, 1761 + w), (2432 - w, 1761 + w)] + door_arc(2385, 1842.8, 51.8 + w, 2432 - w, 2338 + w, True)
    lower = [(2340 + w, 1922 - w)] + door_arc(2385, 1824, 65 + w, 2340 + w, 2430 - w, False) + [(2430 - w, 1922 - w)]
    return upper, lower


def build_doors(b):
    """Closed double doors, transom and carved tympanum in the middle of the opening.

    Leaf layout read off the engraving (left leaf; the right leaf mirrors it): an upper panel
    with a carved sub-panel, and a tall panel holding an arch-cut sub-panel, a four-lobed
    medallion and a second arch-cut sub-panel. Below the tall panel the leaf is plain.
    """
    t = DOOR_LEAF_HALF_T
    b.box("marble", -X_OPEN, X_OPEN, -0.08, 0.08, Z_LEAF_TOP, Z_TRANSOM_TOP)
    fields, insets = door_fields(), door_fields(DOOR_RIM_W)
    for sx in (-1, 1):
        leaf = b.box("bronze", 0, sx * X_OPEN, -t, t, Z_THRESHOLD + 0.03, Z_LEAF_TOP)
        for s in (-1, 1):
            f = s * t

            def out(d):
                return sorted((f, f + s * d))

            b.frame("bronze", sx * hx(2323), sx * hx(2449), vz(1721), vz(1660), *out(0.010), 3 * K)
            b.frame("bronze", sx * hx(2320), sx * hx(2448), vz(1941), vz(1745), *out(0.010), 4 * K)
            b.box("bronze", sx * hx(2341), sx * hx(2431), *out(DOOR_FIELD), vz(1709), vz(1675))
            ff = s * (t + DOOR_FIELD)
            for field, inset in zip(fields, insets):
                def xz(poly):
                    return [(sx * hx(x), vz(y)) for x, y in poly]
                b.prism_xz("bronze", xz(field), *out(DOOR_FIELD))
                # rim: the field outline less its inset, unbroken; where it passes door_band's own
                # carved border the two merge into one band, as in the print
                rim = b.prism_xz("bronze", xz(field), *sorted((ff - s * 0.001, ff + s * DOOR_RIM_H)))
                b.cut_xz(rim, "bronze", xz(inset), *sorted((ff - s * 0.002, ff + s * (DOOR_RIM_H + 0.001))))
        for s in (-1, 1):
            f, ff = s * t, s * (t + DOOR_FIELD)
            for name, (x0, y0), y in (("door_top", (2341, 1675), ff), ("door_band", (2340, 1761), ff),
                                      ("door_low", (2340, 1890), ff), ("door_medallion", (2347, 1800), f)):
                if sx * s > 0:   # +X is the north viewer's left leaf, the south viewer's right leaf
                    b.relief("bronze", name, x0, y0, y, s)
                else:
                    b.relief("bronze", name, 2 * AXIS_X - x0, y0, y, s, mirror=True)
            # The medallion's hatched ground is sunk into the leaf (see huayuanmen_relief.py): cut its
            # pocket a little deeper than the relief's floor so the two never share a face.
            meta = relief_meta("door_medallion")
            b.cut_xz(leaf, "bronze", [(sx * hx(x), vz(y)) for x, y in meta["pocket"]],
                     *sorted((f - s * (meta["recess"] + RELIEF_SINK / 2), f + s * 0.01)))
    # Tympanum: the carving is traced from the engraving (huayuanmen_relief.py). Whether it was
    # openwork or relief on a solid plate cannot be told from the print; a solid plate is assumed.
    b.prism_xz("plaster", arc(0, Z_TRANSOM_TOP, R_ARCH - 0.003, 0, math.pi, 32), -0.012, 0.012)
    for s in (-1, 1):
        b.relief("marble", "tympanum", 2299, 1422, s * 0.012, s)


def build_face(b, s):
    """Pilasters, column, archivolt, keystone and statuary on one face (s=+1 north, -1 south)."""
    f = s * GF

    def proj(d):
        return sorted((f, f + s * d))

    for sx in (-1, 1):
        # door-frame pilaster and impost; above the impost the archivolt leg takes over
        b.box("marble", sx * X_OPEN, sx * X_FRAME, *proj(0.04), 0, Z_IMPOST[0])
        b.box("marble", sx * (X_OPEN - 0.01), sx * (X_FRAME + 0.02), *proj(0.07), *Z_IMPOST)
        b.box("marble", sx * (X_OPEN - 0.01), sx * X_FRAME, *proj(0.04), Z_IMPOST[1], Z_SPRING)
        b.frame("marble", sx * (X_FRAME + 0.05), sx * (X_PIER - 0.05), 0.40, 2.40, *proj(0.025), 0.04)
        # narrow pilaster and outer strip behind the column
        b.box("marble", sx * X_PIER, sx * X_NARROW, *proj(0.05), 0, Z_COL_CAP[1] - 0.03)
        b.box("marble", sx * (X_COL + R_COL), sx * X_BODY, *proj(0.03), 0, Z_CORNICE[0])
        cy = s * (GF + EST_COL_OFFSET)
        build_column(b, sx * X_COL, cy)
        # ressaut above the column: architrave block, block with a leaf crown
        b.box("marble", sx * (X_COL - 0.17), sx * (X_COL + 0.17), *sorted((f, cy + s * 0.17)), *Z_ARCHITRAVE)
        b.box("marble", sx * (X_COL - 0.15), sx * (X_COL + 0.15), *sorted((f, cy + s * 0.15)), *Z_LEAF_BLOCK)
        zl = sum(Z_LEAF_BLOCK) / 2
        for k in range(-2, 3):
            b.ellipsoid("marble", (sx * X_COL + k * 0.05, cy + s * 0.16, zl - 0.01 * abs(k)),
                        (0.03, 0.02, 0.09 - 0.01 * abs(k)), segments=10)
        build_spandrel(b, sx, s, f)
        # lion stands in front of the column (plate 4); depth offset is an estimate
        build_lion(b, sx * X_LION, s * (GF + 0.78), s)
        b.box("marble", sx * (X_BLOCK - 0.09), sx * (X_BLOCK + 0.09), *proj(0.40), 0, Z_BLOCK_TOP)
    band = arc(0, Z_SPRING, R_ARCHIVOLT, 0, math.pi, 32) + arc(0, Z_SPRING, R_ARCH, math.pi, 0, 32)
    b.prism_xz("marble", band, *proj(0.04))
    for sx in (-1, 1):
        b.box("marble", sx * R_ARCH, sx * (X_OPEN - 0.01), *proj(0.04), Z_IMPOST[1], Z_SPRING)
    # wedge keystone reaching up under the raised cornice
    b.prism_xz("marble", [(-0.07, Z_ARCHIVOLT_TOP - 0.10), (0.07, Z_ARCHIVOLT_TOP - 0.10),
                          (0.12, Z_KEYSTONE[1]), (-0.12, Z_KEYSTONE[1])], *proj(0.10))


def build_spandrel(b, sx, s, f):
    """Curved-triangle relief panel between the archivolt and the cornice, with a scroll."""
    zs, zt, xo, rc = Z_SPRING + 0.15, Z_CORNICE[0] - 0.06, X_SPANDREL[1], R_ARCHIVOLT + 0.06
    a_top, a_bot = math.asin((zt - Z_SPRING) / rc), math.asin((zs - Z_SPRING) / rc)
    pts = [(xo, zs), (xo, zt)] + arc(0, Z_SPRING, rc, a_top, a_bot, 16)
    pts = [(sx * x, z) for x, z in pts]
    b.prism_xz("marble", pts if sx > 0 else list(reversed(pts)), *sorted((f, f + s * 0.02)))
    b.tube("marble", spiral(sx * (xo - 0.17), f + s * 0.03, Z_SPRING + 0.55, 0.12, 1.25, flip=sx), 0.018)


def build_column(b, x, y):
    z_base = 0.14
    b.box("marble", x - 0.16, x + 0.16, y - 0.16, y + 0.16, 0, z_base)
    b.lathe("marble", (x, y, z_base), [(0, 0), (R_COL + 0.04, 0), (R_COL + 0.04, 0.04),
                                       (R_COL + 0.02, 0.07), (R_COL, 0.09), (0, 0.09)])
    z0, z1 = z_base + 0.09, Z_COL_CAP[0]
    prof = [(0, 0)] + [(R_COL * (1 - 0.12 * (t / 8) ** 1.5), (z1 - z0) * t / 8) for t in range(9)] + [(0, z1 - z0)]
    b.lathe("marble", (x, y, z0), prof, segments=24)
    zc0, zc1 = Z_COL_CAP
    b.lathe("marble", (x, y, zc0), [(0, 0), (R_COL * 0.95, 0), (R_COL * 1.25, (zc1 - zc0) * 0.5),
                                    (R_COL * 1.3, (zc1 - zc0) * 0.6), (0, (zc1 - zc0) * 0.6)])
    b.box("marble", x - 0.16, x + 0.16, y - 0.16, y + 0.16, zc0 + (zc1 - zc0) * 0.6, zc1)
    for sx in (-1, 1):  # Ionic-like volutes (order is a guess)
        b.cylinder("marble", (x + sx * 0.13, y, zc0 + (zc1 - zc0) * 0.45), 0.05, 0.28, axis="Y", segments=16)


def build_lion(b, x, y, s):
    """Seated lion on a two-tier pedestal, simplified; faces away from the wall."""
    P = H_LION_PEDESTAL
    b.box("marble", x - 0.28, x + 0.28, y - 0.37, y + 0.37, 0, 0.09)
    b.box("marble", x - 0.24, x + 0.24, y - 0.33, y + 0.33, 0.09, P - 0.09)
    b.box("marble", x - 0.27, x + 0.27, y - 0.36, y + 0.36, P - 0.09, P)
    zm = P / 2
    b.ellipsoid("marble", (x, y + s * 0.335, zm), (0.14, 0.015, 0.12))      # relief cartouches
    for ex in (-1, 1):
        b.ellipsoid("marble", (x + ex * 0.245, y, zm), (0.015, 0.18, 0.12))
    b.ellipsoid("marble", (x, y - s * 0.10, P + 0.13), (0.17, 0.21, 0.15))  # haunches
    b.ellipsoid("marble", (x, y + s * 0.05, P + 0.30), (0.14, 0.12, 0.22))  # upright chest
    for lx in (-1, 1):
        b.cylinder("marble", (x + lx * 0.075, y + s * 0.14, P + 0.15), 0.04, 0.30)
        b.ellipsoid("marble", (x + lx * 0.075, y + s * 0.17, P + 0.025), (0.05, 0.07, 0.03))
    zh = P + H_LION - 0.14
    b.ellipsoid("marble", (x, y + s * 0.05, zh), (0.16, 0.10, 0.15))        # mane
    for k in range(14):  # dense, overlapping curls so the mane reads as one mass
        a = 2 * math.pi * k / 14
        b.ellipsoid("marble", (x + 0.12 * math.cos(a), y + s * (0.06 + 0.02 * (k % 2)), zh + 0.115 * math.sin(a)),
                    (0.065, 0.06, 0.065), segments=10)
    b.ellipsoid("marble", (x, y + s * 0.13, zh), (0.10, 0.09, 0.10))        # face
    b.ellipsoid("marble", (x, y + s * 0.22, zh - 0.04), (0.06, 0.05, 0.045))  # muzzle


def build_entablature(b):
    zc0, zc1 = Z_CORNICE
    mid = zc0 + 0.4 * (zc1 - zc0)
    xr = X_COL - 0.2  # the cornice breaks forward over the columns from here outward
    b.box("marble", -xr, xr, -(GF + 0.05), GF + 0.05, zc0, mid)
    b.box("marble", -xr, xr, -(GF + 0.12), GF + 0.12, mid, zc1)
    ry = GF + EST_COL_OFFSET + 0.24
    for sx in (-1, 1):
        b.box("marble", sx * xr, sx * X_CORNICE, -ry, ry, zc0, zc1)
    # central segmental rise of the cornice, standing 5 mm proud so no face is shared
    half = X_SCROLL - 0.25
    top = segment_curve(half, zc1, Z_CORNICE_CROWN)
    bottom = [(x, z - 0.12) for x, z in reversed(top)]
    b.prism_xz("marble", top + bottom, -(GF + 0.125), GF + 0.125)
    # broken pediment: S-curved raking cornice rising inward to a volute, small curl outside
    for sx in (-1, 1):
        n = 16
        top = []
        for i in range(n + 1):
            t = i / n
            x = X_CORNICE + (X_SCROLL + 0.05 - X_CORNICE) * t
            top.append((x, zc1 + 0.12 + (Z_SCROLL + 0.08 - zc1 - 0.12) * (3 * t * t - 2 * t ** 3)))
        bottom = [(x, max(zc1, z - 0.13)) for x, z in reversed(top)]
        pts = [(sx * x, z) for x, z in top + bottom]
        b.prism_xz("marble", pts if sx < 0 else list(reversed(pts)), -(GF + 0.05), GF + 0.05)
        b.cylinder("marble", (sx * X_SCROLL, 0, Z_SCROLL), 0.12, 2 * (GF + 0.07), axis="Y", segments=32)
        b.cylinder("marble", (sx * (X_CORNICE - 0.06), 0, zc1 + 0.12), 0.06, 2 * (GF + 0.07), axis="Y",
                   segments=20)
        for s in (-1, 1):
            y = s * (GF + 0.08)
            b.tube("marble", spiral(sx * X_SCROLL, y, Z_SCROLL, 0.10, 1.5, flip=-sx), 0.016)
            b.tube("marble", [(sx * (X_SCROLL - 0.08), y, Z_SCROLL + 0.05),
                              (sx * (X_SCROLL - 0.17), y, Z_SCROLL - 0.02),
                              (sx * (X_SCROLL - 0.22), y, zc1 + 0.08)], 0.03)


def build_attic(b):
    z0 = Z_CORNICE[1]
    z1 = Z_ATTIC_EAVE[0]
    xb, xt = X_SCROLL - 0.17, X_ATTIC_EAVE - 0.07
    side = [(xt + (xb - xt) * (1 - t / 10) ** 2, z0 + (z1 - z0) * t / 10) for t in range(11)]
    profile = [(-x, z) for x, z in side] + [(x, z) for x, z in reversed(side)]
    b.prism_xz("plaster", list(reversed(profile)), -EST_ATTIC_HALF_DEPTH, EST_ATTIC_HALF_DEPTH)
    ze0, ze1 = Z_ATTIC_EAVE
    b.box("marble", -X_ATTIC_EAVE, X_ATTIC_EAVE, -0.42, 0.42, ze0, ze1)
    top = segment_curve(X_EAVE_ARCH, ze1, Z_EAVE_CROWN)
    bottom = [(x, z - 0.15) for x, z in reversed(top)]
    b.prism_xz("marble", top + bottom, -0.425, 0.425)
    for sx in (-1, 1):
        r = R_CORNER_BALL
        zp = Z_CORNER_BALL_TOP - 2 * r
        b.box("marble", sx * X_CORNER_BALL - 0.10, sx * X_CORNER_BALL + 0.10, -0.10, 0.10, ze1, ze1 + 0.04)
        b.lathe("marble", (sx * X_CORNER_BALL, 0, ze1 + 0.04),
                [(0, 0), (0.07, 0), (0.04, (zp - ze1 - 0.04) * 0.6), (0.05, zp - ze1 - 0.04), (0, zp - ze1 - 0.04)])
        b.ellipsoid("marble", (sx * X_CORNER_BALL, 0, zp + r), (r, r, r))
    for s in (-1, 1):
        y = s * EST_ATTIC_HALF_DEPTH
        # facade cartouche and festoons
        shield = [(-0.22, 4.12), (0.22, 4.12), (0.20, 3.92), (0.0, 3.78), (-0.20, 3.92)]
        b.prism_xz("marble", shield, *sorted((y, y + s * 0.05)))
        for sx in (-1, 1):
            pts = [(sx * (0.25 + 0.5 * t), y + s * 0.06, 4.10 - 0.35 * math.sin(math.pi * t) - 0.25 * t)
                   for t in (i / 12 for i in range(13))]
            for i, p in enumerate(pts[1:-1], 1):  # leafy garland: overlapping flattened leaves
                rr = 0.035 + 0.02 * math.sin(math.pi * i / 12)
                b.ellipsoid("marble", p, (rr * 1.3, rr * 0.5, rr * 0.8), segments=10)
    # crowning cartouche, crown and finial ball
    zc0, zc1 = Z_CARTOUCHE
    w = 170 * K  # cartouche incl. side scrolls, x 2400-2570
    shape = [(-0.10, zc0), (0.10, zc0), (0.24, zc0 + 0.10), (w * 0.5, zc1 - 0.12), (0.20, zc1),
             (-0.20, zc1), (-w * 0.5, zc1 - 0.12), (-0.24, zc0 + 0.10)]
    shape[0], shape[1] = (-0.10, zc0 - 0.06), (0.10, zc0 - 0.06)  # seated into the eave arch
    b.prism_xz("marble", list(reversed(shape)), -0.14, 0.14)
    for s in (-1, 1):
        for sx in (-1, 1):
            b.tube("marble", spiral(sx * 0.30, s * 0.15, zc0 + 0.14, 0.09, 1.25, flip=sx), 0.016)
        b.ellipsoid("marble", (0, s * 0.15, (zc0 + zc1) / 2), (0.07, 0.03, 0.07), segments=12)
    zr0, zr1 = Z_CROWN
    b.lathe("marble", (0, 0, zr0), [(0, 0), (0.15, 0), (0.22, (zr1 - zr0) * 0.6), (0.18, zr1 - zr0), (0, zr1 - zr0)])
    for k in range(12):
        a = 2 * math.pi * k / 12
        b.ellipsoid("marble", (0.2 * math.cos(a), 0.2 * math.sin(a), zr0 + (zr1 - zr0) * 0.6),
                    (0.025, 0.025, (zr1 - zr0) * 0.55), segments=8)
    rb = (Z_TOP - zr1) / 2
    b.ellipsoid("marble", (0, 0, zr1 + rb), (R_TOP_BALL, R_TOP_BALL, rb))


def build_walls(b):
    t = EST_WALL_HALF_THICK
    zc = Z_COPING_BOTTOM
    tc = t + 0.06
    coping = [(-tc, zc), (tc, zc), (tc, zc + 0.06), (0, Z_WALL_TOP), (-tc, zc + 0.06)]  # thin pitched coping
    for sx in (-1, 1):
        x0, x1 = sx * X_BODY, sx * (X_BODY + EST_WALL_LENGTH)
        b.box("plaster", x0, x1, -t, t, 0, zc)
        b.box("marble", x0, x1 + sx * 0.03, -(t + 0.03), t + 0.03, 0, 0.22)  # plinth course (inferred)
        b.prism_yz("marble", coping, x0, x1 + sx * 0.05)
        for s in (-1, 1):
            y = s * t
            centres = [X_FIRST_PANEL + i * WALL_PANEL_PITCH for i in range(4)]
            strips = [X_BODY + 0.14] + [c + WALL_PANEL_PITCH / 2 for c in centres[:-1]] + \
                     [min(centres[-1] + WALL_PANEL_PITCH / 2, X_BODY + EST_WALL_LENGTH - 0.1)]
            for c in strips:
                b.box("marble", sx * (c - 0.085), sx * (c + 0.085), *sorted((y, y + s * 0.03)), 0.22, zc)
            for c in centres:
                build_wall_panel(b, sx * c, y, s)


WALL_PANEL_HALF = (161 * K, 106 * K)    # frame outer edge, x 2964-3286.5 and y 1669-1880.5 px
WALL_PANEL_Z = 1775                     # frame centre row (px)
WALL_FRAME_BAND = 0.02                  # moulding band height off the wall face (estimate)
WALL_FRAME_COVE = 6                     # sections across the cove
WALL_FRAME_ARC = 10                     # samples per traced corner arc
ORNAMENT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huayuanmen_ornament.json")
WALL_ORNAMENT = ("wall_ornament", 140, 1682)  # traced panel: left edge 140 px left of the frame centre
                                              # (ornament axis 3129 px, frame 3125 px, as drawn) and top row


def frame_outline(arcs, origin, scale, a, h):
    """Closed outline (panel-centred metres) of one moulding line from its traced upper-left
    corner: each arc resampled to WALL_FRAME_ARC points so every line has matching samples."""
    corner = []
    for arc in arcs:
        p = np.asarray(arc, dtype=float)
        t = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
        corner += [(np.interp(v, t, p[:, 0]), np.interp(v, t, p[:, 1]))
                   for v in np.linspace(0.0, t[-1], WALL_FRAME_ARC + 1)[:-1]]
    corner.append(tuple(arcs[-1][-1]))
    ul = [(-a + (x - origin[0]) * scale * K, h - (y - origin[1]) * scale * K) for x, y in corner]
    ur = [(-x, z) for x, z in reversed(ul)]
    return ul + ur + [(x, -z) for x, z in reversed(ur)] + [(x, -z) for x, z in reversed(ul)]


def build_wall_panel(b, cx, y, s):
    """Panel with the traced lobed-corner moulding (flat band, then a cove down to the field)
    and the traced cross-and-scroll ornament."""
    with open(ORNAMENT_JSON) as f:
        d = json.load(f)
    fr, zm = d["frame"], vz(WALL_PANEL_Z)
    line = {k: [(cx + x, zm + z) for x, z in frame_outline(fr[k], fr["origin"], d["album_scale"], *WALL_PANEL_HALF)]
            for k in ("outer", "band_inner", "cove_inner")}
    cove = []
    for i in range(1, WALL_FRAME_COVE):
        t = i / WALL_FRAME_COVE
        ring = [(p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
                for p, q in zip(line["band_inner"], line["cove_inner"])]
        cove.append((ring, y + s * WALL_FRAME_BAND * (1 - math.sqrt(1 - (1 - t) ** 2))))
    y_band, sink = y + s * WALL_FRAME_BAND, y - s * RELIEF_SINK
    # the cove's last ring sits just behind the face so it cuts the wall cleanly
    b.loft_xz("marble", [(line["outer"], sink), (line["outer"], y_band), (line["band_inner"], y_band), *cove,
                         (line["cove_inner"], y - s * 0.5 * RELIEF_SINK), (line["cove_inner"], sink)])
    name, left, top = WALL_ORNAMENT
    centre_px = AXIS_X - s * cx / K
    b.relief("marble", name, centre_px - left, top, y, s)


def build_ground(b):
    b.box("grass", -13, 13, -10, 10, -0.05, 0)
    # north: fan-shaped paving in concentric rings, then a straight path toward the bridge
    radii = [3.2, 2.4, 1.6]
    for i, r in enumerate(radii):
        pts = [(r * math.cos(a), GF + r * math.sin(a)) for a in (math.pi * k / 40 for k in range(41))]
        b.prism_xy("paving" if i % 2 == 0 else "pebble", pts, 0, 0.02 + 0.005 * i)
    b.box("pebble", -1.15, 1.15, GF + 3.0, 9.98, 0, 0.018)
    b.box("paving", -0.75, 0.75, GF + 3.0, 10, 0, 0.03)
    for sx in (-1, 1):
        b.box("paving", sx * 1.15, sx * 1.25, GF + 3.0, 10, 0, 0.035)
    b.box("paving", -X_BODY - 0.4, X_BODY + 0.4, GF, GF + 0.9, 0, 0.05)
    # south (not depicted): a plain paved approach, inferred
    b.box("paving", -X_BODY - 0.4, X_BODY + 0.4, -(GF + 0.9), -GF, 0, 0.05)
    b.box("paving", -0.9, 0.9, -10, -(GF + 0.9), 0, 0.03)


def find_shared_faces(col, min_overlap=0.005):
    """List visible flat faces of different parts that lie in the same plane and overlap.

    Coincident faces render as black patches in Cycles (each shadows the other), so the
    build must have none. Faces on the ground plane facing down are ignored.
    """
    buckets = {}
    for ob in col.objects:
        me = ob.data
        for poly in me.polygons:
            if poly.use_smooth or poly.area < 1e-5:
                continue
            n = poly.normal
            co = [me.vertices[i].co for i in poly.vertices]
            d = n.dot(co[0])
            if n.z < -0.99 and abs(d) < 1e-3:
                continue
            axis = max(range(3), key=lambda i: abs(n[i]))
            u, v = [i for i in range(3) if i != axis]
            bb = (min(c[u] for c in co), max(c[u] for c in co), min(c[v] for c in co), max(c[v] for c in co))
            key = (round(n.x, 3), round(n.y, 3), round(n.z, 3), round(d, 3))
            buckets.setdefault(key, []).append((ob.name, bb))
    hits = {}
    for key, faces in buckets.items():
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                (na, a), (nb, bb) = faces[i], faces[j]
                if na == nb:
                    continue
                if min(a[1], bb[1]) - max(a[0], bb[0]) > min_overlap and \
                        min(a[3], bb[3]) - max(a[2], bb[2]) > min_overlap:
                    hits.setdefault(tuple(sorted((na.rsplit(".", 1)[0], nb.rsplit(".", 1)[0]))), set()).add(key)
    return hits


def join_by_material(col):
    """One mesh per material keeps the glb small and the draw-call count low."""
    groups = {}
    for ob in list(col.objects):
        if ".relief." in ob.name:   # shared relief meshes stay instanced
            continue
        groups.setdefault(ob.data.materials[0].name, []).append(ob)
    for name, obs in groups.items():
        bm = bmesh.new()
        expected = 0
        for ob in obs:
            bm.from_mesh(ob.data)
            expected += len(ob.data.vertices)
        if len(bm.verts) != expected:
            raise RuntimeError(f"join {name}: {len(bm.verts)} verts, expected {expected}")
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(bpy.data.materials[name])
        for ob in obs:
            data = ob.data
            bpy.data.objects.remove(ob)
            bpy.data.meshes.remove(data)
        col.objects.link(bpy.data.objects.new(name, me))


# ---------------------------------------------------------------------------
# Scene, light, render, export
# ---------------------------------------------------------------------------
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    return scene


def setup_world_and_light(scene, azimuth_deg, elevation_deg):
    """Physical sky and one sun lamp sharing the same direction.

    Azimuth is a compass bearing from north (+Y) toward east (+X). Measured in Blender
    4.5: Nishita sun_rotation 0 puts the sun at +Y and 90 deg at +X, i.e. the same bearing.
    """
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    world = bpy.data.worlds.new("sky")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_disc = False
    sky.sun_elevation = el
    sky.sun_rotation = az
    sky.air_density = 1.2
    sky.dust_density = 1.5
    bg = nt.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.35
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 4.0
    sun.angle = math.radians(1.5)
    ob = bpy.data.objects.new("sun", sun)
    scene.collection.objects.link(ob)
    to_sun = Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))
    ob.rotation_euler = (-to_sun).to_track_quat("-Z", "Y").to_euler()


def setup_render(scene, samples, res_x, res_y):
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = res_x, res_y
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.render.image_settings.file_format = "WEBP"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.quality = 90


def render_view(scene, out_dir, name, location, target, lens):
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens
    cam_data.clip_end = 500
    cam = bpy.data.objects.new(name, cam_data)
    scene.collection.objects.link(cam)
    cam.location = location
    cam.rotation_euler = (Vector(target) - Vector(location)).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam
    path = os.path.join(out_dir, f"huayuanmen-{name}.webp")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    report_exposure(path)
    return path


def report_exposure(path):
    """Numeric exposure check: mean level and share of clipped pixels."""
    img = bpy.data.images.load(path)
    px = img.pixels[:]
    lum = [0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2] for i in range(0, len(px), 4 * 7)]
    n = len(lum)
    print(f"EXPOSURE {os.path.basename(path)}: mean={sum(lum) / n:.3f} "
          f"white={sum(v > 0.98 for v in lum) / n:.3%} black={sum(v < 0.02 for v in lum) / n:.3%}")
    bpy.data.images.remove(img)


def save_thumb(src, dst, width, height):
    img = bpy.data.images.load(src)
    img.scale(width, height)
    img.file_format = "WEBP"
    img.save(filepath=dst, quality=88)
    bpy.data.images.remove(img)


def export_glb(path, col):
    bpy.ops.object.select_all(action="DESELECT")
    for ob in col.objects:
        ob.select_set(True)
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_apply=True,
                              export_yup=True, export_draco_mesh_compression_enable=False,
                              export_cameras=False, export_lights=False)
    print(f"GLB {path}: {os.path.getsize(path) / 1e6:.2f} MB")


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=96)
    p.add_argument("--no-render", action="store_true")
    return p.parse_args(argv)


# Summer evening in Beijing: low sun in the west-north-west, the only time the north face is lit.
SUN_AZIMUTH, SUN_ELEVATION = 300, 20
RES = (1600, 1040)
THUMB = (720, 468)
VIEWS = {  # name: (camera location, target, lens mm); the documented north face looks to +Y
    "hero": ((-6.5, 11.5, 1.7), (0, 0, 2.4), 35),
    "front": ((0, 13, 2.6), (0, 0, 2.6), 50),
    "side": ((-12.6, 3.4, 2.5), (0, 0.3, 2.6), 50),  # from the west: the wall hides a pure profile
    "oblique": ((11, 13, 9.5), (0, 0, 1.6), 35),
}


def main():
    args = parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    scene = reset_scene()
    mats = make_materials()
    col = bpy.data.collections.new("huayuanmen")
    scene.collection.children.link(col)
    b = Builder(col, mats)
    for section, fn in (("body", build_body), ("doors", build_doors),
                        ("north", lambda bb: build_face(bb, 1)), ("south", lambda bb: build_face(bb, -1)),
                        ("entablature", build_entablature), ("attic", build_attic),
                        ("walls", build_walls), ("ground", build_ground)):
        b.section = section
        fn(b)
    print(f"PARTS {b.count}")
    shared = find_shared_faces(col)
    print(f"SHARED_FACES {sum(len(v) for v in shared.values())}")
    for (pa, pb), keys in sorted(shared.items()):
        for key in sorted(keys):
            print(f"  SHARED {pa} <-> {pb} plane={key}")
    join_by_material(col)
    export_glb(os.path.join(out, "huayuanmen.glb"), col)
    if args.no_render:
        return
    # render-only ground reaching the horizon (the glb keeps its small lawn)
    env = bpy.data.collections.new("environment")
    scene.collection.children.link(env)
    Builder(env, mats).box("grass", -200, 200, -200, 200, -0.06, -0.002)
    setup_world_and_light(scene, SUN_AZIMUTH, SUN_ELEVATION)
    setup_render(scene, args.samples, *RES)
    for name, (loc, target, lens) in VIEWS.items():
        path = render_view(scene, out, name, loc, target, lens)
        if name == "hero":
            save_thumb(path, os.path.join(out, "huayuanmen-thumb.webp"), *THUMB)


if __name__ == "__main__":
    main()
