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
from mathutils.geometry import interpolate_bezier

# ---------------------------------------------------------------------------
# Measurements (album_2.jpg pixels; gate axis x=2475, threshold centre line y=2064)
# ---------------------------------------------------------------------------
K = 0.00471  # m per px at the gate plane; horizon-line estimate, see module docstring
AXIS_X, BASE_Y = 2475, 2064


def hx(px_x, axis=AXIS_X):
    """Horizontal engraving pixel -> metres from the gate axis (absolute)."""
    return abs(px_x - axis) * K


def vz(px_y):
    """Vertical engraving pixel -> metres above the threshold line."""
    return (BASE_Y - px_y) * K


# Horizontal stations, measured (left half, mirrored; symmetry check passed within 6.2 %).
X_OPEN = hx(2294)        # door opening half width (frame inner line)
X_FRAME = hx(2232)       # door-frame pilaster outer edge (sunk panel 2248-2281, see FRAME_PANEL)
X_PIER = hx(2182)        # plain pier face outer edge
X_NARROW = hx(2154)      # narrow pilaster behind the column, outer edge
X_COL = hx(2129)         # column centre
R_COL = 50 * K / 2       # column shaft radius
X_BODY = hx(2056)        # gate body outer edge (side walls start here)
X_CORNICE = hx(2053)     # main cornice outer end
X_LION = hx(2105)        # lion pedestal centre
X_BLOCK = hx(2290)       # small stone block centre at the door jamb

# Vertical stations, measured.
Z_THRESHOLD = vz(2058)
Z_LEAF_TOP = vz(1640)
# Transom over the leaves, top down: (y top px, y bottom px, half-depth m). Fillet, a hatched
# (shaded, so set back) band, a bright projecting torus, a fascia, a vertically hatched frieze,
# a fillet and a plain band. Lines read off the engraving; depths are estimates.
TRANSOM = [(1598, 1601, 0.11), (1601, 1609, 0.09), (1609, 1615, 0.12), (1615, 1620, 0.10),
           (1620, 1628, 0.085), (1628, 1631, 0.10), (1631, 1640, 0.095)]
# Moulded frame round the tympanum carving, between the engraved lines at r 157 and 165 px.
R_TYMPANUM_FRAME = (157 * K, 165 * K)
Z_TRANSOM_TOP = vz(1598)
Z_COL_CAP = (vz(1535), vz(1490))
Z_ARCHITRAVE = (vz(1490), vz(1440))
Z_LEAF_BLOCK = (vz(1440), vz(1385))
Z_CORNICE = (vz(1380), vz(1337))
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

# Arch zone. The engraved archivolt lines were fitted as circles on both flanks at once: centre
# x 2477.2, y 1587.3. Half-widths below are taken about that centre, so the 2 px offset from
# AXIS_X drops out of the symmetric model.
ARCH_C_PX = 2477.2
Z_ARCH_C = vz(1587.3)
Z_ARCHIVOLT_FOOT = vz(1631)   # the legs stand on the impost cap
# Archivolt profile, inside to outside: (radius px, projection m). The radii are the engraved
# lines: inner fillet from the opening edge (X_OPEN) to 194, fascia 196-224, a torus between
# quirks at 225 and 240, outer fillet to 251, edge at 253. Projections are estimates.
ARCHIVOLT = [(X_OPEN / K, 0.035), (194, 0.035), (195.5, 0.05), (224, 0.05), (225, 0.044),
             (226.5, 0.05), (229, 0.062), (232, 0.067), (235, 0.062), (237.5, 0.052), (239.5, 0.044),
             (241.5, 0.055), (251, 0.055), (253, 0.042)]
# Impost under each leg, top down: (y top, y bottom, inner and outer half-width at the top, the
# same at the bottom, projection m); cap, two fasciae, a bell narrowing downward, the door-frame
# cap and a fillet. Half-widths in px about the arch centre; projections are estimates.
IMPOST = [(1631, 1640, 180, 253, 180, 253, 0.09), (1640, 1650, 185, 245, 185, 245, 0.07),
          (1650, 1660, 182, 247, 182, 247, 0.075), (1660, 1680, 181, 247, 194, 239, 0.05),
          (1680, 1690, 175, 251, 175, 251, 0.085), (1690, 1700, 181, 245, 181, 245, 0.06)]
Z_FRAME_TOP = vz(1700)
# Door-frame pilaster below the impost: a sunk panel x 2248-2281, y 1703-2005, standing on a
# plinth from the threshold up to y 2012 that is 6 px wider outward. The pier between the
# pilaster and the column is plain wall on the engraving.
FRAME_PANEL = ((2248, 2281), (2005, 1703))
Z_FRAME_PLINTH = vz(2012)
X_FRAME_PLINTH = hx(2226)
# Raised centre of the main cornice: flat out to 177 px from the axis, then a straight rake
# (slope 0.58, from both flanks) into an arc of radius 205 px over the crown. The bent part
# keeps a normal thickness of 37 px, i.e. about the flat cornice's 43 px measured vertically.
CORNICE_BREAK = 177 * K
CORNICE_RAKE = 0.58
CORNICE_R = 205 * K
CORNICE_T = 37 * K
# Spandrel panels: straight outer side 264.7 px from the arch centre (left 2211-2218, right
# 2742 mirrored), top y 1404, bottom y 1568, inner side on a 255.5 px arc just clear of the
# archivolt. Double rim: outer band 3 px, groove 1.5 px, inner band 2 px.
SPANDREL_X = 264.7 * K
Z_SPANDREL = (vz(1568), vz(1404))
R_SPANDREL = 255.5 * K
# The sprig carved in the field (huayuanmen_relief.py "spandrel", top-left corner of its box);
# moved by the arch centre's offset from AXIS_X, as the panel itself is placed about the arch centre.
SPANDREL_RELIEF_PX = (2209 - (ARCH_C_PX - AXIS_X), 1400)

# Attic. Its left and right edges were measured together (finial balls 2326/2640, eave ends
# 2274/2694, eave arch), which puts the upper part's axis at about x 2482, 7 px right of AXIS_X
# (the scan is slightly tilted); lines read on the left only are taken about ATTIC_AXIS_PX.
ATTIC_AXIS_PX = 2482
# Eave: the top line is flat at y 1134.5 and rises over the centre in a segmental arc to y 1085.
# Every lower line also runs flat and then arches, with its own crown; the break moves inward
# with depth (top line 105 px from the axis, underside 92 px). Profile lines top down as (px
# below the top line where flat, px below it at the crown, end half-width px): fluted band, cyma
# flaring out to the corona, corona, its soffit, cove, bottom fillet. The ends return the
# profile, so each line's front projection is taken equal to its measured overhang at the ends.
EAVE_TOP_PX, EAVE_CROWN_PX = 1134.5, 1085
EAVE_BREAK_PX = (105, 92)
EAVE_PROFILE = [(0, 0, 209), (8.5, 7.5, 209), (17.5, 13, 217), (21.5, 16, 217), (21.5, 16, 201),
                (35.5, 27, 193), (42.5, 36, 193)]
# Attic body: a field framed by a raised band about 19.5 px wide. The band's outer edge (the
# attic's outline against the sky) and its inner edge, left side, top down. At the foot the
# band passes under the outline and curls up into a volute: centre, the radii of the band's
# inner and outer edges at angles measured from straight down, turning outward and up (it ends
# on the right of the eye), and the round eye, ringed by a groove. At the top the band stops
# 97 px from the axis, where the field rises under the arched eave.
ATTIC_FACE_HALF_W_PX = ATTIC_AXIS_PX - 2292.7
ATTIC_OUTLINE = [(2292.7, 1177), (2292.7, 1200), (2292, 1223), (2288, 1250), (2280, 1275),
                 (2268, 1293), (2252, 1302), (2236, 1304), (2228, 1304)]
ATTIC_BAND_INNER = [(2310, 1197), (2311, 1223), (2307, 1250), (2299, 1275), (2287, 1295),
                    (2270, 1312), (2252, 1322), (2240, 1323)]
ATTIC_BAND_END_PX = 2385
ATTIC_VOLUTE = (2217.5, 1296)
ATTIC_VOLUTE_RADII = [(0, 7.75, 27.25), (45, 7.5, 22.4), (90, 7, 18.5), (180, 6.5, 14),
                      (270, 6, 11), (300, 5.8, 10)]
ATTIC_VOLUTE_EYE = 4.5
ATTIC_FIELD_SINK = 0.025      # field set back behind the band (estimate)
# Corner finials, 157.5 px either side of the axis, standing on the eave. Square block, bevel and
# plinth, then a round collar and neck, as (y px, half-width px) bottom up; ball radius 15 px
# centred at y 1093.
X_FINIAL = 157.5 * K
FINIAL_SQUARE = [(1137, 22), (1124, 22), (1121, 17), (1121, 16.5), (1116, 13)]
FINIAL_ROUND = [(1117, 12.5), (1114, 12.5), (1113, 9), (1106, 9), (1105, 7), (1100, 7)]
FINIAL_BALL = (1093, 15)
# Broken pediment over each column: a raking cornice that rises inward from a point on the main
# cornice and curls into a volute, over a sunk triangle. Measured on the left piece; the volute
# ring centres (2174.5 / 2781.5) put its axis at 2478. The cornice's lines from the top: the
# cyma's top (A) and bottom (B), the corona's bottom (C), a shaded cove down to D, a fillet down
# to E, then the sunk triangle, closed on the right by a stem under the volute. Neck lines as
# (x, y) px from the point inward; on the volute each line is a radius per angle (degrees as
# polar_path reads them) about PED_VOLUTE; the corona's face carries the volute disc.
PED_AXIS_PX = 2478
PED_VOLUTE = (2174.5, 1270)
PED_DISC_R, PED_EYE_RING_R, PED_ROSETTE_R = 19.5, 13, 9
PED_BASE_PX = 1338          # 1 px into the main cornice's top (1337)
PED_POINT_X = 2069
PED_NECK = {
    "A": [(2069, 1325), (2073, 1323), (2080, 1319), (2085, 1315), (2090, 1311), (2095, 1307), (2100, 1301),
          (2105, 1295), (2110, 1289), (2115, 1283), (2120, 1275), (2125, 1269), (2130, 1262), (2135, 1256),
          (2140, 1252), (2145, 1247), (2150, 1245), (2155, 1243)],
    "B": [(2075, 1322.5), (2085, 1321), (2090, 1318), (2095, 1314), (2100, 1310), (2105, 1304), (2110, 1298),
          (2115, 1291), (2120, 1284), (2125, 1277), (2130, 1270), (2135, 1265), (2140, 1259), (2145, 1254),
          (2150, 1250), (2155, 1248), (2160, 1246)],
    "C": [(2069, 1333), (2090, 1333), (2097, 1332), (2100, 1331), (2105, 1327), (2110, 1323), (2115, 1317),
          (2120, 1312), (2125, 1305), (2130, 1300), (2135, 1292), (2140, 1286), (2145, 1282), (2150, 1276),
          (2155, 1273)],
    "D": [(2113, 1338), (2115, 1335), (2120, 1332), (2125, 1328), (2130, 1323), (2135, 1316), (2140, 1308),
          (2145, 1300), (2150, 1294), (2155, 1289), (2160, 1285)],
    "E": [(2130, 1338), (2135, 1332), (2140, 1326), (2145, 1321), (2150, 1314), (2155, 1306), (2160, 1298),
          (2165, 1292), (2170, 1288)],
}
PED_SPIRAL = {"A": [(144.2, 33.3), (150, 32.5), (180, 29), (210, 27), (240, 26.5), (270, 26.5), (300, 26.5),
                    (330, 20)],
              "B": [(148.9, 28), (180, 24), (210, 22.5), (240, 22.5), (270, 20.5), (285, 19.5)]}
PED_STEM_PX = (2173, 2185)
# Faces set back from the cyma's front (estimates): corona, cove, fillet and stem, sunk triangle.
PED_SET_BACK = {"corona": 0.02, "cove": 0.05, "fillet": 0.08, "stem": 0.08, "triangle": 0.14}

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

    def lathe(self, mat, center, profile, segments=32, phase=0.0):
        """Revolve a closed (radius, z) profile (starting and ending on the axis) around Z.
        segments=4 with phase=pi/4 gives a square block whose half-width is radius / sqrt(2)."""
        bm = bmesh.new()
        rings = []
        for s in range(segments):
            a = 2 * math.pi * s / segments + phase
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

    def loft_xz(self, mat, rings, open_ends=False):
        """Closed ring solid swept through (outline, y) rings. Outlines are closed XZ loops
        sampled at matching points; the last ring joins back to the first. With open_ends the
        outlines are open paths and both ends are capped (archivolt legs)."""
        bm = bmesh.new()
        vs = [[bm.verts.new((x, y, z)) for x, z in path] for path, y in rings]
        n = len(vs[0])
        for a, b in zip(vs, vs[1:] + vs[:1]):
            for k in range(n - 1 if open_ends else n):
                j = (k + 1) % n
                bm.faces.new((a[k], a[j], b[j], b[k]))
        if open_ends:
            for k in (0, n - 1):
                bm.faces.new([v[k] for v in vs])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        return self._add(mat, bm, smooth=True)

    def stack(self, mat, rings):
        """Closed solid lofted top down through rings. Each ring is an elevation line (x, z),
        sampled left to right with the same count in every ring, run along the front at +y and
        back along the rear at -y; the first and last rings are capped with strips."""
        bm = bmesh.new()
        loops = [[bm.verts.new((x, d, z)) for x, z in pts] + [bm.verts.new((x, -d, z)) for x, z in reversed(pts)]
                 for pts, d in rings]
        n = len(loops[0])
        for a, c in zip(loops, loops[1:]):
            for k in range(n):
                bm.faces.new((a[k], a[(k + 1) % n], c[(k + 1) % n], c[k]))
        for loop in (loops[0], loops[-1]):
            for i in range(n // 2 - 1):
                bm.faces.new((loop[i], loop[i + 1], loop[n - 2 - i], loop[n - 1 - i]))
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


def cornice_line(frac, half_w, n=48):
    """Main-cornice line at a fraction of its height (0 underside, 1 top), sampled left to right
    from -half_w to half_w: flat outside, raked and arched over the centre (see CORNICE_*)."""
    z0, z1 = Z_CORNICE
    flat = z0 + frac * (z1 - z0)
    a = math.atan(CORNICE_RAKE)
    off = frac * CORNICE_T
    zc = z0 + CORNICE_RAKE * (CORNICE_BREAK - CORNICE_R * math.sin(a)) - CORNICE_R * math.cos(a)
    r = CORNICE_R + off
    x_t = r * math.sin(a)                                    # rake meets arc
    x_k = CORNICE_BREAK + (off / math.cos(a) - (flat - z0)) / CORNICE_RAKE   # rake meets flat

    def z(x):
        x = abs(x)
        if x <= x_t:
            return zc + math.sqrt(r * r - x * x)
        return max(flat, z0 + CORNICE_RAKE * (CORNICE_BREAK - x) + off / math.cos(a))
    xs = sorted({half_w * i / n for i in range(n + 1)} | {x_t, x_k})
    xs = [x for x in xs if x <= half_w]
    xs = [-x for x in reversed(xs) if x > 0] + xs
    return [(x, z(x)) for x in xs]


def arch_path(r, n=48):
    """Archivolt line of radius r: up the right leg from the impost, over the crown, down the left."""
    return [(r, Z_ARCHIVOLT_FOOT)] + arc(0, Z_ARCH_C, r, 0, math.pi, n) + [(-r, Z_ARCHIVOLT_FOOT)]


def spandrel_outline(inset):
    """Right-hand spandrel panel outline (x > 0), shrunk by inset."""
    x0, r = SPANDREL_X - inset, R_SPANDREL + inset
    z0, z1 = Z_SPANDREL[0] + inset, Z_SPANDREL[1] - inset
    a_top = math.asin((z1 - Z_ARCH_C) / r)
    if r > x0 and Z_ARCH_C + math.sqrt(r * r - x0 * x0) > z0:
        # Deep insets: the arc meets the straight side above the panel foot, so the outline ends in
        # a point there (left as a rectangle foot, the arc would cross the side and the polygon would
        # turn inside out, which made the inner rim's boolean cut fail and fill the whole field).
        return [(x0, z1)] + arc(0, Z_ARCH_C, r, a_top, math.acos(x0 / r), 24)
    return [(x0, z0), (x0, z1)] + arc(0, Z_ARCH_C, r, a_top, math.asin((z0 - Z_ARCH_C) / r), 24)


def eave_line(f, a, half_w, n=24):
    """Attic eave line f px below the top line where flat and a px below it at the crown,
    sampled left to right from -half_w to half_w (see EAVE_*); always n + 3 points."""
    z_flat, z_crown = vz(EAVE_TOP_PX + f), vz(EAVE_CROWN_PX + a)
    b0, b1 = EAVE_BREAK_PX
    xk = (b0 + (b1 - b0) * f / EAVE_PROFILE[-1][0]) * K
    rise = z_crown - z_flat
    r = (xk * xk + rise * rise) / (2 * rise)
    ak = math.asin(xk / r)
    arch = [(r * math.sin(t), z_crown - r + r * math.cos(t)) for t in (ak * (2 * i / n - 1) for i in range(n + 1))]
    return [(-half_w, z_flat)] + arch + [(half_w, z_flat)]


def smooth_path(points, n=6):
    """Smooth curve through 2D points (Blender's auto-handle Bezier), n samples per span."""
    cu = bpy.data.curves.new("path", "CURVE")
    bez = cu.splines.new("BEZIER").bezier_points
    bez.add(len(points) - 1)
    for bp, (x, y) in zip(bez, points):
        bp.co = (x, y, 0)
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    out = []
    for p, q in zip(bez, bez[1:]):
        out += [(v.x, v.y) for v in interpolate_bezier(p.co, p.handle_right, q.handle_left, q.co, n + 1)[:-1]]
    bpy.data.curves.remove(cu)
    return out + [tuple(points[-1])]


def offset_px(pts, d):
    """Polyline in image axes (y down) moved d px to the left of its direction of travel."""
    p = np.asarray(pts, float)
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    return [tuple(q) for q in p + d * np.stack([t[:, 1], -t[:, 0]], 1)]


def polar_path(centre, radii, a0, a1, grow=0.0, step=10):
    """Album px points round centre from angle a0 to a1 degrees (0 straight down, 90 to the left,
    180 up), the radius interpolated in radii [(angle, r px)] (or a constant) and grown by grow px."""
    cx, cy = centre
    angs, rs = zip(*radii) if isinstance(radii, list) else ((0,), (radii,))
    n = max(1, round(abs(a1 - a0) / step))
    out = []
    for a in (a0 + (a1 - a0) * i / n for i in range(n + 1)):
        r = float(np.interp(a, angs, rs)) + grow
        out.append((cx - r * math.sin(math.radians(a)), cy + r * math.cos(math.radians(a))))
    return out


def volute_edge(k, a0, a1, grow=0.0, step=10):
    """Attic volute: band edge k (1 inner, 2 outer; see ATTIC_VOLUTE_RADII) from angle a0 to a1
    degrees, radius grown by grow px, as album px points."""
    return polar_path(ATTIC_VOLUTE, [(row[0], row[k]) for row in ATTIC_VOLUTE_RADII], a0, a1, grow, step)


def attic_xz(pts_px, sx=-1):
    """Album px about the attic axis -> model (x, z) on side sx; the right side is the mirror."""
    return [(sx * abs(x - ATTIC_AXIS_PX) * K, vz(y)) for x, y in pts_px]


def pediment_outlines():
    """Album px outlines of the left broken-pediment piece's layers (see PED_NECK), front to back.
    A layer's top edge is the line above it raised 1 px, so it starts inside the layer in front
    and no two layers share a face."""
    def neck(k, up=0.0):
        return smooth_path([(x, y - up) for x, y in PED_NECK[k]], n=3)

    disc = lambda a0, a1, r=PED_DISC_R - 1: polar_path(PED_VOLUTE, r, a0, a1)
    # Each lower line runs into the disc radially, at the angle of its last point.
    meet = {k: math.degrees(math.atan2(PED_VOLUTE[0] - PED_NECK[k][-1][0], PED_NECK[k][-1][1] - PED_VOLUTE[1]))
            for k in "CDE"}
    base, x_pt = PED_BASE_PX, PED_POINT_X
    a_end, b_end = PED_SPIRAL["A"][-1][0], PED_SPIRAL["B"][-1][0]
    cyma = (neck("A") + polar_path(PED_VOLUTE, PED_SPIRAL["A"], PED_SPIRAL["A"][0][0], a_end)[1:]
            + disc(a_end, b_end, PED_DISC_R)
            + polar_path(PED_VOLUTE, PED_SPIRAL["B"], b_end, PED_SPIRAL["B"][0][0])[1:] + neck("B")[::-1])
    corona = (neck("B", 1)
              + polar_path(PED_VOLUTE, PED_SPIRAL["B"], PED_SPIRAL["B"][0][0], b_end, grow=1)[1:]
              + disc(b_end, 360 + meet["C"], PED_DISC_R) + neck("C")[::-1])
    cove = (neck("C", 1) + disc(meet["C"], meet["D"]) + neck("D")[::-1]
            + [(x_pt, base)])
    fillet = neck("D", 1) + disc(meet["D"], meet["E"]) + neck("E")[::-1]
    x_stem = PED_STEM_PX[1] - 1   # just inside the stem's right side
    a_stem = -math.degrees(math.asin((x_stem - PED_VOLUTE[0]) / (PED_DISC_R - 1)))
    triangle = neck("E", 1) + disc(meet["E"], a_stem) + [(x_stem, base)]
    return {"cyma": cyma, "corona": corona, "cove": cove, "fillet": fillet, "triangle": triangle}


def ped_xz(pts_px, sx=-1):
    """Album px about the pediment axis -> model (x, z) on side sx; the right side is the mirror."""
    return [(sx * abs(x - PED_AXIS_PX) * K, vz(y)) for x, y in pts_px]


# ---------------------------------------------------------------------------
# Traced relief panels
# ---------------------------------------------------------------------------
RELIEF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huayuanmen_relief")
RELIEF_LIFT = 0.003     # lowest carved level stands this far proud of its face (no shared faces)
RELIEF_SINK = 0.004     # the panel's outer edge dips this far into the face, so no gap shows
RELIEF_DECIMATE = 0.23  # share of triangles kept; flat ground and gentle slopes go first (keeps the glb under 10 MB)
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
    mod.ratio = meta.get("decimate", RELIEF_DECIMATE)
    bpy.context.view_layer.update()
    out = bpy.data.meshes.new_from_object(tmp.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    bpy.data.objects.remove(tmp, do_unlink=True)
    bpy.data.meshes.remove(me)
    drop_hidden_ground(out)
    out.name = f"relief.{name}"
    out.shade_smooth()
    out.set_sharp_from_angle(angle=RELIEF_SHARP)   # keep carved walls crisp: no smoothing across a cliff
    out.materials.append(material)
    return out


def drop_hidden_ground(me):
    """Delete the faces lying wholly on the sunk ground (-RELIEF_SINK): they sit behind the face
    the panel is set on, so no view sees them, and they are about an eighth of a decimated panel's
    triangles. Done after decimation, so the carved forms keep the triangles it gave them; pocket
    floors (recess) lie deeper and stay."""
    bm = bmesh.new()
    bm.from_mesh(me)
    hidden = [f for f in bm.faces if all(abs(v.co.y + RELIEF_SINK) < 1e-5 for v in f.verts)]
    bmesh.ops.delete(bm, geom=hidden, context="FACES")
    bm.to_mesh(me)
    bm.free()


# ---------------------------------------------------------------------------
# Building parts
# ---------------------------------------------------------------------------
def build_body(b):
    """Gate body: two piers and the arch spandrel, through the full depth."""
    top = Z_CORNICE[0]
    for sx in (-1, 1):
        b.prism_xz("plaster", [(sx * X_OPEN, 0), (sx * X_BODY, 0), (sx * X_BODY, top), (sx * X_OPEN, top)],
                   -GF, GF)
    # the head fills the arch up to the underside of the raised cornice
    head = cornice_line(0, X_OPEN) + arc(0, Z_ARCH_C, X_OPEN, 0, math.pi, 48)[1:-1]
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
    for y0, y1, d in TRANSOM:
        b.box("marble", -X_OPEN, X_OPEN, -d, d, vz(y1), vz(y0))
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
    # The plate reaches 2 cm into the masonry all round, so no daylight shows at its edge.
    rp = X_OPEN + 0.02
    b.prism_xz("plaster", [(-rp, Z_TRANSOM_TOP), (rp, Z_TRANSOM_TOP)] + arc(0, Z_ARCH_C, rp, 0, math.pi, 48),
               -0.012, 0.012)
    r0, r1 = R_TYMPANUM_FRAME
    ring = [(r1, Z_TRANSOM_TOP)] + arc(0, Z_ARCH_C, r1, 0, math.pi, 48) + [(-r1, Z_TRANSOM_TOP), (-r0, Z_TRANSOM_TOP)] + \
        arc(0, Z_ARCH_C, r0, math.pi, 0, 48) + [(r0, Z_TRANSOM_TOP)]
    for s in (-1, 1):
        b.relief("marble", "tympanum", 2299, 1422, s * 0.012, s)
        b.prism_xz("marble", ring, *sorted((s * 0.008, s * 0.03)))


def build_face(b, s):
    """Pilasters, column, archivolt, keystone and statuary on one face (s=+1 north, -1 south)."""
    f = s * GF

    def proj(d):
        return sorted((f, f + s * d))

    for sx in (-1, 1):
        # door-frame pilaster and the impost on which the archivolt leg stands
        b.box("marble", sx * X_OPEN, sx * X_FRAME_PLINTH, *proj(0.06), 0, Z_FRAME_PLINTH)
        pil = b.box("marble", sx * X_OPEN, sx * X_FRAME, *proj(0.04), Z_FRAME_PLINTH, Z_FRAME_TOP)
        (px0, px1), (py0, py1) = FRAME_PANEL
        b.cut_xz(pil, "marble", [(sx * hx(px0), vz(py0)), (sx * hx(px1), vz(py0)), (sx * hx(px1), vz(py1)),
                                 (sx * hx(px0), vz(py1))], *sorted((f + s * 0.025, f + s * 0.06)))
        for y0, y1, i0, o0, i1, o1, d in IMPOST:
            pts = [(sx * i1 * K, vz(y1)), (sx * o1 * K, vz(y1)), (sx * o0 * K, vz(y0)), (sx * i0 * K, vz(y0))]
            b.prism_xz("marble", pts if sx > 0 else pts[::-1], *proj(d))
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
    # archivolt: the measured profile swept round the arch; its back sits 1 cm inside the wall
    rings = [(arch_path(r * K), f + s * d) for r, d in ARCHIVOLT]
    rings += [(arch_path(ARCHIVOLT[-1][0] * K), f - s * 0.01), (arch_path(ARCHIVOLT[0][0] * K + 0.004), f - s * 0.01)]
    b.loft_xz("marble", rings, open_ends=True)
    build_keystone(b, s, f)


def build_keystone(b, s, f):
    """Keystone stack on the crown, read top down off the engraving (y px, half-widths px):
    flared bowl 1270-1300 (51 -> 26), gadrooned bulb 1301-1345 (30), round plinth 1345-1360 (32),
    console with a diamond 1360-1387 (25 -> 17), ring 1387-1393 (18), drop 1393-1405 (10).
    The turned parts are centred on the wall face; projections follow from their radii."""
    def y(d):
        return f + s * d
    b.prism_xz("marble", [(-25 * K, vz(1360)), (25 * K, vz(1360)), (17 * K, vz(1387)), (-17 * K, vz(1387))],
               *sorted((y(-0.01), y(0.11))))
    b.prism_xz("marble", [(0, vz(1364)), (10 * K, vz(1373.5)), (0, vz(1383)), (-10 * K, vz(1373.5))],
               *sorted((y(0.105), y(0.122))))
    b.lathe("marble", (0, y(0.03), vz(1393)), [(0, 0), (16 * K, 0), (18 * K, 3 * K), (16 * K, 6 * K), (0, 6 * K)])
    b.lathe("marble", (0, y(0.05), vz(1405)),
            [(0, 0), (3 * K, 2 * K), (8 * K, 6 * K), (10 * K, 9 * K), (9 * K, 12 * K), (0, 12 * K)])
    b.lathe("marble", (0, f, vz(1360)),
            [(0, 0), (28 * K, 0), (32 * K, 4 * K), (32 * K, 10 * K), (29 * K, 15 * K), (0, 15 * K)], segments=48)
    b.lathe("marble", (0, f, vz(1345)),
            [(0, 0), (22 * K, 0), (26 * K, 6 * K), (28 * K, 16 * K), (28 * K, 26 * K), (26 * K, 36 * K),
             (24 * K, 42 * K), (23 * K, 44 * K), (0, 44 * K)], segments=48)
    for k in range(14):  # gadroons
        a = 2 * math.pi * (k + 0.5) / 14
        b.ellipsoid("marble", (26 * K * math.cos(a), f + 26 * K * math.sin(a), vz(1323)), (0.021, 0.021, 20 * K),
                    segments=12)
    b.lathe("marble", (0, f, vz(1300)),
            [(0, 0), (26 * K, 0), (28 * K, 4 * K), (34 * K, 10 * K), (44 * K, 18 * K), (50 * K, 22 * K),
             (51 * K, 26 * K), (51 * K, 30 * K), (0, 30 * K)], segments=48)


def build_spandrel(b, sx, s, f):
    """Spandrel panel with a double rim (SPANDREL_*); the field is the wall face, so it reads sunk."""
    def xz(pts):
        p = [(sx * x, z) for x, z in pts]
        return p if sx > 0 else p[::-1]
    for i0, i1, h, back in ((0, 3, 0.025, 0.005), (4.5, 6.5, 0.018, 0.007)):
        rim = b.prism_xz("marble", xz(spandrel_outline(i0 * K)), *sorted((f - s * back, f + s * h)))
        b.cut_xz(rim, "marble", xz(spandrel_outline(i1 * K)), *sorted((f - s * 0.01, f + s * (h + 0.01))))
    x0, y0 = SPANDREL_RELIEF_PX
    if sx * s > 0:   # the drawing is the left spandrel as the viewer sees it
        b.relief("marble", "spandrel", x0, y0, f, s)
    else:
        b.relief("marble", "spandrel", 2 * AXIS_X - x0, y0, f, s, mirror=True)


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
    xr = X_COL - 0.2  # the cornice breaks forward over the columns from here outward
    # Profile bottom up, as the engraving's bands read near the break: a set-back fillet (dark
    # band under the fascia), the fascia, a set-back cove (dark band under the corona), the corona.
    layers = [(0, 0.28, 0.03), (0.28, 0.58, 0.07), (0.58, 0.72, 0.09), (0.72, 1, 0.12)]
    for t0, t1, d in layers:
        b.prism_xz("marble", cornice_line(t0, xr) + cornice_line(t1, xr)[::-1], -(GF + d), GF + d)
    ry = GF + EST_COL_OFFSET + 0.24
    for sx in (-1, 1):
        b.box("marble", sx * xr, sx * X_CORNICE, -ry, ry, zc0, zc1)
    build_pediment(b, ry - 0.10)


def build_pediment(b, y_front):
    """Broken pediment over each column (PED_*), both faces; y_front is the cyma's front."""
    outlines = pediment_outlines()
    cy = PED_VOLUTE[1]
    for sx in (-1, 1):
        for layer, pts in outlines.items():
            y = y_front - PED_SET_BACK.get(layer, 0)
            b.prism_xz("marble", ped_xz(pts, sx), -y, y)
        x0, x1 = PED_STEM_PX
        y = y_front - PED_SET_BACK["stem"]
        b.prism_xz("marble", ped_xz([(x0, cy), (x1, cy), (x1, PED_BASE_PX), (x0, PED_BASE_PX)], sx), -y, y)
        # On the disc: a ring round the eye and a whirled rosette (eight petals) on a low boss.
        yd = y_front - PED_SET_BACK["corona"]
        (xc, zc), = ped_xz([PED_VOLUTE], sx)
        rb = PED_ROSETTE_R * K
        for s in (-1, 1):
            ring = [(x, s * yd, z) for x, z in ped_xz(polar_path(PED_VOLUTE, PED_EYE_RING_R, 0, 360, step=10), sx)]
            b.tube("marble", ring, 1.2 * K, closed=True)
            b.ellipsoid("marble", (xc, s * yd, zc), (rb, 0.012, rb), segments=16)
            for k in range(8):
                petal = []
                for i in range(7):
                    r = 1.5 + 7 * i / 6
                    a = 45 * k + 70 * i / 6
                    (x, z), = ped_xz(polar_path(PED_VOLUTE, r, a, a, step=1)[:1], sx)
                    petal.append((x, s * (yd + 0.012 * math.sqrt(max(0.0, 1 - (r / PED_ROSETTE_R) ** 2)) + 0.002), z))
                b.tube("marble", petal, 1.0 * K)


def build_attic(b):
    d = EST_ATTIC_HALF_DEPTH
    face_w = ATTIC_FACE_HALF_W_PX
    # Eave: one solid lofted down through the profile lines.
    b.stack("marble", [(eave_line(f, a, w * K), d + (w - face_w) * K) for f, a, w in EAVE_PROFILE])
    # Field: the attic's whole outline (inset 0.5 px behind the band's edges), its top buried in
    # the eave's bottom fillet and its foot in the main cornice.
    left = attic_xz(offset_px(smooth_path(ATTIC_OUTLINE), 0.5) + volute_edge(2, 300, 45, grow=-0.5))
    x_foot = -left[-1][0]
    top = eave_line(EAVE_PROFILE[-1][0] - 3.5, EAVE_PROFILE[-1][1] - 4, (face_w - 0.5) * K)
    right = [(-x, z) for x, z in left]
    field = top + right + cornice_line(0.5, x_foot)[::-1] + left[::-1]
    b.prism_xz("marble", field, -(d - ATTIC_FIELD_SINK), d - ATTIC_FIELD_SINK)
    # Cartouche carved on the field (huayuanmen_attic.json), both faces; the relief is centred on
    # ATTIC_AXIS_PX, so its corner shifts by the attic's offset from AXIS_X.
    x0, y0 = relief_meta("attic")["box"][:2]
    for s in (-1, 1):
        b.relief("marble", "attic", x0 - (ATTIC_AXIS_PX - AXIS_X), y0, s * (d - ATTIC_FIELD_SINK), s)
    # Raised band round the field, ending in the volute; its top runs 3 px up into the eave.
    y_top = ATTIC_OUTLINE[0][1] - 3
    band = ([(ATTIC_BAND_END_PX, y_top), (ATTIC_OUTLINE[0][0], y_top)] + smooth_path(ATTIC_OUTLINE)
            + volute_edge(1, 0, 300) + volute_edge(2, 300, 0) + smooth_path(ATTIC_BAND_INNER)[::-1]
            + [(ATTIC_BAND_END_PX, ATTIC_BAND_INNER[0][1])])
    for sx in (-1, 1):
        pts = attic_xz(band, sx)
        b.prism_xz("marble", pts if sx < 0 else pts[::-1], -d, d)
        (ex, ez), = attic_xz([ATTIC_VOLUTE], sx)
        b.cylinder("marble", (ex, 0, ez), ATTIC_VOLUTE_EYE * K, 2 * d - 0.006, axis="Y", segments=20)
    # Corner finials.
    z0, z1 = vz(FINIAL_SQUARE[0][0]), vz(FINIAL_ROUND[0][0])
    for sx in (-1, 1):
        x = sx * X_FINIAL
        b.lathe("marble", (x, 0, z0), [(0, 0)] + [(h * K * math.sqrt(2), vz(y) - z0) for y, h in FINIAL_SQUARE]
                + [(0, vz(FINIAL_SQUARE[-1][0]) - z0)], segments=4, phase=math.pi / 4)
        b.lathe("marble", (x, 0, z1), [(0, 0)] + [(h * K, vz(y) - z1) for y, h in FINIAL_ROUND]
                + [(0, vz(FINIAL_ROUND[-1][0]) - z1)], segments=24)
        yb, rb = FINIAL_BALL
        b.ellipsoid("marble", (x, 0, vz(yb)), (rb * K,) * 3)
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
                              export_cameras=False, export_lights=False,
                              # No material has a texture, so UVs only add weight (about 0.6 MB).
                              export_texcoords=False)
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
