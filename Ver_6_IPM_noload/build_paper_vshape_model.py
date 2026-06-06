import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import femm


MU0 = 4.0 * math.pi * 1e-7
CORE_MATERIAL_NAME = "350_50A"
CORE_BH_DAT_DEFAULT = "350_50A.dat"
CORE_LAMINATION_THICKNESS_MM = 0.5
CORE_LAMINATION_FILL_FACTOR = 0.95
CORE_LAMINATION_TYPE = 0  # FEMM: 0 = laminated in-plane / not x-y laminated.
SECTOR_BOUNDARY_KIND_DEFAULT = "anti-periodic"
SECTOR_START_DEG_DEFAULT = 0.0
SECTOR_SPAN_DEG_DEFAULT = 60.0

#================================================================================
STATOR_OUTER_RADIUS_MM_DEFAULT = 67.5
SHAFT_RADIUS_MM_DEFAULT = 14.0
CURRENT_RMS_A_DEFAULT = 17.6
RELIEF_FILLET_MM_DEFAULT = 0.0
RED_RELIEF_ANGLE_A_DEG_DEFAULT = 50     #90 - X
V_INCLUDED_ANGLE_DEG_DEFAULT = 145
MAGNET_RADIAL_SHIFT_MM_DEFAULT = -1.16      #base 0.34, giảm là nam châm chạy xuống, wb1 = base + hiện tại
STATOR_ROTATION_DEG_DEFAULT = 45
ROTOR_ROTATION_DEG_DEFAULT = -90
TURNS_PER_LAYER_DEFAULT = 8
SECTOR_OUTPUT_DEFAULT = "paper_ipm_vshape_1over6.FEM"
FULL_OUTPUT_DEFAULT = "base_V_full_model.FEM"

# Default FEMM export is a 1/6 motor sector from 0 to 60 mechanical degrees.
# Keep stator_rotation=45 deg and rotor_rotation=-90 deg for the checked layout.
# A 60 deg sector is one pole pitch, so anti-periodic is the usual symmetry
# boundary. Change SECTOR_BOUNDARY_KIND_DEFAULT to "periodic" if needed.

# Double-layer 36-slot/6-pole lap winding from the provided signed winding
# diagram. Slot 1 is the slot just above the +X axis when stator_rotation=45.
# A/B/C map to Ia/Ib/Ic; a leading "-" means negative turns.
WINDING_UPPER_SEQUENCE = (
    "A", "A", "-C", "-C", "B", "B",
    "-A", "-A", "C", "C", "-B", "-B",
    "A", "A", "-C", "-C", "B", "B",
    "-A", "-A", "C", "C", "-B", "-B",
    "A", "A", "-C", "-C", "B", "B",
    "-A", "-A", "C", "C", "-B", "-B",
)
WINDING_LOWER_SEQUENCE = WINDING_UPPER_SEQUENCE[1:] + WINDING_UPPER_SEQUENCE[:1]
PHASE_TO_CIRCUIT = {"A": "Ia", "B": "Ib", "C": "Ic"}


@dataclass(frozen=True)
class MotorSpec:
    name: str
    rated_power_kw: float
    rated_current_a: float
    rated_speed_rpm: float
    stator_inner_radius: float
    rotor_outer_radius: float
    poles: int
    slots: int
    stack_length: float
    slot_opening_span: float
    slot_span: float
    slot_top_radius: float
    slot_bottom_radius: float
    bridge_width_1: float
    bridge_length_1: float
    bridge_width_2: float
    bridge_length_2: float
    magnet_width: float
    magnet_thickness: float
    magnet_material: str
    magnet_remanence_t: float
    magnet_coercivity_ka_per_m: float


PAPER_SPECS = {
    "vshape": MotorSpec(
        name="vshape",
        rated_power_kw=11.0,
        rated_current_a=17.6,
        rated_speed_rpm=8200,
        stator_inner_radius=40.0,
        rotor_outer_radius=38.8,
        poles=6,
        slots=36,
        stack_length=80.0,
        slot_opening_span=0.05,
        slot_span=0.105,
        slot_top_radius=40.8,
        slot_bottom_radius=60.0,
        bridge_width_1=1.5,
        bridge_length_1=3.6,
        bridge_width_2=1.5,
        bridge_length_2=3.8,
        magnet_width=14.4,
        magnet_thickness=4.0,
        magnet_material="N40UH",
        magnet_remanence_t=1.26,
        magnet_coercivity_ka_per_m=955.0,
    ),
}


PAPER_SOURCE_NOTE = (
    "Geometry values copied from Table I of the paper. "
    "This script draws only the information that is explicit enough to model."
)

# ASSUMED / APPROXIMATE PARTS:
# These are intentionally kept in code because the paper does not give enough
# detail to recreate the exact FEM geometry. Edit these first when checking.
ASSUMPTIONS = (
    "stator_outer_radius: not listed in the paper; default exposed as STATOR_OUTER_RADIUS_MM_DEFAULT.",
    "shaft_radius: not listed in the paper; default exposed as SHAFT_RADIUS_MM_DEFAULT.",
    "stator slot: paper gives equivalent slot radii/spans; this draws that equivalent slot, not the original pear-shaped slot.",
    "rotor bridges: Table I gives bridge width/length, but not all local corner/fillet geometry; bridges are represented indirectly by magnet placement clearance.",
    "V original magnets: drawn from wb2, hb1, lm and hm; V angle and hb1 bridge-line angle are adjustable assumptions.",
    "winding layout: double-layer lap winding is approximated from the supplied screenshot; turns-per-layer defaults to 8.",
    "core material: uses 350_50A.dat B-H curve with in-plane lamination, 0.5 mm sheet thickness, 0.95 fill factor.",
    "magnet material: uses Table I Hc value; Br is noted in Table I but FEMM material call uses Hc directly.",
)


def model_note_text(spec, stator_outer_radius, shaft_radius, turns_per_layer, v_included_angle_deg):
    lines = [
        PAPER_SOURCE_NOTE,
        "",
        f"Topology: {spec.name}",
        f"Explicit Table I values: Rs={spec.stator_inner_radius:g} mm, Rr={spec.rotor_outer_radius:g} mm, "
        f"poles={spec.poles}, slots={spec.slots}, La={spec.stack_length:g} mm.",
        f"Generated values: stator_outer_radius={stator_outer_radius:g} mm, "
        f"shaft_radius={shaft_radius:g} mm, turns_per_layer={turns_per_layer}.",
    ]
    if spec.name == "vshape":
        lines.append(f"Generated V-shape value: v_included_angle_deg={v_included_angle_deg:g}.")
    lines.extend(["", "Assumptions / self-made parts:"])
    lines.extend(f"- {item}" for item in ASSUMPTIONS)
    return "\n".join(lines)


def polar(radius, angle_rad):
    return radius * math.cos(angle_rad), radius * math.sin(angle_rad)


def local_to_xy(theta, radial, tangential):
    er_x, er_y = math.cos(theta), math.sin(theta)
    et_x, et_y = -math.sin(theta), math.cos(theta)
    return radial * er_x + tangential * et_x, radial * er_y + tangential * et_y


def add_segment(p1, p2):
    femm.mi_addnode(*p1)
    femm.mi_addnode(*p2)
    femm.mi_addsegment(*p1, *p2)


def add_arc(radius, start_angle, end_angle, maxseg=2.5):
    if end_angle < start_angle:
        end_angle += 2.0 * math.pi
    p1 = polar(radius, start_angle)
    p2 = polar(radius, end_angle)
    femm.mi_addnode(*p1)
    femm.mi_addnode(*p2)
    femm.mi_addarc(*p1, *p2, math.degrees(end_angle - start_angle), maxseg)


def add_circle(radius, maxseg=2.5, boundary=None, group=0):
    for idx in range(4):
        a0 = idx * math.pi / 2.0
        a1 = (idx + 1) * math.pi / 2.0
        add_arc(radius, a0, a1, maxseg)
        if boundary:
            mid = (a0 + a1) / 2.0
            x, y = polar(radius, mid)
            femm.mi_selectarcsegment(x, y)
            femm.mi_setarcsegmentprop(maxseg, boundary, 0, group)
            femm.mi_clearselected()


def add_sector_arc(radius, start_angle, end_angle, maxseg=2.5, boundary=None, group=0):
    add_arc(radius, start_angle, end_angle, maxseg=maxseg)
    if boundary:
        mid = (start_angle + end_angle) / 2.0
        x, y = polar(radius, mid)
        femm.mi_selectarcsegment(x, y)
        femm.mi_setarcsegmentprop(maxseg, boundary, 0, group)
        femm.mi_clearselected()


def add_radial_segment(radius_start, radius_end, angle_rad, boundary=None, group=0):
    p1 = polar(radius_start, angle_rad)
    p2 = polar(radius_end, angle_rad)
    add_segment(p1, p2)
    if boundary:
        mx, my = polar((radius_start + radius_end) / 2.0, angle_rad)
        femm.mi_selectsegment(mx, my)
        femm.mi_setsegmentprop(boundary, 0, 1, 0, group)
        femm.mi_clearselected()


def angle_in_sector(angle_rad, sector_start_rad, sector_end_rad, eps=1e-9):
    angle = angle_rad % (2.0 * math.pi)
    start = sector_start_rad % (2.0 * math.pi)
    end = sector_end_rad % (2.0 * math.pi)
    if end <= start:
        end += 2.0 * math.pi
    if angle < start:
        angle += 2.0 * math.pi
    return start - eps <= angle <= end + eps


def add_polygon(points):
    for idx, p1 in enumerate(points):
        p2 = points[(idx + 1) % len(points)]
        add_segment(p1, p2)


def add_grouped_polygon(points, group=1):
    add_polygon(points)
    for p1, p2 in zip(points, points[1:] + points[:1]):
        mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
        femm.mi_selectsegment(mx, my)
        femm.mi_setsegmentprop("<None>", 0, 1, 0, group)
        femm.mi_clearselected()


def unit_vector(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length == 0:
        raise ValueError("Cannot normalize zero-length vector.")
    return dx / length, dy / length


def fillet_triangle_corner(prev_point, corner_point, next_point, radius, segments=6):
    if radius <= 0:
        return [prev_point, corner_point, next_point]

    e_prev = unit_vector(corner_point, prev_point)
    e_next = unit_vector(corner_point, next_point)
    dot = max(-1.0, min(1.0, e_prev[0] * e_next[0] + e_prev[1] * e_next[1]))
    angle = math.acos(dot)
    if angle <= 1e-6:
        return [prev_point, corner_point, next_point]

    edge_prev = math.dist(corner_point, prev_point)
    edge_next = math.dist(corner_point, next_point)
    tangent = radius / math.tan(angle / 2.0)
    tangent = min(tangent, edge_prev * 0.45, edge_next * 0.45)
    actual_radius = tangent * math.tan(angle / 2.0)

    t_prev = (
        corner_point[0] + e_prev[0] * tangent,
        corner_point[1] + e_prev[1] * tangent,
    )
    t_next = (
        corner_point[0] + e_next[0] * tangent,
        corner_point[1] + e_next[1] * tangent,
    )

    bisector = (e_prev[0] + e_next[0], e_prev[1] + e_next[1])
    bisector_length = math.hypot(*bisector)
    if bisector_length <= 1e-9:
        return [prev_point, t_prev, t_next, next_point]
    bisector = (bisector[0] / bisector_length, bisector[1] / bisector_length)
    center_distance = actual_radius / math.sin(angle / 2.0)
    center = (
        corner_point[0] + bisector[0] * center_distance,
        corner_point[1] + bisector[1] * center_distance,
    )

    start_angle = math.atan2(t_prev[1] - center[1], t_prev[0] - center[0])
    end_angle = math.atan2(t_next[1] - center[1], t_next[0] - center[0])
    delta = (end_angle - start_angle) % (2.0 * math.pi)
    if delta > math.pi:
        delta -= 2.0 * math.pi

    arc_points = [prev_point, t_prev]
    for idx in range(1, segments):
        fraction = idx / segments
        angle_i = start_angle + delta * fraction
        arc_points.append(
            (
                center[0] + actual_radius * math.cos(angle_i),
                center[1] + actual_radius * math.sin(angle_i),
            )
        )
    arc_points.extend([t_next, next_point])
    return arc_points


def triangle_apex_from_base_angles(point_a, point_c, angle_a_deg, outward_direction):
    if not 1e-6 < angle_a_deg < 89.999:
        raise ValueError("--red-relief-angle-a-deg must be between 0 and 90 degrees.")

    base_length = math.dist(point_a, point_c)
    e_ac = unit_vector(point_a, point_c)
    outward = unit_vector((0.0, 0.0), outward_direction)
    angle_a = math.radians(angle_a_deg)

    along_ac = base_length * (math.cos(angle_a) ** 2)
    outward_height = base_length * math.sin(angle_a) * math.cos(angle_a)
    return (
        point_a[0] + along_ac * e_ac[0] + outward_height * outward[0],
        point_a[1] + along_ac * e_ac[1] + outward_height * outward[1],
    )


def add_short_arc_between(radius, p1, p2, maxseg=1.0):
    a1 = math.atan2(p1[1], p1[0])
    a2 = math.atan2(p2[1], p2[0])
    diff = (a2 - a1) % (2.0 * math.pi)
    if diff <= math.pi:
        add_arc(radius, a1, a2, maxseg=maxseg)
    else:
        add_arc(radius, a2, a1, maxseg=maxseg)


def ray_circle_intersection(point, direction, radius):
    dot = point[0] * direction[0] + point[1] * direction[1]
    point_norm_sq = point[0] ** 2 + point[1] ** 2
    discriminant = dot**2 + radius**2 - point_norm_sq
    if discriminant < 0:
        raise ValueError("Ray does not intersect the rotor circle.")
    distance = -dot + math.sqrt(discriminant)
    return point[0] + distance * direction[0], point[1] + distance * direction[1]


def add_slot(theta, spec):
    rs = spec.stator_inner_radius
    rso = spec.slot_top_radius
    rsb = spec.slot_bottom_radius
    rsm = (rso + rsb) / 2.0
    b0 = spec.slot_opening_span
    b = spec.slot_span

    p0 = polar(rs, theta - b0 / 2.0)
    p1 = polar(rso, theta - b0 / 2.0)
    p2 = polar(rso, theta - b / 2.0)
    p3 = polar(rsb, theta - b / 2.0)
    p4 = polar(rsb, theta + b / 2.0)
    p5 = polar(rso, theta + b / 2.0)
    p6 = polar(rso, theta + b0 / 2.0)
    p7 = polar(rs, theta + b0 / 2.0)

    add_segment(p0, p1)
    add_arc(rso, theta - b / 2.0, theta - b0 / 2.0)
    add_segment(p2, p3)
    add_arc(rsb, theta - b / 2.0, theta + b / 2.0)
    add_segment(p4, p5)
    add_arc(rso, theta + b0 / 2.0, theta + b / 2.0)
    add_segment(p6, p7)
    add_arc(rs, theta - b0 / 2.0, theta + b0 / 2.0)
    add_segment(polar(rsm, theta - b / 2.0), polar(rsm, theta + b / 2.0))


def add_rotated_rectangle(theta, center_r, center_t, length, thickness, axis_angle, group=1):
    ux_r = math.sin(axis_angle)
    ux_t = math.cos(axis_angle)
    vx_r = math.cos(axis_angle)
    vx_t = -math.sin(axis_angle)
    pts = []
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        radial = center_r + sx * length * 0.5 * ux_r + sy * thickness * 0.5 * vx_r
        tangent = center_t + sx * length * 0.5 * ux_t + sy * thickness * 0.5 * vx_t
        pts.append(local_to_xy(theta, radial, tangent))
    add_grouped_polygon(pts, group=group)
    return local_to_xy(theta, center_r, center_t)


def read_bh_points(dat_path):
    points = []
    for line_number, line in enumerate(Path(dat_path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "%", "//")):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            b_value = float(parts[0])
            h_value = float(parts[1])
        except ValueError as exc:
            raise ValueError(f"Invalid B-H data at {dat_path}:{line_number}: {line}") from exc
        points.append((b_value, h_value))

    if not points:
        raise ValueError(f"No B-H points found in {dat_path}")
    return points


def add_core_material_from_bh_dat(material_name, dat_path):
    bh_path = Path(dat_path)
    if not bh_path.exists():
        raise FileNotFoundError(f"Core B-H file not found: {bh_path}")

    femm.mi_addmaterial(
        material_name,
        1000,
        1000,
        0,
        0,
        0,
        CORE_LAMINATION_THICKNESS_MM,
        0,
        CORE_LAMINATION_FILL_FACTOR,
        CORE_LAMINATION_TYPE,
        0,
        0,
    )
    for b_value, h_value in read_bh_points(bh_path):
        femm.mi_addbhpoint(material_name, b_value, h_value)


def add_materials(spec):
    femm.mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    add_core_material_from_bh_dat(CORE_MATERIAL_NAME, CORE_BH_DAT_DEFAULT)
    femm.mi_addmaterial("Copper", 1, 1, 0, 0, 58, 0, 0, 1, 0, 0, 0)
    femm.mi_addmaterial(
        spec.magnet_material,
        1.05,
        1.05,
        spec.magnet_coercivity_ka_per_m * 1000.0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
    )


def phase_currents_from_rms(current_rms_a):
    current_peak = current_rms_a * math.sqrt(2.0)
    return {
        "Ia": current_peak * math.sin(0.0),
        "Ib": current_peak * math.sin(-2.0 * math.pi / 3.0),
        "Ic": current_peak * math.sin(2.0 * math.pi / 3.0),
    }


def sector_boundary_format(boundary_kind):
    normalized = boundary_kind.strip().lower()
    if normalized in {"anti-periodic", "antiperiodic", "anti"}:
        return 5
    if normalized in {"periodic", "period"}:
        return 4
    raise ValueError("--sector-boundary-kind must be 'anti-periodic' or 'periodic'.")


def add_sector_boundary_property(boundary_name, boundary_kind):
    femm.mi_addboundprop(
        boundary_name,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        sector_boundary_format(boundary_kind),
    )


def add_sector_side_boundaries(
    sector_start_rad,
    sector_end_rad,
    shaft_radius,
    rotor_outer_radius,
    stator_inner_radius,
    stator_outer_radius,
    boundary_name,
    boundary_kind,
):
    radial_spans = (
        (0.0, shaft_radius),
        (shaft_radius, rotor_outer_radius),
        (rotor_outer_radius, stator_inner_radius),
        (stator_inner_radius, stator_outer_radius),
    )
    for idx, (radius_start, radius_end) in enumerate(radial_spans, start=1):
        pair_boundary_name = f"{boundary_name}_{idx}"
        add_sector_boundary_property(pair_boundary_name, boundary_kind)
        for angle_rad in (sector_start_rad, sector_end_rad):
            add_radial_segment(radius_start, radius_end, angle_rad, boundary=pair_boundary_name)


def add_block_label(x, y, material, circuit="<None>", magdir=0, group=0, turns=0, mesh=0):
    femm.mi_addblocklabel(x, y)
    femm.mi_selectlabel(x, y)
    femm.mi_setblockprop(material, 1, mesh, circuit, magdir, group, turns)
    femm.mi_clearselected()


def slot_layer_phases(slot_index):
    def decode(token):
        sign = -1 if token.startswith("-") else 1
        phase = token[1:] if sign < 0 else token
        return PHASE_TO_CIRCUIT[phase], sign

    return decode(WINDING_UPPER_SEQUENCE[slot_index]), decode(WINDING_LOWER_SEQUENCE[slot_index])


def draw_stator(
    spec,
    stator_outer_radius,
    turns_per_layer,
    stator_rotation_rad,
    sector_start_rad=None,
    sector_end_rad=None,
):
    # ASSUMED: stator outer radius is not in Table I.
    if sector_start_rad is None:
        add_circle(stator_outer_radius, maxseg=2.0, boundary="A0")
        add_circle(spec.stator_inner_radius, maxseg=1.0)
        label_angle = 0.0
    else:
        add_sector_arc(stator_outer_radius, sector_start_rad, sector_end_rad, maxseg=2.0, boundary="A0")
        add_sector_arc(spec.stator_inner_radius, sector_start_rad, sector_end_rad, maxseg=1.0)
        label_angle = (sector_start_rad + sector_end_rad) / 2.0

    for i in range(spec.slots):
        theta = stator_rotation_rad + 2.0 * math.pi * i / spec.slots
        if sector_start_rad is not None and not angle_in_sector(theta, sector_start_rad, sector_end_rad):
            continue
        add_slot(theta, spec)

    add_block_label(
        *polar((spec.slot_bottom_radius + stator_outer_radius) / 2.0, label_angle),
        CORE_MATERIAL_NAME,
        group=10,
    )

    for i in range(spec.slots):
        theta_unrotated = 2.0 * math.pi * i / spec.slots
        theta = stator_rotation_rad + theta_unrotated
        if sector_start_rad is not None and not angle_in_sector(theta, sector_start_rad, sector_end_rad):
            continue
        rsm = (spec.slot_top_radius + spec.slot_bottom_radius) / 2.0
        # Slot 1 in the winding diagram is the slot just above +X. Convert the
        # geometric slot index to that signed winding table index.
        slot_angle = (theta % (2.0 * math.pi))
        slot_number_zero_based = round((slot_angle - math.radians(5.0)) / (2.0 * math.pi / spec.slots)) % spec.slots
        (upper_phase, upper_sign), (lower_phase, lower_sign) = slot_layer_phases(slot_number_zero_based)

        upper_r = (spec.slot_top_radius + rsm) / 2.0
        lower_r = (rsm + spec.slot_bottom_radius) / 2.0
        add_block_label(
            *polar(upper_r, theta),
            "Copper",
            circuit=upper_phase,
            group=20,
            turns=upper_sign * turns_per_layer,
        )
        add_block_label(
            *polar(lower_r, theta),
            "Copper",
            circuit=lower_phase,
            group=20,
            turns=lower_sign * turns_per_layer,
        )


def draw_rotor(spec, shaft_radius, sector_start_rad=None, sector_end_rad=None):
    # ASSUMED: shaft radius is not in Table I.
    if sector_start_rad is None:
        add_circle(spec.rotor_outer_radius, maxseg=1.0)
        add_circle(shaft_radius, maxseg=2.5)
        label_angle = 0.0
        add_block_label(0, 0, "Air", group=0)
    else:
        add_sector_arc(spec.rotor_outer_radius, sector_start_rad, sector_end_rad, maxseg=1.0)
        add_sector_arc(shaft_radius, sector_start_rad, sector_end_rad, maxseg=2.5)
        label_angle = (sector_start_rad + sector_end_rad) / 2.0
        add_block_label(*polar(shaft_radius / 2.0, label_angle), "Air", group=0)
    add_block_label(
        *polar((shaft_radius + spec.rotor_outer_radius) / 2.0, label_angle),
        CORE_MATERIAL_NAME,
        group=1,
    )


def local_vector_angle_deg(theta, radial_component, tangential_component):
    vx = radial_component * math.cos(theta) - tangential_component * math.sin(theta)
    vy = radial_component * math.sin(theta) + tangential_component * math.cos(theta)
    return math.degrees(math.atan2(vy, vx))


def draw_vshape_magnets(
    spec,
    v_included_angle_deg,
    bridge_angle_deg,
    relief_fillet_mm,
    red_relief_angle_a_deg,
    magnet_radial_shift_mm,
    rotor_rotation_rad,
    sector_start_rad=None,
    sector_end_rad=None,
):
    # ASSUMED: this draws the original V-shape in Fig. 1(a), not the equivalent
    # curved magnet. The exact V included angle is not listed in Table I.
    #
    # Construction for one pole in local coordinates:
    # - local radial axis points outward, local tangential axis points to the
    #   right side of the pole;
    # - the inner top magnet corner starts at +/- wb2/2;
    # - v_included_angle_deg is the total included angle between the two magnet
    #   branches. The internal gamma is measured from the local tangential axis.
    # - the red outer relief is triangle A-X-C outside the outer magnet side AC.
    #   The angle at A is adjustable; angle C is 90 deg - angle A, so angle X
    #   remains 90 deg. At 45 deg this returns to the symmetric triangle.
    # - the purple inner relief is an Air triangle outside the inner side,
    #   constructed with local radial/tangential guide lines.
    # ASSUMED: the paper does not give exact coordinates for all reliefs, so
    # the guide-line construction is used for FEMM geometry checking.
    if not 0.0 < v_included_angle_deg < 180.0:
        raise ValueError("--v-included-angle-deg must be between 0 and 180 degrees.")

    pole_pitch = 2.0 * math.pi / spec.poles
    gamma = math.radians(90.0 - v_included_angle_deg / 2.0)
    bridge_angle = math.radians(bridge_angle_deg)
    bridge_r = math.cos(bridge_angle)
    bridge_t = math.sin(bridge_angle)

    inner_top_t = spec.bridge_width_2 / 2.0
    # Point A is the lower point of the magnet edge AC near the rotor. Place the
    # magnet so AA1 has length hb1. Point C is the higher point on the same edge.
    a_t = inner_top_t + spec.magnet_width * math.cos(gamma) + spec.magnet_thickness * math.sin(gamma)
    rotor_touch_t = a_t + spec.bridge_length_1 * bridge_t
    max_touch_t = spec.rotor_outer_radius * math.sin(pole_pitch / 2.0)
    if rotor_touch_t >= max_touch_t:
        raise ValueError(
            "V-shape construction exceeds one pole pitch. "
            "Adjust --v-included-angle-deg or --v-bridge-angle-deg."
        )

    rotor_touch_r = math.sqrt(spec.rotor_outer_radius**2 - rotor_touch_t**2)
    a_r = rotor_touch_r - spec.bridge_length_1 * bridge_r
    outer_top_r = a_r + spec.magnet_thickness * math.cos(gamma)
    inner_top_r = outer_top_r - spec.magnet_width * math.sin(gamma)

    # ASSUMED/USER-TUNABLE: positive shift moves magnets outward toward the
    # rotor surface; negative shift moves them inward.
    inner_top_r += magnet_radial_shift_mm

    if inner_top_r <= 0:
        raise ValueError("V-shape construction put the magnet through the shaft area.")

    for pole in range(spec.poles):
        theta = rotor_rotation_rad + pole * pole_pitch
        if sector_start_rad is not None and not angle_in_sector(theta, sector_start_rad, sector_end_rad):
            continue
        polarity = 0 if pole % 2 == 0 else 180
        for side in [-1, 1]:
            u_r = math.sin(gamma)
            u_t = side * math.cos(gamma)
            n_r = math.cos(gamma)
            n_t = -side * math.sin(gamma)
            bridge_vec = (bridge_r, side * bridge_t)

            inner_top = (inner_top_r, side * inner_top_t)
            outer_top = (
                inner_top[0] + spec.magnet_width * u_r,
                inner_top[1] + spec.magnet_width * u_t,
            )
            outer_bottom = (
                outer_top[0] - spec.magnet_thickness * n_r,
                outer_top[1] - spec.magnet_thickness * n_t,
            )
            inner_bottom = (
                inner_top[0] - spec.magnet_thickness * n_r,
                inner_top[1] - spec.magnet_thickness * n_t,
            )
            local_points = [inner_top, outer_top, outer_bottom, inner_bottom]
            points = [local_to_xy(theta, radial, tangential) for radial, tangential in local_points]
            add_grouped_polygon(points, group=1)

            # Red Air triangle outside the outer magnet side AC.
            # A is the lower point and C is the higher point of the magnet edge
            # nearest the rotor. X is chosen so AX and CX are both 45 deg to AC.
            a_point = outer_bottom
            c_point = outer_top
            x_point = triangle_apex_from_base_angles(a_point, c_point, red_relief_angle_a_deg, (u_r, u_t))
            red_air = fillet_triangle_corner(a_point, x_point, c_point, relief_fillet_mm)
            red_air_points = [local_to_xy(theta, radial, tangential) for radial, tangential in red_air]
            add_grouped_polygon(red_air_points, group=1)

            red_label_r = sum(point[0] for point in red_air) / len(red_air)
            red_label_t = sum(point[1] for point in red_air) / len(red_air)
            add_block_label(*local_to_xy(theta, red_label_r, red_label_t), "Air", group=1)

            # Purple Air triangle outside the inner magnet side, using local
            # radial/tangential guide lines as in the sketch.
            purple_corner = (inner_bottom[0], inner_top[1])
            purple_air = fillet_triangle_corner(inner_top, purple_corner, inner_bottom, relief_fillet_mm)
            purple_air_points = [local_to_xy(theta, radial, tangential) for radial, tangential in purple_air]
            add_grouped_polygon(purple_air_points, group=1)
            purple_label_r = sum(point[0] for point in purple_air) / len(purple_air)
            purple_label_t = sum(point[1] for point in purple_air) / len(purple_air)
            add_block_label(*local_to_xy(theta, purple_label_r, purple_label_t), "Air", group=1)

            label_r = sum(point[0] for point in local_points) / len(local_points)
            label_t = sum(point[1] for point in local_points) / len(local_points)
            label = local_to_xy(theta, label_r, label_t)
            magdir = local_vector_angle_deg(theta, n_r, n_t) + polarity
            add_block_label(*label, spec.magnet_material, magdir=magdir, group=1)


def build_model(
    topology,
    output_path,
    stator_outer_radius=STATOR_OUTER_RADIUS_MM_DEFAULT,
    shaft_radius=SHAFT_RADIUS_MM_DEFAULT,
    current_rms_a=CURRENT_RMS_A_DEFAULT,
    turns_per_layer=TURNS_PER_LAYER_DEFAULT,
    v_included_angle_deg=V_INCLUDED_ANGLE_DEG_DEFAULT,
    v_bridge_angle_deg=45,
    relief_fillet_mm=RELIEF_FILLET_MM_DEFAULT,
    red_relief_angle_a_deg=RED_RELIEF_ANGLE_A_DEG_DEFAULT,
    magnet_radial_shift_mm=MAGNET_RADIAL_SHIFT_MM_DEFAULT,
    stator_rotation_deg=STATOR_ROTATION_DEG_DEFAULT,
    rotor_rotation_deg=ROTOR_ROTATION_DEG_DEFAULT,
    sector_start_deg=SECTOR_START_DEG_DEFAULT,
    sector_span_deg=SECTOR_SPAN_DEG_DEFAULT,
    sector_boundary_kind=SECTOR_BOUNDARY_KIND_DEFAULT,
    full_model=False,
):
    spec = PAPER_SPECS[topology]
    stator_rotation_rad = math.radians(stator_rotation_deg)
    rotor_rotation_rad = math.radians(rotor_rotation_deg)
    sector_start_rad = None
    sector_end_rad = None
    sector_boundary_name = "SectorPeriodic"
    if not full_model:
        sector_start_rad = math.radians(sector_start_deg)
        sector_end_rad = math.radians(sector_start_deg + sector_span_deg)

    phase_currents = phase_currents_from_rms(current_rms_a)

    femm.openfemm(1)
    try:
        femm.newdocument(0)
        femm.mi_probdef(0, "millimeters", "planar", 1e-8, spec.stack_length, 30)
        femm.mi_addboundprop("A0", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        femm.mi_addcircprop("Ia", phase_currents["Ia"], 1)
        femm.mi_addcircprop("Ib", phase_currents["Ib"], 1)
        femm.mi_addcircprop("Ic", phase_currents["Ic"], 1)
        add_materials(spec)

        draw_stator(
            spec,
            stator_outer_radius,
            turns_per_layer,
            stator_rotation_rad,
            sector_start_rad,
            sector_end_rad,
        )
        draw_rotor(spec, shaft_radius, sector_start_rad, sector_end_rad)
        if not full_model:
            add_sector_side_boundaries(
                sector_start_rad,
                sector_end_rad,
                shaft_radius,
                spec.rotor_outer_radius,
                spec.stator_inner_radius,
                stator_outer_radius,
                sector_boundary_name,
                sector_boundary_kind,
            )
            airgap_label_angle = (sector_start_rad + sector_end_rad) / 2.0
        else:
            airgap_label_angle = rotor_rotation_rad + math.radians(3)
        add_block_label(
            *polar((spec.rotor_outer_radius + spec.stator_inner_radius) / 2.0, airgap_label_angle),
            "Air",
        )

        draw_vshape_magnets(
            spec,
            v_included_angle_deg,
            v_bridge_angle_deg,
            relief_fillet_mm,
            red_relief_angle_a_deg,
            magnet_radial_shift_mm,
            rotor_rotation_rad,
            sector_start_rad,
            sector_end_rad,
        )

        femm.mi_zoomnatural()
        output = Path(output_path).absolute()
        femm.mi_saveas(str(output))
    finally:
        femm.closefemm()


def main():
    parser = argparse.ArgumentParser(description="Build approximate V-shape FEMM model from Table I of the IPM paper.")
    parser.add_argument("--turns-per-layer", type=int, default=TURNS_PER_LAYER_DEFAULT)
    parser.add_argument("--turns-per-slot", type=int, dest="turns_per_layer", help=argparse.SUPPRESS)
    parser.add_argument("--stator-outer-radius", type=float, default=STATOR_OUTER_RADIUS_MM_DEFAULT)
    parser.add_argument("--shaft-radius", type=float, default=SHAFT_RADIUS_MM_DEFAULT)
    parser.add_argument("--current-rms-a", type=float, default=CURRENT_RMS_A_DEFAULT)
    parser.add_argument("--v-included-angle-deg", type=float, default=V_INCLUDED_ANGLE_DEG_DEFAULT)
    parser.add_argument("--v-bridge-angle-deg", type=float, default=45.0)
    parser.add_argument("--relief-fillet-mm", type=float, default=RELIEF_FILLET_MM_DEFAULT)
    parser.add_argument("--red-relief-angle-a-deg", type=float, default=RED_RELIEF_ANGLE_A_DEG_DEFAULT)
    parser.add_argument("--magnet-radial-shift-mm", type=float, default=MAGNET_RADIAL_SHIFT_MM_DEFAULT)
    parser.add_argument("--stator-rotation-deg", type=float, default=STATOR_ROTATION_DEG_DEFAULT)
    parser.add_argument("--rotor-rotation-deg", type=float, default=ROTOR_ROTATION_DEG_DEFAULT)
    parser.add_argument("--sector-start-deg", type=float, default=SECTOR_START_DEG_DEFAULT)
    parser.add_argument("--sector-span-deg", type=float, default=SECTOR_SPAN_DEG_DEFAULT)
    parser.add_argument(
        "--sector-boundary-kind",
        choices=("anti-periodic", "periodic"),
        default=SECTOR_BOUNDARY_KIND_DEFAULT,
    )
    parser.add_argument("--output-sector", default=SECTOR_OUTPUT_DEFAULT)
    parser.add_argument("--output-full", default=FULL_OUTPUT_DEFAULT)
    parser.add_argument("--sector-only", action="store_true", help="Export only the 1/6 sector model.")
    parser.add_argument("--full-model", action="store_true", help="Export only the full model.")
    parser.add_argument("--full-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.sector_only and (args.full_model or args.full_only):
        parser.error("--sector-only cannot be used with --full-model or --full-only")

    common_kwargs = dict(
        stator_outer_radius=args.stator_outer_radius,
        shaft_radius=args.shaft_radius,
        current_rms_a=args.current_rms_a,
        turns_per_layer=args.turns_per_layer,
        v_included_angle_deg=args.v_included_angle_deg,
        v_bridge_angle_deg=args.v_bridge_angle_deg,
        relief_fillet_mm=args.relief_fillet_mm,
        red_relief_angle_a_deg=args.red_relief_angle_a_deg,
        magnet_radial_shift_mm=args.magnet_radial_shift_mm,
        stator_rotation_deg=args.stator_rotation_deg,
        rotor_rotation_deg=args.rotor_rotation_deg,
        sector_start_deg=args.sector_start_deg,
        sector_span_deg=args.sector_span_deg,
        sector_boundary_kind=args.sector_boundary_kind,
    )

    export_sector = not (args.full_model or args.full_only)
    export_full = not args.sector_only

    saved = []
    if export_sector:
        build_model("vshape", args.output_sector, full_model=False, **common_kwargs)
        saved.append(
            f"{args.output_sector} "
            f"(sector={args.sector_start_deg:g}..{args.sector_start_deg + args.sector_span_deg:g}deg, "
            f"sector_boundary_kind={args.sector_boundary_kind})"
        )
    if export_full:
        build_model("vshape", args.output_full, full_model=True, **common_kwargs)
        saved.append(f"{args.output_full} (full_model)")

    print(
        "saved "
        + "; ".join(saved)
        + " "
        + f"(stator_outer_radius={args.stator_outer_radius:g}, "
        f"shaft_radius={args.shaft_radius:g}, "
        f"current_rms_a={args.current_rms_a:g}, "
        f"v_included_angle_deg={args.v_included_angle_deg:g}, "
        f"turns_per_layer={args.turns_per_layer:g}, "
        f"relief_fillet_mm={args.relief_fillet_mm:g}, "
        f"red_relief_angle_a_deg={args.red_relief_angle_a_deg:g}, "
        f"magnet_radial_shift_mm={args.magnet_radial_shift_mm:g}, "
        f"stator_rotation_deg={args.stator_rotation_deg:g}, "
        f"rotor_rotation_deg={args.rotor_rotation_deg:g})"
    )


if __name__ == "__main__":
    main()
