import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import femm


MU0 = 4.0 * math.pi * 1e-7


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
    "flat": MotorSpec(
        name="flat",
        rated_power_kw=15.0,
        rated_current_a=28.3,
        rated_speed_rpm=20000,
        stator_inner_radius=35.0,
        rotor_outer_radius=33.7,
        poles=4,
        slots=18,
        stack_length=110.0,
        slot_opening_span=0.057,
        slot_span=0.188,
        slot_top_radius=36.0,
        slot_bottom_radius=53.0,
        bridge_width_1=1.5,
        bridge_length_1=4.5,
        bridge_width_2=2.0,
        bridge_length_2=4.0,
        magnet_width=15.0,
        magnet_thickness=4.0,
        magnet_material="N42UH",
        magnet_remanence_t=1.3,
        magnet_coercivity_ka_per_m=987.0,
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
    "stator_outer_radius: not listed in the paper; default = slot_bottom_radius + 10 mm.",
    "shaft_radius: not listed in the paper; default = 0.35 * rotor_outer_radius.",
    "stator slot: paper gives equivalent slot radii/spans; this draws that equivalent slot, not the original pear-shaped slot.",
    "rotor bridges: Table I gives bridge width/length, but not all local corner/fillet geometry; bridges are represented indirectly by magnet placement clearance.",
    "flat magnets: drawn as rectangular magnets using lm/hm, with a center bridge gap wb2.",
    "winding layout: phase assignment is synthetic from electrical slot angle; turns-per-slot defaults to 10.",
    "core material: linear mu=1000 placeholder; no B-H curve is reconstructed from Fig. 7.",
    "magnet material: uses Table I Hc value; Br is noted in Table I but FEMM material call uses Hc directly.",
)


def model_note_text(spec, stator_outer_radius, shaft_radius, turns_per_slot, v_angle_deg):
    lines = [
        PAPER_SOURCE_NOTE,
        "",
        f"Topology: {spec.name}",
        f"Explicit Table I values: Rs={spec.stator_inner_radius:g} mm, Rr={spec.rotor_outer_radius:g} mm, "
        f"poles={spec.poles}, slots={spec.slots}, La={spec.stack_length:g} mm.",
        f"Generated values: stator_outer_radius={stator_outer_radius:g} mm, "
        f"shaft_radius={shaft_radius:g} mm, turns_per_slot={turns_per_slot}.",
    ]
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


def add_slot(theta, spec):
    rs = spec.stator_inner_radius
    rso = spec.slot_top_radius
    rsb = spec.slot_bottom_radius
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


def add_materials(spec):
    femm.mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    femm.mi_addmaterial("PaperCore_linear_mu1000", 1000, 1000, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    femm.mi_addmaterial("PaperCopper", 1, 1, 0, 0, 58, 0, 0, 1, 0, 0, 0)
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


def add_block_label(x, y, material, circuit="<None>", magdir=0, group=0, turns=0, mesh=0):
    femm.mi_addblocklabel(x, y)
    femm.mi_selectlabel(x, y)
    femm.mi_setblockprop(material, 1, mesh, circuit, magdir, group, turns)
    femm.mi_clearselected()


def slot_phase(theta, pole_pairs):
    # ASSUMED: the paper does not list the actual winding table. This synthetic
    # layout only assigns slots to the nearest three-phase electrical axis.
    electrical = (pole_pairs * theta) % (2.0 * math.pi)
    axes = [0.0, -2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0]
    names = ["Ia", "Ib", "Ic"]
    values = [math.cos(electrical - axis) for axis in axes]
    idx = max(range(3), key=lambda i: abs(values[i]))
    sign = 1 if values[idx] >= 0 else -1
    return names[idx], sign


def draw_stator(spec, stator_outer_radius, turns_per_slot):
    # ASSUMED: stator outer radius is not in Table I.
    add_circle(stator_outer_radius, maxseg=2.0, boundary="A0")
    add_circle(spec.stator_inner_radius, maxseg=1.0)

    for i in range(spec.slots):
        theta = 2.0 * math.pi * i / spec.slots
        add_slot(theta, spec)

    add_block_label(
        (spec.slot_bottom_radius + stator_outer_radius) / 2.0,
        0,
        "PaperCore_linear_mu1000",
        group=10,
    )

    pole_pairs = spec.poles // 2
    for i in range(spec.slots):
        theta = 2.0 * math.pi * i / spec.slots
        phase, sign = slot_phase(theta, pole_pairs)
        r = (spec.slot_top_radius + spec.slot_bottom_radius) / 2.0
        x, y = polar(r, theta)
        add_block_label(x, y, "PaperCopper", circuit=phase, group=20, turns=sign * turns_per_slot)


def draw_rotor(spec, shaft_radius):
    # ASSUMED: shaft radius is not in Table I.
    add_circle(spec.rotor_outer_radius, maxseg=1.0)
    add_circle(shaft_radius, maxseg=2.5)
    add_block_label((shaft_radius + spec.rotor_outer_radius) / 2.0, 0, "PaperCore_linear_mu1000", group=1)
    add_block_label(0, 0, "Air", group=0)


def draw_flat_magnets(spec):
    # APPROXIMATE: Table I gives lm/hm and bridge sizes, but not complete
    # corner/fillet positions. Draw two rectangular magnet blocks per pole.
    pole_pitch = 2.0 * math.pi / spec.poles
    r_outer = spec.rotor_outer_radius - spec.bridge_length_1
    r_mid = r_outer - spec.magnet_thickness / 2.0
    offset = spec.bridge_width_2 / 2.0 + spec.magnet_width / 2.0
    for pole in range(spec.poles):
        theta = pole * pole_pitch
        polarity = 0 if pole % 2 == 0 else 180
        for side in [-1, 1]:
            label = add_rotated_rectangle(
                theta,
                r_mid,
                side * offset,
                spec.magnet_width,
                spec.magnet_thickness,
                axis_angle=0.0,
                group=1,
            )
            add_block_label(*label, spec.magnet_material, magdir=math.degrees(theta) + polarity, group=1)


def build_model(output_path, stator_outer_radius=None, shaft_radius=None, turns_per_slot=10):
    spec = PAPER_SPECS["flat"]

    # ASSUMED: these defaults are not from the paper.
    stator_outer_radius = stator_outer_radius or (spec.slot_bottom_radius + 10.0)
    shaft_radius = shaft_radius or (0.35 * spec.rotor_outer_radius)

    femm.openfemm(1)
    try:
        femm.newdocument(0)
        femm.mi_probdef(0, "millimeters", "planar", 1e-8, spec.stack_length, 30)
        femm.mi_addboundprop("A0", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        femm.mi_addcircprop("Ia", 0, 1)
        femm.mi_addcircprop("Ib", 0, 1)
        femm.mi_addcircprop("Ic", 0, 1)
        add_materials(spec)

        draw_stator(spec, stator_outer_radius, turns_per_slot)
        draw_rotor(spec, shaft_radius)
        add_block_label((spec.rotor_outer_radius + spec.stator_inner_radius) / 2.0, math.radians(3), "Air")

        draw_flat_magnets(spec)

        femm.mi_zoomnatural()
        output = Path(output_path).absolute()
        femm.mi_saveas(str(output))
    finally:
        femm.closefemm()


def main():
    parser = argparse.ArgumentParser(description="Build approximate flat-shape FEMM model from Table I of the IPM paper.")
    parser.add_argument("--turns-per-slot", type=int, default=10)
    args = parser.parse_args()

    output = "paper_ipm_flat_approx.FEM"
    build_model(output, turns_per_slot=args.turns_per_slot)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
