import argparse
import math
from pathlib import Path

import femm

import build_paper_vshape_model as base


EQUIVALENT_OUTPUT_DEFAULT = "paper_ipm_vshape_equivalent_1over6.FEM"
EQUIVALENT_W1_MM_DEFAULT = 2.5
EQUIVALENT_W2_MM_DEFAULT = 1.2
EQUIVALENT_CURRENT_RMS_A_DEFAULT = 0
EQUIVALENT_RADIAL_SHIFT_MM_DEFAULT = 1.7            #base 2.1 tương đương 1.5 real
DRAW_ANALYTICAL_SUBDOMAIN_BOUNDARIES_DEFAULT = True
ANALYTICAL_SUBDOMAIN_GROUP_DEFAULT = 1
GROUP_STATOR_YOKE = 1
GROUP_STATOR_SLOT_LOWER = 2
GROUP_STATOR_SLOT_UPPER = 3
GROUP_STATOR_TOOTH_TIP = 4
GROUP_AIRGAP = 5
GROUP_ROTOR_OUTER = 6
GROUP_ROTOR_SIDE_BRIDGE = 7
GROUP_ROTOR_RADIAL_PM = 8
GROUP_ROTOR_INNER_PM = 9
GROUP_ROTOR_INNER_CORE = 10
GROUP_SHAFT_AIR = 11


def equivalent_pm_dimensions(
    spec,
    alpha_deg=None,
    w1_mm=EQUIVALENT_W1_MM_DEFAULT,
    w2_mm=EQUIVALENT_W2_MM_DEFAULT,
    wb1_mm=None,
    hb1_mm=None,
    wb2_mm=None,
    radial_shift_mm=EQUIVALENT_RADIAL_SHIFT_MM_DEFAULT,
):
    """Paper Fig. 1(a) / Eq. (1)-(4) V-shape equivalent rotor dimensions."""
    wb1 = spec.bridge_width_1 if wb1_mm is None else wb1_mm
    hb1 = spec.bridge_length_1 if hb1_mm is None else hb1_mm
    wb2 = spec.bridge_width_2 if wb2_mm is None else wb2_mm

    pole_pitch = 2.0 * math.pi / spec.poles
    if alpha_deg is None:
        # ASSUMED: alpha is the available PM span after the two outer bridge
        # widths wb1 are removed from one pole pitch.
        alpha = pole_pitch - 2.0 * wb1 / spec.rotor_outer_radius
    else:
        alpha = math.radians(alpha_deg)

    rl = spec.rotor_outer_radius - hb1
    denominator = alpha / 2.0 - 1.0 - spec.magnet_thickness / (rl - w1_mm)
    numerator = spec.magnet_width - rl + w1_mm + w2_mm + wb2 / 2.0
    if abs(denominator) < 1e-9:
        raise ValueError("Equivalent PM equation is singular. Adjust alpha/w1/hb1.")

    rm = numerator / denominator
    rf = rm - spec.magnet_thickness
    alpha1 = alpha - 2.0 * spec.magnet_thickness / (rl - w1_mm)
    lx = rl - rm - w1_mm

    # Positive shift moves the whole equivalent PM + Air pocket outward.
    rf += radial_shift_mm
    rm += radial_shift_mm
    rl += radial_shift_mm

    if not (0.0 < rf < rm < rl < spec.rotor_outer_radius):
        raise ValueError(
            "Equivalent PM radii are invalid: "
            f"Rf={rf:.4g}, Rm={rm:.4g}, Rl={rl:.4g}, Rr={spec.rotor_outer_radius:.4g}. "
            "Adjust alpha/w1/w2/hb1/radial_shift."
        )
    if lx <= 0.0 or alpha1 <= 0.0:
        raise ValueError(
            f"Equivalent PM dimensions are invalid: lx={lx:.4g}, alpha1={math.degrees(alpha1):.4g}deg."
        )

    return {
        "alpha_rad": alpha,
        "alpha_deg": math.degrees(alpha),
        "alpha1_rad": alpha1,
        "alpha1_deg": math.degrees(alpha1),
        "delta_rad": 0.5 * (alpha - alpha1),
        "delta_deg": math.degrees(0.5 * (alpha - alpha1)),
        "rf": rf,
        "rm": rm,
        "rl": rl,
        "lx": lx,
        "w1": w1_mm,
        "w2": w2_mm,
        "wb1": wb1,
        "hb1": hb1,
        "wb2": wb2,
        "radial_shift": radial_shift_mm,
    }


def add_grouped_annular_sector(r_inner, r_outer, start_angle, end_angle, group=1, maxseg=1.0):
    p_inner_start = base.polar(r_inner, start_angle)
    p_outer_start = base.polar(r_outer, start_angle)
    p_outer_end = base.polar(r_outer, end_angle)
    p_inner_end = base.polar(r_inner, end_angle)

    base.add_segment(p_inner_start, p_outer_start)
    base.add_arc(r_outer, start_angle, end_angle, maxseg=maxseg)
    base.add_segment(p_outer_end, p_inner_end)
    base.add_arc(r_inner, start_angle, end_angle, maxseg=maxseg)

    for p1, p2 in ((p_inner_start, p_outer_start), (p_outer_end, p_inner_end)):
        mx = 0.5 * (p1[0] + p2[0])
        my = 0.5 * (p1[1] + p2[1])
        femm.mi_selectsegment(mx, my)
        femm.mi_setsegmentprop("<None>", 0, 1, 0, group)
        femm.mi_clearselected()

    mid_angle = 0.5 * (start_angle + end_angle)
    for radius in (r_inner, r_outer):
        femm.mi_selectarcsegment(*base.polar(radius, mid_angle))
        femm.mi_setarcsegmentprop(maxseg, "<None>", 0, group)
        femm.mi_clearselected()


def set_selected_segment_group(point_a, point_b, group):
    mx = 0.5 * (point_a[0] + point_b[0])
    my = 0.5 * (point_a[1] + point_b[1])
    femm.mi_selectsegment(mx, my)
    femm.mi_setsegmentprop("<None>", 0, 1, 0, group)
    femm.mi_clearselected()


def set_selected_arc_group(radius, start_angle, end_angle, group, maxseg):
    mid_angle = 0.5 * (start_angle + end_angle)
    femm.mi_selectarcsegment(*base.polar(radius, mid_angle))
    femm.mi_setarcsegmentprop(maxseg, "<None>", 0, group)
    femm.mi_clearselected()


def add_grouped_radial_segment(radius_start, radius_end, angle_rad, group=1):
    p1 = base.polar(radius_start, angle_rad)
    p2 = base.polar(radius_end, angle_rad)
    base.add_segment(p1, p2)
    set_selected_segment_group(p1, p2, group)


def add_grouped_arc(radius, start_angle, end_angle, group=1, maxseg=1.0):
    a0, a1 = sorted((start_angle, end_angle))
    base.add_arc(radius, a0, a1, maxseg=maxseg)
    set_selected_arc_group(radius, a0, a1, group, maxseg)


def add_grouped_u_air_cavity(rf, rm, rl, center_angle, mid_angle, outer_angle, group=1, maxseg=1.0):
    """Draw one L/U-shaped PM pocket instead of one filled annular sector."""
    rm_start, rm_end = sorted((center_angle, mid_angle))
    rl_start, rl_end = sorted((mid_angle, outer_angle))
    rf_start, rf_end = sorted((center_angle, outer_angle))

    add_grouped_radial_segment(rf, rm, center_angle, group=group)
    add_grouped_arc(rm, rm_start, rm_end, group=group, maxseg=maxseg)
    add_grouped_radial_segment(rm, rl, mid_angle, group=group)
    add_grouped_arc(rl, rl_start, rl_end, group=group, maxseg=maxseg)
    add_grouped_radial_segment(rf, rl, outer_angle, group=group)
    add_grouped_arc(rf, rf_start, rf_end, group=group, maxseg=maxseg)


def add_pm_sector_label(spec, radius, angle, magdir_deg, group=GROUP_ROTOR_INNER_PM):
    base.add_block_label(
        *base.polar(radius, angle),
        spec.magnet_material,
        magdir=magdir_deg,
        group=group,
    )


def add_air_cavity_label(radius, angle, group=GROUP_ROTOR_INNER_PM):
    base.add_block_label(*base.polar(radius, angle), "Air", group=group)


def angle_in_sector_window(angle_rad, sector_start_rad, sector_end_rad):
    angle = angle_rad
    while angle < sector_start_rad - 1e-9:
        angle += 2.0 * math.pi
    while angle > sector_end_rad + 1e-9:
        angle -= 2.0 * math.pi
    return angle


def add_slot_without_layer_chord(theta, spec):
    rs = spec.stator_inner_radius
    rso = spec.slot_top_radius
    rsb = spec.slot_bottom_radius
    rsm = 0.5 * (spec.slot_top_radius + spec.slot_bottom_radius)
    b0 = spec.slot_opening_span
    b = spec.slot_span

    p0 = base.polar(rs, theta - b0 / 2.0)
    p1 = base.polar(rso, theta - b0 / 2.0)
    p2 = base.polar(rso, theta - b / 2.0)
    p3 = base.polar(rsm, theta - b / 2.0)
    p3b = base.polar(rsb, theta - b / 2.0)
    p4 = base.polar(rsb, theta + b / 2.0)
    p4b = base.polar(rsm, theta + b / 2.0)
    p5 = base.polar(rso, theta + b / 2.0)
    p6 = base.polar(rso, theta + b0 / 2.0)
    p7 = base.polar(rs, theta + b0 / 2.0)

    base.add_segment(p0, p1)
    base.add_segment(p2, p3)
    base.add_segment(p3, p3b)
    base.add_segment(p4, p4b)
    base.add_segment(p4b, p5)
    base.add_segment(p6, p7)
    base.add_arc(rs, theta - b0 / 2.0, theta + b0 / 2.0)


def add_stator_inner_core_arcs(spec, stator_rotation_rad, sector_start_rad, sector_end_rad):
    """Draw Rs arcs only between slot openings so slot nodes close the stator."""
    opening_edges = []
    for slot_index in range(spec.slots):
        raw_theta = stator_rotation_rad + 2.0 * math.pi * slot_index / spec.slots
        if not base.angle_in_sector(raw_theta, sector_start_rad, sector_end_rad):
            continue
        theta = angle_in_sector_window(raw_theta, sector_start_rad, sector_end_rad)
        start = theta - spec.slot_opening_span / 2.0
        end = theta + spec.slot_opening_span / 2.0
        if start < sector_start_rad - 1e-9 or end > sector_end_rad + 1e-9:
            continue
        opening_edges.append((start, end))

    cursor = sector_start_rad
    for start, end in sorted(opening_edges):
        if start > cursor + 1e-9:
            base.add_sector_arc(spec.stator_inner_radius, cursor, start, maxseg=1.0)
        cursor = max(cursor, end)
    if sector_end_rad > cursor + 1e-9:
        base.add_sector_arc(spec.stator_inner_radius, cursor, sector_end_rad, maxseg=1.0)


def add_partition_arcs(radius, break_angles, group=ANALYTICAL_SUBDOMAIN_GROUP_DEFAULT, maxseg=1.0):
    angles = sorted(set(round(angle, 12) for angle in break_angles))
    for start, end in zip(angles, angles[1:]):
        if end > start + 1e-9:
            add_grouped_arc(radius, start, end, group=group, maxseg=maxseg)


def draw_stator_partition_boundaries(
    spec,
    stator_rotation_rad,
    sector_start_rad,
    sector_end_rad,
    group=ANALYTICAL_SUBDOMAIN_GROUP_DEFAULT,
):
    """Draw the checked stator radial partitions: Rso, Rsm, and Rsb."""
    rs = spec.stator_inner_radius
    rso = spec.slot_top_radius
    rsm = 0.5 * (spec.slot_top_radius + spec.slot_bottom_radius)
    rsb = spec.slot_bottom_radius
    b0 = spec.slot_opening_span
    b = spec.slot_span

    rs_breaks = [sector_start_rad, sector_end_rad]
    rso_breaks = [sector_start_rad, sector_end_rad]
    slot_breaks = [sector_start_rad, sector_end_rad]
    for slot_index in range(spec.slots):
        raw_theta = stator_rotation_rad + 2.0 * math.pi * slot_index / spec.slots
        if not base.angle_in_sector(raw_theta, sector_start_rad, sector_end_rad):
            continue
        theta = angle_in_sector_window(raw_theta, sector_start_rad, sector_end_rad)

        rs_breaks.extend((theta - b0 / 2.0, theta + b0 / 2.0))
        rso_breaks.extend((theta - b / 2.0, theta - b0 / 2.0, theta + b0 / 2.0, theta + b / 2.0))
        slot_breaks.extend((theta - b / 2.0, theta + b / 2.0))

    add_partition_arcs(rs, rs_breaks, group=group, maxseg=1.0)
    add_partition_arcs(rso, rso_breaks, group=group, maxseg=1.0)
    add_partition_arcs(rsm, slot_breaks, group=group, maxseg=1.0)
    add_partition_arcs(rsb, slot_breaks, group=group, maxseg=1.0)


def draw_stator_without_layer_chords(
    spec,
    stator_outer_radius,
    turns_per_layer,
    stator_rotation_rad,
    sector_start_rad,
    sector_end_rad,
):
    base.add_sector_arc(stator_outer_radius, sector_start_rad, sector_end_rad, maxseg=2.0, boundary="A0")

    for slot_index in range(spec.slots):
        raw_theta = stator_rotation_rad + 2.0 * math.pi * slot_index / spec.slots
        if base.angle_in_sector(raw_theta, sector_start_rad, sector_end_rad):
            theta = angle_in_sector_window(raw_theta, sector_start_rad, sector_end_rad)
            add_slot_without_layer_chord(theta, spec)
    draw_stator_partition_boundaries(
        spec,
        stator_rotation_rad,
        sector_start_rad,
        sector_end_rad,
        group=ANALYTICAL_SUBDOMAIN_GROUP_DEFAULT,
    )

    label_angle = 0.5 * (sector_start_rad + sector_end_rad)
    base.add_block_label(
        *base.polar((spec.slot_bottom_radius + stator_outer_radius) / 2.0, label_angle),
        base.CORE_MATERIAL_NAME,
        group=GROUP_STATOR_YOKE,
    )

    for slot_index in range(spec.slots):
        raw_theta = stator_rotation_rad + 2.0 * math.pi * slot_index / spec.slots
        if not base.angle_in_sector(raw_theta, sector_start_rad, sector_end_rad):
            continue
        theta = angle_in_sector_window(raw_theta, sector_start_rad, sector_end_rad)

        rsm = 0.5 * (spec.slot_top_radius + spec.slot_bottom_radius)
        base.add_block_label(
            *base.polar(0.5 * (spec.stator_inner_radius + spec.slot_top_radius), theta),
            "Air",
            group=GROUP_STATOR_TOOTH_TIP,
        )
        slot_angle = theta % (2.0 * math.pi)
        slot_number_zero_based = round((slot_angle - math.radians(5.0)) / (2.0 * math.pi / spec.slots)) % spec.slots
        (upper_phase, upper_sign), (lower_phase, lower_sign) = base.slot_layer_phases(slot_number_zero_based)

        base.add_block_label(
            *base.polar(0.5 * (spec.slot_top_radius + rsm), theta),
            "Copper",
            circuit=upper_phase,
            group=GROUP_STATOR_SLOT_UPPER,
            turns=upper_sign * turns_per_layer,
        )
        base.add_block_label(
            *base.polar(0.5 * (rsm + spec.slot_bottom_radius), theta),
            "Copper",
            circuit=lower_phase,
            group=GROUP_STATOR_SLOT_LOWER,
            turns=lower_sign * turns_per_layer,
        )


def add_stator_core_subdomain_labels(
    spec,
    stator_outer_radius,
    stator_rotation_rad,
    sector_start_rad,
    sector_end_rad,
):
    slot_pitch = 2.0 * math.pi / spec.slots
    rso = spec.slot_top_radius
    rsm = 0.5 * (spec.slot_top_radius + spec.slot_bottom_radius)
    rsb = spec.slot_bottom_radius
    core_regions = (
        (0.5 * (spec.stator_inner_radius + rso), GROUP_STATOR_TOOTH_TIP),
        (0.5 * (rso + rsm), GROUP_STATOR_SLOT_UPPER),
        (0.5 * (rsm + rsb), GROUP_STATOR_SLOT_LOWER),
    )
    slot_centers = []
    for slot_index in range(spec.slots):
        slot_theta = stator_rotation_rad + slot_index * slot_pitch
        if not base.angle_in_sector(slot_theta, sector_start_rad, sector_end_rad):
            continue
        slot_theta = angle_in_sector_window(slot_theta, sector_start_rad, sector_end_rad)
        slot_centers.append(slot_theta)

    slot_centers = sorted(slot_centers)
    tooth_angles = []
    for left, right in zip(slot_centers, slot_centers[1:]):
        tooth_angles.append(0.5 * (left + right))

    if slot_centers:
        tooth_angles.append(0.5 * (sector_start_rad + slot_centers[0] - 0.5 * spec.slot_span))
        tooth_angles.append(0.5 * (slot_centers[-1] + 0.5 * spec.slot_span + sector_end_rad))

    for tooth_theta in sorted(tooth_angles):
        if not (sector_start_rad + 1e-9 < tooth_theta < sector_end_rad - 1e-9):
            continue
        for radius, group in core_regions:
            base.add_block_label(
                *base.polar(radius, tooth_theta),
                base.CORE_MATERIAL_NAME,
                group=group,
            )


def draw_rotor_without_center_core_label(spec, shaft_radius, dims, sector_start_rad, sector_end_rad):
    base.add_sector_arc(spec.rotor_outer_radius, sector_start_rad, sector_end_rad, maxseg=1.0)
    base.add_sector_arc(shaft_radius, sector_start_rad, sector_end_rad, maxseg=2.5)
    label_angle = 0.5 * (sector_start_rad + sector_end_rad)
    base.add_block_label(*base.polar(shaft_radius / 2.0, label_angle), "Air", group=GROUP_SHAFT_AIR)


def add_rotor_core_subdomain_labels(spec, dims, rotor_rotation_rad, sector_start_rad, sector_end_rad, shaft_radius):
    alpha = dims["alpha_rad"]
    alpha1 = dims["alpha1_rad"]
    rf = dims["rf"]
    rm = dims["rm"]
    rl = dims["rl"]
    w1 = dims["w1"]
    pole_pitch = 2.0 * math.pi / spec.poles

    for pole in range(spec.poles):
        theta = rotor_rotation_rad + pole * pole_pitch
        if not base.angle_in_sector(theta, sector_start_rad, sector_end_rad):
            continue

        center_gap = dims["wb2"] / (2.0 * rm)
        tangential_start = center_gap + dims["w2"] / rm
        angles = [
            theta - alpha / 2.0,
            theta - alpha1 / 2.0,
            theta - tangential_start,
            theta - center_gap,
            theta + center_gap,
            theta + tangential_start,
            theta + alpha1 / 2.0,
            theta + alpha / 2.0,
        ]

        def add_if_inside(radius, angle, material, group):
            if base.angle_in_sector(angle, sector_start_rad, sector_end_rad):
                base.add_block_label(*base.polar(radius, angle), material, group=group)

        add_if_inside(0.5 * (shaft_radius + rf), theta, base.CORE_MATERIAL_NAME, GROUP_ROTOR_INNER_CORE)
        add_if_inside(0.5 * (rf + rm), theta, base.CORE_MATERIAL_NAME, GROUP_ROTOR_INNER_PM)
        add_if_inside(0.5 * (rl + spec.rotor_outer_radius), theta, base.CORE_MATERIAL_NAME, GROUP_ROTOR_OUTER)
        add_if_inside(0.5 * (rm + rl - w1), theta, base.CORE_MATERIAL_NAME, GROUP_ROTOR_RADIAL_PM)
        add_if_inside(0.5 * (rl - w1 + rl), theta, base.CORE_MATERIAL_NAME, GROUP_ROTOR_SIDE_BRIDGE)


def add_rotor_center_gap_air_labels(spec, dims, rotor_rotation_rad, sector_start_rad, sector_end_rad):
    rf = dims["rf"]
    rm = dims["rm"]
    pole_pitch = 2.0 * math.pi / spec.poles
    for pole in range(spec.poles):
        theta = rotor_rotation_rad + pole * pole_pitch
        if base.angle_in_sector(theta, sector_start_rad, sector_end_rad):
            base.add_block_label(*base.polar(0.5 * (rf + rm), theta), base.CORE_MATERIAL_NAME, group=GROUP_ROTOR_INNER_PM)


def add_rotor_edge_core_labels(spec, dims, rotor_rotation_rad, sector_start_rad, sector_end_rad):
    alpha = dims["alpha_rad"]
    rf = dims["rf"]
    rm = dims["rm"]
    rl = dims["rl"]
    w1 = dims["w1"]
    pole_pitch = 2.0 * math.pi / spec.poles

    for pole in range(spec.poles):
        theta = rotor_rotation_rad + pole * pole_pitch
        if not base.angle_in_sector(theta, sector_start_rad, sector_end_rad):
            continue

        pole_start = theta - alpha / 2.0
        pole_end = theta + alpha / 2.0
        edge_angle_ranges = (
            (sector_start_rad, pole_start),
            (pole_end, sector_end_rad),
        )
        edge_radius_groups = (
            (0.5 * (rf + rm), GROUP_ROTOR_INNER_PM),
            (0.5 * (rm + rl - w1), GROUP_ROTOR_RADIAL_PM),
            (0.5 * (rl - w1 + rl), GROUP_ROTOR_SIDE_BRIDGE),
        )
        for start, end in edge_angle_ranges:
            if end <= start:
                continue
            angle = 0.5 * (start + end)
            for radius, group in edge_radius_groups:
                base.add_block_label(*base.polar(radius, angle), base.CORE_MATERIAL_NAME, group=group)


def add_sector_side_boundaries_with_subdomain_cuts(
    sector_start_rad,
    sector_end_rad,
    shaft_radius,
    spec,
    stator_outer_radius,
    dims,
    boundary_name,
    boundary_kind,
):
    cut_radii = {
        0.0,
        shaft_radius,
        dims["rf"],
        dims["rm"],
        dims["rl"] - dims["w1"],
        dims["rl"],
        spec.rotor_outer_radius,
        spec.stator_inner_radius,
        spec.slot_top_radius,
        0.5 * (spec.slot_top_radius + spec.slot_bottom_radius),
        spec.slot_bottom_radius,
        stator_outer_radius,
    }
    sorted_radii = sorted(radius for radius in cut_radii if 0.0 <= radius <= stator_outer_radius)
    for idx, (radius_start, radius_end) in enumerate(zip(sorted_radii, sorted_radii[1:]), start=1):
        if radius_end - radius_start <= 1e-6:
            continue
        pair_boundary_name = f"{boundary_name}_{idx}"
        base.add_sector_boundary_property(pair_boundary_name, boundary_kind)
        for angle_rad in (sector_start_rad, sector_end_rad):
            base.add_radial_segment(radius_start, radius_end, angle_rad, boundary=pair_boundary_name)


def add_sector_edge_subdomain_labels(
    spec,
    stator_outer_radius,
    shaft_radius,
    dims,
    sector_start_rad,
    sector_end_rad,
):
    rsm = 0.5 * (spec.slot_top_radius + spec.slot_bottom_radius)
    radial_bands = [
        (0.5 * shaft_radius, "Air", 0),
        (0.5 * (shaft_radius + dims["rf"]), base.CORE_MATERIAL_NAME, 1),
        (0.5 * (dims["rf"] + dims["rm"]), base.CORE_MATERIAL_NAME, 1),
        (0.5 * (dims["rm"] + dims["rl"] - dims["w1"]), base.CORE_MATERIAL_NAME, 1),
        (0.5 * (dims["rl"] - dims["w1"] + dims["rl"]), base.CORE_MATERIAL_NAME, 1),
        (0.5 * (dims["rl"] + spec.rotor_outer_radius), base.CORE_MATERIAL_NAME, 1),
        (0.5 * (spec.rotor_outer_radius + spec.stator_inner_radius), "Air", 0),
        (0.5 * (spec.stator_inner_radius + spec.slot_top_radius), base.CORE_MATERIAL_NAME, 10),
        (0.5 * (spec.slot_top_radius + rsm), base.CORE_MATERIAL_NAME, 10),
        (0.5 * (rsm + spec.slot_bottom_radius), base.CORE_MATERIAL_NAME, 10),
        (0.5 * (spec.slot_bottom_radius + stator_outer_radius), base.CORE_MATERIAL_NAME, 10),
    ]
    edge_margin = math.radians(0.15)
    for angle in (sector_start_rad + edge_margin, sector_end_rad - edge_margin):
        for radius, material, group in radial_bands:
            base.add_block_label(*base.polar(radius, angle), material, group=group)


def draw_stator_analytical_subdomain_boundaries(
    spec,
    stator_rotation_rad,
    sector_start_rad,
    sector_end_rad,
    group=ANALYTICAL_SUBDOMAIN_GROUP_DEFAULT,
):
    """Stator partitions are required geometry and are drawn with the stator."""


def draw_rotor_analytical_subdomain_boundaries(
    spec,
    rotor_rotation_rad,
    dims,
    sector_start_rad,
    sector_end_rad,
    group=ANALYTICAL_SUBDOMAIN_GROUP_DEFAULT,
):
    """Overlay paper Fig. 3 rotor subdomain split lines on the curved FEMM sector."""
    alpha = dims["alpha_rad"]
    alpha1 = dims["alpha1_rad"]
    rf = dims["rf"]
    rm = dims["rm"]
    rl = dims["rl"]
    w1 = dims["w1"]
    w2 = dims["w2"]
    wb2 = dims["wb2"]

    pole_pitch = 2.0 * math.pi / spec.poles
    center_gap = wb2 / (2.0 * rm)
    center_air_gap = w2 / rm
    tangential_start = center_gap + center_air_gap
    if alpha1 / 2.0 <= tangential_start:
        return

    for pole in range(spec.poles):
        theta = rotor_rotation_rad + pole * pole_pitch
        if not base.angle_in_sector(theta, sector_start_rad, sector_end_rad):
            continue

        pole_start = theta - alpha / 2.0
        pole_end = theta + alpha / 2.0
        pm_start = theta - alpha1 / 2.0
        pm_end = theta + alpha1 / 2.0
        center_left = theta - center_gap
        center_right = theta + center_gap
        curved_pm_left = theta - tangential_start
        curved_pm_right = theta + tangential_start

        # Main rotor subdomain rings.  In the checked FEMM reference these
        # rings close the whole 1/6 sector, not only the magnet pole span.
        for radius in (rf, rm, rl - w1, rl):
            add_grouped_arc(radius, sector_start_rad, sector_end_rad, group=group, maxseg=1.0)

        # Outer bridge / PM end cuts close only the rotor subdomain stack up to
        # Rl.  The Rl..Rr bridge band stays continuous, matching the reference.
        for angle in (pole_start, pm_start, pm_end, pole_end):
            add_grouped_radial_segment(rf, rl, angle, group=group)

        # Local PM and center-bridge interfaces exist only across the inner PM
        # radial span Rf..Rm.
        for angle in (curved_pm_left, center_left, center_right, curved_pm_right):
            add_grouped_radial_segment(rf, rm, angle, group=group)


def draw_analytical_subdomain_boundaries(
    spec,
    stator_rotation_rad,
    rotor_rotation_rad,
    dims,
    sector_start_rad,
    sector_end_rad,
    group=ANALYTICAL_SUBDOMAIN_GROUP_DEFAULT,
):
    draw_stator_analytical_subdomain_boundaries(
        spec,
        stator_rotation_rad,
        sector_start_rad,
        sector_end_rad,
        group=group,
    )
    draw_rotor_analytical_subdomain_boundaries(
        spec,
        rotor_rotation_rad,
        dims,
        sector_start_rad,
        sector_end_rad,
        group=group,
    )


def draw_equivalent_vshape_magnets(
    spec,
    rotor_rotation_rad,
    dims,
    sector_start_rad=None,
    sector_end_rad=None,
):
    alpha = dims["alpha_rad"]
    alpha1 = dims["alpha1_rad"]
    rf = dims["rf"]
    rm = dims["rm"]
    rl = dims["rl"]
    w1 = dims["w1"]
    w2 = dims["w2"]
    wb2 = dims["wb2"]

    pole_pitch = 2.0 * math.pi / spec.poles
    center_gap = wb2 / (2.0 * rm)
    center_air_gap = w2 / rm
    tangential_start = center_gap + center_air_gap
    tangential_end = alpha1 / 2.0
    if tangential_end <= tangential_start:
        raise ValueError(
            "Tangential equivalent PM span is not positive. "
            "Adjust alpha/w2/wb2."
        )

    for pole in range(spec.poles):
        theta = rotor_rotation_rad + pole * pole_pitch
        if sector_start_rad is not None and not base.angle_in_sector(theta, sector_start_rad, sector_end_rad):
            continue
        polarity = 0 if pole % 2 == 0 else 180

        for side in (-1, 1):
            if side < 0:
                cavity_center = theta - center_gap
                cavity_mid = theta - alpha1 / 2.0
                cavity_outer = theta - alpha / 2.0
            else:
                cavity_center = theta + center_gap
                cavity_mid = theta + alpha1 / 2.0
                cavity_outer = theta + alpha / 2.0

            # One connected PM pocket per side. Its outline is U/L-shaped:
            # a curved lower arm for the tangential PM and a radial side arm
            # for the radial PM. The corner outside those arms remains core.
            add_grouped_u_air_cavity(
                rf,
                rm,
                rl,
                cavity_center,
                cavity_mid,
                cavity_outer,
                group=1,
                maxseg=1.0,
            )
            add_air_cavity_label(
                0.5 * (rf + rm),
                theta + side * (center_gap + 0.5 * center_air_gap),
                group=GROUP_ROTOR_INNER_PM,
            )
            add_air_cavity_label(
                0.5 * (rf + rm),
                theta + side * (0.5 * (alpha1 / 2.0 + alpha / 2.0)),
                group=GROUP_ROTOR_INNER_PM,
            )

            if side < 0:
                start = theta - tangential_end
                end = theta - tangential_start
            else:
                start = theta + tangential_start
                end = theta + tangential_end

            add_grouped_annular_sector(rf, rm, start, end, group=1, maxseg=1.0)
            mid_angle = 0.5 * (start + end)
            add_pm_sector_label(
                spec,
                0.5 * (rf + rm),
                mid_angle,
                math.degrees(mid_angle) + polarity,
                group=GROUP_ROTOR_INNER_PM,
            )

            if side < 0:
                side_start = theta - alpha / 2.0
                side_end = theta - alpha1 / 2.0
            else:
                side_start = theta + alpha1 / 2.0
                side_end = theta + alpha / 2.0

            add_grouped_annular_sector(rm, rl - w1, side_start, side_end, group=1, maxseg=1.0)
            side_mid = 0.5 * (side_start + side_end)
            add_air_cavity_label(rl - 0.5 * w1, side_mid, group=GROUP_ROTOR_SIDE_BRIDGE)
            # Side PMs are the paper's radial PM parts. Their magnetization is
            # tangential and points toward the pole center for the north pole.
            add_pm_sector_label(
                spec,
                0.5 * (rm + rl - w1),
                side_mid,
                math.degrees(side_mid - side * math.pi / 2.0) + polarity,
                group=GROUP_ROTOR_RADIAL_PM,
            )


def build_model(
    output_path=EQUIVALENT_OUTPUT_DEFAULT,
    stator_outer_radius=base.STATOR_OUTER_RADIUS_MM_DEFAULT,
    shaft_radius=base.SHAFT_RADIUS_MM_DEFAULT,
    current_rms_a=EQUIVALENT_CURRENT_RMS_A_DEFAULT,
    turns_per_layer=base.TURNS_PER_LAYER_DEFAULT,
    w1_mm=EQUIVALENT_W1_MM_DEFAULT,
    w2_mm=EQUIVALENT_W2_MM_DEFAULT,
    equivalent_radial_shift_mm=EQUIVALENT_RADIAL_SHIFT_MM_DEFAULT,
    wb1_mm=None,
    hb1_mm=None,
    wb2_mm=None,
    alpha_deg=None,
    stator_rotation_deg=base.STATOR_ROTATION_DEG_DEFAULT,
    rotor_rotation_deg=base.ROTOR_ROTATION_DEG_DEFAULT,
    sector_start_deg=base.SECTOR_START_DEG_DEFAULT,
    sector_span_deg=base.SECTOR_SPAN_DEG_DEFAULT,
    sector_boundary_kind=base.SECTOR_BOUNDARY_KIND_DEFAULT,
    draw_analytical_subdomains=DRAW_ANALYTICAL_SUBDOMAIN_BOUNDARIES_DEFAULT,
    analytical_subdomain_group=ANALYTICAL_SUBDOMAIN_GROUP_DEFAULT,
):
    spec = base.PAPER_SPECS["vshape"]
    stator_rotation_rad = math.radians(stator_rotation_deg)
    rotor_rotation_rad = math.radians(rotor_rotation_deg)
    sector_start_rad = math.radians(sector_start_deg)
    sector_end_rad = math.radians(sector_start_deg + sector_span_deg)
    sector_boundary_name = "SectorPeriodic"
    phase_currents = base.phase_currents_from_rms(current_rms_a)
    dims = equivalent_pm_dimensions(
        spec,
        alpha_deg=alpha_deg,
        w1_mm=w1_mm,
        w2_mm=w2_mm,
        wb1_mm=wb1_mm,
        hb1_mm=hb1_mm,
        wb2_mm=wb2_mm,
        radial_shift_mm=equivalent_radial_shift_mm,
    )

    femm.openfemm(1)
    try:
        femm.newdocument(0)
        femm.mi_probdef(0, "millimeters", "planar", 1e-8, spec.stack_length, 30)
        femm.mi_addboundprop("A0", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        femm.mi_addcircprop("Ia", phase_currents["Ia"], 1)
        femm.mi_addcircprop("Ib", phase_currents["Ib"], 1)
        femm.mi_addcircprop("Ic", phase_currents["Ic"], 1)
        base.add_materials(spec)

        draw_stator_without_layer_chords(
            spec,
            stator_outer_radius,
            turns_per_layer,
            stator_rotation_rad,
            sector_start_rad,
            sector_end_rad,
        )
        draw_rotor_without_center_core_label(spec, shaft_radius, dims, sector_start_rad, sector_end_rad)
        add_sector_side_boundaries_with_subdomain_cuts(
            sector_start_rad,
            sector_end_rad,
            shaft_radius,
            spec,
            stator_outer_radius,
            dims,
            sector_boundary_name,
            sector_boundary_kind,
        )
        base.add_block_label(
            *base.polar((spec.rotor_outer_radius + spec.stator_inner_radius) / 2.0, 0.5 * (sector_start_rad + sector_end_rad)),
            "Air",
            group=GROUP_AIRGAP,
        )

        draw_equivalent_vshape_magnets(
            spec,
            rotor_rotation_rad,
            dims,
            sector_start_rad,
            sector_end_rad,
        )
        if draw_analytical_subdomains:
            draw_analytical_subdomain_boundaries(
                spec,
                stator_rotation_rad,
                rotor_rotation_rad,
                dims,
                sector_start_rad,
                sector_end_rad,
                group=analytical_subdomain_group,
            )
            add_stator_core_subdomain_labels(
                spec,
                stator_outer_radius,
                stator_rotation_rad,
                sector_start_rad,
                sector_end_rad,
            )
            add_rotor_core_subdomain_labels(
                spec,
                dims,
                rotor_rotation_rad,
                sector_start_rad,
                sector_end_rad,
                shaft_radius,
            )
            add_rotor_edge_core_labels(
                spec,
                dims,
                rotor_rotation_rad,
                sector_start_rad,
                sector_end_rad,
            )

        femm.mi_zoomnatural()
        output = Path(output_path).absolute()
        femm.mi_saveas(str(output))
    finally:
        femm.closefemm()

    return dims


def main():
    parser = argparse.ArgumentParser(
        description="Build the paper Fig. 1(a) equivalent V-shape IPM FEMM sector model."
    )
    parser.add_argument("--output", default=EQUIVALENT_OUTPUT_DEFAULT)
    parser.add_argument("--turns-per-layer", type=int, default=base.TURNS_PER_LAYER_DEFAULT)
    parser.add_argument("--stator-outer-radius", type=float, default=base.STATOR_OUTER_RADIUS_MM_DEFAULT)
    parser.add_argument("--shaft-radius", type=float, default=base.SHAFT_RADIUS_MM_DEFAULT)
    parser.add_argument("--current-rms-a", type=float, default=EQUIVALENT_CURRENT_RMS_A_DEFAULT)
    parser.add_argument("--w1-mm", type=float, default=EQUIVALENT_W1_MM_DEFAULT)
    parser.add_argument("--w2-mm", type=float, default=EQUIVALENT_W2_MM_DEFAULT)
    parser.add_argument(
        "--equivalent-radial-shift-mm",
        "--magnet-radial-shift-mm",
        dest="equivalent_radial_shift_mm",
        type=float,
        default=EQUIVALENT_RADIAL_SHIFT_MM_DEFAULT,
        help="Move the whole equivalent PM + Air pocket radially; positive is outward.",
    )
    parser.add_argument("--wb1-mm", type=float)
    parser.add_argument("--hb1-mm", type=float)
    parser.add_argument("--wb2-mm", type=float)
    parser.add_argument("--alpha-deg", type=float, help="Default: pole pitch minus two wb1/Rr edge bridge spans.")
    parser.add_argument("--stator-rotation-deg", type=float, default=base.STATOR_ROTATION_DEG_DEFAULT)
    parser.add_argument("--rotor-rotation-deg", type=float, default=base.ROTOR_ROTATION_DEG_DEFAULT)
    parser.add_argument("--sector-start-deg", type=float, default=base.SECTOR_START_DEG_DEFAULT)
    parser.add_argument("--sector-span-deg", type=float, default=base.SECTOR_SPAN_DEG_DEFAULT)
    parser.add_argument(
        "--sector-boundary-kind",
        choices=("anti-periodic", "periodic"),
        default=base.SECTOR_BOUNDARY_KIND_DEFAULT,
    )
    parser.add_argument(
        "--analytical-subdomains",
        action="store_true",
        dest="draw_analytical_subdomains",
        default=DRAW_ANALYTICAL_SUBDOMAIN_BOUNDARIES_DEFAULT,
        help="Draw the paper-style internal subdomain boundary overlay.",
    )
    parser.add_argument(
        "--no-analytical-subdomains",
        action="store_false",
        dest="draw_analytical_subdomains",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--analytical-subdomain-group",
        type=int,
        default=ANALYTICAL_SUBDOMAIN_GROUP_DEFAULT,
        help="FEMM group assigned to the internal analytical subdomain boundary lines.",
    )
    args = parser.parse_args()

    dims = build_model(
        output_path=args.output,
        stator_outer_radius=args.stator_outer_radius,
        shaft_radius=args.shaft_radius,
        current_rms_a=args.current_rms_a,
        turns_per_layer=args.turns_per_layer,
        w1_mm=args.w1_mm,
        w2_mm=args.w2_mm,
        equivalent_radial_shift_mm=args.equivalent_radial_shift_mm,
        wb1_mm=args.wb1_mm,
        hb1_mm=args.hb1_mm,
        wb2_mm=args.wb2_mm,
        alpha_deg=args.alpha_deg,
        stator_rotation_deg=args.stator_rotation_deg,
        rotor_rotation_deg=args.rotor_rotation_deg,
        sector_start_deg=args.sector_start_deg,
        sector_span_deg=args.sector_span_deg,
        sector_boundary_kind=args.sector_boundary_kind,
        draw_analytical_subdomains=args.draw_analytical_subdomains,
        analytical_subdomain_group=args.analytical_subdomain_group,
    )
    print(
        f"saved {args.output} "
        f"(alpha={dims['alpha_deg']:.6g}deg, alpha1={dims['alpha1_deg']:.6g}deg, "
        f"Rf={dims['rf']:.6g}mm, Rm={dims['rm']:.6g}mm, Rl={dims['rl']:.6g}mm, "
        f"lx={dims['lx']:.6g}mm, w1={dims['w1']:.6g}mm, w2={dims['w2']:.6g}mm, "
        f"radial_shift={dims['radial_shift']:.6g}mm)"
    )


if __name__ == "__main__":
    main()
