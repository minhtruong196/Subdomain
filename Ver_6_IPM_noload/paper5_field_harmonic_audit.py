import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULT_DIR = "results/paper5_harmonic_audit"
DEFAULT_DATASETS = [
    "slotless FEMM=results/paper5_slotless_equivalent_1over6/br_bt_arc.csv",
    "slotted FEMM=results/paper_vshape_equivalent_1over6/br_bt_arc.csv",
    "Structure2 h7=results/paper5_structure2_offset30_h7/structure2_br_bt.csv",
    "S2 h7 + slotting=results/paper5_slotting_structure2_h7_empirical_lh6/slotting_corrected_br_bt.csv",
]


def parse_dataset(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("dataset must be name=csv_path")
    name, path = text.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("dataset name and path must not be empty")
    return name, path


def read_field_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "angle_deg": np.array([float(row["angle_deg"]) for row in rows]),
        "Br_T": np.array([float(row["Br_T"]) for row in rows]),
        "Bt_T": np.array([float(row["Bt_T"]) for row in rows]),
    }


def resample(data, angle_deg):
    if len(data["angle_deg"]) == len(angle_deg) and np.max(np.abs(data["angle_deg"] - angle_deg)) < 1e-9:
        return data
    return {
        "angle_deg": angle_deg,
        "Br_T": np.interp(angle_deg, data["angle_deg"], data["Br_T"]),
        "Bt_T": np.interp(angle_deg, data["angle_deg"], data["Bt_T"]),
    }


def harmonic_coefficients(angle_deg, values, max_harmonic):
    theta = 2.0 * np.pi * (angle_deg - angle_deg[0]) / (angle_deg[-1] - angle_deg[0])
    span = theta[-1] - theta[0]
    a0 = (1.0 / span) * np.trapezoid(values, theta)
    cos_coeff = np.zeros(max_harmonic + 1)
    sin_coeff = np.zeros(max_harmonic + 1)
    mag = np.zeros(max_harmonic + 1)
    cos_coeff[0] = a0
    mag[0] = abs(a0)
    for h in range(1, max_harmonic + 1):
        a = (2.0 / span) * np.trapezoid(values * np.cos(h * theta), theta)
        b = (2.0 / span) * np.trapezoid(values * np.sin(h * theta), theta)
        cos_coeff[h] = a
        sin_coeff[h] = b
        mag[h] = np.hypot(a, b)
    return cos_coeff, sin_coeff, mag


def write_coeff_csv(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "component",
                "harmonic",
                "cos_coeff_T",
                "sin_coeff_T",
                "magnitude_T",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output


def plot_spectra(out_dir, spectra, max_harmonic):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "field_harmonic_spectrum.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    harmonics = np.arange(max_harmonic + 1)
    for name, components in spectra.items():
        axes[0].semilogy(harmonics, components["Br_T"], marker="o", markersize=2.4, linewidth=1.2, label=name)
        axes[1].semilogy(harmonics, components["Bt_T"], marker="o", markersize=2.4, linewidth=1.2, label=name)
    axes[0].set_ylabel("Br harmonic magnitude (T)")
    axes[1].set_ylabel("Bt harmonic magnitude (T)")
    axes[1].set_xlabel("Fourier harmonic over 60 deg sector")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Audit harmonic content of FEMM and analytical airgap fields.")
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--max-harmonic", type=int, default=60)
    parser.add_argument("--dataset", action="append", type=parse_dataset)
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = args.dataset or [parse_dataset(item) for item in DEFAULT_DATASETS]
    base_data = read_field_csv(datasets[0][1])
    angle_deg = base_data["angle_deg"]
    spectra = {}
    rows = []

    for name, path in datasets:
        data = resample(read_field_csv(path), angle_deg)
        spectra[name] = {}
        for component in ("Br_T", "Bt_T"):
            cos_coeff, sin_coeff, mag = harmonic_coefficients(
                angle_deg, data[component], args.max_harmonic
            )
            spectra[name][component] = mag
            for h in range(args.max_harmonic + 1):
                rows.append(
                    {
                        "dataset": name,
                        "component": component,
                        "harmonic": h,
                        "cos_coeff_T": cos_coeff[h],
                        "sin_coeff_T": sin_coeff[h],
                        "magnitude_T": mag[h],
                    }
                )

    coeff_csv = write_coeff_csv(Path(args.result_dir) / "field_harmonic_coefficients.csv", rows)
    plot_path = plot_spectra(args.result_dir, spectra, args.max_harmonic)
    print(f"Harmonic coefficients: {coeff_csv}")
    print(f"Harmonic spectrum plot: {plot_path}")


if __name__ == "__main__":
    main()
