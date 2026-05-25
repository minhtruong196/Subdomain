import numpy as np
import matplotlib
from me_boundary_matrix import UnknownLayout, BoundaryMatrixParams, solve_boundary_matrix, selected_segment
from pathlib import Path
from dataclasses import dataclass

matplotlib.use("Agg")
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class FluxDensityParams:
    boundary: BoundaryMatrixParams = BoundaryMatrixParams()
    airgap_radius_m: float = 88.4 * 1e-3
    sample_count: int = 721 #Số điểm lấy mẫu dọc theo airgap để vẽ đồ thị
    output_dir: str = "outputs/step4_flux_density"

#=================================
# Tính toán các công thức liên quan đến P và E từ equa. (5) nhưng tránh bị nổ do số quá lớn
#Note: (1), thay vì thay số trực tiếp quy hết về numpy tránh nổ
def Ez_x_y(order: float, x: float, y: float) -> tuple[float, float]:
    a = float(order * np.log(x / y)) 
    if np.isclose(a, 0.0):      #if a = 0
        return 0.0, -np.inf
    sign = 1.0 if a > 0 else -1.0
    abs_a = abs(a)
    if abs_a > 25.0:            # if a > 25 or a < -25
        return sign, abs_a
    return sign, float(np.log(abs(2.0 * np.sinh(a))))

def p_over_e(order: float, p_x: float, p_y: float, e_x: float, e_y: float) -> float:
    a = float(order * np.log(p_x / p_y))
    log_p = float(np.logaddexp(a, -a))  #Tượng tự Ez_x_y cũng quy đổi qua numpy để tránh nổ
    sign_e, log_abs_Ez = Ez_x_y(order, e_x, e_y)
    return float(sign_e * np.exp(log_p - log_abs_Ez))

#Vì có 2 E nên cần tính 2 sign và 2 log_abs_Ez
def e_over_e(order: float, e1_x: float, e1_y: float, e2_x: float, e2_y: float) -> float:
    sign_e1, log_abs_Ez1 = Ez_x_y(order, e1_x, e1_y)
    if sign_e1 == 0.0:
        return 0.0          #Nếu tử là 0 thì kết quả là 0
    sign_e2, log_abs_Ez2 = Ez_x_y(order, e2_x, e2_y)
    return float(sign_e1 * sign_e2 * np.exp(log_abs_Ez1 - log_abs_Ez2))
#=================================

# Viết một số term để tính toán sau
def az2_term(r: float, Ru: float, Rs: float, mc_value: np.ndarray, 
             Am2: np.ndarray, Bm2: np.ndarray, Cm2: np.ndarray, Dm2: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # Note: (2)
    alpha = np.empty_like(mc_value, dtype=float)
    beta = np.empty_like(mc_value, dtype=float)
    alpha_derivative_r = np.empty_like(mc_value, dtype=float)
    beta_derivative_r = np.empty_like(mc_value, dtype=float)

    for idx, mc in enumerate(mc_value):
        a_part = Ru * p_over_e(mc, r, Rs, Ru, Rs) / mc
        b_part = Rs * p_over_e(mc, r, Ru, Rs, Ru) / mc
        a_derivative_r_part = Ru * e_over_e(mc, r, Rs, Ru, Rs) / r
        b_derivative_r_part = Rs * e_over_e(mc, r, Ru, Rs, Ru) / r

        alpha[idx] = Am2[idx] * a_part + Bm2[idx] * b_part
        beta[idx] = Cm2[idx] * a_part + Dm2[idx] * b_part
        alpha_derivative_r[idx] = Am2[idx] * a_derivative_r_part + Bm2[idx] * b_derivative_r_part
        beta_derivative_r[idx] = Cm2[idx] * a_derivative_r_part + Dm2[idx] * b_derivative_r_part

    return alpha, beta, alpha_derivative_r, beta_derivative_r

#Flux density của một segment, segment được ẩn trong solution từ boundary matrix
def segment_flux_density(r: float, theta_mech_rad: np.ndarray, solution: np.ndarray, layout: UnknownLayout,
                         mc_value: np.ndarray, Ru: float, Rs: float) -> tuple[np.ndarray, np.ndarray]:

    # Lấy phần tử fourier của ma trận
    Am2 = solution[layout.started_pos_Am2 : layout.started_pos_Am2 + layout.m_count]
    Bm2 = solution[layout.started_pos_Bm2 : layout.started_pos_Bm2 + layout.m_count]
    Cm2 = solution[layout.started_pos_Cm2 : layout.started_pos_Cm2 + layout.m_count]
    Dm2 = solution[layout.started_pos_Dm2 : layout.started_pos_Dm2 + layout.m_count]
    alpha, beta, alpha_derivative_r, beta_derivative_r = az2_term(r, Ru, Rs, mc_value, Am2, Bm2, Cm2, Dm2)

    phase = np.outer(mc_value, theta_mech_rad)

    #Note: (3), Equa. (28)
    Br = np.sum((- mc_value * alpha / r)[:, None] * np.sin(phase) + (mc_value * beta / r)[:, None] * np.cos(phase), axis=0)
    Btheta = -np.sum(alpha_derivative_r[:, None] * np.cos(phase) + beta_derivative_r[:, None] * np.sin(phase), axis=0)

    return Br, Btheta

#Equa. (29)
def total_flux_density(params: FluxDensityParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    p = params.boundary.geometry.pole_pairs
    #Với góc điện một cực luôn là 180 độ, hai cực sẽ là 360 độ với tất cả động cơ
    #Chọn -90 đến 90 để tâm là 0 độ cho đối xứng
    theta_elec_deg = np.linspace(-90, 90, params.sample_count)
    theta_mech_rad = np.deg2rad(theta_elec_deg) / p  

    Br_total = np.zeros_like(theta_mech_rad)
    Btheta_total = np.zeros_like(theta_mech_rad)
    residuals = []  #Để nhét phần tử

    for segment_j in range(1, params.boundary.geometry.Nz + 1):
        boundary_params = BoundaryMatrixParams(
                                               segment_j=segment_j, 
                                               geometry=params.boundary.geometry, pole_harmonic=params.boundary.pole_harmonic,
                                               slot_harmonic=params.boundary.slot_harmonic, magnetization_model=params.boundary.magnetization_model,
                                               Br_T=params.boundary.Br_T, mu_0=params.boundary.mu_0, mu_r=params.boundary.mu_r, 
                                               stator_inner_radius_m=params.boundary.stator_inner_radius_m, slot_depth_m=params.boundary.slot_depth_m,
                                               slot_width_m=params.boundary.slot_width_m, Jui_A_per_m2=params.boundary.Jui_A_per_m2,
                                               Jdi_A_per_m2=params.boundary.Jdi_A_per_m2, delta_rad=params.boundary.delta_rad, 
                                               output_dir=params.boundary.output_dir
                                               )

        solution, _, _, layout, meta, residual = solve_boundary_matrix(boundary_params)
        residuals.append(residual)
        Ru, _ = selected_segment(boundary_params)
        #Chuyển thành Numpy để tính toán
        mc_value = np.asarray(meta["mc_value"], dtype=float)

        Br_j, Btheta_j = segment_flux_density(params.airgap_radius_m, theta_mech_rad, solution, 
                                              layout, mc_value, Ru, boundary_params.stator_inner_radius_m)
        
        Br_total += Br_j
        Btheta_total += Btheta_j

    return theta_elec_deg, Br_total, Btheta_total, residuals

def save_flux_outputs(theta_elec_deg: np.ndarray, Br_total: np.ndarray, Btheta_total: np.ndarray,
                      residuals: list[float], params: FluxDensityParams) -> tuple[Path, Path]:
    output_dir = Path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    PM_model = params.boundary.magnetization_model

    #Save data
    data_path = output_dir / f"flux_density_{PM_model}_data.npz"
    np.savez_compressed(data_path, theta_elec_deg=theta_elec_deg, Br_total=Br_total, Btheta_total=Btheta_total, 
                        residuals=np.asarray(residuals), airgap_radius_m=params.airgap_radius_m, magnetizing_model=PM_model)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    axes[0].plot(theta_elec_deg, Br_total)
    axes[0].set_ylabel("Br (T)")
    axes[0].grid(True)

    axes[1].plot(theta_elec_deg, Btheta_total)
    axes[1].set_ylabel("Btheta (T)")
    axes[1].set_xlabel("Theta (deg)")
    axes[1].grid(True)

    fig.tight_layout()

    plot_path = output_dir / f"flux_density_{PM_model}.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return plot_path, data_path

def main() -> None:
    params = FluxDensityParams()
    theta_elec_deg, Br_total, Btheta_total, residuals = total_flux_density(params)
    plot_path, data_path = save_flux_outputs(theta_elec_deg, Br_total, Btheta_total, residuals, params)

    print("=" * 60)
    print("Computed airgap flux density.")
    print("=" * 60)
    print()
    print(f"Number of segments:     {params.boundary.geometry.Nz}")
    print(f"Magnetization model:    {params.boundary.magnetization_model}")
    print(f"Airgap radius:          {params.airgap_radius_m:.9f} m")
    print(f"Br min/max:             {Br_total.min():.6e} T / {Br_total.max():.6e} T")
    print(f"Btheta min/max:         {Btheta_total.min():.6e} T / {Btheta_total.max():.6e} T")
    print(f"Max residual:           {max(residuals):.6e}")
    print()
    print(f"Plot saved to:          {plot_path.resolve()}")
    print(f"Data saved to:          {data_path.resolve()}")

if __name__ == "__main__":
    main()