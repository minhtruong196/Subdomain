import numpy as np
from pathlib import Path
from dataclasses import dataclass
from matplotlib import pyplot as plt

#Hướng đối tượng
@dataclass(frozen = True)
class GeometryParams:

    slot: int = 72
    pole: int = 12
    Nz: int = 16    #number of magnet segments
    alpha_pm: float = 0.922    #magnet coverage
    h: float = 57.0    #center offset (mm)
    Rp: float = 29.8   #radius of center offset (mm)
    hp: float = 77.05  #distance between center and magnet bottom (mm)
    pm_side_length: float = 2.42   #side length of magnet (mm)

    @property
    def pole_pairs(self) -> int:
        return self.pole // 2
    @property
    def zeta0_rad(self) -> float:   #góc của từng segments
        return self.alpha_pm * np.pi / (2 * self.pole_pairs * (self.Nz - 1))    #equa. (1)
    @property
    def half_pole_arc_rad(self) -> float:    #nửa góc mở pole
        return self.alpha_pm * np.pi / (2 * self.pole_pairs)   
    
@dataclass(frozen=True)    
class MagnetizationParams:
    geometry: GeometryParams = GeometryParams()
    Br_T: float = 1.0
    mu0: float = 4.0 * np.pi * 1.0e-7
    #dùng trong tất cả các công thức fourier mà có xích ma vô cùng
    max_harmonic: int = 200  
    delta_rad: float = 0 #rotor initial position
    magnetization_model: str = "parallel"
    sample_count: int = 4000 #Số điểm lấy mẫu dọc theo airgap để vẽ đồ thị
    output_path: str = "outputs/step2_magnetization"    #Đường dẫn để lưu đồ thị kết quả

    @property
    def M0(self) -> float:
        return self.Br_T / self.mu0 #biên độ từ hóa
    
#Bậc sóng hài 1, 3, 5,...
def odd_harmonic_orders(params: MagnetizationParams) -> np.ndarray:
    return np.arange(1, params.max_harmonic + 1, 2, dtype=int)

# mc/p = nu => mc = nu * p
def mechanical_harmonic_orders(params: MagnetizationParams) -> np.ndarray:
    nu = odd_harmonic_orders(params)
    return nu * params.geometry.pole_pairs  #mc = nu * p

# for mc khác 1
def ka_kb(mc: np.ndarray, j: int, zeta0: float) -> tuple[np.ndarray, np.ndarray]:
    a0 = (j - 1) * zeta0
    a1 = j * zeta0

    ka = (np.sin((mc + 1.0) * a1) - np.sin((mc + 1.0) * a0)) / (mc + 1.0)   #equa. (11)
    kb = np.empty_like(ka)

    singular = np.isclose(mc, 1.0) #phẩn tử true nếu mc = 1 và ngược lại
    kb[~singular] = (np.sin((mc[~singular] - 1.0)*a1) - np.sin((mc[~singular] - 1.0)*a0)) / (mc[~singular] - 1.0)   #equa. (11)
    # dòng này hiện tại không có ý nghĩa lắm vì bản thân mc = 1 đã có công thức riêng rồi, 
    # nhưng để đảm bảo tính tổng quát thì vẫn phải được tính và được tính toán bằng L'Hospital
    kb[singular] = a1 - a0

    return ka, kb

#Biên độ của các sóng hài từ hóa theo mô hình parallel hoặc radial
def magnetization_coefficients(j: int, params: MagnetizationParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 1 <= j <= params.geometry.Nz:
        raise ValueError(f"j must be in [1, {params.geometry.Nz}] segments")
    
    mc = mechanical_harmonic_orders(params).astype(float)   #float để lát tính toán chia
    zeta0 = params.geometry.zeta0_rad

    if params.magnetization_model == "radial":
        nu = odd_harmonic_orders(params).astype(float)
        Mr_m = 4 * params.Br_T * (np.sin(mc * j * zeta0) - np.sin(mc * (j-1) * zeta0)) / (nu * np.pi * params.mu0)  #equa. (15)
        #Nhờ sự xuất hiện của c, m được quy đổi thành nu (1, 3, 5, ...)

        Mt_m = np.zeros_like(Mr_m)
        return mc, Mr_m, Mt_m
    
    if params.magnetization_model != "parallel":
        raise ValueError("magnetization_model must be parallel or radial")
    
    factor = 2 * params.geometry.pole_pairs * params.Br_T / (np.pi * params.mu0)
    
    Mr_m = np.empty_like(mc)
    Mt_m = np.empty_like(mc)

    singular = np.isclose(mc, 1.0) #phẩn tử true nếu mc = 1 và ngược lại
    # For mc khác 1
    if np.any(~singular):
        ka, kb = ka_kb(mc[~singular], j, zeta0)
        Mr_m[~singular] = factor * (ka + kb) #equa. (10)
        Mt_m[~singular] = factor * (ka - kb) #equa. (10)

    # For mc = 1
    if np.any(singular):
        a0 = (j - 1) * zeta0
        a1 = j * zeta0
        Mr_m[singular] = params.geometry.pole_pairs * params.Br_T * (np.sin(2 * a1) - np.sin(2 * a0)) / (np.pi * params.mu0)  #equa. (12)
        Mt_m[singular] = Mr_m[singular]  #equa. (12)

    return mc, Mr_m, Mt_m

#Tạo vùng không gian mẫu cho nửa nam châm
def theta_grid_half_pole(params: MagnetizationParams) -> np.ndarray:
    return np.linspace(0, np.pi / (2 * params.geometry.pole_pairs), params.sample_count, endpoint=True)

def reconstruct_Mr(theta_rad: np.ndarray, mc: np.ndarray, Mr_m: np.ndarray, params: MagnetizationParams) -> np.ndarray:
    phase = np.outer(mc, theta_rad - params.delta_rad)  #(θi−Δ), 3(θi−Δ), 5(θi−Δ), ... cho từng điểm rời rạc theta_rad
    #Vì giờ có thêm chiều vị trí nên cần tăng số chiều của Mr_m
    #Mỗi cột là một chuỗi fourier theo từng vị trí rời rạc nên cộng lại theo cột
    return np.sum(Mr_m[:, None] * np.cos(phase), axis=0)  #equa. (9)

# Equa. (7). Note (1)
def mr_ground_truth(theta_rad: np.ndarray, j: int, params: MagnetizationParams) -> np.ndarray:
    pole_pitch = np.pi / params.geometry.pole_pairs
    theta_local = np.mod(theta_rad - params.delta_rad, pole_pitch)  
    theta_active = np.minimum(theta_local, pole_pitch - theta_local)

    #Calculate angle that contains J-th segment
    lower_ang = (j - 1) * params.geometry.zeta0_rad
    upper_ang = j * params.geometry.zeta0_rad
    segment_active = (theta_active >= lower_ang) & (theta_active <= upper_ang)

    Mr = np.zeros_like(theta_rad)
    Mr[segment_active] = params.M0 * np.cos(theta_active[segment_active])
    return Mr
    
def save_plot(theta_rad: np.ndarray, Mr_fourier: np.ndarray, Mr_ground_truth: np.ndarray, j: int, params: MagnetizationParams) -> Path: 
    output_path = Path(params.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    theta_deg = np.rad2deg(theta_rad)
    
    plt.figure()
    plt.plot(theta_deg, Mr_fourier/params.M0, label="Fourier Reconstruction")   #Chuẩn hóa dữ liệu để dễ so sánh
    plt.plot(theta_deg, Mr_ground_truth/params.M0, label="Ground Truth")        #Chuẩn hóa dữ liệu để dễ so sánh
    plt.legend()
    plt.xlabel("Theta [deg]")
    plt.ylabel("Mr / M0")
    plt.title(f"{params.magnetization_model.capitalize()} Magnetization waveform in one half pole with j={j} segments")
    plt.grid()

    path = output_path / f"Mr_waveform_j{j}_m{params.max_harmonic}.png"
    plt.savefig(path)
    plt.close()
    return path


def main() -> None:
    params = MagnetizationParams()
    j = 1                                                                       # Chọn J mong muốn, Modify here

    mc, Mr_m, Mt_m = magnetization_coefficients(j, params)
    theta_rad = theta_grid_half_pole(params)
    Mr_fourier = reconstruct_Mr(theta_rad, mc, Mr_m, params)
    Mr_ground_truth = mr_ground_truth(theta_rad, j, params)
    plot_path = save_plot(theta_rad, Mr_fourier, Mr_ground_truth, j, params)

    print("=" * 60)
    print("Compared Fourier reconstruction with ground truth.")
    print("=" * 60)
    print() #Cách dòng
    print(f"Segment chosen: j = {j}")
    print(f"Magnetization model: {params.magnetization_model}")
    print(f"Max harmonic order: {params.max_harmonic}")
    print(f"Slot, Pole: {params.geometry.slot}, {params.geometry.pole}")
    print(f"M0 (Br/mu0): {params.M0:.6e} [A/m]")
    print() #Cách dòng
    print("First 5 Fourier coefficients")
    for order, mr, mt in zip(mc[:5].astype(int), Mr_m[:5]/params.M0, Mt_m[:5]/params.M0):   #Được chuẩn hóa dữ liệu
        print(f"mc = {order:.1f}: Mr_m = {mr:.6f}, Mt_m = {mt:.6f}")
    print() #Cách dòng
    print(f"Plot saved to: {plot_path.resolve()}")
    print() #Cách dòng

if __name__ == "__main__":
    main()






