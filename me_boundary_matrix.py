import numpy as np
from dataclasses import dataclass
from math import gcd
from me_magnetization import GeometryParams, MagnetizationParams, magnetization_coefficients
from pathlib import Path

@dataclass(frozen=True)
class BoundaryMatrixParams:
    #Pole
    segment_j: int = 1                                                                                  #Chỉ đang chạy với 1 segment duy nhất
    Br_T: float = 1.065
    mu_0: float = 4.0 * np.pi * 1.0e-7
    mu_r: float = 1.0   #Assumed
    magnetization_model: str = "parallel"
    
    geometry: GeometryParams = GeometryParams()
    pole_harmonic: int = 200
    slot_harmonic: int = 10

    #Geometry
    stator_inner_radius_m: float = 90.0 * 1e-3
    slot_depth_m: float = 33 * 1e-3
    slot_width_m: float = 4.2 * 1e-3

    Jui_A_per_m2: float = 0.0  # Current density of III
    Jdi_A_per_m2: float = 0.0  # Current density of IV

    delta_rad: float = 0.0 #rotor initial position
    output_dir: str = "outputs/step3_boundary_matrix"

    @property
    def b_s_rad(self) -> float:
        # For rectangular slots only.
        return float(2.0 * np.arcsin(self.slot_width_m / (2.0 * self.stator_inner_radius_m))) 
    
    @property   # Radius of domain IV, Rsa
    def R_slot_1_m(self) -> float:
        return self.stator_inner_radius_m + self.slot_depth_m
    
    @property   # Radius of domain III, Rsc
    def R_slot_2_m(self) -> float:
        return self.stator_inner_radius_m + 0.5 * self.slot_depth_m
    
    @property
    def c_periods(self) -> int:
        return gcd(self.geometry.slot, self.geometry.pole_pairs)
    
    @property #Z/2c, equa. (25)
    def reduced_slot(self) -> int:
        return self.geometry.slot // (2 * self.c_periods)

#Note: (4)
@dataclass(frozen=True)
class UnknownLayout:
    m_count: int
    n_count: int
    slot_count: int

    @property
    def total_furier_slot_coef(self) -> int:
        return self.n_count * self.slot_count
    
    #Note: (4.1)
    @property
    def started_pos_Am1(self) -> int:
        return 0
    
    @property
    def started_pos_Cm1(self) -> int:
        return self.m_count
    
    @property
    def started_pos_Am2(self) -> int:
        return 2 * self.m_count
    
    @property
    def started_pos_Bm2(self) -> int:
        return 3 * self.m_count
    
    @property
    def started_pos_Cm2(self) -> int:
        return 4 * self.m_count
    
    @property
    def started_pos_Dm2(self) -> int:
        return 5 * self.m_count
    
    # Fourier coefficients for slot harmonics
    @property
    def started_pos_An3(self) -> int:
        return 6 * self.m_count
    
    @property
    def started_pos_B03(self) -> int:
        return self.started_pos_An3 + self.total_furier_slot_coef
    
    @property
    def started_pos_An4(self) -> int:
        return self.started_pos_B03 + self.slot_count
    
    @property
    def started_pos_B04(self) -> int:
        return self.started_pos_An4 + self.total_furier_slot_coef
    
    @property
    def total_coefficients(self) -> int:
        return self.started_pos_B04 + self.slot_count
    
    #Gán hệ số cho từng vị trí (index) trong ma trận K và vector Y
    def Am1(self, m_idx: int) -> int:
        return self.started_pos_Am1 + m_idx
    
    def Cm1(self, m_idx: int) -> int:
        return self.started_pos_Cm1 + m_idx
    
    def Am2(self, m_idx: int) -> int:
        return self.started_pos_Am2 + m_idx
    
    def Bm2(self, m_idx: int) -> int:
        return self.started_pos_Bm2 + m_idx
    
    def Cm2(self, m_idx: int) -> int:
        return self.started_pos_Cm2 + m_idx
    
    def Dm2(self, m_idx: int) -> int:
        return self.started_pos_Dm2 + m_idx
    
    def An3(self, n_idx: int, slot_idx: int) -> int:
        return self.started_pos_An3 + slot_idx * self.n_count + n_idx
    
    def B03(self, slot_idx: int) -> int:
        return self.started_pos_B03 + slot_idx
    
    def An4(self, n_idx: int, slot_idx: int) -> int:
        return self.started_pos_An4 + slot_idx * self.n_count + n_idx
    
    def B04(self, slot_idx: int) -> int:
        return self.started_pos_B04 + slot_idx

#=================================
# Ru and Ri Calculations
def zeta_j_rad(j: int, params: GeometryParams) -> float:
    if not 1 <= j <= params.Nz - 1:
        raise ValueError(f"Segment j must be in [1, {params.Nz - 1}]")
    return (2.0 * j - 1.0) * params.zeta0_rad / 2.0    #equa. (2)

def upper_radius_mm(zeta: float, params: GeometryParams) -> float:
    # Note: (1)
    delta = params.Rp**2 - (params.h * np.sin(zeta)) ** 2
    if delta < 0.0:
        raise ValueError("Error: Negative solution")
    return float(params.h * np.cos(zeta) + np.sqrt(delta))  #equa. (3)

def lower_radius_mm(zeta: float, params: GeometryParams) -> float:
    return float(params.hp / np.cos(zeta))  #equa. (4)

def last_segment_mm(params: GeometryParams) -> tuple[float, float]:
    #Note: (2)
    zeta_edge = params.half_pole_arc_rad    #Nửa góc mở pole
    radius_upper_point = upper_radius_mm(zeta_edge, params)
    upper_edge_x = radius_upper_point * np.sin(zeta_edge)   #Tọa độ x của C
    upper_edge_y = radius_upper_point * np.cos(zeta_edge)   #Tọa độ y của C
    PM_side_length = upper_edge_y - params.hp

    Ru = np.hypot(upper_edge_x, params.hp + 0.5 * PM_side_length)

    lower_edge_x = lower_radius_mm(zeta_edge, params) * np.sin(zeta_edge)   #Tọa độ x của A
    Rl = np.hypot((lower_edge_x + upper_edge_x) /2, params.hp)

    print(f"PM side length calculated from geometry: {PM_side_length:.3f} mm")  #Kiểm tra với FEA
    print(f"PM width length calculated from geometry:{upper_edge_x:.3f} mm")
    return float(Ru), float(Rl)

def Ru_l_segment_mm(params: GeometryParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orders = np.arange(1, params.Nz + 1, dtype=int) # 1 to Nz
    Ru = np.empty(params.Nz, dtype=float)
    Rl = np.empty(params.Nz, dtype=float)

    for idx, j in enumerate(orders[:-1]):   #Trừ phần tử cuối
        zeta = zeta_j_rad(int(j), params)
        Ru[idx] = upper_radius_mm(zeta, params)
        Rl[idx] = lower_radius_mm(zeta, params)

    Ru[-1], Rl[-1] = last_segment_mm(params)    #gán cho segment cuối
    return orders, Ru, Rl
#=================================

#=================================
# Tính toán các công thức liên quan đến P và E từ equa. (5) nhưng tránh bị nổ do số quá lớn
#Note: (3)
def ratio_p_over_e(orders: float, x: float, y: float) -> float:
    a = float(orders * np.log(x /y))    #Order là bậc hài p, 3p, 5p,...
    if a > 25.0:
        return 1.0
    if a < -25.0:
        return -1.0
    return float(np.cosh(a) / np.sinh(a)) 

def ratio_e_over_p(orders: float, x: float, y: float) -> float:
    return float(np.tanh(orders * np.log(x / y)))

#Tính 2/E: Tượng tự như def ratio_p_over_e
def ratio_two_over_e(order: float, x: float, y: float) -> float:
    a = float(order * np.log(x / y))
    if a > 25.0:
        return float(2.0 * np.exp(-a))
    if a < -25.0:
        return float(-2.0 * np.exp(a))
    return float(1.0 / np.sinh(a))

#Tính P/P: Tránh bị nổ bằng cách dùng np.logaddexp có sẵn của numpy
def ratio_p_over_p_same_y(order: float, x: float, x_ref: float, y: float) -> float:
    a = float(order * np.log(x / y))
    b = float(order * np.log(x_ref / y))
    return float(np.exp(np.logaddexp(a, -a) - np.logaddexp(b, -b))) #exp(ln(P(x,y)) - ln(P(x_ref,y)))
#=================================

#Equa. (20)
def k_mj(r: float, mc: float, Mr_m: float, Mt_m: float, mu0: float) -> float:
    if np.isclose(mc, 1.0):
        return float(-0.5 * mu0 * (Mr_m + Mt_m) * r * np.log(r))    #Phương trình nếu mc = 1
    return float(mu0 * (mc * Mr_m + Mt_m) * r / (mc**2 - 1.0))

#Đạo hàm của kmj theo biến r
def k_mj_derivative(r: float, mc: float, Mr_m: float, Mt_m: float, mu0: float) -> float:
    if np.isclose(mc, 1.0):
        return float(-0.5 * mu0 * (Mr_m + Mt_m) * (np.log(r) + 1))  # đạo hàm rln(r) là ln(r) + 1
    return float(mu0 * (mc * Mr_m + Mt_m) / (mc**2 - 1.0))  # đạo hàm r là 1

#Equa. (19)
def x_mj(r: float, mc: float, Mr_m: float, Mt_m: float, mu0: float, Ru: float, Rl: float) -> float:
    term_1 = Rl * (Rl / r) ** mc * (k_mj_derivative(Rl, mc, Mr_m, Mt_m, mu0) + mu0 * Mt_m)/ mc
    term_2 = Rl * (Rl / Ru) ** mc * (k_mj_derivative(Rl, mc, Mr_m, Mt_m, mu0) + mu0 * Mt_m) / mc + k_mj(Ru, mc, Mr_m, Mt_m, mu0)
    return float(term_1 + k_mj(r, mc, Mr_m, Mt_m, mu0) - ratio_p_over_p_same_y(mc, r, Ru, Rl) * term_2) #Biến r khác Ru và Rl

#Đạo hàm của xmj và thay r bằng Ru
def x_mj_derivative_at_Ru(mc: float, Mr_m: float, Mt_m: float, mu0: float, Ru: float, Rl: float) -> float:
    A = k_mj_derivative(Rl, mc, Mr_m, Mt_m, mu0) + mu0 * Mt_m
    U = Rl * (Rl / Ru) ** mc * A / mc + k_mj(Ru, mc, Mr_m, Mt_m, mu0)
    term_1 = -Rl * (Rl / Ru) ** mc * A / Ru + k_mj_derivative(Ru, mc, Mr_m, Mt_m, mu0)
    term_2 = -mc * ratio_e_over_p(mc, Ru, Rl) * U / Ru

    return term_1 + term_2

#Equa. (25)
def theta_i(slot_index: int, params: BoundaryMatrixParams) -> float:
    return float(np.pi * (2 * slot_index - 1) / params.geometry.slot)

#n là slot_harmonic
def G_n(n: int, params: BoundaryMatrixParams) -> float:
    return float(n * np.pi / params.b_s_rad)

#=================================
# APPENDIX
# Appendix 3
def f_mi(mc: float, slot_index: int, params: BoundaryMatrixParams) -> float:
    theta_i_var = theta_i(slot_index, params)
    return float((np.sin(mc * (theta_i_var + 0.5 * params.b_s_rad)) - np.sin(mc * (theta_i_var - 0.5 * params.b_s_rad))) / mc)

def g_mi(mc: float, slot_index: int, params: BoundaryMatrixParams) -> float:
    theta_i_var = theta_i(slot_index, params)
    return float(-(np.cos(mc * (theta_i_var + 0.5 * params.b_s_rad)) - np.cos(mc * (theta_i_var - 0.5 * params.b_s_rad))) / mc)

def f_mni(n: int, mc: float, slot_index: int, params: BoundaryMatrixParams) -> float:
    theta_i_var = theta_i(slot_index, params)
    bs_var = params.b_s_rad
    gn_var = G_n(n, params)
    if np.isclose(mc, gn_var):
        return float(bs_var * np.cos(mc * (theta_i_var - 0.5 * bs_var)) / 2)
    return float(mc * (np.sin(mc * (theta_i_var - 0.5 * bs_var)) - ((-1)**n)*np.sin(mc * (theta_i_var + 0.5 * bs_var))) / (gn_var**2 - mc**2))

def g_mni(n: int, mc: float, slot_index: int, params: BoundaryMatrixParams) -> float:
    theta_i_var = theta_i(slot_index, params)
    bs_var = params.b_s_rad
    gn_var = G_n(n, params)
    if np.isclose(mc, gn_var):
        return float(bs_var * np.sin(mc * (theta_i_var - 0.5 * bs_var)) / 2)
    return float(mc * (((-1)**n)*np.cos(mc * (theta_i_var + 0.5 * bs_var)) - np.cos(mc * (theta_i_var - 0.5 * bs_var))) / (gn_var**2 - mc**2))

def K_c(params: BoundaryMatrixParams) -> float:
    Rs = params.stator_inner_radius_m # Domain II
    R_slot_2 = params.R_slot_2_m # Domain III
    R_slot_1 = params.R_slot_1_m # Domain IV
    Jui = params.Jui_A_per_m2   #Domain III
    Jdi = params.Jdi_A_per_m2   #Domain IV

    term_1 = -params.mu_0 * Jui * Rs / (2 * np.pi)
    term_2 = (params.mu_0 * (R_slot_2**2) * (Jui - Jdi) / (2 * np.pi) + params.mu_0 * (R_slot_1**2) * Jdi / (2 * np.pi)) / Rs
    return term_1 + term_2

# Appendix 5
def B04_minus_BO3(params: BoundaryMatrixParams) -> float:
    Rs = params.stator_inner_radius_m # Domain II
    R_slot_2 = params.R_slot_2_m # Domain III
    R_slot_1 = params.R_slot_1_m # Domain IV
    Jui = params.Jui_A_per_m2   #Domain III
    Jdi = params.Jdi_A_per_m2   #Domain IV

    term_1 = params.mu_0 * Jui * (Rs**2 - R_slot_2**2) / 4
    term_2 = (params.mu_0 * (R_slot_2**2) * (Jui - Jdi) / 2 + params.mu_0 * (R_slot_1**2) * Jdi / 2) * np.log(R_slot_2 / Rs)
    return term_1 + term_2
#=================================

#Lấy Ru và Rl của segment đã chọn dựa trên ma trận Ru và Rl đã tính toán cho tất cả các segment
def selected_segment(params: BoundaryMatrixParams) -> tuple[float, float]:
    _, Ru, Rl = Ru_l_segment_mm(params.geometry)
    idx = params.segment_j - 1 #Thực chất là segment đã chọn nhưng python index bắt đầu từ 0 nên phải trừ đi 1
    if idx < 0 or idx >= len(Ru):
        raise ValueError(f"Selected segment j must be in [1, {len(Ru)}]")
    return float(Ru[idx] * 1e-3), float(Rl[idx] * 1e-3) #Convert mm to m

def build_boundary_matrix_KXY(params: BoundaryMatrixParams) -> tuple[np.ndarray, np.ndarray, UnknownLayout, dict[str, np.ndarray | float]]:
    #Chuyển thông tin từ BoundaryMatrixParams sang MagnetizationParams để tính toán vì hệ số có thể khác nhau
    mag_params = MagnetizationParams(geometry=params.geometry, Br_T=params.Br_T, mu0=params.mu_0, max_harmonic=params.pole_harmonic, 
                                     delta_rad=params.delta_rad, magnetization_model=params.magnetization_model)
    mc_value, Mr_value, Mt_value = magnetization_coefficients(params.segment_j, mag_params)

    layout = UnknownLayout(m_count = len(mc_value), n_count = params.slot_harmonic, slot_count = params.reduced_slot)   #Gán value vào class

    #Note: (5)
    K = np.zeros((layout.total_coefficients, layout.total_coefficients), dtype=float)   #Kích thước NxN
    Y = np.zeros(layout.total_coefficients, dtype=float)                                #Kích thước N

    Ru, Rl = selected_segment(params)
    Rs = params.stator_inner_radius_m
    Rsa = params.R_slot_1_m
    Rsc = params.R_slot_2_m
    c = params.c_periods
    bs = params.b_s_rad
    kc = K_c(params)
    b04_b03 = B04_minus_BO3(params)

    #Note: (6)
    row = 0
    #Note: (6.1)
    #Appendix 1: Equations of Am1 and Cm1
    for m_idx, mc in enumerate(mc_value):
        a_coef = Ru * ratio_p_over_e(mc, Ru, Rs) / mc
        b_coef = Rs * ratio_two_over_e(mc, Rs, Ru) / mc

        K[row, layout.Am1(m_idx)] = 1.0
        K[row, layout.Am2(m_idx)] = -a_coef
        K[row, layout.Bm2(m_idx)] = -b_coef
        row += 1    # 1st Equation

        K[row, layout.Cm1(m_idx)] = 1.0
        K[row, layout.Cm2(m_idx)] = -a_coef
        K[row, layout.Dm2(m_idx)] = -b_coef
        row += 1    # 2nd Equation

    #Appendix 2: Equations of Am2 and Cm2
    #Lấy từng giá trị cụ thể của mc, Mr_m, Mt_m trong 1 lần for.
    for m_idx, (mc, Mr_m, Mt_m) in enumerate(zip(mc_value, Mr_value, Mt_value)):
        boundary_source = x_mj_derivative_at_Ru(mc, Mr_m, Mt_m, params.mu_0, Ru, Rl) + params.mu_0 * Mt_m   #Thêm nguồn do parallel
        a_coef = mc * ratio_e_over_p(mc, Ru, Rl) / (params.mu_r * Ru)

        K[row, layout.Am2(m_idx)] = 1.0
        K[row, layout.Am1(m_idx)] = -a_coef
        Y[row] = - boundary_source * np.sin(mc * params.delta_rad) / params.mu_r
        row += 1    # 1st Equation

        K[row, layout.Cm2(m_idx)] = 1.0
        K[row, layout.Cm1(m_idx)] = -a_coef
        Y[row] = boundary_source * np.cos(mc * params.delta_rad) / params.mu_r
        row += 1    # 2nd Equation
        
    #Appendix 3: Odd periodic equations of Bm2 and Dm2
    for m_idx, mc in enumerate(mc_value):
        K[row, layout.Bm2(m_idx)] = 1.0
        Y[row] = sum(2 * c * kc * f_mi(mc, slot_i, params) for slot_i in range(1, layout.slot_count + 1))
        #Chỗ nào có An3, An4 thì sẽ có 2 for
        for slot_idx, slot_i in enumerate(range(1, layout.slot_count + 1)):
            for n_idx, n in enumerate(range(1, layout.n_count + 1)):
                gn = G_n(n, params)
                a_coef = gn * ratio_e_over_p(gn, Rs, Rsa) * 2 * c * f_mni(n, mc, slot_i, params) / (Rs * np.pi)
                K[row, layout.An3(n_idx, slot_idx)] -= a_coef   #Trừ đi qua từng vòng lặp for sẽ thành tổng sigma
        row += 1    # 1st Equation

        K[row, layout.Dm2(m_idx)] = 1.0
        Y[row] = sum(2 * c * kc * g_mi(mc, slot_i, params) for slot_i in range(1, layout.slot_count + 1))
        #Chỗ nào có An3, An4 thì sẽ có 2 for
        for slot_idx, slot_i in enumerate(range(1, layout.slot_count + 1)):
            for n_idx, n in enumerate(range(1, layout.n_count + 1)):
                gn = G_n(n, params)
                a_coef = gn * ratio_e_over_p(gn, Rs, Rsa) * 2 * c * g_mni(n, mc, slot_i, params) / (Rs * np.pi)
                K[row, layout.An3(n_idx, slot_idx)] -= a_coef   #Trừ đi qua từng vòng lặp for sẽ thành tổng sigma
        row += 1    # 2nd Equation

    #Appendix 4: Equations of An3 and An4
    #Chỗ nào có An3, An4 thì sẽ có 2 for
    for slot_idx, slot_i in enumerate(range(1, layout.slot_count + 1)):
        for n_idx, n in enumerate(range(1, layout.n_count + 1)):
            #An3 và An4 cùng giá trị nên dùng 2 vòng lặp giống nhau
            for target in (layout.An3(n_idx, slot_idx), layout.An4(n_idx, slot_idx)):
                K[row, target] = 1.0
                for m_idx, mc in enumerate(mc_value):
                    f_coef = 2 * f_mni(n, mc, slot_i, params) / bs
                    g_coef = 2 * g_mni(n, mc, slot_i, params) / bs

                    a_part = Ru * ratio_two_over_e(mc, Ru, Rs) / mc
                    b_part = Rs * ratio_p_over_e(mc, Rs, Ru) / mc

                    #Trừ đi qua từng vòng lặp for sẽ thành tổng sigma
                    K[row, layout.Am2(m_idx)] -= a_part * f_coef
                    K[row, layout.Bm2(m_idx)] -= b_part * f_coef
                    K[row, layout.Cm2(m_idx)] -= a_part * g_coef
                    K[row, layout.Dm2(m_idx)] -= b_part * g_coef
                row += 1    # Equation for An3 and An4

    #Appendix 5: Equations of B03 and B04
    for slot_idx, slot_i in enumerate(range(1, layout.slot_count + 1)):
        K[row, layout.B03(slot_idx)] = 1.0
        for m_idx, mc in enumerate(mc_value):
            f_coef = f_mi(mc, slot_i, params) / bs
            g_coef = g_mi(mc, slot_i, params) / bs

            a_part = Ru * ratio_two_over_e(mc, Ru, Rs) / mc
            b_part = Rs * ratio_p_over_e(mc, Rs, Ru) / mc

            #Trừ đi qua từng vòng lặp for sẽ thành tổng sigma
            K[row, layout.Am2(m_idx)] -= a_part * f_coef
            K[row, layout.Bm2(m_idx)] -= b_part * f_coef
            K[row, layout.Cm2(m_idx)] -= a_part * g_coef
            K[row, layout.Dm2(m_idx)] -= b_part * g_coef
        row += 1   # Equation for B03

        K[row, layout.B04(slot_idx)] = 1.0
        K[row, layout.B03(slot_idx)] = -1.0
        Y[row] = b04_b03
        row += 1   # Equation for B04

    if row != layout.total_coefficients:
        raise ValueError(f"Row count {row} does not match expected total coefficients {layout.total_coefficients}")
    
    meta: dict[str, np.ndarray | float] = {"mc_value": mc_value, "Mr_value": Mr_value, "Mt_value": Mt_value, 
                                           "Ru": Ru, "Rl": Rl, "Rs": Rs, "Rsa": Rsa, "Rsc": Rsc, "bs_rad": bs, "kc": kc}
    
    return K, Y, layout, meta

def solve_boundary_matrix(params: BoundaryMatrixParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, UnknownLayout, dict[str, np.ndarray | float], float]:
    K, Y, layout, meta = build_boundary_matrix_KXY(params)
    solution = np.linalg.solve(K, Y)    #Giải ma trận hệ số X, X = K^-1 * Y
    #Note: (7)
    residual = np.linalg.norm(K @ solution - Y) / max(1.0, np.linalg.norm(Y))   #Sai số tương đối
    return solution, K, Y, layout, meta, float(residual)

def save_solution(solution: np.ndarray, K: np.ndarray, Y: np.ndarray, layout: UnknownLayout, meta: dict[str, np.ndarray | float], 
                  residual: float, params: BoundaryMatrixParams) -> Path:
    output_dir = Path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"Solution_j{params.segment_j}_m{params.pole_harmonic}_n{params.slot_harmonic}.npz"
    np.savez_compressed(path, solution=solution, K=K, Y=Y, residual=residual, total_fourier_coef=layout.total_coefficients,
                        m_count=layout.m_count,         #odd pole harmonic count
                        n_count=layout.n_count,         #odd slot harmonic count
                        slot_count=layout.slot_count,   #reduced slot count
                        **meta)
    return path
    
def main() -> None:
    params = BoundaryMatrixParams()
    solution, K, Y, layout, meta, residual = solve_boundary_matrix(params)
    output_path = save_solution(solution, K, Y, layout, meta, residual, params)

    print("=" * 60)
    print("Boundary Matrix Solution")
    print("=" * 60)
    print()
    print(f"Selected segment j:         {params.segment_j}")
    print(f"Periodicity c:              {params.c_periods}")
    print(f"reduced slot count:         {params.reduced_slot}")
    print(f"Odd pole harmonic count m:  {layout.m_count} <= Max {params.pole_harmonic}")
    print(f"slot harmonic count n:      {layout.n_count} <= Max {params.slot_harmonic}")
    print(f"Magnetization model:        {params.magnetization_model}")
    print(f"Matrix size:                {K.shape}")
    print(f"Residual of solution:       {residual:.6e}")
    print(f"Rs, Rsc(III), Rsa(IV):      {meta['Rs']:.9f} m, {meta['Rsc']:.9f} m, {meta['Rsa']:.9f} m")
    print(f"bs_rad:                     {meta['bs_rad']:.6e} rad")
    print(f"mu_r:                       {params.mu_r}")
    print(f"Jui (III), Jdi(IV):         {params.Jui_A_per_m2:.6e} A/m^2, {params.Jdi_A_per_m2:.6e} A/m^2")
    print()
    print(f"Solution saved to:          {output_path.resolve()}")

    print("first solved coefficients")
    print(f"Am1[0]                    : {solution[layout.Am1(0)]: .6e}")
    print(f"Cm1[0]                    : {solution[layout.Cm1(0)]: .6e}")
    print(f"Am2[0]                    : {solution[layout.Am2(0)]: .6e}")
    print(f"Bm2[0]                    : {solution[layout.Bm2(0)]: .6e}")
    print(f"Cm2[0]                    : {solution[layout.Cm2(0)]: .6e}")
    print(f"Dm2[0]                    : {solution[layout.Dm2(0)]: .6e}")

if __name__ == "__main__":
    main()