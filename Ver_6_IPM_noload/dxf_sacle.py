import ezdxf
from ezdxf.math import Matrix44

def fix_flux_dxf_scale(input_file, output_file):
    try:
        # 1. Đọc file DXF hệ mét từ Flux
        doc = ezdxf.readfile(input_file)
        msp = doc.modelspace()
        
        # 2. Tạo ma trận phóng to 1000 lần (X, Y, Z)
        scale_matrix = Matrix44.scale(1000, 1000, 1000)
        
        # 3. Duyệt qua từng đường nét con bên trong và phóng to riêng từng thằng
        for entity in msp:
            entity.transform(scale_matrix)
        
        # 4. Lưu thành file mới chuẩn mm
        doc.saveas(output_file)
        print(f"🎉 Sửa lỗi scale thành công! File mới sẵn sàng dùng: {output_file}")
        
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")

# --- ĐƯỜNG DẪN FILE CỦA BẠN ---
file_tu_flux = "flux.dxf"
file_cho_motorcad = "fix_ok.dxf"

fix_flux_dxf_scale(file_tu_flux, file_cho_motorcad)