# scripts/generate_architecture_diagram.py
"""
Script ve so do kien truc he thong PetroForecast AI chuan Fintech SaaS.
Luu file tai reports/system_architecture_pipeline.png de chen vao README.md.
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_diagram():
    fig = plt.figure(figsize=(20, 24), dpi=220, facecolor='#0B0F19')
    ax = fig.add_axes([0, 0, 1, 1], facecolor='#0B0F19')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette
    card_bg = '#131B2E'
    card_border = '#2E3D5B'
    blue_accent = '#38BDF8'
    emerald_accent = '#34D399'
    purple_accent = '#A78BFA'
    amber_accent = '#FBBF24'
    rose_accent = '#FB7185'
    text_white = '#FFFFFF'
    text_muted = '#94A3B8'

    def draw_card(x, y, w, h, title="", subtitle="", badge="", badge_color=blue_accent, border_color=card_border, bg_color=card_bg, corner_radius=1.2, is_narrow=False):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle=f"round,pad=0,rounding_size={corner_radius}",
                             linewidth=1.4, edgecolor=border_color, facecolor=bg_color,
                             zorder=2)
        ax.add_patch(box)
        
        # Wide cards: badge at top-right
        if not is_narrow:
            if title:
                ax.text(x + 2.0, y + h - 2.2, title, fontsize=11.5, fontweight='bold', color=text_white, zorder=3, va='top')
            if badge:
                badge_len = max(len(badge) * 0.88, 9.0)
                badge_box = FancyBboxPatch((x + w - badge_len - 2.0, y + h - 3.4), badge_len, 2.2,
                                           boxstyle="round,pad=0,rounding_size=0.6",
                                           linewidth=1, edgecolor=badge_color, facecolor='#090D16',
                                           zorder=3)
                ax.add_patch(badge_box)
                ax.text(x + w - badge_len/2 - 2.0, y + h - 2.3, badge, fontsize=8.2, fontweight='bold', color=badge_color, ha='center', va='center', zorder=4)
            if subtitle:
                ax.text(x + 2.0, y + h - 5.0, subtitle, fontsize=9.2, color=text_muted, zorder=3, va='top', linespacing=1.42)
        else:
            # Narrow cards: title at top, badge below title, then subtitle
            if title:
                ax.text(x + 1.6, y + h - 1.8, title, fontsize=10.5, fontweight='bold', color=text_white, zorder=3, va='top')
            if badge:
                badge_len = max(len(badge) * 0.85, 8.5)
                badge_box = FancyBboxPatch((x + 1.6, y + h - 4.4), badge_len, 1.8,
                                           boxstyle="round,pad=0,rounding_size=0.5",
                                           linewidth=1, edgecolor=badge_color, facecolor='#090D16',
                                           zorder=3)
                ax.add_patch(badge_box)
                ax.text(x + 1.6 + badge_len/2, y + h - 3.5, badge, fontsize=7.8, fontweight='bold', color=badge_color, ha='center', va='center', zorder=4)
            if subtitle:
                ax.text(x + 1.6, y + h - 5.4, subtitle, fontsize=8.6, color=text_muted, zorder=3, va='top', linespacing=1.35)

    def draw_arrow_down(x, y_start, y_end, color=blue_accent):
        ax.annotate('', xy=(x, y_end), xytext=(x, y_start),
                    arrowprops=dict(arrowstyle="->,head_width=0.45,head_length=0.7",
                                    lw=2.2, color=color), zorder=5)

    # =========================================================================
    # HEADER BANNER
    # =========================================================================
    draw_card(4, 93.5, 92, 5.2, 
              title="PETROFORECAST AI — END-TO-END SYSTEM PIPELINE ARCHITECTURE",
              subtitle="Hệ thống Trí tuệ Nhân tạo Dự báo Giá Xăng dầu Singapore (MoPS) & Bán lẻ Việt Nam (Nghị định 80/2023/NĐ-CP)",
              badge="v3.0 MASTER SOTA", badge_color=emerald_accent, border_color='#2563EB', bg_color='#1E1B4B')

    # =========================================================================
    # STAGE 1: DỮ LIỆU ĐA NGUỒN (DATA INGESTION)
    # =========================================================================
    draw_card(4, 80.0, 92, 12.0, border_color='#3B82F6', bg_color='#0F172A')
    ax.text(6, 90.3, "1. TẦNG THU THẬP & ĐỒNG BỘ DỮ LIỆU ĐA NGUỒN (MULTI-SOURCE INGESTION)", fontsize=11, fontweight='bold', color=blue_accent)

    # Subcard 1A: Singapore Platts
    draw_card(6, 81.0, 43.5, 7.8,
              title="[MoPS] Thị trường Singapore (FOB Platts)",
              subtitle="• Tập tin: price_petroleum.xlsx (4.735 dòng, 2008–2026)\n• 4 sản phẩm: MOGAS 95, MOGAS 92, DO 0.001%, DO 0.05%\n• Xử lý ngày nghỉ lễ tài chính Singapore (ffill / bfill quốc tế)",
              badge="PLATTS MoPS", badge_color=amber_accent, border_color='#D97706')

    # Subcard 1B: 10 Global Markets
    draw_card(50.5, 81.0, 43.5, 7.8,
              title="[Global] 10 Thị trường Tài chính Toàn cầu",
              subtitle="• Tập tin: external_market.csv (6.519 dòng, CME / NYMEX / ICE)\n• Năng lượng: Brent, WTI, RBOB Gasoline, Heating Oil, Natural Gas\n• Vĩ mô & Tiền tệ: DXY, USD/VND, USD/SGD, VIX Volatility, Vàng",
              badge="10 MARKETS", badge_color=emerald_accent, border_color='#059669')

    draw_arrow_down(50, 80.0, 77.2, color=blue_accent)

    # =========================================================================
    # STAGE 2: KỸ THUẬT ĐẶC TRƯNG CHUYÊN NGÀNH (135 FEATURES)
    # =========================================================================
    draw_card(4, 63.8, 92, 13.4, border_color='#8B5CF6', bg_color='#0F172A')
    ax.text(6, 75.6, "2. TẦNG KỸ THUẬT ĐẶC TRƯNG CHUYÊN NGÀNH LỌC DẦU (135 ECONOMETRIC FEATURES)", fontsize=11, fontweight='bold', color=purple_accent)

    draw_card(6, 64.8, 21.2, 9.4,
              title="Biên Lọc Dầu Spreads",
              subtitle="• Crack Gasoline: MG95-Brent\n• Crack Diesel: DO-Brent\n• Premium: MG95 - MG92\n• Quality: DO 10ppm - 500ppm",
              badge="Crack Spreads", badge_color=purple_accent, is_narrow=True)

    draw_card(28.2, 64.8, 21.2, 9.4,
              title="Chênh Lệch Á – Mỹ",
              subtitle="• Trans-Pacific Arbitrage:\n  MG95 vs NYMEX RBOB\n  DO vs NYMEX Heating Oil\n• Bắt luồng luân chuyển hàng",
              badge="Arbitrage", badge_color=blue_accent, is_narrow=True)

    draw_card(50.4, 64.8, 21.2, 9.4,
              title="Động Lượng & Đảo Chiều",
              subtitle="• Z-Score 20d (Mean-Reversion)\n• Realized Volatility (10d, 20d)\n• Bollinger Bands, RSI, MACD\n• Bắt điểm hồi quy trung bình",
              badge="Mean-Reversion", badge_color=amber_accent, is_narrow=True)

    draw_card(72.6, 64.8, 21.4, 9.4,
              title="Kháng Ngoại Lệ 2026",
              subtitle="• Relative Log-Delta Returns\n• Sin/Cos Day, Month, Quarter\n• StandardScaler chuẩn hóa\n• Khắc phục sốc Diesel 292$",
              badge="Scale-Invariant", badge_color=emerald_accent, is_narrow=True)

    draw_arrow_down(50, 63.8, 61.2, color=purple_accent)

    # =========================================================================
    # STAGE 3: THIẾT KẾ ĐẦU RA ĐA NHIỆM VỤ (DUAL-HORIZON)
    # =========================================================================
    draw_card(4, 49.8, 92, 11.4, border_color='#EC4899', bg_color='#0F172A')
    ax.text(6, 59.6, "3. THIẾT KẾ ĐẦU RA ĐA NHIỆM VỤ THEO THỰC TIỄN & TÀI CHÍNH (DUAL-TASK FORMULATION)", fontsize=11, fontweight='bold', color=rose_accent)

    # Task 1
    draw_card(6, 50.8, 43.5, 7.5,
              title="[Nhiệm Vụ 1] Chu Kỳ Điều Hành 7 Ngày (NĐ 80/2023/NĐ-CP)",
              subtitle="• Mục tiêu: Giá bình quân 7 ngày giữa 2 kỳ Thứ Năm: P_avg(1..7)\n• Bản chất: Triệt tiêu nhiễu giao ngay, phản ánh đúng công thức nhà nước\n• Hiệu năng: R² = 0.9242 | MAPE = 3.02% (Đạt chuẩn xuất sắc)",
              badge="R² = 0.9242", badge_color=emerald_accent, border_color='#059669', bg_color='#064E3B')

    # Task 2
    draw_card(50.5, 50.8, 43.5, 7.5,
              title="[Nhiệm Vụ 2] Giá Giao Ngay Đa Mốc (Multi-Horizon Spot)",
              subtitle="• Mục tiêu: Dự báo mức giá đóng cửa tại T+1, T+3, T+7, T+20 ngày\n• Bản chất: Phục vụ lướt sóng ngắn hạn và phòng vệ rủi ro hàng hóa\n• Hiệu năng T+7: R² = 0.8095 | MAPE = 4.95% (Chạm ngưỡng trần Martingale)",
              badge="R² = 0.8095", badge_color=amber_accent, border_color='#D97706')

    draw_arrow_down(50, 49.8, 47.2, color=rose_accent)

    # =========================================================================
    # STAGE 4: TỔ HỢP 5 MÔ HÌNH SOTA & STACKING ENSEMBLE
    # =========================================================================
    draw_card(4, 29.8, 92, 17.4, border_color='#10B981', bg_color='#0F172A')
    ax.text(6, 45.6, "4. TỔ HỢP 5 MÔ HÌNH HỌC MÁY & HỌC SÂU SOTA (HYBRID STACKING COMMITTEE)", fontsize=11, fontweight='bold', color=emerald_accent)

    # 5 Models
    model_w = 17.2
    gap = 1.0
    start_x = 6.0

    models_info = [
        ("XGBoost", "Cây tăng cường gradient bậc 2", "Bắt ngưỡng phi tuyến Spreads", "R² = 0.9217", blue_accent),
        ("HistGBoost", "Phân thùng dữ liệu siêu tốc", "Kháng nhiễu ngoại lai dạng bảng", "R² = 0.9202", emerald_accent),
        ("RidgeCV", "Hồi quy tuyến tính co L2", "Chống đa cộng tuyến, ổn định", "R² = 0.8602", amber_accent),
        ("BiGRU-Attn", "Ngữ cảnh 2 chiều 30 ngày", "Self-Attention trọng số thời gian", "R² = 0.9221", purple_accent),
        ("Dilated TCN", "Tích chập nhân quả mở rộng", "Receptive field cực đại", "R² = 0.9242", rose_accent),
    ]

    for i, (m_title, m_desc1, m_desc2, m_r2, m_color) in enumerate(models_info):
        mx = start_x + i * (model_w + gap)
        draw_card(mx, 37.6, model_w, 6.8,
                  title=m_title,
                  subtitle=f"• {m_desc1}\n• {m_desc2}",
                  badge=m_r2, badge_color=m_color, border_color=m_color, is_narrow=True)

    # Meta Stacking Box
    draw_card(6, 30.6, 88, 5.8,
              title="BỘ TỐI ƯU HÓA RÀNG BUỘC SLSQP META-LEARNER (CONSTRAINED ENSEMBLE STACKING)",
              subtitle="Giải thuật SLSQP tìm phân bổ trọng số tối ưu: min Σ|y - Σ wᵢ ŷᵢ| thỏa mãn Σ wᵢ = 1, wᵢ ≥ 0\nTriệt tiêu sai số riêng lẻ của từng mô hình, đẩy toàn diện: R² = 0.9242 | MAPE = 3.02% | MAE = 3.44 USD/thùng",
              badge="QUÁN QUÂN SOTA", badge_color=emerald_accent, border_color='#10B981', bg_color='#064E3B')

    draw_arrow_down(50, 29.8, 27.2, color=emerald_accent)

    # =========================================================================
    # STAGE 5: BỘ QUY ĐỔI GIÁ BÁN LẺ VN & MÁY CHỦ SẢN XUẤT
    # =========================================================================
    draw_card(4, 15.8, 92, 11.4, border_color='#06B6D4', bg_color='#0F172A')
    ax.text(6, 25.6, "5. BỘ QUY ĐỔI GIÁ BÁN LẺ VIỆT NAM & TRẠM MÁY CHỦ FASTAPI (SERVING ENGINE)", fontsize=11, fontweight='bold', color=blue_accent)

    # Vietnam Pricing
    draw_card(6, 16.8, 48.0, 7.6,
              title="Bộ Quy Đổi Giá Cơ Sở Việt Nam (NĐ 80/2023/NĐ-CP)",
              subtitle="• Công thức: Giá Bán Lẻ = [(MoPS × Tỷ giá / 158.987 + CIF) × (1+NK) × (1+TTĐB) + BVMT + CPKD] × (1+VAT)\n• Tự động bóc tách 6 cấu phần thuế phí định mức và dự báo xu thế tăng/giảm (đ/lít)\n• Chu kỳ cập nhật Thứ Năm hàng tuần đồng bộ kỳ điều hành nhà nước",
              badge="NĐ 80/2023/NĐ-CP", badge_color=rose_accent, border_color='#E11D48')

    # FastAPI Server
    draw_card(55.0, 16.8, 39.0, 7.6,
              title="Trạm Dịch Vụ API FastAPI Hiệu Năng Cao",
              subtitle="• Inference tăng tốc trên GPU NVIDIA RTX 4060 Ti (< 10ms)\n• RESTful API Endpoints: /api/overview, /api/forecast, /api/vietnam\n• Kiểm định dữ liệu nghiêm ngặt với Pydantic Schemas",
              badge="LATENCY < 10ms", badge_color=emerald_accent, border_color='#059669')

    draw_arrow_down(50, 15.8, 13.2, color=blue_accent)

    # =========================================================================
    # STAGE 6: FINTECH SAAS DASHBOARD
    # =========================================================================
    draw_card(4, 2.5, 92, 10.7, border_color='#F59E0B', bg_color='#0F172A')
    ax.text(6, 11.6, "6. GIAO DIỆN NGƯỜI DÙNG FINTECH SAAS DASHBOARD (PRESENTATION LAYER)", fontsize=11, fontweight='bold', color=amber_accent)

    draw_card(6, 3.5, 28.5, 6.8,
              title="Biểu Đồ Tương Tác ApexCharts",
              subtitle="• Lịch sử giá 4 sản phẩm MoPS\n• Quỹ đạo dự báo Fan Chart 20 ngày\n• Bộ lọc trực tiếp mốc T+1, T+3, T+7, T+20",
              badge="ApexCharts", badge_color=blue_accent, is_narrow=True)

    draw_card(35.5, 3.5, 28.5, 6.8,
              title="Phòng Thí Nghiệm Cú Sốc",
              subtitle="• Stress Testing giả lập biến động\n• Gasoline Shock %, Diesel Shock %\n• Biến động độ lệch chuẩn Volatility (%)",
              badge="Stress Testing", badge_color=rose_accent, is_narrow=True)

    draw_card(65.0, 3.5, 29.0, 6.8,
              title="Bản Tin Điều Hành Giá Thứ Năm",
              subtitle="• Dự báo biến động giá bán lẻ trong nước\n• Tăng / Giảm bao nhiêu đồng/lít\n• Khuyến nghị tiêu dùng & phòng vệ tồn kho",
              badge="Petrolimex", badge_color=emerald_accent, is_narrow=True)

    # Save diagram
    output_path = "reports/system_architecture_pipeline.png"
    plt.savefig(output_path, dpi=220, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"SUCCESS: Exported diagram to {output_path}")

if __name__ == "__main__":
    create_diagram()
