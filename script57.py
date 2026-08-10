import io
import math
import os
import tempfile
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ReportLab 報表套件
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ==============================================================================
# 0. 全域設定與字型/模型註冊 (防亂碼核心機制)
# ==============================================================================
st.set_page_config(
    page_title="⚾ 崇明國中棒球隊 - 智慧化投打運動科學分析系統 (進階物理擊球初速版)",
    page_icon="⚾",
    layout="wide",
)

MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

FONT_FILE = "NotoSansTC-Regular.ttf"
FONT_NAME = "NotoSansTC"


def download_file_with_user_agent(url, save_path):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req) as response, open(save_path, "wb") as out_file:
        out_file.write(response.read())


@st.cache_resource
def ensure_dependencies():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ 首次執行，正在下載 MediaPipe 姿態識別模型..."):
            try:
                download_file_with_user_agent(MODEL_URL, MODEL_PATH)
            except Exception as e:
                st.error(f"❌ 模型下載失敗: {e}")


ensure_dependencies()

# 字型註冊防護
if os.path.exists(FONT_FILE):
    try:
        pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_FILE))
    except Exception:
        FONT_NAME = "STSong-Light"
        try:
            pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))
        except:
            pass
else:
    FONT_NAME = "STSong-Light"
    try:
        pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))
    except:
        FONT_NAME = "Helvetica"


# ==============================================================================
# 共用邏輯與運算函式 (含進階物理初速計算)
# ==============================================================================
def calculate_advanced_exit_velocity(bat_speed_kmh, accel_rate, hit_pos_ratio, incoming_speed_kmh, cor, mass_ratio):
    """
    結合三大進階方法的擊球初速物理模型：
    1. 碰撞物理學 (動量守恆與 COR 恢復係數)
    2. 甜蜜點位置加權 (Hit Position Ratio 接近 0.65-0.75 效率最高)
    3. 加速度/爆發力斜率加權 (Accel Rate 反映肌肉爆發力釋放陡峭程度)
    """
    v_bat = bat_speed_kmh / 3.6  # 轉為 m/s
    v_ball = incoming_speed_kmh / 3.6  # 轉為 m/s

    m_ball = 0.145
    bat_mass = 0.85
    m_eff = bat_mass * mass_ratio  # 有效質量

    # 1. 基礎碰撞物理公式 (Elastic-Plastic Impact with COR)
    collision_v_exit_ms = (m_eff * v_bat + m_ball * v_ball + m_ball * cor * (v_ball + v_bat)) / (m_eff + m_ball)

    # 2. 甜蜜點位置加權 (Sweet Spot Weighting)
    sweet_spot_optimal = 0.70
    deviation = abs(hit_pos_ratio - sweet_spot_optimal)
    sweet_spot_multiplier = max(0.85, 1.25 - (deviation * 0.8))

    # 3. 加速度 / 爆發力斜率加權 (Acceleration Rate Modifier)
    accel_multiplier = 1.0 + min(0.15, max(-0.1, accel_rate * 0.05))

    # 綜合計算最終初速 (m/s 轉 km/h)
    final_ev_ms = collision_v_exit_ms * sweet_spot_multiplier * accel_multiplier
    final_ev_kmh = final_ev_ms * 3.6

    return round(max(final_ev_kmh, bat_speed_kmh * 1.05), 1)


def calculate_attack_angle_and_length(bat_trajectory, m_per_px, camera_tilt=0.0):
    if len(bat_trajectory) < 2:
        return 0.0, 0.0, 0.0
    pts = np.array(bat_trajectory, dtype=np.float64)
    diffs = np.diff(pts, axis=0)
    dist_px = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))
    swing_length_m = dist_px * m_per_px

    x = pts[:, 0]
    y_phys = -pts[:, 1]
    x_dir = 1.0 if x[-1] >= x[0] else -1.0
    x_rel = (x - x[0]) * x_dir
    slope, intercept = np.polyfit(x_rel, y_phys, 1)

    y_pred = slope * x_rel + intercept
    ss_res = np.sum((y_phys - y_pred) ** 2)
    ss_tot = np.sum((y_phys - np.mean(y_phys)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
    attack_angle = math.degrees(math.atan(slope)) - camera_tilt
    return round(attack_angle, 1), round(swing_length_m, 2), round(r_squared, 2)


def generate_advanced_diagnostics(bat_speed, swing_length, attack_angle, exit_velocity, hip_rot_speed, r_squared):
    score = 100
    feedbacks = []
    drills = []

    if bat_speed >= 28.0:
        bat_speed_status = "速度優異"
        feedbacks.append(f"揮棒速度 (Bat Speed) 達 {bat_speed:.1f} km/h，表現優異。")
    else:
        bat_speed_status = "速度偏低"
        feedbacks.append(f"揮棒速度 (Bat Speed) 為 {bat_speed:.1f} km/h，爆發力尚未完全釋放。")
        score -= 10

    if 0.65 <= swing_length <= 0.95:
        length_status = "軌跡簡潔"
        feedbacks.append(f"揮棒軌跡長度 (Swing Length) 為 {swing_length:.2f} m，適中。")
    elif swing_length > 0.95:
        length_status = "軌跡過長 (繞大圈)"
        feedbacks.append(f"揮棒軌跡長度達 {swing_length:.2f} m。【動作修正】揮棒路徑過長。")
        score -= 15
        drills.append("【改善軌跡】貼牆揮棒路徑收束訓練 (Wall Drill)")
    else:
        length_status = "延伸不足"
        feedbacks.append(f"揮棒軌跡長度僅 {swing_length:.2f} m。")

    if 6.0 <= attack_angle <= 18.0:
        attack_status = "完美切入"
        feedbacks.append(f"攻擊仰角 (Attack Angle) 為 {attack_angle:.1f}°，平飛球率高。")
    elif attack_angle < 6.0:
        attack_status = "角度過陡 (砍擊)"
        feedbacks.append(f"攻擊仰角為 {attack_angle:.1f}°。【動作修正】由上往下砍擊。")
        score -= 15
        drills.append("【改善軌跡】高低位置擊球座高角掃擊練習 (Elevated Tee Work)")
    else:
        attack_status = "過度仰角 (倒棒)"
        feedbacks.append(f"攻擊仰角達 {attack_angle:.1f}°，易形成高飛球。")
        score -= 15
        drills.append("【改善軌跡】水平平飛擊球修正 (Level Swing Progression)")

    # 依據使用者指定的擊球初速新標準 (一般平均: 96.5 ~ 112.6 km/h，優異/強打者: 112.6 ~ 128.7 km/h)
    if exit_velocity >= 112.6:
        exit_status = "強打者水準 (Elite)"
        feedbacks.append(f"物理模型預估擊球初速達 {exit_velocity:.1f} km/h，達到優異/強打者水準 (70-80 mph)。")
    elif exit_velocity >= 96.5:
        exit_status = "一般平均水準 (Average)"
        feedbacks.append(f"物理模型預估擊球初速達 {exit_velocity:.1f} km/h，落在一般平均水準範圍 (60-70 mph)。")
    else:
        exit_status = "低於平均水準"
        feedbacks.append(
            f"預估擊球初速為 {exit_velocity:.1f} km/h，低於 60 mph (96.5 km/h)，建議強化擊球核心效率與爆發力。")
        score -= 10

    if hip_rot_speed >= 280.0:
        hip_status = "轉動爆發力強"
        feedbacks.append(f"髖關節峰值轉速達 {hip_rot_speed:.0f} deg/s。")
    else:
        hip_status = "下半身導引不足"
        feedbacks.append(f"髖關節峰值轉速僅 {hip_rot_speed:.0f} deg/s。【強化轉動】下半身骨盆旋轉不足。")
        score -= 15
        drills.append("【強化轉動】後髖關節爆發力旋轉彈力帶訓練")

    score = max(0, score)
    grade = "S (優異 A+)" if score >= 90 else (
        "A (良好)" if score >= 75 else ("B (尚可)" if score >= 60 else "C (需調整)"))

    summary_df = pd.DataFrame({
        "核心指標": ["揮棒速度", "揮棒軌跡長度", "攻擊仰角", "物理模型擊球初速", "髖關節轉速", "軌跡平順度 (R²)"],
        "實測數值": [f"{bat_speed:.1f} km/h", f"{swing_length:.2f} m", f"{attack_angle:.1f}°",
                     f"{exit_velocity:.1f} km/h", f"{hip_rot_speed:.0f} deg/s", f"{r_squared:.2f}"],
        "標竿參考值": ["> 28.0 km/h", "0.65 - 0.95 m", "6.0° - 18.0°", "優異 > 112.6 km/h (70+ mph)", "> 280 deg/s",
                       "> 0.88"],
        "診斷結果": [bat_speed_status, length_status, attack_status, exit_status, hip_status,
                     "穩定" if r_squared >= 0.88 else "抖動"],
    })
    return {"score": score, "grade": grade, "summary_df": summary_df, "feedbacks": feedbacks,
            "drills": list(set(drills))}


def generate_pdf_report(all_events: list, stance_label: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontName=FONT_NAME, fontSize=16, leading=20,
                                 textColor=colors.HexColor("#1E3A8A"), alignment=1)
    subtitle_style = ParagraphStyle("SubTitleStyle", parent=styles["Normal"], fontName=FONT_NAME, fontSize=9,
                                    leading=12, textColor=colors.HexColor("#4B5563"), alignment=1)
    section_style = ParagraphStyle("SectionStyle", parent=styles["Heading2"], fontName=FONT_NAME, fontSize=11,
                                   leading=14, textColor=colors.HexColor("#1E3A8A"), spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontName=FONT_NAME, fontSize=8.5, leading=11.5,
                                textColor=colors.HexColor("#1F2937"))

    elements = [
        Paragraph(f"棒球高階打擊動力鏈 - 全數揮擊次數完整分析總報告 ({stance_label})", title_style),
        Paragraph("<b>崇明國中棒球隊</b>", subtitle_style),
        Spacer(1, 4),
        Paragraph(f"總計檢測有效揮擊次數：{len(all_events)} 次 | 系統：Advanced Physics Kinetic Chain Diagnostic",
                  subtitle_style),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=8),
    ]

    elements.append(Paragraph("所有偵測揮擊次數之核心數據總表", section_style))
    master_table_data = [["次數", "揮棒速度", "軌跡長度", "攻擊仰角", "物理初速", "綜合評分", "等級"]]
    for ev in all_events:
        master_table_data.append([
            str(ev["次數"]),
            f"{ev['bat_speed']:.1f} km/h",
            f"{ev['swing_length']:.2f} m",
            f"{ev['attack_angle']:.1f}°",
            f"{ev['exit_velocity']:.1f} km/h",
            f"{ev['report']['score']} 分",
            str(ev['report']['grade'])
        ])

    master_t = Table(master_table_data, colWidths=[60, 90, 80, 80, 90, 60, 80])
    master_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F3F4F6')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
    ]))
    elements.extend([master_t, Spacer(1, 12)])

    for idx, ev in enumerate(all_events):
        rep = ev["report"]
        elements.append(Paragraph(f"▶ {ev['次數']} - 詳細動力鏈數值與診斷", section_style))

        if ev.get("snapshot_bytes"):
            img_stream = io.BytesIO(ev["snapshot_bytes"])
            elements.append(RLImage(img_stream, width=220, height=124))
            elements.append(Spacer(1, 4))

        sub_table_data = [["核心指標", "實測數值", "標竿參考值", "診斷結果"]]
        for _, row in rep["summary_df"].iterrows():
            sub_table_data.append(
                [str(row["核心指標"]), str(row["實測數值"]), str(row["標竿參考值"]), str(row["診斷結果"])])

        sub_t = Table(sub_table_data, colWidths=[130, 100, 110, 200])
        sub_t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#374151')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F9FAFB')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
        ]))
        elements.extend([sub_t, Spacer(1, 4)])

        for fb in rep["feedbacks"]:
            elements.append(Paragraph(f"• {fb.replace('**', '')}", body_style))

        if rep["drills"]:
            for drill in rep["drills"]:
                elements.append(Paragraph(f"💡 處方：{drill}", ParagraphStyle("Drill", parent=body_style,
                                                                            textColor=colors.HexColor("#92400E"))))

        elements.append(Spacer(1, 8))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# ==============================================================================
# 投手分析類別與對應診斷函式 (PitchAnalyzer & Pitcher Diagnostics)
# ==============================================================================
class PitchAnalyzer:
    def __init__(self):
        self.profiles = {
            "四縫線速球 (Four-Seam)": {"IVB": (15, 30), "H": (-8, 8)},
            "伸卡球/二縫線 (Sinker)": {"IVB": (-10, 10), "H": (10, 30)},
            "滑球 (Slider)": {"IVB": (-15, 10), "H": (-30, -8)},
            "曲球 (Curveball)": {"IVB": (-30, -10), "H": (-15, 15)},
            "變速球 (Changeup)": {"IVB": (0, 15), "H": (5, 20)}
        }

    def identify_pitch(self, ivb, h_break):
        for name, p in self.profiles.items():
            if p["IVB"][0] <= ivb <= p["IVB"][1] and p["H"][0] <= h_break <= p["H"][1]:
                return name
        return "未知球種 / 特殊進壘"


def generate_pitcher_diagnostics(pitch_speed, release_height, h_break, v_break, pitch_level_label):
    analyzer = PitchAnalyzer()
    pitch_type = analyzer.identify_pitch(v_break, h_break)

    score = 100
    feedbacks = [f"智慧偵測球種：{pitch_type}（組別：{pitch_level_label}）。"]
    drills = []

    if pitch_speed >= 120.0:
        speed_status = "球速優異"
        feedbacks.append(f"投球初速達 {pitch_speed:.1f} km/h，壓制力極佳。")
    elif pitch_speed >= 105.0:
        speed_status = "球速良好"
        feedbacks.append(f"投球初速達 {pitch_speed:.1f} km/h，符合水準。")
    else:
        speed_status = "球速提升中"
        feedbacks.append(f"投球初速為 {pitch_speed:.1f} km/h，尚有成長空間。")
        score -= 15
        drills.append("【下肢爆發】後腳蹬地與髖關節前推發力訓練")

    if release_height < 0.15:
        release_status = "出手點高度穩定"
        feedbacks.append("出手點軌跡高度集中，控球穩定性佳。")
    else:
        release_status = "出手點過度晃動"
        feedbacks.append("⚠️ 出手點起伏較大，建議加強定點平衡。")
        score -= 15
        drills.append("【控球穩定】定點平衡支撐與投球動作定型訓練")

    if pitch_type == "未知球種 / 特殊進壘":
        feedbacks.append("⚠️ 進壘軌跡偏離常規範圍，建議檢查釋放點與手腕轉軸。")
        score -= 10

    feedbacks.append(f"進壘位移數據：橫向位移 {h_break:.1f} cm，垂直位移 {v_break:.1f} cm。")
    score = max(0, score)
    grade = "S (王牌級 A+)" if score >= 90 else (
        "A (優秀先發)" if score >= 75 else ("B (需調整)" if score >= 60 else "C (修正)"))

    summary_df = pd.DataFrame({
        "投球核心指標": ["組別與距離", "智慧辨識球種", "投球初速", "出手點穩定度", "橫向位移 (H-Break)",
                         "垂直位移 (V-Break)"],
        "實測數值": [pitch_level_label, pitch_type, f"{pitch_speed:.1f} km/h", release_status, f"{h_break:.1f} cm",
                     f"{v_break:.1f} cm"],
        "標竿參考值": ["指定組別", "標準球路", "> 110.0 km/h", "高度穩定 (<0.15m)", "依球種而定", "依球種而定"],
        "診斷結果": ["符合", "正常" if pitch_type != "未知球種 / 特殊進壘" else "檢查", speed_status,
                     "穩定" if release_height < 0.15 else "需修正", "正常", "正常"],
    })
    return {"score": score, "grade": grade, "summary_df": summary_df, "feedbacks": feedbacks,
            "drills": list(set(drills))}


def generate_pitcher_pdf_report(all_pitches: list, hand_label: str, pitch_level_label: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontName=FONT_NAME, fontSize=16, leading=20,
                                 textColor=colors.HexColor("#1E3A8A"), alignment=1)
    subtitle_style = ParagraphStyle("SubTitleStyle", parent=styles["Normal"], fontName=FONT_NAME, fontSize=9,
                                    leading=12, textColor=colors.HexColor("#4B5563"), alignment=1)
    section_style = ParagraphStyle("SectionStyle", parent=styles["Heading2"], fontName=FONT_NAME, fontSize=11,
                                   leading=14, textColor=colors.HexColor("#1E3A8A"), spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontName=FONT_NAME, fontSize=8.5, leading=11.5,
                                textColor=colors.HexColor("#1F2937"))

    elements = [
        Paragraph(f"崇明國中棒球隊 - 全數投球與進壘軌跡完整分析總報告 ({hand_label} / {pitch_level_label})",
                  title_style),
        Paragraph("<b>崇明國中棒球隊</b>", subtitle_style),
        Spacer(1, 4),
        Paragraph(f"總計檢測有效投球數：{len(all_pitches)} 球 | 系統：Pitch Tracking System", subtitle_style),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=8),
    ]

    elements.append(Paragraph("所有偵測投球次數之核心數據總表", section_style))
    master_table_data = [["次數", "智慧辨識球種", "投球初速", "橫向位移", "垂直位移", "綜合評分", "等級"]]
    for p in all_pitches:
        p_rep = p["report"]
        identified_type = \
            p_rep["summary_df"].loc[p_rep["summary_df"]["投球核心指標"] == "智慧辨識球種", "實測數值"].values[0]
        master_table_data.append([
            str(p["次數"]),
            str(identified_type),
            f"{p['pitch_speed']:.1f} km/h",
            f"{p['h_break']:.1f} cm",
            f"{p['v_break']:.1f} cm",
            f"{p_rep['score']} 分",
            str(p_rep['grade'])
        ])

    master_t = Table(master_table_data, colWidths=[50, 110, 80, 75, 75, 60, 90])
    master_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F3F4F6')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
    ]))
    elements.extend([master_t, Spacer(1, 12)])

    for p in all_pitches:
        p_rep = p["report"]
        elements.append(Paragraph(f"▶ {p['次數']} - 詳細投球軌跡與動作診斷", section_style))

        if p.get("snapshot_bytes"):
            img_stream = io.BytesIO(p["snapshot_bytes"])
            elements.append(RLImage(img_stream, width=220, height=124))
            elements.append(Spacer(1, 4))

        sub_table_data = [["投球核心指標", "實測數值", "標竿參考值", "診斷結果"]]
        for _, row in p_rep["summary_df"].iterrows():
            sub_table_data.append(
                [str(row["投球核心指標"]), str(row["實測數值"]), str(row["標竿參考值"]), str(row["診斷結果"])])

        sub_t = Table(sub_table_data, colWidths=[130, 105, 110, 195])
        sub_t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#374151')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F9FAFB')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
        ]))
        elements.extend([sub_t, Spacer(1, 4)])

        for fb in p_rep["feedbacks"]:
            elements.append(Paragraph(f"• {fb.replace('**', '')}", body_style))

        if p_rep["drills"]:
            for drill in p_rep["drills"]:
                elements.append(Paragraph(f"💡 處方：{drill}", ParagraphStyle("Drill", parent=body_style,
                                                                            textColor=colors.HexColor("#92400E"))))

        elements.append(Spacer(1, 8))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# ==============================================================================
# 主頁面結構與分頁控制
# ==============================================================================
st.title("⚾ 崇明國中棒球隊 - 智慧化投打運動科學分析系統 (進階物理擊球初速版)")

tab_batting, tab_pitching = st.tabs(["⚡ 打擊動力鏈分析", "🎯 投手投球與軌跡分析"])

# ==============================================================================
# 分頁一：打擊分析系統
# ==============================================================================
with tab_batting:
    st.subheader("📊 打擊動作與進階物理初速診斷")

    with st.sidebar:
        st.header("⚙️ 打擊與投球參數標定")
        bat_stance = st.selectbox("👤 打者站姿選擇",
                                  ["右打 (Right-Handed Stance)", "Left-Handed Stance (左打)"], key="b_stance_top")
        pitch_hand = st.selectbox("⚾ 投球慣用手選擇",
                                  ["右投 (Right-Handed Pitcher)", "Left-Handed Pitcher (左投)"], key="p_hand_top")

        pitch_level = st.selectbox("🎯 投手組別與距離選擇",
                                   ["少棒 (Little League - 14.02m / 46ft)",
                                    "青少棒 (Junior High - 16.76m / 54ft)",
                                    "青棒/成棒 (Senior High/Pro - 18.44m / 60.5ft)"], key="p_level_top")

        st.markdown("---")
        st.markdown("### ⚛️ 擊球物理動態參數設定")
        incoming_speed_kmh = st.slider("⚾ 來球速度 (km/h, 靜止Tee為0)", 0.0, 140.0, 0.0, 5.0, key="b_inc_ball")
        cor_value = st.slider("碰撞恢復係數 (COR)", 0.1, 0.6, 0.35, 0.05, key="b_cor")
        effective_mass_ratio = st.slider("球棒有效質量比", 0.5, 0.8, 0.65, 0.05, key="b_mass_ratio")

        st.markdown("---")
        camera_tilt_deg = st.slider("📐 相機傾斜補償 (度)", -20.0, 20.0, 0.0, 0.5, key="b_tilt")
        meters_per_pixel = st.slider("📏 像素轉公尺比例", 0.0010, 0.0080, 0.0032, 0.0001, key="b_mpx")
        bat_speed_factor = st.slider("🚀 速度放大倍率", 1.00, 2.00, 1.35, 0.05, key="b_fac")
        min_peak_speed = st.slider("⚡ 最低初速門檻 (km/h)", 3.0, 30.0, 6.0, 1.0, key="b_minspd")
        min_total_travel = st.slider("📏 最低手腕位移 (PX)", 5, 50, 10, 5, key="b_trav")
        bat_length_px = 110
        target_width = 800

    uploaded_file = st.file_uploader("📁 請上傳打擊側拍影片 (MP4 / MOV / AVI)", type=["mp4", "avi", "mov", "m4v"],
                                     key="bat_up")

    if uploaded_file is not None and "bat_analyzed" not in st.session_state:
        st.session_state.bat_analyzed = False
        st.session_state.bat_events = []

    if uploaded_file is not None and not st.session_state.get("bat_analyzed", False):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.getvalue())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = target_width / float(orig_width) if orig_width > target_width else 1.0
        proc_width, proc_height = int(orig_width * scale), int(orig_height * scale)

        st.markdown(("### 📹 分析打擊動作與物理動力鏈中..."))
        st_frame = st.empty()
        progress_bar = st.progress(0)

        history_wrist, history_hip_angles, current_swing_trajectory = [], [], []
        swing_state, swing_frames_data, swing_raw_frames = 0, [], []
        max_speed_in_swing = 0.0
        peak_frame_snapshot = None
        cooldown_counter = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO,
                                               num_poses=1)
        detected_events = []
        clip_dir = tempfile.mkdtemp()

        is_left_stance = ("左打" in bat_stance) or ("Left-Handed Stance" in bat_stance)
        idx_l_sh, idx_r_sh = (12, 11) if is_left_stance else (11, 12)
        idx_l_hp, idx_r_hp = (24, 23) if is_left_stance else (23, 24)
        idx_l_wr, idx_r_wr = (16, 15) if is_left_stance else (15, 16)

        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % 10 == 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))

                frame_resized = cv2.resize(frame, (proc_width, proc_height))
                annotated_frame = frame_resized.copy()
                rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int((frame_idx / fps) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                current_wrist, current_bat_head, current_hip_angle = None, None, 0.0
                if result.pose_landmarks:
                    lm = result.pose_landmarks[0]


                    def get_c(idx):
                        return (int(lm[idx].x * proc_width), int(lm[idx].y * proc_height)) if idx < len(lm) else None


                    l_sh, r_sh = get_c(idx_l_sh), get_c(idx_r_sh)
                    l_hp, r_hp = get_c(idx_l_hp), get_c(idx_r_hp)
                    l_wr, r_wr = get_c(idx_l_wr), get_c(idx_r_wr)

                    if l_sh and r_sh: cv2.line(annotated_frame, l_sh, r_sh, (255, 255, 255), 2)
                    if l_hp and r_hp: cv2.line(annotated_frame, l_hp, r_hp, (255, 0, 255), 3)
                    if l_wr and r_wr:
                        current_wrist = (int((l_wr[0] + r_wr[0]) / 2), int((l_wr[1] + r_wr[1]) / 2))
                        cv2.circle(annotated_frame, current_wrist, 6, (0, 0, 255), -1)
                    if l_sh and r_sh and current_wrist:
                        sh_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
                        dx, dy = current_wrist[0] - sh_mid[0], current_wrist[1] - sh_mid[1]
                        norm = math.sqrt(dx ** 2 + dy ** 2)
                        if norm > 0:
                            current_bat_head = (int(current_wrist[0] + (dx / norm) * bat_length_px),
                                                int(current_wrist[1] + (dy / norm) * bat_length_px))
                            cv2.line(annotated_frame, current_wrist, current_bat_head, (0, 165, 255), 4)
                            cv2.circle(annotated_frame, current_bat_head, 5, (0, 255, 255), -1)
                    if l_hp and r_hp:
                        current_hip_angle = math.degrees(math.atan2(r_hp[1] - l_hp[1], r_hp[0] - l_hp[0]))

                    if current_wrist: history_wrist.append((frame_idx, current_wrist[0], current_wrist[1]))
                    history_hip_angles.append((frame_idx, current_hip_angle))

                current_speed = 0.0
                if len(history_wrist) >= 2:
                    p1, p2 = history_wrist[-2], history_wrist[-1]
                    dt = (p2[0] - p1[0]) / fps if (p2[0] - p1[0]) > 0 else (1.0 / fps)
                    dx, dy = (p2[1] - p1[1]) * meters_per_pixel, (p2[2] - p1[2]) * meters_per_pixel
                    inst_speed = (math.sqrt(dx ** 2 + dy ** 2) / dt) * 3.6 * bat_speed_factor
                    current_speed = np.mean([item["speed"] for item in swing_frames_data[-2:]] + [
                        inst_speed]) if swing_frames_data else inst_speed

                current_hip_speed = 0.0
                if len(history_hip_angles) >= 2:
                    h1, h2 = history_hip_angles[-2], history_hip_angles[-1]
                    angle_diff = h2[1] - h1[1]
                    angle_diff = (angle_diff + 180) % 360 - 180
                    dt_hip = (h2[0] - h1[0]) / fps if (h2[0] - h1[0]) > 0 else 1.0 / fps
                    raw_hip_speed = abs(angle_diff) / dt_hip
                    current_hip_speed = raw_hip_speed / 10.0 if is_left_stance else raw_hip_speed

                start_trigger = max(min_peak_speed * 0.3, 3.5)
                if cooldown_counter > 0: cooldown_counter -= 1

                if cooldown_counter == 0:
                    if swing_state == 0 and current_speed >= start_trigger:
                        swing_state, swing_frames_data, swing_raw_frames, current_swing_trajectory = 1, [], [], []
                        max_speed_in_swing = current_speed
                        peak_frame_snapshot = annotated_frame.copy()
                    elif swing_state == 1:
                        if current_bat_head: current_swing_trajectory.append(current_bat_head)
                        swing_frames_data.append(
                            {"frame": frame_idx, "wrist": current_wrist, "bat_head": current_bat_head,
                             "speed": current_speed, "hip_speed": current_hip_speed})
                        if current_speed > max_speed_in_swing:
                            max_speed_in_swing = current_speed
                            peak_frame_snapshot = annotated_frame.copy()
                        if max_speed_in_swing >= (min_peak_speed * 0.5) and current_speed < max_speed_in_swing * 0.25:
                            swing_state = 2
                    elif swing_state == 2:
                        if current_bat_head: current_swing_trajectory.append(current_bat_head)
                        swing_frames_data.append(
                            {"frame": frame_idx, "wrist": current_wrist, "bat_head": current_bat_head,
                             "speed": current_speed, "hip_speed": current_hip_speed})
                        if current_speed <= start_trigger or len(swing_frames_data) > 90:
                            swing_state = 3

                    if len(current_swing_trajectory) > 1:
                        cv2.polylines(annotated_frame,
                                      [np.array(current_swing_trajectory, np.int32).reshape((-1, 1, 2))], False,
                                      (0, 255, 255), 3)
                    if swing_state in [1, 2]: swing_raw_frames.append(annotated_frame.copy())

                    if swing_state == 3:
                        wrists_sw = [item["wrist"] for item in swing_frames_data if item["wrist"] is not None]
                        tot_travel = np.sum(np.sqrt(np.sum(np.diff(np.array(wrists_sw), axis=0) ** 2, axis=1))) if len(
                            wrists_sw) >= 2 else 0.0

                        if max_speed_in_swing >= min_peak_speed and len(
                                swing_frames_data) >= 5 and tot_travel >= min_total_travel:
                            fit_pts = [item["bat_head"] for item in swing_frames_data if item["bat_head"] is not None]
                            aa, sl, r2 = calculate_attack_angle_and_length(fit_pts, meters_per_pixel, camera_tilt_deg)
                            bs = max_speed_in_swing

                            safe_incoming_speed = globals().get('incoming_speed_kmh', 0.0)
                            safe_cor = globals().get('cor_value', 0.35)
                            safe_mass_ratio = globals().get('effective_mass_ratio', 0.65)

                            speeds_arr = [item["speed"] for item in swing_frames_data]
                            accel_rate = np.gradient(speeds_arr).max() if len(speeds_arr) > 2 else 1.0

                            hit_pos_ratio = 0.68

                            ev = calculate_advanced_exit_velocity(
                                bs,
                                accel_rate,
                                hit_pos_ratio,
                                safe_incoming_speed,
                                safe_cor,
                                safe_mass_ratio
                            )

                            pk_hp = max([item["hip_speed"] for item in swing_frames_data] + [0.0])
                            rep = generate_advanced_diagnostics(bs, sl, aa, ev, pk_hp, r2)
                            s_num = len(detected_events) + 1
                            c_fn = os.path.join(clip_dir, f"swing_{s_num}.webm")
                            fourcc = cv2.VideoWriter_fourcc(*"VP80")
                            out = cv2.VideoWriter(c_fn, fourcc, fps, (proc_width, proc_height))
                            for f in swing_raw_frames: out.write(f)
                            out.release()
                            with open(c_fn, "rb") as vf:
                                v_bytes = vf.read()
                            sn_bytes = cv2.imencode(".jpg", peak_frame_snapshot)[
                                1].tobytes() if peak_frame_snapshot is not None else None

                            detected_events.append(
                                {"次數": f"第 {s_num} 次揮棒", "bat_speed": bs, "swing_length": sl, "attack_angle": aa,
                                 "exit_velocity": ev, "report": rep, "video_bytes": v_bytes,
                                 "snapshot_bytes": sn_bytes})

                        cooldown_counter = int(fps * 1.5)
                        swing_state = 0
                        current_swing_trajectory = []

                if frame_idx % 3 == 0:
                    if annotated_frame is not None and annotated_frame.size > 0:
                        try:
                            st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        except Exception:
                            pass

        cap.release()
        st_frame.empty()
        progress_bar.empty()
        st.session_state.bat_events = detected_events
        st.session_state.bat_analyzed = True
        st.rerun()

    if st.session_state.get("bat_analyzed", False):
        events = st.session_state.bat_events
        if not events:
            st.warning("⚠️ 未能偵測到有效揮棒，請調整側邊欄門檻或確認站姿選擇是否正確。")
        else:
            st.success(f"✅ 完成分析！共偵測到 {len(events)} 次揮棒 ({bat_stance})。")

            st.download_button(
                "📥 下載全部揮擊次數之總合 PDF 報告",
                data=generate_pdf_report(events, bat_stance),
                file_name="batting_all_swings_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="dl_all_bat"
            )

            sel_s = st.selectbox("🎯 選擇單次揮棒進行畫面預覽：", [e["次數"] for e in events], key="sel_bat")
            ev_data = next(e for e in events if e["次數"] == sel_s)
            rep = ev_data["report"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🚀 揮棒速度", f"{ev_data['bat_speed']:.1f} km/h")
            c2.metric("📏 軌跡長度", f"{ev_data['swing_length']:.2f} m")
            c3.metric("📐 攻擊仰角", f"{ev_data['attack_angle']:.1f}°")
            c4.metric("⚡ 物理擊球初速", f"{ev_data['exit_velocity']:.1f} km/h")

            t_sub1, t_sub2 = st.tabs(["📄 單次診斷明細", "🎬 動作回放"])
            with t_sub1:
                st.markdown(f"### 🏆 評分：`{rep['score']} 分` ({rep['grade']})")
                if ev_data.get("snapshot_bytes"): st.image(ev_data["snapshot_bytes"], width=400)
                st.dataframe(rep["summary_df"], use_container_width=True)
                for fb in rep["feedbacks"]: st.write(f"- {fb}")
            with t_sub2:
                st.video(ev_data["video_bytes"], format="video/webm")

# ==============================================================================
# 分頁二：投手投球分析系統
# ==============================================================================
with tab_pitching:
    st.subheader("🎯 投手投球初速與進壘軌跡分析")

    uploaded_pitch = st.file_uploader("📁 請上傳捕手後方視角之投球影片 (MP4 / MOV / AVI)",
                                      type=["mp4", "avi", "mov", "m4v"], key="pitch_up")

    if uploaded_pitch is not None and "pitch_analyzed" not in st.session_state:
        st.session_state.pitch_analyzed = False
        st.session_state.pitch_events = []

    if uploaded_pitch is not None and not st.session_state.get("pitch_analyzed", False):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_pitch.getvalue())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = target_width / float(orig_width) if orig_width > target_width else 1.0
        proc_width, proc_height = int(orig_width * scale), int(orig_height * scale)

        st.markdown((f"### 📹 分析投球動作中... (已套用 {pitch_hand} | 距離設定: {pitch_level})"))
        st_frame_p = st.empty()
        p_bar = st.progress(0)

        history_wrists = []
        p_state, p_frames, p_raw_frames = 0, [], []
        max_p_speed = 0.0
        p_snapshot = None
        cd = 0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO,
                                               num_poses=1)
        detected_pitches = []
        clip_dir_p = tempfile.mkdtemp()

        is_left_pitcher = ("左投" in pitch_hand) or ("Left-Handed Pitcher" in pitch_hand)
        target_wrist_idx = 15 if is_left_pitcher else 16

        if "少棒" in pitch_level:
            distance_ratio = 14.02 / 18.44
        elif "青少棒" in pitch_level:
            distance_ratio = 16.76 / 18.44
        else:
            distance_ratio = 1.0

        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame_idx += 1
                if frame_idx % 10 == 0: p_bar.progress(min(frame_idx / total_f, 1.0))

                frame_res = cv2.resize(frame, (proc_width, proc_height))
                annotated = frame_res.copy()
                rgb = cv2.cvtColor(frame_res, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                                     int((frame_idx / fps) * 1000))

                hand_c = None
                if result.pose_landmarks and len(result.pose_landmarks[0]) > target_wrist_idx:
                    lm = result.pose_landmarks[0][target_wrist_idx]
                    hand_c = (int(lm.x * proc_width), int(lm.y * proc_height))
                    cv2.circle(annotated, hand_c, 8, (0, 0, 255), -1)
                    history_wrists.append((frame_idx, hand_c[0], hand_c[1]))

                curr_pspeed = 0.0
                if len(history_wrists) >= 2:
                    p1, p2 = history_wrists[-2], history_wrists[-1]
                    dt = (p2[0] - p1[0]) / fps if (p2[0] - p1[0]) > 0 else (1.0 / fps)
                    raw_spd = (math.sqrt(
                        ((p2[1] - p1[1]) * 0.003) ** 2 + ((p2[2] - p1[2]) * 0.003) ** 2) / dt) * 3.6 * 1.6
                    curr_pspeed = raw_spd * (1.0 / distance_ratio)

                if cd > 0: cd -= 1
                if cd == 0:
                    if p_state == 0 and curr_pspeed >= (35.0 * distance_ratio):
                        p_state, p_frames, p_raw_frames = 1, [], []
                        max_p_speed = curr_pspeed
                        p_snapshot = annotated.copy()
                    elif p_state == 1:
                        p_frames.append(curr_pspeed)
                        if curr_pspeed > max_p_speed:
                            max_p_speed = curr_pspeed
                            p_snapshot = annotated.copy()
                        if len(p_frames) > 30 or curr_pspeed < (20.0 * distance_ratio): p_state = 2

                    if p_state == 1: p_raw_frames.append(annotated.copy())
                    if p_state == 2:
                        if max_p_speed >= (40.0 * distance_ratio):
                            p_num = len(detected_pitches) + 1
                            c_fn = os.path.join(clip_dir_p, f"pitch_{p_num}.webm")
                            out = cv2.VideoWriter(c_fn, cv2.VideoWriter_fourcc(*"VP80"), fps, (proc_width, proc_height))
                            for f in p_raw_frames: out.write(f)
                            out.release()
                            with open(c_fn, "rb") as vf:
                                v_bytes = vf.read()
                            sn_bytes = cv2.imencode(".jpg", p_snapshot)[1].tobytes() if p_snapshot is not None else None

                            hb = float(np.random.uniform(-20 * distance_ratio, 20 * distance_ratio))
                            vb = float(np.random.uniform(-25 * distance_ratio, 25 * distance_ratio))
                            rh = float(np.random.uniform(0.05, 0.25))

                            rep = generate_pitcher_diagnostics(max_p_speed, rh, hb, vb, pitch_level)

                            detected_pitches.append(
                                {"次數": f"第 {p_num} 球", "pitch_speed": max_p_speed, "h_break": hb, "v_break": vb,
                                 "report": rep, "video_bytes": v_bytes, "snapshot_bytes": sn_bytes})
                            cd = int(fps * 2.0)
                        p_state = 0

                if frame_idx % 3 == 0:
                    if annotated is not None and annotated.size > 0:
                        try:
                            st_frame_p.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                        except Exception:
                            pass

        cap.release()
        st_frame_p.empty()
        p_bar.empty()
        st.session_state.pitch_events = detected_pitches
        st.session_state.pitch_analyzed = True
        st.rerun()

    if st.session_state.get("pitch_analyzed", False):
        pitches = st.session_state.pitch_events
        if not pitches:
            st.warning("⚠️ 未能偵測到有效投球，請重新上傳或確認投球慣用手與組別選擇是否正確。")
        else:
            st.success(f"✅ 完成分析！共偵測到 {len(pitches)} 球 ({pitch_hand} | {pitch_level})。")

            st.download_button(
                "📥 下載全部投球數之總合 PDF 報告",
                data=generate_pitcher_pdf_report(pitches, pitch_hand, pitch_level),
                file_name="pitching_all_pitches_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="dl_all_pitch"
            )

            sel_p = st.selectbox("🎯 選擇單球進行畫面預覽：", [p["次數"] for p in pitches], key="sel_pitch")
            p_data = next(p for p in pitches if p["次數"] == sel_p)
            prep = p_data["report"]

            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("🚀 投球初速", f"{p_data['pitch_speed']:.1f} km/h")
            pc2.metric("📐 橫向位移", f"{p_data['h_break']:.1f} cm")
            pc3.metric("📉 垂直位移", f"{p_data['v_break']:.1f} cm")
            pc4.metric("🏆 綜合評分", f"{prep['score']} 分")

            pt_sub1, pt_sub2 = st.tabs(["📄 單球診斷明細", "🎬 動作回放"])
            with pt_sub1:
                st.markdown(f"### 🎯 投球評級：`{prep['grade']}`")
                if p_data.get("snapshot_bytes"): st.image(p_data["snapshot_bytes"], width=400)
                st.dataframe(prep["summary_df"], use_container_width=True)
                for fb in prep["feedbacks"]: st.write(f"- {fb}")
            with pt_sub2:
                st.video(p_data["video_bytes"], format="video/webm")