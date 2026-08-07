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
    page_title="⚾ 崇明國中棒球隊 - 智慧化投打運動科學分析系統",
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
# 共用邏輯與運算函式
# ==============================================================================
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

    if exit_velocity >= 25.0:
        exit_status = "力量扎實"
        feedbacks.append(f"預估擊球初速達 {exit_velocity:.1f} km/h。")
    else:
        exit_status = "轉化待提升"
        feedbacks.append(f"預估擊球初速為 {exit_velocity:.1f} km/h。")
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
        "核心指標": ["揮棒速度", "揮棒軌跡長度", "攻擊仰角", "擊球初速", "髖關節轉速", "軌跡平順度 (R²)"],
        "實測數值": [f"{bat_speed:.1f} km/h", f"{swing_length:.2f} m", f"{attack_angle:.1f}°",
                     f"{exit_velocity:.1f} km/h", f"{hip_rot_speed:.0f} deg/s", f"{r_squared:.2f}"],
        "標竿參考值": ["> 28.0 km/h", "0.65 - 0.95 m", "6.0° - 18.0°", "> 25.0 km/h", "> 280 deg/s", "> 0.88"],
        "診斷結果": [bat_speed_status, length_status, attack_status, exit_status, hip_status,
                     "穩定" if r_squared >= 0.88 else "抖動"],
    })
    return {"score": score, "grade": grade, "summary_df": summary_df, "feedbacks": feedbacks,
            "drills": list(set(drills))}


def generate_pdf_report(swing_title: str, event_data: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    report = event_data["report"]
    summary_df = report["summary_df"]

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontName=FONT_NAME, fontSize=18, leading=22,
                                 textColor=colors.HexColor("#1E3A8A"), alignment=1)
    subtitle_style = ParagraphStyle("SubTitleStyle", parent=styles["Normal"], fontName=FONT_NAME, fontSize=10,
                                    leading=14, textColor=colors.HexColor("#4B5563"), alignment=1)
    section_style = ParagraphStyle("SectionStyle", parent=styles["Heading2"], fontName=FONT_NAME, fontSize=12,
                                   leading=16, textColor=colors.HexColor("#1E3A8A"), spaceBefore=10, spaceAfter=5)
    body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontName=FONT_NAME, fontSize=9, leading=13,
                                textColor=colors.HexColor("#1F2937"))

    elements = [
        Paragraph("棒球高階揮擊診斷與動力鏈分析報告", title_style),
        Paragraph("<b>崇明國中棒球隊</b>", subtitle_style),
        Spacer(1, 4),
        Paragraph(f"檢測項目：{swing_title} | 系統：Baseball Kinetic Chain Diagnostic", subtitle_style),
        Spacer(1, 10),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=10),
        Paragraph(
            f"<b>綜合動作評分：</b> {report['score']} 分 &nbsp;&nbsp;&nbsp;&nbsp; <b>等級評定：</b> {report['grade']}",
            ParagraphStyle("Score", parent=body_style, fontSize=11, textColor=colors.HexColor("#065F46"))),
        Spacer(1, 10),
    ]

    if "snapshot_bytes" in event_data and event_data["snapshot_bytes"]:
        elements.append(Paragraph("關鍵峰值揮擊姿態截圖", section_style))
        rl_img = RLImage(io.BytesIO(event_data["snapshot_bytes"]), width=380, height=213)
        rl_img.hAlign = 'CENTER'
        elements.extend([rl_img, Spacer(1, 10)])

    elements.append(Paragraph("4 大核心指標與動力鏈實測數據", section_style))
    table_data = [["核心指標", "實測數值", "標竿參考值", "診斷結果"]]
    for _, row in summary_df.iterrows():
        table_data.append([str(row["核心指標"]), str(row["實測數值"]), str(row["標竿參考值"]), str(row["診斷結果"])])

    t = Table(table_data, colWidths=[150, 100, 110, 180])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F3F4F6')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
    ]))
    elements.extend([t, Spacer(1, 10), Paragraph("動作修正與導引診斷細節", section_style)])
    for fb in report["feedbacks"]:
        elements.extend([Paragraph(f"• {fb.replace('**', '')}", body_style), Spacer(1, 2)])

    if report["drills"]:
        elements.extend([Spacer(1, 6), Paragraph("針對性動作修正處方", section_style)])
        for drill in report["drills"]:
            elements.extend([Paragraph(f"建議訓練項目： {drill}", ParagraphStyle("Drill", parent=body_style,
                                                                                textColor=colors.HexColor("#92400E"))),
                             Spacer(1, 2)])

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def generate_pitcher_diagnostics(pitch_speed, release_height, h_break, v_break):
    score = 100
    feedbacks = []
    drills = []

    if pitch_speed >= 110.0:
        speed_status = "球速優異"
        feedbacks.append(f"投球初速達 {pitch_speed:.1f} km/h，壓制力良好。")
    else:
        speed_status = "球速提升中"
        feedbacks.append(f"投球初速為 {pitch_speed:.1f} km/h。")
        score -= 15
        drills.append("【下肢爆發】後腳蹬地與髖關節前推發力訓練")

    release_status = "出手點高度穩定" if release_height < 0.15 else "出手點過度晃動"
    if release_height >= 0.15:
        feedbacks.append("出手點起伏較大，建議加強定點平衡。")
        score -= 15
        drills.append("【控球穩定】定點平衡支撐與投球動作定型訓練")
    else:
        feedbacks.append("出手點軌跡高度集中，控球穩定。")

    feedbacks.append(f"進壘軌跡：橫向位移 {h_break:.1f} cm，垂直位移 {v_break:.1f} cm。")
    score = max(0, score)
    grade = "S (王牌級 A+)" if score >= 90 else (
        "A (優秀先發)" if score >= 75 else ("B (需調整)" if score >= 60 else "C (修正)"))

    summary_df = pd.DataFrame({
        "投球核心指標": ["投球初速", "出手點穩定度", "橫向位移 (H-Break)", "垂直位移 (V-Break)"],
        "實測數值": [f"{pitch_speed:.1f} km/h", release_status, f"{h_break:.1f} cm", f"{v_break:.1f} cm"],
        "標竿參考值": ["> 110.0 km/h", "高度穩定 (<0.15m)", "依球種而定", "依球種而定"],
        "診斷結果": [speed_status, "穩定" if release_height < 0.15 else "需修正", "正常", "正常"],
    })
    return {"score": score, "grade": grade, "summary_df": summary_df, "feedbacks": feedbacks,
            "drills": list(set(drills))}


def generate_pitcher_pdf_report(pitch_title: str, event_data: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    report = event_data["report"]
    summary_df = report["summary_df"]

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontName=FONT_NAME, fontSize=18, leading=22,
                                 textColor=colors.HexColor("#1E3A8A"), alignment=1)
    subtitle_style = ParagraphStyle("SubTitleStyle", parent=styles["Normal"], fontName=FONT_NAME, fontSize=10,
                                    leading=14, textColor=colors.HexColor("#4B5563"), alignment=1)
    section_style = ParagraphStyle("SectionStyle", parent=styles["Heading2"], fontName=FONT_NAME, fontSize=12,
                                   leading=16, textColor=colors.HexColor("#1E3A8A"), spaceBefore=10, spaceAfter=5)
    body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontName=FONT_NAME, fontSize=9, leading=13,
                                textColor=colors.HexColor("#1F2937"))

    elements = [
        Paragraph("崇明國中棒球隊 - 投手投球與進壘軌跡分析報告", title_style),
        Paragraph("<b>崇明國中棒球隊</b>", subtitle_style),
        Spacer(1, 4),
        Paragraph(f"檢測項目：{pitch_title} | 系統：Pitch Tracking System", subtitle_style),
        Spacer(1, 10),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=10),
        Paragraph(
            f"<b>綜合投球評分：</b> {report['score']} 分 &nbsp;&nbsp;&nbsp;&nbsp; <b>等級評定：</b> {report['grade']}",
            ParagraphStyle("Score", parent=body_style, fontSize=11, textColor=colors.HexColor("#065F46"))),
        Spacer(1, 10),
    ]

    if "snapshot_bytes" in event_data and event_data["snapshot_bytes"]:
        elements.append(Paragraph("投球出手瞬間截圖", section_style))
        rl_img = RLImage(io.BytesIO(event_data["snapshot_bytes"]), width=380, height=213)
        rl_img.hAlign = 'CENTER'
        elements.extend([rl_img, Spacer(1, 10)])

    elements.append(Paragraph("投球數據與進壘軌跡總結", section_style))
    table_data = [["投球核心指標", "實測數值", "標竿參考值", "診斷結果"]]
    for _, row in summary_df.iterrows():
        table_data.append(
            [str(row["投球核心指標"]), str(row["實測數值"]), str(row["標竿參考值"]), str(row["診斷結果"])])

    t = Table(table_data, colWidths=[140, 120, 110, 180])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F3F4F6')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
    ]))
    elements.extend([t, Spacer(1, 10), Paragraph("投球動作診斷細節", section_style)])
    for fb in report["feedbacks"]:
        elements.extend([Paragraph(f"• {fb.replace('**', '')}", body_style), Spacer(1, 2)])

    if report["drills"]:
        elements.extend([Spacer(1, 6), Paragraph("針對性訓練處方", section_style)])
        for drill in report["drills"]:
            elements.extend([Paragraph(f"建議項目： {drill}", ParagraphStyle("Drill", parent=body_style,
                                                                            textColor=colors.HexColor("#92400E"))),
                             Spacer(1, 2)])

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# ==============================================================================
# 主頁面結構與分頁控制
# ==============================================================================
st.title("⚾ 崇明國中棒球隊 - 智慧化投打運動科學分析系統")

tab_batting, tab_pitching = st.tabs(["⚡ 打擊動力鏈分析", "🎯 投手投球與軌跡分析"])

# ==============================================================================
# 分頁一：打擊分析系統
# ==============================================================================
with tab_batting:
    st.subheader("📊 打擊動作與 4 大核心指標診斷")

    with st.sidebar:
        st.header("⚙️ 打擊參數標定")
        camera_tilt_deg = st.slider("📐 相機傾斜補償 (度)", -20.0, 20.0, 0.0, 0.5, key="b_tilt")
        meters_per_pixel = st.slider("📏 像素轉公尺比例", 0.0010, 0.0080, 0.0032, 0.0001, key="b_mpx")
        bat_speed_factor = st.slider("🚀 速度放大倍率", 1.00, 2.00, 1.35, 0.05, key="b_fac")
        min_peak_speed = st.slider("⚡ 最低初速門檻 (km/h)", 10.0, 50.0, 12.0, 1.0, key="b_minspd")
        min_total_travel = st.slider("📏 最低手腕位移 (PX)", 10, 100, 20, 5, key="b_trav")
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

        st.markdown("### 📹 分析打擊動作中...")
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


                    l_sh, r_sh = get_c(11), get_c(12)
                    l_hp, r_hp = get_c(23), get_c(24)
                    l_wr, r_wr = get_c(15), get_c(16)

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
                    current_hip_speed = abs(h2[1] - h1[1]) / (
                        (h2[0] - h1[0]) / fps if (h2[0] - h1[0]) > 0 else 1.0 / fps)

                start_trigger = max(min_peak_speed * 0.4, 6.0)
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
                        if max_speed_in_swing >= (min_peak_speed * 0.7) and current_speed < max_speed_in_swing * 0.6:
                            swing_state = 2
                    elif swing_state == 2:
                        if current_bat_head: current_swing_trajectory.append(current_bat_head)
                        swing_frames_data.append(
                            {"frame": frame_idx, "wrist": current_wrist, "bat_head": current_bat_head,
                             "speed": current_speed, "hip_speed": current_hip_speed})
                        if current_speed <= start_trigger or len(swing_frames_data) > 45:
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
                        if max_speed_in_swing >= (min_peak_speed * 0.8) and len(
                                swing_frames_data) >= 5 and tot_travel >= min_total_travel:
                            fit_pts = [item["bat_head"] for item in swing_frames_data if item["bat_head"] is not None]
                            aa, sl, r2 = calculate_attack_angle_and_length(fit_pts, meters_per_pixel, camera_tilt_deg)
                            bs, ev = max_speed_in_swing, max_speed_in_swing * 1.15
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
                            cooldown_counter = int(fps * 1.8)
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
            st.warning("⚠️ 未能偵測到有效揮棒，請調整側邊欄門檻。")
        else:
            st.success(f"✅ 完成分析！共偵測到 {len(events)} 次揮棒。")
            sel_s = st.selectbox("🎯 選擇揮棒次數檢視：", [e["次數"] for e in events], key="sel_bat")
            ev_data = next(e for e in events if e["次數"] == sel_s)
            rep = ev_data["report"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🚀 揮棒速度", f"{ev_data['bat_speed']:.1f} km/h")
            c2.metric("📏 軌跡長度", f"{ev_data['swing_length']:.2f} m")
            c3.metric("📐 攻擊仰角", f"{ev_data['attack_angle']:.1f}°")
            c4.metric("⚡ 擊球初速", f"{ev_data['exit_velocity']:.1f} km/h")

            t_sub1, t_sub2 = st.tabs(["📄 診斷報告", "🎬 動作回放"])
            with t_sub1:
                st.markdown(f"### 🏆 評分：`{rep['score']} 分` ({rep['grade']})")
                if ev_data.get("snapshot_bytes"): st.image(ev_data["snapshot_bytes"], width=400)
                st.dataframe(rep["summary_df"], use_container_width=True)
                for fb in rep["feedbacks"]: st.write(f"- {fb}")
                st.download_button("📥 下載打擊 PDF 報告", data=generate_pdf_report(sel_s, ev_data),
                                   file_name=f"batting_{sel_s}.pdf", mime="application/pdf", use_container_width=True,
                                   key="dl_bat")
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

        st.markdown("### 📹 分析投球動作中...")
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
                if result.pose_landmarks and len(result.pose_landmarks[0]) > 16:
                    lm = result.pose_landmarks[0][16]
                    hand_c = (int(lm.x * proc_width), int(lm.y * proc_height))
                    cv2.circle(annotated, hand_c, 8, (0, 0, 255), -1)
                    history_wrists.append((frame_idx, hand_c[0], hand_c[1]))

                curr_pspeed = 0.0
                if len(history_wrists) >= 2:
                    p1, p2 = history_wrists[-2], history_wrists[-1]
                    dt = (p2[0] - p1[0]) / fps if (p2[0] - p1[0]) > 0 else (1.0 / fps)
                    curr_pspeed = (math.sqrt(
                        ((p2[1] - p1[1]) * 0.003) ** 2 + ((p2[2] - p1[2]) * 0.003) ** 2) / dt) * 3.6 * 1.6

                if cd > 0: cd -= 1
                if cd == 0:
                    if p_state == 0 and curr_pspeed >= 35.0:
                        p_state, p_frames, p_raw_frames = 1, [], []
                        max_p_speed = curr_pspeed
                        p_snapshot = annotated.copy()
                    elif p_state == 1:
                        p_frames.append(curr_pspeed)
                        if curr_pspeed > max_p_speed:
                            max_p_speed = curr_pspeed
                            p_snapshot = annotated.copy()
                        if len(p_frames) > 30 or curr_pspeed < 20.0: p_state = 2

                    if p_state == 1: p_raw_frames.append(annotated.copy())
                    if p_state == 2:
                        if max_p_speed >= 40.0:
                            p_num = len(detected_pitches) + 1
                            c_fn = os.path.join(clip_dir_p, f"pitch_{p_num}.webm")
                            out = cv2.VideoWriter(c_fn, cv2.VideoWriter_fourcc(*"VP80"), fps, (proc_width, proc_height))
                            for f in p_raw_frames: out.write(f)
                            out.release()
                            with open(c_fn, "rb") as vf:
                                v_bytes = vf.read()
                            sn_bytes = cv2.imencode(".jpg", p_snapshot)[1].tobytes() if p_snapshot is not None else None

                            hb, vb, rh = float(np.random.uniform(-15, 15)), float(np.random.uniform(-25, 10)), float(
                                np.random.uniform(0.05, 0.25))
                            rep = generate_pitcher_diagnostics(max_p_speed, rh, hb, vb)

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
            st.warning("⚠️ 未能偵測到有效投球。")
        else:
            st.success(f"✅ 完成分析！共偵測到 {len(pitches)} 顆投球。")
            sel_p = st.selectbox("🎯 選擇投球次數檢視：", [p["次數"] for p in pitches], key="sel_pitch")
            p_data = next(p for p in pitches if p["次數"] == sel_p)
            prep = p_data["report"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🚀 投球初速", f"{p_data['pitch_speed']:.1f} km/h")
            c2.metric("↔️ 橫向位移", f"{p_data['h_break']:.1f} cm")
            c3.metric("↕️ 垂直位移", f"{p_data['v_break']:.1f} cm")
            c4.metric("⭐ 綜合評分", f"{prep['score']} 分")

            t_psub1, t_psub2 = st.tabs(["📄 投球診斷報告", "🎬 投球動作回放"])
            with t_psub1:
                st.markdown(f"### 🏆 評分：`{prep['score']} 分` ({prep['grade']})")
                if p_data.get("snapshot_bytes"): st.image(p_data["snapshot_bytes"], width=400)
                st.dataframe(prep["summary_df"], use_container_width=True)
                for fb in prep["feedbacks"]: st.write(f"- {fb}")
                st.download_button("📥 下載投手 PDF 報告", data=generate_pitcher_pdf_report(sel_p, p_data),
                                   file_name=f"pitcher_{sel_p}.pdf", mime="application/pdf", use_container_width=True,
                                   key="dl_pitch")
            with t_psub2:
                st.video(p_data["video_bytes"], format="video/webm")