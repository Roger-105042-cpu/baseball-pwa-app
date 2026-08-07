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
# 0. 頁面設定與模型/字型註冊 (防亂碼核心機制)
# ==============================================================================
st.set_page_config(
    page_title="⚾ 棒球高階揮擊診斷與動力鏈分析系統",
    page_icon="⚾",
    layout="wide",
)

MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

FONT_FILE = "NotoSansTC-Regular.ttf"
FONT_NAME = "NotoSansTC"


def download_file_with_user_agent(url, save_path):
    """加上完整的 User-Agent 避免下載遭到阻擋"""
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
    with urllib.request.urlopen(req) as response, open(
        save_path, "wb"
    ) as out_file:
        out_file.write(response.read())


@st.cache_resource
def ensure_dependencies():
    # 1. 下載 MediaPipe 模型
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ 首次執行，正在下載 MediaPipe 姿態識別模型..."):
            try:
                download_file_with_user_agent(MODEL_URL, MODEL_PATH)
            except Exception as e:
                st.error(f"❌ 模型下載失敗: {e}")


ensure_dependencies()

# 2. 強制註冊繁體中文字型（優先讀取本地，次之備援）
if os.path.exists(FONT_FILE):
    try:
        pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_FILE))
    except Exception:
        # 若 TTFont 註冊失敗，改用內建 CID 中文字型
        FONT_NAME = "STSong-Light"
        try:
            pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))
        except:
            pass
else:
    # 本地若無字型檔案，嘗試註冊 ReportLab 內建中文字型
    FONT_NAME = "STSong-Light"
    try:
        pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))
    except:
        FONT_NAME = "Helvetica"  # 最後防線

st.title("⚾ 棒球高階揮擊診斷與動力鏈分析系統")
st.caption(
    "整合 4 大核心指標（揮棒速度、揮棒軌跡長度、攻擊仰角、擊球初速）與下半身髖關節旋轉動力鏈診斷"
)

# 顯示核心數據與動作修正指標說明區塊
with st.expander("📖 查看 4 大核心數據說明與動作修正重點", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        #### 📊 核心參考數據
        1. **揮棒速度 (Bat Speed)**：代表球棒通過本壘板前緣的移動速率，直接決定擊球的爆發力與遠度。
        2. **揮棒軌跡長度 (Swing Length)**：衡量球棒從啟動到擊球點移動的距離，軌跡越短、越能有效應對高速球。
        3. **攻擊仰角 (Attack Angle)**：球棒迎向擊球區的垂直傾斜角度，決定球被打出去是強勁平飛球還是無力滾地球。
        4. **擊球初速 (Exit Velocity)**：球離開球棒瞬間的速度，是用來檢驗揮棒力量是否扎實轉化為實際破壞力的指標。
        """)
    with col_b:
        st.markdown("""
        #### ⚙️ 動作修正方向
        * **改善軌跡**：若揮棒路徑過長或角度過陡（由上往下砍），會降低容錯率且容易揮空。
        * **強化轉動**：透過後髖關節與骨盆旋轉速度的數據，確保力量由下而上完整傳遞到棒頭。
        """)

# ==============================================================================
# 1. 側邊欄控制項
# ==============================================================================
with st.sidebar:
    st.header("⚙️ 系統與物理參數標定")

    camera_tilt_deg = st.slider(
        "📐 相機傾斜補償 (度)",
        -20.0,
        20.0,
        0.0,
        0.5,
        help="修正拍攝傾斜角度，維持絕對地平線基準。",
    )

    meters_per_pixel = st.slider(
        "📏 像素轉公尺比例 (Meters/PX)",
        0.0010,
        0.0080,
        0.0032,
        0.0001,
        help="標定空間距離，影響揮棒速度與揮棒軌跡長度之精確計算。",
    )

    bat_speed_factor = st.slider(
        "🚀 手腕至棒頭速度放大倍率",
        1.00,
        2.00,
        1.35,
        0.05,
        help="以手腕點推算棒頭通過本壘板前緣離心速度之補償係數。",
    )

    st.subheader("揮棒偵測門檻")
    min_peak_speed = st.slider("⚡ 最低初速門檻 (km/h)", 10.0, 50.0, 12.0, 1.0)
    min_total_travel = st.slider("📏 最低手腕位移 (PX)", 10, 100, 20, 5)

    bat_length_px = 110
    target_width = 800

    if st.button("🔄 重置分析", use_container_width=True):
        st.session_state.is_analyzed = False
        st.session_state.swing_events = []
        st.rerun()

# ==============================================================================
# 2. 幾何演算法與核心診斷引擎
# ==============================================================================


def calculate_attack_angle_and_length(
    bat_trajectory: list[tuple[int, int]],
    m_per_px: float,
    camera_tilt: float = 0.0,
) -> tuple[float, float, float]:
    """計算攻擊仰角 (Attack Angle)、揮棒軌跡長度 (Swing Length) 與軌跡平順度 (R²)"""
    if len(bat_trajectory) < 2:
        return 0.0, 0.0, 0.0

    pts = np.array(bat_trajectory, dtype=np.float64)

    # 1. 計算累積揮棒軌跡長度
    diffs = np.diff(pts, axis=0)
    dist_px = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
    swing_length_m = dist_px * m_per_px

    # 2. 擊球前攻擊仰角擬合
    x = pts[:, 0]
    y_phys = -pts[:, 1]  # Y軸轉為物理向上為正

    x_dir = 1.0 if x[-1] >= x[0] else -1.0
    x_rel = (x - x[0]) * x_dir

    slope, intercept = np.polyfit(x_rel, y_phys, 1)

    y_pred = slope * x_rel + intercept
    ss_res = np.sum((y_phys - y_pred) ** 2)
    ss_tot = np.sum((y_phys - np.mean(y_phys)) ** 2)
    r_squared = (
        1.0 - (ss_res / ss_tot)
        if ss_tot > 0
        else (1.0 if ss_res == 0 else 0.0)
    )

    attack_angle = math.degrees(math.atan(slope)) - camera_tilt

    return round(attack_angle, 1), round(swing_length_m, 2), round(r_squared, 2)


def generate_advanced_diagnostics(
    bat_speed: float,
    swing_length: float,
    attack_angle: float,
    exit_velocity: float,
    hip_rot_speed: float,
    r_squared: float,
) -> dict:
    """整合 4 大指標與下半身轉動動力鏈之自動化報告引擎"""
    score = 100
    feedbacks = []
    drills = []

    # 1. 揮棒速度
    if bat_speed >= 28.0:
        bat_speed_status = "速度優異"
        feedbacks.append(
            f"揮棒速度 (Bat Speed) 達 {bat_speed:.1f} km/h，球棒通過本壘板前緣移動速率優異，展現良好擊球爆發力。"
        )
    else:
        bat_speed_status = "速度偏低"
        feedbacks.append(
            f"揮棒速度 (Bat Speed) 為 {bat_speed:.1f} km/h，揮棒爆發力尚未完全釋放，影響最終擊球遠度。"
        )
        score -= 10

    # 2. 揮棒軌跡長度
    if 0.65 <= swing_length <= 0.95:
        length_status = "軌跡簡潔"
        feedbacks.append(
            f"揮棒軌跡長度 (Swing Length) 為 {swing_length:.2f} m，從啟動至擊球點距離適中，能有效應對高振幅速球。"
        )
    elif swing_length > 0.95:
        length_status = "軌跡過長 (繞大圈)"
        feedbacks.append(
            f"揮棒軌跡長度 (Swing Length) 達 {swing_length:.2f} m (建議 < 0.95m)。【動作修正】揮棒路徑過長（繞大圈），會大幅降低反應時間與容錯率，容易揮空。"
        )
        score -= 15
        drills.append("【改善軌跡】貼牆揮棒路徑收束訓練 (Wall Drill)")
    else:
        length_status = "延伸不足"
        feedbacks.append(
            f"揮棒軌跡長度 (Swing Length) 僅 {swing_length:.2f} m，推棒成份較多，影響擊球後段延伸與力量貫穿。"
        )

    # 3. 攻擊仰角
    if 6.0 <= attack_angle <= 18.0:
        attack_status = "完美切入 (微向上揚)"
        feedbacks.append(
            f"攻擊仰角 (Attack Angle) 為 {attack_angle:.1f}°，精確迎向擊球區軌跡，最易創造強勁平飛球。"
        )
    elif attack_angle < 6.0:
        attack_status = "角度過陡 (砍擊)"
        feedbacks.append(
            f"攻擊仰角 (Attack Angle) 為 {attack_angle:.1f}°。【動作修正】揮棒角度過陡（由上往下砍），會嚴重縮小擊球容錯區間，容易產生揮空或無力滾地球。"
        )
        score -= 15
        drills.append("【改善軌跡】高低位置擊球座高角掃擊練習 (Elevated Tee Work)")
    else:
        attack_status = "過度仰角 (倒棒)"
        feedbacks.append(
            f"攻擊仰角 (Attack Angle) 達 {attack_angle:.1f}°，角度過大易導致倒棒，容易形成無效高飛球。"
        )
        score -= 15
        drills.append("【改善軌跡】水平平飛擊球修正 (Level Swing Progression)")

    # 4. 擊球初速
    if exit_velocity >= 25.0:
        exit_status = "力量扎實"
        feedbacks.append(
            f"預估擊球初速 (Exit Velocity) 達 {exit_velocity:.1f} km/h，揮棒動能已扎實轉化為實際破壞力。"
        )
    else:
        exit_status = "轉化待提升"
        feedbacks.append(
            f"預估擊球初速 (Exit Velocity) 為 {exit_velocity:.1f} km/h，揮棒力量未能完全轉化，建議優化擊球甜蜜點對齊。"
        )
        score -= 10

    # 5. 髖關節轉速
    if hip_rot_speed >= 280.0:
        hip_status = "轉動爆發力強"
        feedbacks.append(
            f"髖關節峰值轉速達 {hip_rot_speed:.0f} deg/s，下半身後髖關節與骨盆旋轉優異，能量鏈傳遞完整。"
        )
    else:
        hip_status = "下半身導引不足"
        feedbacks.append(
            f"髖關節峰值轉速僅 {hip_rot_speed:.0f} deg/s。【強化轉動】下半身骨盆旋轉速度不足，無法確保力量由下而上完整傳遞到棒頭，導致力量斷層。"
        )
        score -= 15
        drills.append("【強化轉動】後髖關節爆發力旋轉彈力帶訓練 (Hip Hinge Band Rotation)")

    # 6. 等級評定
    score = max(0, score)
    if score >= 90:
        grade = "S (優異 A+)"
    elif score >= 75:
        grade = "A (良好)"
    elif score >= 60:
        grade = "B (尚可/需修正)"
    else:
        grade = "C (基礎動作需調整)"

    summary_df = pd.DataFrame({
        "核心指標": [
            "揮棒速度 (Bat Speed)",
            "揮棒軌跡長度 (Swing Length)",
            "攻擊仰角 (Attack Angle)",
            "預估擊球初速 (Exit Velocity)",
            "髖關節轉速 (Hip Rotation)",
            "軌跡平順度 (R²)",
        ],
        "實測數值": [
            f"{bat_speed:.1f} km/h",
            f"{swing_length:.2f} m",
            f"{attack_angle:.1f}°",
            f"{exit_velocity:.1f} km/h",
            f"{hip_rot_speed:.0f} deg/s",
            f"{r_squared:.2f}",
        ],
        "標竿參考值": [
            "> 28.0 km/h",
            "0.65 - 0.95 m",
            "6.0° - 18.0°",
            "> 25.0 km/h",
            "> 280 deg/s",
            "> 0.88",
        ],
        "診斷結果": [
            bat_speed_status,
            length_status,
            attack_status,
            exit_status,
            hip_status,
            "穩定" if r_squared >= 0.88 else "抖動",
        ],
    })

    return {
        "score": score,
        "grade": grade,
        "summary_df": summary_df,
        "feedbacks": feedbacks,
        "drills": list(set(drills)),
    }


def generate_pdf_report(swing_title: str, event_data: dict) -> bytes:
    """使用 ReportLab 產出包含防亂碼字型與關鍵揮擊截圖之 PDF 診斷報告"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )

    report = event_data["report"]
    summary_df = report["summary_df"]

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontName=FONT_NAME,
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1E3A8A"),
        alignment=1,
    )
    subtitle_style = ParagraphStyle(
        "SubTitleStyle",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4B5563"),
        alignment=1,
    )
    section_style = ParagraphStyle(
        "SectionStyle",
        parent=styles["Heading2"],
        fontName=FONT_NAME,
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=10,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1F2937"),
    )

    elements = []

    # 1. 頁眉標題區
    elements.append(Paragraph("棒球高階揮擊診斷與動力鏈分析報告", title_style))
    elements.append(Paragraph(f"檢測項目：{swing_title} | 報告產出系統：Baseball Kinetic Chain Diagnostic System", subtitle_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=10))

    # 2. 綜合評分與等級區
    score_text = f"<b>綜合動作評分：</b> {report['score']} 分 &nbsp;&nbsp;&nbsp;&nbsp; <b>等級評定：</b> {report['grade']}"
    elements.append(Paragraph(score_text, ParagraphStyle("Score", parent=body_style, fontSize=11, textColor=colors.HexColor("#065F46"))))
    elements.append(Spacer(1, 10))

    # 3. 關鍵揮擊瞬間截圖 (Snapshot)
    if "snapshot_bytes" in event_data and event_data["snapshot_bytes"]:
        elements.append(Paragraph("關鍵峰值揮擊姿態截圖", section_style))
        img_buffer = io.BytesIO(event_data["snapshot_bytes"])
        rl_img = RLImage(img_buffer, width=380, height=213)
        rl_img.hAlign = 'CENTER'
        elements.append(rl_img)
        elements.append(Spacer(1, 10))

    # 4. 4 大核心數據與動力鏈表格
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
        ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
        ('TOPPADDING', (0, 0), (-1, 0), 5),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F3F4F6')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 10))

    # 5. 動作修正與導引診斷
    elements.append(Paragraph("動作修正與導引診斷細節", section_style))
    for fb in report["feedbacks"]:
        clean_fb = fb.replace("**", "")
        elements.append(Paragraph(f"• {clean_fb}", body_style))
        elements.append(Spacer(1, 2))

    elements.append(Spacer(1, 6))

    # 6. 針對性動作修正處方
    if report["drills"]:
        elements.append(Paragraph("針對性動作修正處方 (Recommended Drills)", section_style))
        for drill in report["drills"]:
            elements.append(Paragraph(f"建議訓練項目： {drill}", ParagraphStyle("Drill", parent=body_style, textColor=colors.HexColor("#92400E"))))
            elements.append(Spacer(1, 2))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def process_frame_landmarks(
    frame: np.ndarray,
    pose_landmarks,
    width: int,
    height: int,
    bat_length: int,
):
    """擷取關節座標、計算髖關節夾角與手腕/棒頭位置"""
    if not pose_landmarks:
        return None, None, None

    landmarks = pose_landmarks[0]

    def get_coords(idx):
        if idx < len(landmarks):
            lm = landmarks[idx]
            return int(lm.x * width), int(lm.y * height)
        return None

    l_shoulder = get_coords(11)
    r_shoulder = get_coords(12)
    l_hip = get_coords(23)
    r_hip = get_coords(24)
    l_wrist = get_coords(15)
    r_wrist = get_coords(16)

    # 繪製肩線與骨盆線
    if l_shoulder and r_shoulder:
        cv2.line(frame, l_shoulder, r_shoulder, (255, 255, 255), 2)
    if l_hip and r_hip:
        cv2.line(frame, l_hip, r_hip, (255, 0, 255), 3)

    wrist_center = None
    if l_wrist and r_wrist:
        wrist_center = (
            int((l_wrist[0] + r_wrist[0]) / 2),
            int((l_wrist[1] + r_wrist[1]) / 2),
        )
        cv2.circle(frame, wrist_center, 6, (0, 0, 255), -1)

    bat_head = None
    if l_shoulder and r_shoulder and wrist_center:
        shoulder_mid = (
            (l_shoulder[0] + r_shoulder[0]) / 2,
            (l_shoulder[1] + r_shoulder[1]) / 2,
        )
        dx = wrist_center[0] - shoulder_mid[0]
        dy = wrist_center[1] - shoulder_mid[1]
        norm = math.sqrt(dx**2 + dy**2)
        if norm > 0:
            bat_head = (
                int(wrist_center[0] + (dx / norm) * bat_length),
                int(wrist_center[1] + (dy / norm) * bat_length),
            )
            cv2.line(frame, wrist_center, bat_head, (0, 165, 255), 4)
            cv2.circle(frame, bat_head, 5, (0, 255, 255), -1)

    # 計算髖關節連線方向角
    hip_angle = 0.0
    if l_hip and r_hip:
        dx_h = r_hip[0] - l_hip[0]
        dy_h = r_hip[1] - l_hip[1]
        hip_angle = math.degrees(math.atan2(dy_h, dx_h))

    return wrist_center, bat_head, hip_angle


# ==============================================================================
# 3. 主程序：影片分析與姿態偵測
# ==============================================================================
uploaded_file = st.file_uploader(
    "📁 請上傳揮棒側拍影片 (MP4 / MOV / AVI)", type=["mp4", "avi", "mov", "m4v"]
)

if uploaded_file is not None:
    if "is_analyzed" not in st.session_state:
        st.session_state.is_analyzed = False
        st.session_state.swing_events = []

if uploaded_file is not None and not st.session_state.get("is_analyzed", False):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.getvalue())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = target_width / float(orig_width) if orig_width > target_width else 1.0
    proc_width = int(orig_width * scale)
    proc_height = int(orig_height * scale)

    st.markdown("### 📹 進行 4 大指標與動力鏈姿態分析中...")
    st_frame = st.empty()
    progress_bar = st.progress(0)

    history_wrist = []
    history_hip_angles = []
    current_swing_trajectory = []
    swing_state = 0
    swing_frames_data = []
    swing_raw_frames = []
    max_speed_in_swing = 0.0
    peak_frame_snapshot = None
    cooldown_counter = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

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
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb_frame
            )
            timestamp_ms = int((frame_idx / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            current_wrist = None
            current_bat_head = None
            current_hip_angle = 0.0

            if result.pose_landmarks:
                wrist, bat_head, hip_angle = process_frame_landmarks(
                    annotated_frame,
                    result.pose_landmarks,
                    proc_width,
                    proc_height,
                    bat_length_px,
                )
                if wrist:
                    current_wrist = wrist
                    current_bat_head = bat_head
                    current_hip_angle = hip_angle
                    history_wrist.append((frame_idx, wrist[0], wrist[1]))
                    history_hip_angles.append((frame_idx, hip_angle))

            # 計算即時揮棒速度 (滑動平均)
            current_speed = 0.0
            if len(history_wrist) >= 2:
                p1, p2 = history_wrist[-2], history_wrist[-1]
                dt = (
                    (p2[0] - p1[0]) / fps
                    if (p2[0] - p1[0]) > 0
                    else (1.0 / fps)
                )
                dx = (p2[1] - p1[1]) * meters_per_pixel
                dy = (p2[2] - p1[2]) * meters_per_pixel
                inst_speed = (
                    math.sqrt(dx**2 + dy**2) / dt
                ) * 3.6 * bat_speed_factor

                if len(swing_frames_data) > 0:
                    prev_speeds = [
                        item["speed"] for item in swing_frames_data[-2:]
                    ]
                    current_speed = np.mean(prev_speeds + [inst_speed])
                else:
                    current_speed = inst_speed

            # 計算髖關節角速度 (deg/s)
            current_hip_speed = 0.0
            if len(history_hip_angles) >= 2:
                h1, h2 = history_hip_angles[-2], history_hip_angles[-1]
                dt_h = (
                    (h2[0] - h1[0]) / fps
                    if (h2[0] - h1[0]) > 0
                    else (1.0 / fps)
                )
                d_ang = abs(h2[1] - h1[1])
                current_hip_speed = d_ang / dt_h

            # 揮棒偵測邏輯
            start_trigger_speed = max(min_peak_speed * 0.40, 6.0)

            if cooldown_counter > 0:
                cooldown_counter -= 1

            if cooldown_counter == 0:
                if swing_state == 0:
                    if current_speed >= start_trigger_speed:
                        swing_state = 1
                        swing_frames_data = []
                        swing_raw_frames = []
                        current_swing_trajectory = []
                        max_speed_in_swing = current_speed
                        peak_frame_snapshot = annotated_frame.copy()

                elif swing_state == 1:
                    if current_bat_head:
                        current_swing_trajectory.append(current_bat_head)

                    swing_frames_data.append({
                        "frame": frame_idx,
                        "wrist": current_wrist,
                        "bat_head": current_bat_head,
                        "speed": current_speed,
                        "hip_speed": current_hip_speed,
                    })

                    if current_speed > max_speed_in_swing:
                        max_speed_in_swing = current_speed
                        peak_frame_snapshot = annotated_frame.copy()

                    if (
                        max_speed_in_swing >= (min_peak_speed * 0.7)
                        and current_speed < max_speed_in_swing * 0.6
                    ):
                        swing_state = 2

                elif swing_state == 2:
                    if current_bat_head:
                        current_swing_trajectory.append(current_bat_head)

                    swing_frames_data.append({
                        "frame": frame_idx,
                        "wrist": current_wrist,
                        "bat_head": current_bat_head,
                        "speed": current_speed,
                        "hip_speed": current_hip_speed,
                    })

                    if (
                        current_speed <= start_trigger_speed
                        or len(swing_frames_data) > 45
                    ):
                        swing_state = 3

                if len(current_swing_trajectory) > 1:
                    pts = np.array(
                        current_swing_trajectory, np.int32
                    ).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame, [pts], False, (0, 255, 255), 3
                    )

                if swing_state in [1, 2]:
                    swing_raw_frames.append(annotated_frame.copy())

                # 結算揮棒並產出分析數據
                if swing_state == 3:
                    wrists_in_swing = [
                        item["wrist"]
                        for item in swing_frames_data
                        if item["wrist"] is not None
                    ]
                    total_wrist_travel = 0.0
                    if len(wrists_in_swing) >= 2:
                        w_pts = np.array(wrists_in_swing)
                        total_wrist_travel = np.sum(
                            np.sqrt(np.sum(np.diff(w_pts, axis=0) ** 2, axis=1))
                        )

                    if (
                        max_speed_in_swing >= (min_peak_speed * 0.8)
                        and len(swing_frames_data) >= 5
                        and total_wrist_travel >= min_total_travel
                    ):
                        fit_pts = [
                            item["bat_head"]
                            for item in swing_frames_data
                            if item["bat_head"] is not None
                        ]

                        (
                            attack_angle,
                            swing_length,
                            r_squared,
                        ) = calculate_attack_angle_and_length(
                            fit_pts, meters_per_pixel, camera_tilt_deg
                        )

                        bat_speed = max_speed_in_swing
                        exit_velocity = bat_speed * 1.15
                        peak_hip_speed = max(
                            [item["hip_speed"] for item in swing_frames_data]
                            + [0.0]
                        )

                        report = generate_advanced_diagnostics(
                            bat_speed=bat_speed,
                            swing_length=swing_length,
                            attack_angle=attack_angle,
                            exit_velocity=exit_velocity,
                            hip_rot_speed=peak_hip_speed,
                            r_squared=r_squared,
                        )

                        swing_num = len(detected_events) + 1
                        clip_filename = os.path.join(
                            clip_dir, f"swing_{swing_num}.webm"
                        )

                        # 儲存短影片
                        fourcc = cv2.VideoWriter_fourcc(*"VP80")
                        out = cv2.VideoWriter(
                            clip_filename,
                            fourcc,
                            fps,
                            (proc_width, proc_height),
                        )
                        for f in swing_raw_frames:
                            out.write(f)
                        out.release()

                        with open(clip_filename, "rb") as vf:
                            video_bytes = vf.read()

                        # 編碼峰值截圖 (JPG)
                        snapshot_bytes = None
                        if peak_frame_snapshot is not None:
                            _, buf = cv2.imencode(".jpg", peak_frame_snapshot)
                            snapshot_bytes = buf.tobytes()

                        detected_events.append({
                            "次數": f"第 {swing_num} 次揮棒",
                            "bat_speed": bat_speed,
                            "swing_length": swing_length,
                            "attack_angle": attack_angle,
                            "exit_velocity": exit_velocity,
                            "hip_speed": peak_hip_speed,
                            "report": report,
                            "video_bytes": video_bytes,
                            "snapshot_bytes": snapshot_bytes,
                        })

                        cooldown_counter = int(fps * 1.8)

                    swing_state = 0
                    current_swing_trajectory = []

            if frame_idx % 3 == 0:
                st_frame.image(
                    cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )

    cap.release()
    st_frame.empty()
    progress_bar.empty()

    st.session_state.swing_events = detected_events
    st.session_state.is_analyzed = True
    st.rerun()

# ==============================================================================
# 4. 分析報告展示
# ==============================================================================
if st.session_state.get("is_analyzed", False):
    events = st.session_state.swing_events

    if not events:
        st.warning("⚠️ 未能偵測到有效揮棒，請調整側邊欄門檻（如調低「最低初速門檻」或「手腕位移」）。")
    else:
        st.success(f"✅ 完成分析！共偵測到 {len(events)} 次揮棒。")

        selected_swing_name = st.selectbox(
            "🎯 選擇揮棒次數檢視詳細診斷：", [e["次數"] for e in events]
        )
        event = next(e for e in events if e["次數"] == selected_swing_name)
        report = event["report"]

        st.markdown("---")
        # 展示 4 大核心指標
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "🚀 揮棒速度",
            f"{event['bat_speed']:.1f} km/h",
            help="球棒通過本壘板前緣的移動速率，直接決定擊球爆發力與遠度。",
        )
        c2.metric(
            "📏 揮棒軌跡長度",
            f"{event['swing_length']:.2f} m",
            help="衡量球棒從啟動到擊球點移動距離，軌跡越短越能應對高振幅速球。",
        )
        c3.metric(
            "📐 攻擊仰角",
            f"{event['attack_angle']:.1f}°",
            help="迎向擊球區垂直傾斜角度，決定球被打出是平飛球還是滾地球。",
        )
        c4.metric(
            "⚡ 預估擊球初速",
            f"{event['exit_velocity']:.1f} km/h",
            help="球離開球棒瞬間速度，檢驗揮棒力量是否扎實轉化為實際破壞力。",
        )

        tab1, tab2 = st.tabs(
            ["📄 揮擊診斷與動力鏈修正報告", "🎬 慢動作姿態回放"]
        )

        with tab1:
            st.markdown(
                f"### 🏆 揮棒綜合評分：`{report['score']} 分` (等級: {report['grade']})"
            )

            if event.get("snapshot_bytes"):
                st.image(
                    event["snapshot_bytes"],
                    caption=f"{selected_swing_name} 關鍵擊球瞬間姿態骨架截圖",
                    width=500,
                )

            st.subheader("📊 4 大核心數據與動力鏈標竿對比")
            st.dataframe(report["summary_df"], use_container_width=True)

            st.subheader("💡 動作修正與導引診斷")
            for fb in report["feedbacks"]:
                st.write(f"- {fb}")

            if report["drills"]:
                st.subheader("🎯 針對性動作修正處方 (Recommended Drills)")
                for drill in report["drills"]:
                    st.info(f"👉 **建議訓練：** {drill}")

            st.markdown("---")
            pdf_bytes = generate_pdf_report(selected_swing_name, event)
            st.download_button(
                label="📥 下載完整 PDF 診斷修正報告 (含動作截圖)",
                data=pdf_bytes,
                file_name=f"baseball_swing_report_{selected_swing_name}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with tab2:
            st.video(event["video_bytes"], format="video/webm")