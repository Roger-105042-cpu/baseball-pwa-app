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

# ==============================================================================
# 0. 頁面設定與 MediaPipe 核心模型自動下載
# ==============================================================================
st.set_page_config(
    page_title="棒球揮棒姿態與 AI 診斷報告系統",
    page_icon="⚾",
    layout="wide",
)

MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"


@st.cache_resource
def ensure_model_file():
    """自動下載 MediaPipe Pose Landmarker 核心模型檔"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ 首次執行，正在下載 MediaPipe 姿態識別模型檔..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


ensure_model_file()

st.title("⚾ 崇明國中-棒球揮棒姿態、畫面絕對座標仰角與 AI 診斷報告系統")
st.caption(
    "整合 Kinovea 式畫面絕對水平基準、多點軌跡擬合仰角、相機傾斜補償與標竿自動診斷報告"
)

# ==============================================================================
# 1. 側邊欄：參數控制（相機傾斜、實體尺寸標定與偵測門檻）
# ==============================================================================
with st.sidebar:
    st.header("⚙️ 系統參數與 Kinovea 校正")

    st.subheader("1. 畫面水平與相機傾斜校正")
    camera_tilt_deg = st.slider(
        "📐 相機傾斜補償角度 (度)",
        min_value=-20.0,
        max_value=20.0,
        value=0.0,
        step=0.5,
        help="以畫面絕對 X 軸為 0° 基準。若手持拍攝或相機未架平，可調整此值使綠線完全平行於地面。",
    )

    st.subheader("2. 距離與比例標定")
    meters_per_pixel = st.slider(
        "📏 像素轉換公尺比例 (Meters/PX)",
        min_value=0.0010,
        max_value=0.0080,
        value=0.0032,
        step=0.0001,
        help="遠景拍攝請調大（如 0.0040~0.0050）；近景拍攝請調小。",
    )

    bat_speed_factor = st.slider(
        "🚀 手腕至棒頭速度放大係數",
        min_value=1.0,
        max_value=2.0,
        value=1.35,
        step=0.05,
    )

    st.subheader("3. 揮棒觸發門檻")
    min_peak_speed = st.slider(
        "⚡ 最低有效初速 (km/h)",
        min_value=10.0,
        max_value=50.0,
        value=15.0,
        step=1.0,
        help="低於此初速不結算為揮棒。若經常漏抓揮棒請調低。",
    )

    min_total_travel = st.slider(
        "📏 最低手腕總位移量 (像素)",
        min_value=10,
        max_value=100,
        value=20,
        step=5,
    )

    st.markdown("---")
    bat_length_px = 110  # 棒頭延伸長度 (像素)
    target_width = 800  # 圖像處理最大寬度

    if st.button("🔄 重置並重新分析", use_container_width=True):
        st.session_state.is_analyzed = False
        st.session_state.swing_events = []
        st.rerun()

# ==============================================================================
# 2. 核心幾何演算法與自動報告產出模組
# ==============================================================================


def calculate_launch_angle_multipoint(
    trajectory_points: list[tuple[int, int]], camera_tilt: float = 0.0
) -> tuple[float, float]:
    """使用畫面絕對座標系與多點最小二乘法計算擊球仰角與平順度 (R²)

    :return: (擊球仰角_度, 平順度_R2)
    """
    if len(trajectory_points) < 2:
        return 0.0, 0.0

    pts = np.array(trajectory_points, dtype=np.float64)
    x = pts[:, 0]
    y_phys = -pts[:, 1]  # 影像 Y 軸向下為正，轉為物理 Y 軸向上為正

    # 歸一化 X 軸移動方向，消除左打/右打方向影響
    x_dir = 1.0 if x[-1] >= x[0] else -1.0
    x_rel = (x - x[0]) * x_dir

    # 一階線性擬合 y = m * x + c
    slope, intercept = np.polyfit(x_rel, y_phys, 1)

    # 計算擬合平順度 (R² Score)
    y_pred = slope * x_rel + intercept
    ss_res = np.sum((y_phys - y_pred) ** 2)
    ss_tot = np.sum((y_phys - np.mean(y_phys)) ** 2)
    r_squared = (
        1.0 - (ss_res / ss_tot)
        if ss_tot > 0
        else (1.0 if ss_res == 0 else 0.0)
    )
    r_squared = max(0.0, min(1.0, r_squared))

    # 弧度轉角度並扣除相機傾斜角
    raw_angle_deg = math.degrees(math.atan(slope))
    final_launch_angle = raw_angle_deg - camera_tilt

    return round(final_launch_angle, 1), round(r_squared, 2)


def draw_kinovea_protractor(
    frame: np.ndarray, wrist_pos: tuple[int, int], tilt_angle: float
):
    """基於畫面絕對水平線繪製 Kinovea 量角規與 15°-35° 甜蜜區扇形"""
    if not wrist_pos:
        return

    cx, cy = wrist_pos
    rad = math.radians(tilt_angle)

    # 1. 絕對地平參考線 (綠線)
    dx = int(600 * math.cos(rad))
    dy = int(600 * math.sin(rad))
    cv2.line(
        frame,
        (cx - dx, cy - dy),
        (cx + dx, cy + dy),
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # 2. 鉛直軸線 (粉紫色線)
    p_up = (cx + dy, cy - dx)
    p_down = (cx - dy, cy + dx)
    cv2.line(frame, p_up, p_down, (255, 0, 255), 1, cv2.LINE_AA)

    # 3. 甜蜜仰角扇形區 (15° 到 35°)
    overlay = frame.copy()
    radius = 120
    start_ang = -(tilt_angle + 35)
    end_ang = -(tilt_angle + 15)

    cv2.ellipse(
        overlay,
        (cx, cy),
        (radius, radius),
        0,
        start_ang,
        end_ang,
        (0, 215, 255),
        -1,
    )
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # 文字標籤
    cv2.putText(
        frame,
        f"Frame Horizon ({tilt_angle:+.1f} deg)",
        (cx + 80, cy - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Sweet Spot (15-35 deg)",
        (cx + 60, cy - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )


def process_frame(
    frame: np.ndarray,
    pose_landmarks,
    width: int,
    height: int,
    bat_length: int,
    tilt_angle: float,
):
    """擷取關節座標、繪製骨架與延伸棒頭線"""
    if not pose_landmarks:
        return None, None

    landmarks = pose_landmarks[0]

    def get_coords(idx):
        if idx < len(landmarks):
            lm = landmarks[idx]
            return int(lm.x * width), int(lm.y * height)
        return None

    l_shoulder = get_coords(11)
    r_shoulder = get_coords(12)
    l_wrist = get_coords(15)
    r_wrist = get_coords(16)

    if l_shoulder and r_shoulder:
        cv2.line(frame, l_shoulder, r_shoulder, (255, 255, 255), 2)
    if l_shoulder and l_wrist:
        cv2.line(frame, l_shoulder, l_wrist, (0, 255, 0), 2)
    if r_shoulder and r_wrist:
        cv2.line(frame, r_shoulder, r_wrist, (0, 255, 0), 2)

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

    if wrist_center:
        draw_kinovea_protractor(frame, wrist_center, tilt_angle)

    return wrist_center, bat_head


def generate_swing_report(
    peak_speed: float,
    launch_angle: float,
    swing_duration: float,
    r_squared: float,
) -> dict:
    """自動比對標竿數據並產出詳細評分與訓練診斷報告"""
    score = 100
    feedbacks = []
    recommended_drills = []

    # 1. 擊球仰角評估 (15° - 35°)
    if 15.0 <= launch_angle <= 35.0:
        angle_status = "✅ 優秀 (甜蜜角)"
        feedbacks.append(
            f"仰角為 **{launch_angle}°**，處於的最佳平飛長打區間 (15°~35°)。"
        )
    elif launch_angle < 15.0:
        angle_status = "⚠️ 偏低 (易滾地)"
        feedbacks.append(
            f"仰角僅 **{launch_angle}°**，容易打成滾地球。請調整擊球點前後位置，保持擊球時身體不過度前傾。"
        )
        score -= 15
        recommended_drills.append(
            "高低擊球座標竿訓練 (Tee Work for Elevated Contact)"
        )
    else:
        angle_status = "⚠️ 偏高 (易高飛)"
        feedbacks.append(
            f"仰角達 **{launch_angle}°**，容易打成無效高飛球。請注意揮棒過程中是否有過度倒肩或倒棒現象。"
        )
        score -= 15
        recommended_drills.append("水平掃擊與平飛打擊修正訓練 (Level Swing Drill)")

    # 2. 出棒時間評估 (理想 <= 0.22 秒)
    if swing_duration <= 0.22:
        speed_status = "✅ 揮棒簡潔"
        feedbacks.append(
            f"啟動至擊球耗時 **{swing_duration:.2f} 秒**，出棒路徑非常簡潔俐落。"
        )
    else:
        speed_status = "⚠️ 揮棒歷時偏長"
        feedbacks.append(
            f"耗時 **{swing_duration:.2f} 秒** (標竿 < 0.22 秒)，出棒可能存在拉棒或軌跡繞大圈的現象。"
        )
        score -= 15
        recommended_drills.append("貼牆揮棒路徑收束訓練 (Wall Drill)")

    # 3. 軌跡平順度評估 (R² >= 0.88)
    if r_squared >= 0.88:
        plane_status = "✅ 軌跡平順"
        feedbacks.append(
            f"軌跡平順度 R² 為 **{r_squared:.2f}**，棒頭揮擊平面非常穩定。"
        )
    else:
        plane_status = "⚠️ 軌跡波動較大"
        feedbacks.append(
            f"軌跡平順度 R² 僅 **{r_squared:.2f}**，揮棒過程中有抖動或脫軌現象，可能擊球瞬間手腕提早翻轉。"
        )
        score -= 10
        recommended_drills.append("單手控棒穩定度訓練 (One-Handed Swing Drill)")

    # 4. 擊球初速評估
    if peak_speed >= 25.0:
        power_status = "✅ 爆發力佳"
    else:
        power_status = "⚠️ 速度待提升"

    # 5. 等級評定
    score = max(0, score)
    if score >= 90:
        grade = "S (優異 A+)"
    elif score >= 75:
        grade = "A (良好)"
    elif score >= 60:
        grade = "B (尚可/需微調)"
    else:
        grade = "C (建議調整基礎動作)"

    summary_df = pd.DataFrame({
        "檢測指標": [
            "擊球初速 (Peak Speed)",
            "擊球仰角 (Launch Angle)",
            "出棒耗時 (Swing Duration)",
            "軌跡平順度 (R² Score)",
        ],
        "實測數據": [
            f"{peak_speed:.1f} km/h",
            f"{launch_angle:.1f}°",
            f"{swing_duration:.2f} 秒",
            f"{r_squared:.2f}",
        ],
        "最佳標竿": ["> 25.0 km/h", "15.0° - 35.0°", "≤ 0.22 秒", "≥ 0.88"],
        "診斷結果": [power_status, angle_status, speed_status, plane_status],
    })

    return {
        "score": score,
        "grade": grade,
        "summary_df": summary_df,
        "feedbacks": feedbacks,
        "drills": recommended_drills,
    }


def save_swing_clip(
    frames: list[np.ndarray],
    fps: float,
    width: int,
    height: int,
    output_path: str,
) -> str:
    """將揮棒影像串流存為 WebM 影片檔"""
    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for f in frames:
        out.write(f)
    out.release()
    return output_path


# ==============================================================================
# 3. Session 狀態初始化與影片上傳
# ==============================================================================
if "swing_events" not in st.session_state:
    st.session_state.swing_events = []
if "is_analyzed" not in st.session_state:
    st.session_state.is_analyzed = False
if "current_file_key" not in st.session_state:
    st.session_state.current_file_key = None

uploaded_file = st.file_uploader(
    "📁 選擇或上傳揮棒影片 (MP4 / MOV / AVI)", type=["mp4", "avi", "mov", "m4v"]
)

if uploaded_file is not None:
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.current_file_key != file_key:
        st.session_state.current_file_key = file_key
        st.session_state.swing_events = []
        st.session_state.is_analyzed = False

# ==============================================================================
# 4. 影片分析與揮棒事件偵測核心流程
# ==============================================================================
if uploaded_file is not None and not st.session_state.is_analyzed:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.getvalue())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if orig_width > target_width:
        scale = target_width / float(orig_width)
        proc_width = target_width
        proc_height = int(orig_height * scale)
    else:
        scale = 1.0
        proc_width = orig_width
        proc_height = orig_height

    st.markdown("### 📹 全程 AI 姿態追蹤與仰角校正中...")
    st_frame = st.empty()
    progress_bar = st.progress(0)

    history_wrist = []
    current_swing_trajectory = []
    swing_state = 0  # 0: 準備, 1: 揮棒中, 2: 減速中, 3: 結算
    swing_frames_data = []
    swing_raw_frames = []
    max_speed_in_swing = 0.0
    peak_frame_in_swing = 0
    cooldown_counter = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    clip_dir = tempfile.mkdtemp()
    detected_events = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            if frame_idx % 10 == 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

            frame_resized = (
                cv2.resize(
                    frame,
                    (proc_width, proc_height),
                    interpolation=cv2.INTER_AREA,
                )
                if scale != 1.0
                else frame.copy()
            )
            annotated_frame = frame_resized.copy()
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb_frame
            )
            timestamp_ms = int((frame_idx / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            current_wrist = None
            current_bat_head = None

            if result.pose_landmarks:
                wrist, bat_head = process_frame(
                    annotated_frame,
                    result.pose_landmarks,
                    proc_width,
                    proc_height,
                    bat_length_px,
                    camera_tilt_deg,
                )
                if wrist:
                    current_wrist = wrist
                    current_bat_head = bat_head
                    history_wrist.append((frame_idx, wrist[0], wrist[1]))

            # 計算即時加速度
            current_speed = 0.0
            if len(history_wrist) >= 2:
                p1 = history_wrist[-2]
                p2 = history_wrist[-1]
                frame_diff = p2[0] - p1[0]
                dt = (frame_diff / fps) if frame_diff > 0 else (1.0 / fps)

                dx = (p2[1] - p1[1]) * meters_per_pixel
                dy = (p2[2] - p1[2]) * meters_per_pixel
                dist_m = math.sqrt(dx**2 + dy**2)
                current_speed = (dist_m / dt) * 3.6 * bat_speed_factor

            if cooldown_counter > 0:
                cooldown_counter -= 1

            start_trigger_speed = min_peak_speed * 0.35

            if cooldown_counter == 0:
                if swing_state == 0:
                    if current_speed >= start_trigger_speed:
                        swing_state = 1
                        swing_frames_data = []
                        swing_raw_frames = []
                        current_swing_trajectory = []
                        max_speed_in_swing = current_speed
                        peak_frame_in_swing = frame_idx

                elif swing_state == 1:
                    if current_bat_head:
                        current_swing_trajectory.append(current_bat_head)

                    swing_frames_data.append({
                        "frame": frame_idx,
                        "wrist": current_wrist,
                        "bat_head": current_bat_head,
                        "speed": current_speed,
                    })

                    if current_speed > max_speed_in_swing:
                        max_speed_in_swing = current_speed
                        peak_frame_in_swing = frame_idx

                    if (
                        max_speed_in_swing >= min_peak_speed
                        and current_speed < max_speed_in_swing * 0.6
                    ):
                        swing_state = 2
                    elif (
                        current_speed < start_trigger_speed
                        and max_speed_in_swing < min_peak_speed
                        and len(swing_frames_data) > 12
                    ):
                        swing_state = 0
                        current_swing_trajectory = []

                elif swing_state == 2:
                    if current_bat_head:
                        current_swing_trajectory.append(current_bat_head)

                    swing_frames_data.append({
                        "frame": frame_idx,
                        "wrist": current_wrist,
                        "bat_head": current_bat_head,
                        "speed": current_speed,
                    })

                    if (
                        current_speed <= start_trigger_speed * 0.8
                        or len(swing_frames_data) > 35
                    ):
                        x_coords = [
                            item["wrist"][0]
                            for item in swing_frames_data
                            if item["wrist"]
                        ]
                        y_coords = [
                            item["wrist"][1]
                            for item in swing_frames_data
                            if item["wrist"]
                        ]

                        total_disp = 0
                        if x_coords and y_coords:
                            total_disp = math.sqrt(
                                (max(x_coords) - min(x_coords)) ** 2
                                + (max(y_coords) - min(y_coords)) ** 2
                            )

                        if total_disp >= min_total_travel:
                            swing_state = 3
                        else:
                            swing_state = 0
                            current_swing_trajectory = []
                            cooldown_counter = int(fps * 0.3)

                if len(current_swing_trajectory) > 1:
                    pts = np.array(
                        current_swing_trajectory, np.int32
                    ).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame, [pts], False, (0, 255, 255), 3
                    )

                if swing_state in [1, 2]:
                    swing_raw_frames.append(annotated_frame.copy())

                # 結算本次揮棒並計算報告
                if swing_state == 3:
                    peak_sub_idx = [
                        i
                        for i, item in enumerate(swing_frames_data)
                        if item["frame"] == peak_frame_in_swing
                    ]
                    launch_angle = 0.0
                    r_squared = 0.0

                    if peak_sub_idx:
                        p_idx = peak_sub_idx[0]
                        start_idx = max(0, p_idx - 1)
                        end_idx = min(len(swing_frames_data), p_idx + 4)

                        fit_pts = [
                            item["bat_head"]
                            for item in swing_frames_data[
                                start_idx:end_idx
                            ]
                            if item["bat_head"] is not None
                        ]

                        (
                            launch_angle,
                            r_squared,
                        ) = calculate_launch_angle_multipoint(
                            fit_pts, camera_tilt_deg
                        )

                    duration = len(swing_frames_data) / fps

                    # 自動產出 AI 診斷報告
                    report = generate_swing_report(
                        max_speed_in_swing, launch_angle, duration, r_squared
                    )

                    frame_logs = []
                    start_f = (
                        swing_frames_data[0]["frame"]
                        if swing_frames_data
                        else 0
                    )
                    for item in swing_frames_data:
                        rel_time = (item["frame"] - start_f) / fps
                        w_x = item["wrist"][0] if item["wrist"] else None
                        w_y = item["wrist"][1] if item["wrist"] else None
                        b_x = item["bat_head"][0] if item["bat_head"] else None
                        b_y = item["bat_head"][1] if item["bat_head"] else None
                        frame_logs.append({
                            "相對時間 (秒)": round(rel_time, 2),
                            "即時速度 (km/h)": round(item["speed"], 1),
                            "手腕 X": w_x,
                            "手腕 Y": w_y,
                            "棒頭 X": b_x,
                            "棒頭 Y": b_y,
                        })
                    detailed_df = pd.DataFrame(frame_logs)

                    swing_num = len(detected_events) + 1
                    clip_filename = os.path.join(
                        clip_dir, f"swing_{swing_num}.webm"
                    )
                    actual_clip_path = save_swing_clip(
                        swing_raw_frames,
                        fps,
                        proc_width,
                        proc_height,
                        clip_filename,
                    )

                    with open(actual_clip_path, "rb") as vf:
                        video_bytes = vf.read()

                    detected_events.append({
                        "次數": f"第 {swing_num} 次揮棒",
                        "初速": f"{max_speed_in_swing:.1f} km/h",
                        "仰角": f"{launch_angle:.1f}°",
                        "耗時": f"{duration:.2f} 秒",
                        "r_squared": f"{r_squared:.2f}",
                        "total_frames": len(swing_frames_data),
                        "video_bytes": video_bytes,
                        "detailed_df": detailed_df,
                        "report": report,
                    })

                    swing_state = 0
                    current_swing_trajectory = []
                    cooldown_counter = int(fps * 0.8)

            if frame_idx % 3 == 0:
                frame_display = cv2.cvtColor(
                    annotated_frame, cv2.COLOR_BGR2RGB
                )
                st_frame.image(
                    frame_display, channels="RGB", use_container_width=True
                )

    cap.release()
    try:
        os.remove(tfile.name)
    except Exception:
        pass

    st_frame.empty()
    progress_bar.empty()

    st.session_state.swing_events = detected_events
    st.session_state.is_analyzed = True
    st.rerun()

# ==============================================================================
# 5. 分析結果與 AI 診斷報告展示區
# ==============================================================================
if st.session_state.is_analyzed:
    events = st.session_state.swing_events

    if not events:
        st.warning(
            "⚠️ 未能偵測到揮棒。請嘗試調整左側邊欄：\n"
            "1. 將【最低有效初速】調低（如 12 km/h）。\n"
            "2. 將【像素轉換公尺比例】調大（如 0.0040~0.0050）。"
        )
    else:
        st.success(f"✅ 分析完成！共偵測到 {len(events)} 次有效揮棒。")

        option_list = [e["次數"] for e in events]
        selected_swing_name = st.selectbox(
            "🎯 請選擇揮棒次數以檢視詳細分析與報告：",
            options=option_list,
            index=len(option_list) - 1,
        )

        selected_event = next(
            (e for e in events if e["次數"] == selected_swing_name), events[0]
        )

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("⚡ 擊球初速", selected_event["初速"])
        col2.metric("📐 絕對座標擊球仰角", selected_event["仰角"])
        col3.metric("⏱️ 揮棒耗時", selected_event["耗時"])
        col4.metric("📈 軌跡平順度 (R²)", selected_event["r_squared"])

        tab1, tab2, tab3 = st.tabs([
            "🎬 Kinovea 慢動作回放",
            "📄 揮擊診斷建議報告",
            "📊 逐幀軌跡數據表",
        ])

        with tab1:
            st.markdown(
                f"#### 🎬 {selected_event['次數']} - 慢動作回放（綠線：畫面絕對水平基準；黃區：15°-35° 甜蜜角區域）"
            )
            st.video(selected_event["video_bytes"], format="video/webm")

        with tab2:
            report = selected_event["report"]
            st.markdown(
                f"### 🏆 揮棒綜合評分：`{report['score']} 分` (評級: {report['grade']})"
            )

            st.subheader("📊 指標檢測與標竿比對")
            st.dataframe(report["summary_df"], use_container_width=True)

            st.subheader("💡 專業動作診斷與建議")
            for fb in report["feedbacks"]:
                st.write(f"- {fb}")

            if report["drills"]:
                st.subheader("🎯 建議針對性訓練處方 (Recommended Drills)")
                for drill in report["drills"]:
                    st.info(f"👉 **推薦處方：** {drill}")
            else:
                st.success(
                    "🎉 動作非常標準！請保持當前的揮棒節奏與軌跡延伸。"
                )

        with tab3:
            st.markdown(f"#### 📋 {selected_event['次數']} 明細數據")
            st.dataframe(
                selected_event["detailed_df"],
                use_container_width=True,
                height=300,
            )
            csv_data = (
                selected_event["detailed_df"]
                .to_csv(index=False)
                .encode("utf-8-sig")
            )
            st.download_button(
                label=f"📥 下載 {selected_event['次數']} CSV 數據檔",
                data=csv_data,
                file_name=f"{selected_event['次數']}_swing_detail.csv",
                mime="text/csv",
            )