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
# 0. 系統與 MediaPipe Pose Landmarker 模型配置
# ==============================================================================
st.set_page_config(
    page_title="⚾ 棒球高階揮擊診斷與動力鏈分析系統",
    page_icon="⚾",
    layout="wide",
)

MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"


@st.cache_resource
def ensure_model_file():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ 首次執行，正在下載 MediaPipe 姿態識別模型..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


ensure_model_file()

st.title("⚾ 棒球高階揮擊診斷與動力鏈分析系統")
st.caption(
    "整合 4 大核心參考數據與下半身骨盆轉動診斷（含畫面角度與姿態繪製）"
)

# ==============================================================================
# 1. 側邊欄控制項：標定與門檻參數
# ==============================================================================
with st.sidebar:
    st.header("⚙️ 系統與物理參數標定")

    camera_tilt_deg = st.slider(
        "📐 相機傾斜補償 (度)",
        -20.0,
        20.0,
        0.0,
        0.5,
        help="修正拍攝傾斜角度，維護畫面絕對地平線基準。",
    )

    meters_per_pixel = st.slider(
        "📏 像素轉公尺比例 (Meters/PX)",
        0.0010,
        0.0080,
        0.0032,
        0.0001,
        help="標定空間距離，影響揮棒速度與軌跡長度計算。",
    )

    bat_speed_factor = st.slider(
        "🚀 手腕至棒頭速度放大倍率",
        1.00,
        2.00,
        1.35,
        0.05,
        help="以手腕點推算棒頭離心速度之補償係數。",
    )

    st.subheader("揮棒偵測門檻")
    min_peak_speed = st.slider("⚡ 最低初速門檻 (km/h)", 10.0, 50.0, 15.0, 1.0)

    bat_length_px = 110
    target_width = 800

    if st.button("🔄 重置分析", use_container_width=True):
        st.session_state.is_analyzed = False
        st.session_state.swing_events = []
        st.rerun()

# ==============================================================================
# 2. 幾何計算與 4 大指標 / 轉動診斷模組
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

    # 1. 揮棒軌跡長度 (Swing Length)
    diffs = np.diff(pts, axis=0)
    dist_px = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
    swing_length_m = dist_px * m_per_px

    # 2. 攻擊仰角 (Attack Angle)
    x = pts[:, 0]
    y_phys = -pts[:, 1]  # 轉為物理向上為正

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
    """整合 4 大指標與動作修正方向（軌跡改善、下半身轉動強化）之自動化診斷報告"""
    score = 100
    feedbacks = []
    drills = []

    # 1. 揮棒速度 (Bat Speed)
    if bat_speed >= 28.0:
        bat_speed_status = "✅ 爆發力佳"
        feedbacks.append(
            f"揮棒速度 **{bat_speed:.1f} km/h**，棒頭通過本壘板前緣具備足夠速率與擊球貫穿力。"
        )
    else:
        bat_speed_status = "⚠️ 速度偏低"
        feedbacks.append(
            f"揮棒速度 **{bat_speed:.1f} km/h**，揮棒速度不足會直接影響擊球距離與遠度。"
        )
        score -= 10

    # 2. 揮棒軌跡長度 (Swing Length)
    if 0.65 <= swing_length <= 0.95:
        length_status = "✅ 軌跡緊湊敏捷"
        feedbacks.append(
            f"揮棒軌跡長度 **{swing_length:.2f} m**，從啟動到擊球點路徑簡潔，能有效應對高振幅速球。"
        )
    elif swing_length > 0.95:
        length_status = "⚠️ 軌跡過長 (繞大圈)"
        feedbacks.append(
            f"揮棒軌跡長達 **{swing_length:.2f} m** (標竿 < 0.95m)，揮棒路徑過長或繞大圈，會大幅降低面對高速球的反應容錯率。"
        )
        score -= 15
        drills.append("貼牆揮棒路徑收束訓練 (Wall Drill)")
    else:
        length_status = "⚠️ 揮棒延伸不足"
        feedbacks.append(
            f"揮棒軌跡僅 **{swing_length:.2f} m**，手臂過早收縮，影響擊球後段貫穿力與延伸。"
        )

    # 3. 攻擊仰角 (Attack Angle)
    if 6.0 <= attack_angle <= 18.0:
        attack_status = "✅ 完美迎球切入"
        feedbacks.append(
            f"攻擊仰角 **{attack_angle:.1f}°**，球棒精確切入擊球區，有利於創造強勁平飛球。"
        )
    elif attack_angle < 6.0:
        attack_status = "⚠️ 角度過陡 (由上往下砍)"
        feedbacks.append(
            f"攻擊仰角僅 **{attack_angle:.1f}°**，揮棒角度過陡（由上往下砍），會嚴重縮小擊球容錯區間，且極易打成無力滾地球或揮空。"
        )
        score -= 15
        drills.append("高低位置擊球座高角掃擊練習 (Elevated Tee Work)")
    else:
        attack_status = "⚠️ 倒棒/過度仰角"
        feedbacks.append(
            f"攻擊仰角達 **{attack_angle:.1f}°**，過度仰角容易造成倒棒並打成無效高飛球。"
        )
        score -= 15
        drills.append("水平平飛擊球修正 (Level Swing Progression)")

    # 4. 擊球初速 (Exit Velocity)
    if exit_velocity >= 25.0:
        exit_status = "✅ 力量扎實轉化"
        feedbacks.append(
            f"預估擊球初速 **{exit_velocity:.1f} km/h**，展現揮棒力量扎實轉化為實際破壞力的效率。"
        )
    else:
        exit_status = "⚠️ 動能轉化待提升"
        feedbacks.append(
            f"預估擊球初速 **{exit_velocity:.1f} km/h**，揮棒動能傳遞至球體的轉化效率需補強。"
        )
        score -= 10

    # 5. 動力鏈與轉動強化診斷
    if hip_rot_speed >= 280.0:
        hip_status = "✅ 骨盆轉動爆發力強"
        feedbacks.append(
            f"髖關節峰值轉速達 **{hip_rot_speed:.0f} deg/s**，展現優秀的後髖關節與骨盆旋轉帶動。"
        )
    else:
        hip_status = "⚠️ 下半身轉動引導不足"
        feedbacks.append(
            f"髖關節峰值轉速僅 **{hip_rot_speed:.0f} deg/s**，下半身轉動發力不足，未能將力量由下而上完整傳遞至手腕與棒頭。"
        )
        score -= 15
        drills.append("後髖關節爆發力旋轉彈力帶訓練 (Hip-Hinge Band Rotation)")

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
        "核心參考指標": [
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
        "標竿參考標準": [
            "> 28.0 km/h",
            "0.65 - 0.95 m",
            "6.0° - 18.0°",
            "> 25.0 km/h",
            "> 280 deg/s",
            "> 0.88",
        ],
        "診斷評定": [
            bat_speed_status,
            length_status,
            attack_status,
            exit_status,
            hip_status,
            "✅ 穩定" if r_squared >= 0.88 else "⚠️ 抖動",
        ],
    })

    return {
        "score": score,
        "grade": grade,
        "summary_df": summary_df,
        "feedbacks": feedbacks,
        "drills": list(set(drills)),
    }


def draw_detection_annotations(
    frame: np.ndarray,
    pose_landmarks,
    width: int,
    height: int,
    bat_length: int,
):
    """於畫面上實體繪製肩線、髖關節角度線、棒頭向量與角度數值"""
    if not pose_landmarks:
        return None, None, 0.0

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

    # 1. 繪製肩膀連線 (白色)
    if l_shoulder and r_shoulder:
        cv2.line(frame, l_shoulder, r_shoulder, (255, 255, 255), 2)
        cv2.circle(frame, l_shoulder, 4, (255, 255, 255), -1)
        cv2.circle(frame, r_shoulder, 4, (255, 255, 255), -1)

    # 2. 繪製骨盆/髖關節旋轉連線 (洋紅色) 與角度標示
    hip_angle = 0.0
    if l_hip and r_hip:
        cv2.line(frame, l_hip, r_hip, (255, 0, 255), 3)
        cv2.circle(frame, l_hip, 5, (255, 0, 255), -1)
        cv2.circle(frame, r_hip, 5, (255, 0, 255), -1)

        dx_h = r_hip[0] - l_hip[0]
        dy_h = r_hip[1] - l_hip[1]
        hip_angle = math.degrees(math.atan2(dy_h, dx_h))

        # 標註髖關節角度文字
        mid_hip = (int((l_hip[0] + r_hip[0]) / 2), int((l_hip[1] + r_hip[1]) / 2))
        cv2.putText(
            frame,
            f"Hip: {abs(hip_angle):.1f}deg",
            (mid_hip[0] - 40, mid_hip[1] + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # 3. 手腕中心與棒頭向量繪製
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
            # 繪製球棒本體 (橘色) 與棒頭 (黃色)
            cv2.line(frame, wrist_center, bat_head, (0, 165, 255), 4)
            cv2.circle(frame, bat_head, 6, (0, 255, 255), -1)

    return wrist_center, bat_head, hip_angle


# ==============================================================================
# 3. 影片分析與視訊畫面檢測線條繪製主程序
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

    st.markdown("### 📹 全程 4 大數據與檢測線條繪製分析中...")
    st_frame = st.empty()
    progress_bar = st.progress(0)

    history_wrist = []
    history_hip_angles = []
    current_swing_trajectory = []
    swing_state = 0
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

            # 繪製姿態骨架與檢測線條
            if result.pose_landmarks:
                wrist, bat_head, hip_angle = draw_detection_annotations(
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

            # 計算即時速度
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
                current_speed = (
                    math.sqrt(dx**2 + dy**2) / dt
                ) * 3.6 * bat_speed_factor

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

                elif swing_state in [1, 2]:
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
                        peak_frame_in_swing = frame_idx

                    if (
                        swing_state == 1
                        and max_speed_in_swing >= min_peak_speed
                        and current_speed < max_speed_in_swing * 0.6
                    ):
                        swing_state = 2

                    if (
                        swing_state == 2
                        and (
                            current_speed <= start_trigger_speed * 0.8
                            or len(swing_frames_data) > 35
                        )
                    ):
                        swing_state = 3

                # 繪製揮棒黃色軌跡連續線
                if len(current_swing_trajectory) > 1:
                    pts = np.array(
                        current_swing_trajectory, np.int32
                    ).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame, [pts], False, (0, 255, 255), 3
                    )

                # 畫面 OSD 即時數據儀表板 overlay
                cv2.rectangle(
                    annotated_frame, (10, 10), (280, 85), (0, 0, 0), -1
                )
                cv2.putText(
                    annotated_frame,
                    f"Bat Speed: {current_speed:.1f} km/h",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated_frame,
                    f"Hip Rot: {current_hip_speed:.0f} deg/s",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                if swing_state in [1, 2]:
                    swing_raw_frames.append(annotated_frame.copy())

                # 結算揮棒並產出分析數據
                if swing_state == 3:
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

                    detected_events.append({
                        "次數": f"第 {swing_num} 次揮棒",
                        "bat_speed": bat_speed,
                        "swing_length": swing_length,
                        "attack_angle": attack_angle,
                        "exit_velocity": exit_velocity,
                        "hip_speed": peak_hip_speed,
                        "report": report,
                        "video_bytes": video_bytes,
                    })

                    swing_state = 0
                    current_swing_trajectory = []
                    cooldown_counter = int(fps * 0.8)

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
# 4. 分析結果與診斷報告展示區
# ==============================================================================
if st.session_state.get("is_analyzed", False):
    events = st.session_state.swing_events

    if not events:
        st.warning("⚠️ 未能偵測到有效揮棒，請調整左側邊欄門檻。")
    else:
        st.success(f"✅ 完成分析！共偵測到 {len(events)} 次揮棒。")

        selected_swing_name = st.selectbox(
            "🎯 選擇揮棒次數檢視詳細診斷報告：", [e["次數"] for e in events]
        )
        event = next(e for e in events if e["次數"] == selected_swing_name)
        report = event["report"]

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🚀 揮棒速度 (Bat Speed)", f"{event['bat_speed']:.1f} km/h")
        c2.metric(
            "📏 揮棒軌跡長度 (Swing Length)",
            f"{event['swing_length']:.2f} m",
        )
        c3.metric("📐 攻擊仰角 (Attack Angle)", f"{event['attack_angle']:.1f}°")
        c4.metric(
            "⚡ 預估擊球初速 (Exit Velocity)",
            f"{event['exit_velocity']:.1f} km/h",
        )

        tab1, tab2 = st.tabs(
            ["📄 揮擊診斷與動作修正報告", "🎬 慢動作姿態回放（含檢測線條）"]
        )

        with tab1:
            st.markdown(
                f"### 🏆 揮棒綜合評分：`{report['score']} 分` (等級: {report['grade']})"
            )

            st.subheader("📊 4 大核心參考數據與動力鏈指標對比")
            st.dataframe(report["summary_df"], use_container_width=True)

            st.subheader("💡 動作修正方向診斷 (軌跡改善與下半身轉動)")
            for fb in report["feedbacks"]:
                st.write(f"- {fb}")

            if report["drills"]:
                st.subheader("🎯 建議針對性動作修正處方 (Recommended Drills)")
                for drill in report["drills"]:
                    st.info(f"👉 **建議訓練：** {drill}")

        with tab2:
            st.video(event["video_bytes"], format="video/webm")