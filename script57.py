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
# 0. 頁面設定與 MediaPipe 模型下載
# ==============================================================================
st.set_page_config(
    page_title="崇明國中-棒球揮棒姿態與 Kinovea 式仰角分析系統",
    page_icon="⚾",
    layout="wide",
)

MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"


@st.cache_resource
def ensure_model_file():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ 下載 MediaPipe 核心分析模型檔..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


ensure_model_file()

st.title("⚾ 崇明國中-棒球揮棒姿態與 Kinovea 式仰角校正系統")
st.caption(
    "結合 Kinovea 地平基準線校正、自訂實體長度標定與 AI 骨架軌跡追蹤"
)

# ==============================================================================
# 1. 側邊欄：Kinovea 式水平線與實體尺寸校正
# ==============================================================================
with st.sidebar:
    st.header("📐 Kinovea 角度與物理校正")

    st.subheader("1. 地平線與相機傾斜校正")
    manual_horizon_offset = st.slider(
        "🎯 地平線角度微調 (度)",
        min_value=-20.0,
        max_value=20.0,
        value=0.0,
        step=0.5,
        help="如畫面相機傾斜，可手動微調此數值，使綠色基準線對齊地面或圍欄。",
    )

    st.subheader("2. 實體長度標定 (Scale)")
    real_bat_meters = st.number_input(
        "🏏 球棒實際長度 (公尺)",
        min_value=0.5,
        max_value=1.1,
        value=0.84,
        step=0.01,
        help="一般成人球棒約 0.84m (33吋)，青少棒約 0.75m~0.80m。",
    )

    meters_per_pixel = st.slider(
        "📏 像素轉公尺比例 (Meters/PX)",
        min_value=0.0010,
        max_value=0.0080,
        value=0.0030,
        step=0.0001,
    )

    bat_speed_factor = st.slider(
        "🚀 手腕至棒頭速度倍率",
        min_value=1.0,
        max_value=2.0,
        value=1.35,
        step=0.05,
    )

    st.subheader("3. 偵測靈敏度門檻")
    min_peak_speed = st.slider(
        "⚡ 最低有效初速 (km/h)",
        min_value=10.0,
        max_value=50.0,
        value=15.0,
        step=1.0,
    )

    min_total_travel = st.slider(
        "📏 手腕最低總位移 (像素)",
        min_value=10,
        max_value=100,
        value=20,
        step=5,
    )

    st.markdown("---")
    bat_length_px = 110
    target_width = 800

    if st.button("🔄 重置並重新分析", use_container_width=True):
        st.session_state.is_analyzed = False
        st.session_state.swing_events = []
        st.rerun()


# ==============================================================================
# 2. 核心幾何運算與 Kinovea 繪圖函式
# ==============================================================================
def calculate_auto_horizon(landmarks, width, height):
    """偵測兩肩角度作為自動地平線基礎"""
    if not landmarks:
        return 0.0
    l_s = landmarks[11] if len(landmarks) > 11 else None
    r_s = landmarks[12] if len(landmarks) > 12 else None

    if l_s and r_s and l_s.visibility > 0.5 and r_s.visibility > 0.5:
        dx = (r_s.x - l_s.x) * width
        dy = (r_s.y - l_s.y) * height
        return math.degrees(math.atan2(dy, dx))
    return 0.0


def draw_kinovea_protractor(frame, wrist_pos, total_horizon_angle):
    """繪製 Kinovea 風格的地平線、垂直軸線與最佳擊球仰角扇形區域 (15°-35°)"""
    if not wrist_pos:
        return

    cx, cy = wrist_pos
    rad = math.radians(total_horizon_angle)

    # 1. 地平基準線 (綠色實線)
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

    # 2. 鉛直 90 度軸線 (紫色虛線)
    p_up = (cx + dy, cy - dx)
    p_down = (cx - dy, cy + dx)
    cv2.line(frame, p_up, p_down, (255, 0, 255), 1, cv2.LINE_AA)

    # 3. 繪製甜蜜擊球仰角區間扇形 (15° 到 35°)
    overlay = frame.copy()
    radius = 120

    # 轉為 OpenCV 角度定義 (Y 軸向下)
    start_ang = -(total_horizon_angle + 35)
    end_ang = -(total_horizon_angle + 15)

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

    # 標籤文字
    cv2.putText(
        frame,
        f"0 deg Horizon ({total_horizon_angle:.1f}deg)",
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
    frame, pose_landmarks, width, height, bat_length, manual_offset
):
    if not pose_landmarks:
        return None, None, None, 0.0

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

    # 計算總地平線角度 = 自動骨架角度 + 手動微調值
    auto_angle = calculate_auto_horizon(landmarks, width, height)
    total_horizon_angle = auto_angle + manual_offset

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

    # 畫上 Kinovea 量角規
    if wrist_center:
        draw_kinovea_protractor(frame, wrist_center, total_horizon_angle)

    return wrist_center, bat_head, total_horizon_angle


def save_swing_clip(frames, fps, width, height, output_path):
    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for f in frames:
        out.write(f)
    out.release()
    return output_path


# ==============================================================================
# 3. Session 狀態與介面建立
# ==============================================================================
if "swing_events" not in st.session_state:
    st.session_state.swing_events = []
if "is_analyzed" not in st.session_state:
    st.session_state.is_analyzed = False
if "current_file_key" not in st.session_state:
    st.session_state.current_file_key = None

uploaded_file = st.file_uploader(
    "📁 選擇或上傳揮棒影片", type=["mp4", "avi", "mov", "m4v"]
)

if uploaded_file is not None:
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.current_file_key != file_key:
        st.session_state.current_file_key = file_key
        st.session_state.swing_events = []
        st.session_state.is_analyzed = False

# ==============================================================================
# 4. 核心分析與 Kinovea 仰角計算
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

    st.markdown("### 📹 Kinovea 圖像分析與水平校正中...")
    st_frame = st.empty()
    progress_bar = st.progress(0)

    history_wrist = []
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
            total_horizon = 0.0

            if result.pose_landmarks:
                wrist, bat_head, total_horizon = process_frame(
                    annotated_frame,
                    result.pose_landmarks,
                    proc_width,
                    proc_height,
                    bat_length_px,
                    manual_horizon_offset,
                )
                if wrist:
                    current_wrist = wrist
                    current_bat_head = bat_head
                    history_wrist.append((frame_idx, wrist[0], wrist[1]))

            # 計算即時速度
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
                        "horizon": total_horizon,
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
                        "horizon": total_horizon,
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

                # 結算本次揮棒仰角 (Kinovea 相對角度計算)
                if swing_state == 3:
                    launch_angle = 0.0
                    peak_sub_idx = [
                        i
                        for i, item in enumerate(swing_frames_data)
                        if item["frame"] == peak_frame_in_swing
                    ]

                    if peak_sub_idx:
                        p_idx = peak_sub_idx[0]
                        post_idx = min(len(swing_frames_data) - 1, p_idx + 2)
                        p_c = swing_frames_data[p_idx]["bat_head"]
                        p_post = swing_frames_data[post_idx]["bat_head"]
                        h_angle = swing_frames_data[p_idx].get("horizon", 0.0)

                        if p_c and p_post:
                            dx_launch = p_post[0] - p_c[0]
                            dy_launch = -(p_post[1] - p_c[1])

                            if (
                                abs(dx_launch) > 0.0001
                                or abs(dy_launch) > 0.0001
                            ):
                                raw_angle = math.degrees(
                                    math.atan2(dy_launch, dx_launch)
                                )
                                launch_angle = raw_angle - h_angle

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
                        "耗時": f"{(len(swing_frames_data) / fps):.2f} 秒",
                        "total_frames": len(swing_frames_data),
                        "video_bytes": video_bytes,
                        "detailed_df": detailed_df,
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
# 5. 結果展現
# ==============================================================================
if st.session_state.is_analyzed:
    events = st.session_state.swing_events

    if not events:
        st.warning(
            "⚠️ 未偵測到有效揮棒。請嘗試調低【最低有效初速】或微調【像素轉公尺比例】。"
        )
    else:
        st.success(f"✅ 分析完成！共偵測到 {len(events)} 次揮棒。")

        option_list = [e["次數"] for e in events]
        selected_swing_name = st.selectbox(
            "🎯 選擇揮棒次數：",
            options=option_list,
            index=len(option_list) - 1,
        )

        selected_event = next(
            (e for e in events if e["次數"] == selected_swing_name), events[0]
        )

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("⚡ 估算初速", selected_event["初速"])
        col2.metric("📐 校正後仰角", selected_event["仰角"])
        col3.metric("⏱️ 揮棒耗時", selected_event["耗時"])
        col4.metric("🎞️ 記錄點數", f"{selected_event['total_frames']} 點")

        tab1, tab2 = st.tabs(["🎬 Kinovea 回放 (含基準線/扇形區)", "📊 數據明細表"])

        with tab1:
            st.markdown(
                f"#### 🎬 {selected_event['次數']} - 慢動作與綠色地平線/黃色 15°-35° 甜蜜角區域"
            )
            st.video(selected_event["video_bytes"], format="video/webm")

        with tab2:
            st.dataframe(selected_event["detailed_df"], use_container_width=True)