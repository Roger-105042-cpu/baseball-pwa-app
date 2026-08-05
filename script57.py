import math
import os
import tempfile
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. PWA 與 行動裝置 響應式 CSS 注入
# ==============================================================================
st.set_page_config(
    page_title="棒球打擊分析 PWA",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",  # 手機預設收合側邊欄，增加可視面積
)

# 注入 PWA Web App Meta 標籤與行動端優化 CSS
pwa_html = """
    <!-- PWA Web App 設定 -->
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="theme-color" content="#0E1117">
    <link rel="manifest" href="data:application/json;base64,ewogICJzaG9ydF9uYW1lIjogIuajkmnDicaAIiwKICAibmFtZSI6ICLmo5Jp
    eLp24o2A44S056CU5L2P44SuIiwKICAiaWNvbnMiOiBbCiAgICB7CiAgICAgICJzcmMiOiAiaHR0cHM6Ly9lbW9qaWNkbi5lbW9qaS51cy9zbm
    Fwc2hvdC92MTEvYmFzZWJhbGxfMWY2Y2UucG5nIiwKICAgICAgInR5cGUiOiAiaW1hZ2UvcG5nIiwKICAgICAgInNpemVzIjogIjE5MngxOTIi
    CiAgICB9CiAgXSwKICAic3RhcnRfdXJsIjogIi4iLAogICJkaXNwbGF5IjogInN0YW5kYWxvbmUiLAogICJiYWNrZ3JvdW5kX2NvbG9yIjog
    IiMwRTExMTciLAogICJ0aGVtZV9jb2xvciI6ICIjMEUxMTE3Igp9">
"""
st.markdown(pwa_html, unsafe_allow_html=True)

# 手機/平板動態樣式微調
st.markdown(
    """
    <style>
        /* 隱藏預設的主選單與頁尾以達全螢幕 APP 效果 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* 調整行動端容器間距 */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        /* 提升觸控按鈕體驗 */
        .stButton>button {
            width: 100%;
            height: 3rem;
            font-size: 1.1rem;
            border-radius: 10px;
        }

        /* 揮棒數據卡片樣式 */
        .metric-card {
            background-color: #1E222A;
            border-radius: 10px;
            padding: 12px;
            text-align: center;
            border: 1px solid #31363F;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. MediaPipe 與計算核心
# ==============================================================================
MODEL_PATH = "pose_landmarker.task"
POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]


def process_frame(image, landmarks_list, width, height):
    """擷取骨架、重心與手腕座標"""
    if not landmarks_list:
        return None, None, None

    landmarks = landmarks_list[0]

    for start_idx, end_idx in POSE_CONNECTIONS:
        pt1 = (
            int(landmarks[start_idx].x * width),
            int(landmarks[start_idx].y * height),
        )
        pt2 = (
            int(landmarks[end_idx].x * width),
            int(landmarks[end_idx].y * height),
        )
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)

    def get_pt(idx):
        return np.array([landmarks[idx].x * width, landmarks[idx].y * height])

    try:
        left_shoulder, right_shoulder = get_pt(11), get_pt(12)
        left_hip, right_hip = get_pt(23), get_pt(24)
        left_knee, right_knee = get_pt(25), get_pt(26)
        left_ankle, right_ankle = get_pt(27), get_pt(28)
        wrist_pt = get_pt(16)  # 右手腕

        torso_center = (
            left_shoulder + right_shoulder + left_hip + right_hip
        ) / 4.0
        left_leg_center = (left_hip + left_knee + left_ankle) / 3.0
        right_leg_center = (right_hip + right_knee + right_ankle) / 3.0
        arms_center = (left_shoulder + right_shoulder) / 2.0

        com = (
            torso_center * 0.50
            + left_leg_center * 0.15
            + right_leg_center * 0.15
            + arms_center * 0.20
        )
        cx, cy = int(com[0]), int(com[1])
        cv2.circle(image, (cx, cy), 8, (0, 0, 255), -1)

        shoulder_mid = (left_shoulder + right_shoulder) / 2.0
        return (
            (cx, cy),
            (int(wrist_pt[0]), int(wrist_pt[1])),
            (int(shoulder_mid[0]), int(shoulder_mid[1])),
        )
    except Exception:
        return None, None, None


def save_swing_clip(frames, fps, width, height, output_path):
    """匯出單次揮棒短片 (.mp4)"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for f in frames:
        out.write(f)
    out.release()


# ==============================================================================
# 3. Streamlit UI 介面設定
# ==============================================================================
st.title("⚾ 棒球打擊分析 PWA 系統")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ 找不到模型檔案 `{MODEL_PATH}`！請確保模型檔案在同一目錄。")
    st.stop()

# 側邊欄設定
st.sidebar.header("⚙️ 實體尺寸與參數校正")
ref_pixel = st.sidebar.number_input(
    "參考物像素長度 (Pixels)", value=150.0, step=10.0
)
ref_meters = st.sidebar.number_input(
    "參考物實際長度 (公尺)", value=1.8, step=0.1
)
meters_per_pixel = ref_meters / ref_pixel if ref_pixel > 0 else 0.01

bat_speed_factor = st.sidebar.slider(
    "手腕到球速轉換係數", min_value=1.0, max_value=2.0, value=1.3, step=0.05
)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ 防多算軌跡門檻")
min_peak_speed = st.sidebar.slider(
    "最低揮棒峰值速度 (km/h)",
    min_value=20.0,
    max_value=80.0,
    value=35.0,
    step=5.0,
)
min_x_travel = st.sidebar.slider(
    "手腕最小水平位移 (Pixels)",
    min_value=50.0,
    max_value=400.0,
    value=150.0,
    step=10.0,
)

# 行動端 Friendly 檔案上傳
uploaded_file = st.file_uploader(
    "📁 選擇或拍攝打擊影片", type=["mp4", "avi", "mov", "m4v"]
)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 針對行動裝置切換單/雙欄版面
    st.markdown("### 📹 打擊分析動態")
    st_frame = st.empty()

    st.markdown("---")
    st.markdown("### ⚡ 揮棒數據與短片回放")
    metrics_placeholder = st.empty()
    table_placeholder = st.empty()
    replay_placeholder = st.empty()

    # 數據變數
    history_wrist = []
    history_speeds = []
    swing_events = []

    # 狀態機變數
    swing_state = 0
    swing_frames_data = []
    swing_raw_frames = []
    max_speed_in_swing = 0.0
    peak_frame_in_swing = 0
    cooldown_counter = 0

    dt = 1.0 / fps

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    clip_dir = tempfile.mkdtemp()

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            annotated_frame = frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb_frame
            )
            timestamp_ms = int((frame_idx / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            current_wrist = None
            if result.pose_landmarks:
                com, wrist, shoulder_mid = process_frame(
                    annotated_frame, result.pose_landmarks, width, height
                )
                if wrist:
                    current_wrist = wrist
                    history_wrist.append((frame_idx, wrist[0], wrist[1]))

            # 即時速度計算
            current_speed = 0.0
            if len(history_wrist) >= 2:
                p1 = history_wrist[-2]
                p2 = history_wrist[-1]
                dx = (p2[1] - p1[1]) * meters_per_pixel
                dy = (p2[2] - p1[2]) * meters_per_pixel
                dist_m = math.sqrt(dx**2 + dy**2)
                current_speed = (dist_m / dt) * 3.6 * bat_speed_factor

            history_speeds.append((frame_idx, current_speed))

            # 揮棒狀態機
            if cooldown_counter > 0:
                cooldown_counter -= 1

            start_trigger_speed = min_peak_speed * 0.45

            if cooldown_counter == 0:
                if swing_state == 0:
                    if current_speed >= start_trigger_speed:
                        swing_state = 1
                        swing_frames_data = []
                        swing_raw_frames = []
                        max_speed_in_swing = current_speed
                        peak_frame_in_swing = frame_idx

                elif swing_state == 1:
                    swing_frames_data.append(
                        {
                            "frame": frame_idx,
                            "wrist": current_wrist,
                            "speed": current_speed,
                        }
                    )
                    swing_raw_frames.append(annotated_frame)

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
                    ):
                        swing_state = 0

                elif swing_state == 2:
                    swing_frames_data.append(
                        {
                            "frame": frame_idx,
                            "wrist": current_wrist,
                            "speed": current_speed,
                        }
                    )
                    swing_raw_frames.append(annotated_frame)

                    if current_speed <= start_trigger_speed * 0.8:
                        x_coords = [
                            item["wrist"][0]
                            for item in swing_frames_data
                            if item["wrist"] is not None
                        ]
                        total_x_displacement = (
                            max(x_coords) - min(x_coords) if x_coords else 0
                        )

                        if total_x_displacement >= min_x_travel:
                            swing_state = 3
                        else:
                            swing_state = 0
                            cooldown_counter = int(fps * 0.3)

                if swing_state == 3:
                    launch_angle = 0.0
                    peak_sub_idx = [
                        i
                        for i, item in enumerate(swing_frames_data)
                        if item["frame"] == peak_frame_in_swing
                    ]
                    if peak_sub_idx:
                        p_idx = peak_sub_idx[0]
                        post_idx = min(len(swing_frames_data) - 1, p_idx + 3)
                        p_c = swing_frames_data[p_idx]["wrist"]
                        p_post = swing_frames_data[post_idx]["wrist"]
                        if p_c and p_post:
                            dx_launch = (
                                p_post[0] - p_c[0]
                            ) * meters_per_pixel
                            dy_launch = (
                                p_post[1] - p_c[1]
                            ) * meters_per_pixel
                            if (
                                abs(dx_launch) > 0.0001
                                or abs(dy_launch) > 0.0001
                            ):
                                launch_angle = math.degrees(
                                    math.atan2(-dy_launch, dx_launch)
                                )

                    swing_num = len(swing_events) + 1
                    clip_filename = os.path.join(
                        clip_dir, f"swing_{swing_num}.mp4"
                    )
                    save_swing_clip(
                        swing_raw_frames, fps, width, height, clip_filename
                    )

                    event_data = {
                        "次數": f"第 {swing_num} 次",
                        "初速 (km/h)": round(max_speed_in_swing, 1),
                        "仰角 (度)": round(launch_angle, 1),
                        "耗時 (秒)": round(len(swing_frames_data) / fps, 2),
                        "clip_path": clip_filename,
                    }
                    swing_events.append(event_data)

                    # UI 更新
                    with metrics_placeholder.container():
                        st.success(f"🎉 成功偵測第 {swing_num} 次完整揮棒！")
                        m1, m2 = st.columns(2)
                        m1.metric(
                            label="估算初速",
                            value=f"{max_speed_in_swing:.1f} km/h",
                        )
                        m2.metric(label="預測仰角", value=f"{launch_angle:.1f}°")

                    df_events = pd.DataFrame(swing_events)
                    table_placeholder.dataframe(
                        df_events.drop(columns=["clip_path"]),
                        use_container_width=True,
                    )

                    with replay_placeholder.container():
                        st.subheader(f"🎬 第 {swing_num} 次揮棒動作回放")
                        if os.path.exists(clip_filename):
                            st.video(clip_filename)

                    swing_state = 0
                    cooldown_counter = int(fps * 1.0)

            # 畫面繪製與更新
            cv2.putText(
                annotated_frame,
                f"STATE: {swing_state}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )

            # 轉換為 RGB 格式並確保型態為 uint8
            frame_display = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_display = np.asarray(frame_display, dtype=np.uint8)

            # 修正 use_column_width 改用 use_container_width
            st_frame.image(
                frame_display,
                channels="RGB",
                use_container_width=True,
            )

    cap.release()