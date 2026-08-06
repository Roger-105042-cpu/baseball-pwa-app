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
    initial_sidebar_state="collapsed",
)

pwa_html = """
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="theme-color" content="#0E1117">
"""
st.markdown(pwa_html, unsafe_allow_html=True)

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .stButton>button {
            width: 100%;
            height: 3rem;
            font-size: 1.1rem;
            border-radius: 10px;
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


def process_frame(image, landmarks_list, width, height, bat_length_px):
    """擷取骨架、重心、手腕與棒頭座標"""
    if not landmarks_list:
        return None, None, None, None

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
        right_elbow = get_pt(14)
        wrist_pt = get_pt(16)  # 右手腕
        left_hip, right_hip = get_pt(23), get_pt(24)
        left_knee, right_knee = get_pt(25), get_pt(26)
        left_ankle, right_ankle = get_pt(27), get_pt(28)

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

        # 棒頭向量延伸
        arm_vector = wrist_pt - right_elbow
        arm_norm = np.linalg.norm(arm_vector)

        if arm_norm > 0:
            unit_vector = arm_vector / arm_norm
            bat_head_pt = wrist_pt + unit_vector * bat_length_px
        else:
            bat_head_pt = wrist_pt

        bat_head = (int(bat_head_pt[0]), int(bat_head_pt[1]))
        wrist = (int(wrist_pt[0]), int(wrist_pt[1]))

        cv2.line(image, wrist, bat_head, (255, 0, 255), 3)
        cv2.circle(image, bat_head, 6, (0, 255, 255), -1)

        shoulder_mid = (left_shoulder + right_shoulder) / 2.0
        return (
            (cx, cy),
            wrist,
            bat_head,
            (int(shoulder_mid[0]), int(shoulder_mid[1])),
        )
    except Exception:
        return None, None, None, None


def save_swing_clip(frames, fps, width, height, output_path):
    """匯出單次揮棒短片 (.webm)"""
    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    if not output_path.endswith(".webm"):
        output_path = output_path.replace(".mp4", ".webm")

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for f in frames:
        out.write(f)
    out.release()
    return output_path


# ==============================================================================
# 3. Streamlit UI 介面設定
# ==============================================================================
st.title("⚾ 棒球打擊分析 PWA 系統")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ 找不到模型檔案 `{MODEL_PATH}`！請確保模型檔案在同一目錄。")
    st.stop()

# 側邊欄設定
st.sidebar.header("🚀 效能與分析加速設定")
target_width = st.sidebar.selectbox(
    "分析處理解析度 (影響處理速度)",
    options=[480, 640, 720, 1080],
    index=1,
    help="降低解析度可大幅縮短分析時間",
)

frame_skip_step = st.sidebar.slider(
    "抽幀採訪步階 (Frame Skip)",
    min_value=1,
    max_value=3,
    value=1,
    help="1: 每幀分析(最精準); 2: 每2幀分析(速度翻倍)",
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 實體尺寸與球棒設定")
ref_pixel = st.sidebar.number_input(
    "參考物像素長度 (Pixels)", value=150.0, step=10.0
)
ref_meters = st.sidebar.number_input(
    "參考物實際長度 (公尺)", value=1.8, step=0.1
)
meters_per_pixel = ref_meters / ref_pixel if ref_pixel > 0 else 0.01

bat_length_px = st.sidebar.slider(
    "棒長估算延伸量 (Pixels)",
    min_value=30.0,
    max_value=250.0,
    value=100.0,
    step=10.0,
)

bat_speed_factor = st.sidebar.slider(
    "手腕到球速轉換係數", min_value=1.0, max_value=2.0, value=1.3, step=0.05
)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ 防多算軌跡門檻")
min_peak_speed = st.sidebar.slider(
    "最低揮棒峰值速度 (km/h)",
    min_value=10.0,
    max_value=80.0,
    value=20.0,
    step=5.0,
)
min_x_travel = st.sidebar.slider(
    "手腕最小水平位移 (Pixels)",
    min_value=30.0,
    max_value=400.0,
    value=80.0,
    step=10.0,
)

uploaded_file = st.file_uploader(
    "📁 選擇或拍攝打擊影片", type=["mp4", "avi", "mov", "m4v"]
)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # [建議 1: 降採樣計算] 計算縮放比例
    if orig_width > target_width:
        scale = target_width / float(orig_width)
        proc_width = target_width
        proc_height = int(orig_height * scale)
    else:
        scale = 1.0
        proc_width = orig_width
        proc_height = orig_height

    st.markdown("### 📹 全程動態分析預覽")
    st_frame = st.empty()

    st.markdown("---")
    st.markdown("### ⚡ 揮棒分段分析與獨立數據表格")

    if "swing_events" not in st.session_state or st.sidebar.button(
        "🔄 重新分析影片"
    ):
        st.session_state.swing_events = []

    history_wrist = []
    current_swing_trajectory = []

    swing_state = 0
    swing_frames_data = []
    swing_raw_frames = []
    max_speed_in_swing = 0.0
    peak_frame_in_swing = 0
    cooldown_counter = 0

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    clip_dir = tempfile.mkdtemp()

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        last_wrist_time = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # [建議 2: 跳幀機制] 依照步階跳過非關鍵幀
            if frame_skip_step > 1 and (frame_idx % frame_skip_step != 0):
                continue

            # [建議 1: 影像降採樣] 將圖像縮放後再丟入 MediaPipe 與繪圖，極大減輕運算負荷
            if scale != 1.0:
                frame_resized = cv2.resize(
                    frame, (proc_width, proc_height), interpolation=cv2.INTER_AREA
                )
            else:
                frame_resized = frame.copy()

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
                com, wrist, bat_head, shoulder_mid = process_frame(
                    annotated_frame,
                    result.pose_landmarks,
                    proc_width,
                    proc_height,
                    bat_length_px,
                )
                if wrist:
                    current_wrist = wrist
                    current_bat_head = bat_head
                    history_wrist.append((frame_idx, wrist[0], wrist[1]))

            # 計算即時速度（動態考慮跳幀的時間差 dt）
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

            # 揮棒狀態機
            if cooldown_counter > 0:
                cooldown_counter -= 1

            start_trigger_speed = min_peak_speed * 0.4

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

                    swing_frames_data.append(
                        {
                            "frame": frame_idx,
                            "wrist": current_wrist,
                            "bat_head": current_bat_head,
                            "speed": current_speed,
                        }
                    )

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
                        and len(swing_frames_data) > (15 // frame_skip_step)
                    ):
                        swing_state = 0
                        current_swing_trajectory = []

                elif swing_state == 2:
                    if current_bat_head:
                        current_swing_trajectory.append(current_bat_head)

                    swing_frames_data.append(
                        {
                            "frame": frame_idx,
                            "wrist": current_wrist,
                            "bat_head": current_bat_head,
                            "speed": current_speed,
                        }
                    )

                    if (
                        current_speed <= start_trigger_speed * 0.8
                        or len(swing_frames_data) > (40 // frame_skip_step)
                    ):
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
                            current_swing_trajectory = []
                            cooldown_counter = int((fps * 0.3) // frame_skip_step)

                # 繪製棒頭軌跡 (亮黃色)
                if len(current_swing_trajectory) > 1:
                    pts = np.array(current_swing_trajectory, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated_frame,
                        [pts],
                        isClosed=False,
                        color=(0, 255, 255),
                        thickness=3,
                    )

                if swing_state in [1, 2]:
                    swing_raw_frames.append(annotated_frame.copy())

                # 結算與建立數據表
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
                        b_x = (
                            item["bat_head"][0] if item["bat_head"] else None
                        )
                        b_y = (
                            item["bat_head"][1] if item["bat_head"] else None
                        )
                        frame_logs.append(
                            {
                                "相對時間 (秒)": round(rel_time, 2),
                                "即時速度 (km/h)": round(item["speed"], 1),
                                "手腕 X": w_x,
                                "手腕 Y": w_y,
                                "棒頭 X": b_x,
                                "棒頭 Y": b_y,
                            }
                        )
                    detailed_df = pd.DataFrame(frame_logs)

                    swing_num = len(st.session_state.swing_events) + 1
                    clip_filename = os.path.join(
                        clip_dir, f"swing_{swing_num}.webm"
                    )
                    actual_clip_path = save_swing_clip(
                        swing_raw_frames,
                        fps / frame_skip_step,
                        proc_width,
                        proc_height,
                        clip_filename,
                    )

                    event_data = {
                        "次數": f"第 {swing_num} 次揮棒",
                        "初速": f"{max_speed_in_swing:.1f} km/h",
                        "仰角": f"{launch_angle:.1f}°",
                        "耗時": f"{((len(swing_frames_data) * frame_skip_step) / fps):.2f} 秒",
                        "total_frames": len(swing_frames_data),
                        "clip_path": actual_clip_path,
                        "detailed_df": detailed_df,
                    }
                    st.session_state.swing_events.append(event_data)

                    swing_state = 0
                    current_swing_trajectory = []
                    cooldown_counter = int((fps * 0.8) // frame_skip_step)

            # 畫面資訊標記
            cv2.putText(
                annotated_frame,
                f"STATE: {swing_state} | SPEED: {current_speed:.1f} km/h",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # [建議 3: UI 畫面渲染控制] 減少畫面刷新的 Overhead (每2次採樣才更新一次預覽畫面)
            if frame_idx % (frame_skip_step * 2) == 0:
                frame_display = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_display = np.ascontiguousarray(frame_display, dtype=np.uint8)
                st_frame.image(
                    frame_display,
                    channels="RGB",
                    use_container_width=True,
                )

    cap.release()

    # ==============================================================================
    # 4. 獨立揮棒數據表格與短片回放區域
    # ==============================================================================
    if st.session_state.swing_events:
        events = st.session_state.swing_events
        st.success(f"✅ 影片分析完成！一共偵測出 {len(events)} 次有效揮棒。")

        option_list = [e["次數"] for e in events]
        selected_swing_name = st.selectbox(
            "🎯 請選擇要查看與回放的揮棒次數：",
            options=option_list,
            index=len(option_list) - 1,
        )

        selected_event = next(
            (e for e in events if e["次數"] == selected_swing_name), events[0]
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("⚡ 估算初速", selected_event["初速"])
        col2.metric("📐 預測仰角", selected_event["仰角"])
        col3.metric("⏱️ 揮棒耗時", selected_event["耗時"])
        col4.metric("🎞️ 記錄點數", f"{selected_event['total_frames']} 點")

        tab1, tab2 = st.tabs(["🎬 慢動作影片回放", "📊 該次揮棒獨立數據表"])

        with tab1:
            st.markdown(f"#### 🎬 {selected_event['次數']} 慢動作回放")
            if os.path.exists(selected_event["clip_path"]):
                st.video(selected_event["clip_path"])

        with tab2:
            st.markdown(f"#### 📋 {selected_event['次數']} - 逐幀速度與軌跡詳細數據表")
            st.dataframe(
                selected_event["detailed_df"],
                use_container_width=True,
                height=300,
            )
            csv_data = selected_event["detailed_df"].to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label=f"📥 下載 {selected_event['次數']} 數據明細 (CSV)",
                data=csv_data,
                file_name=f"{selected_event['次數']}_detail.csv",
                mime="text/csv",
            )

        st.markdown("---")
        with st.expander("📊 查看所有歷史揮棒數據彙整表"):
            df_all = pd.DataFrame(events)[["次數", "初速", "仰角", "耗時", "total_frames"]]
            df_all.columns = ["次數", "初速", "仰角", "耗時", "記錄點數"]
            st.dataframe(df_all, use_container_width=True)