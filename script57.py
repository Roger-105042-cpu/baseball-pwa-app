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
# 0. 頁面設定與 MediaPipe 模型自動下載機制 (修復 FileNotFoundError)
# ==============================================================================
st.set_page_config(
    page_title="棒球揮棒動作姿態與初速分析系統",
    page_icon="⚾",
    layout="wide",
)

MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"


@st.cache_resource
def ensure_model_file():
    """自動檢測並下載 MediaPipe 缺失的模型檔"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ 首次於雲端執行，正在從官方下載 MediaPipe 模型檔..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# 執行下載檢測 (只會在伺服器缺少模型時自動下載 1 次)
ensure_model_file()

# 系統與演算預設參數
meters_per_pixel = 0.0025  # 像素轉公尺比例
bat_speed_factor = 1.35  # 手腕轉棒頭速度放大係數
min_peak_speed = 35.0  # 判定為有效揮棒的最低峰值速度 (km/h)
min_x_travel = 45  # 手腕最低 X 軸位移量 (px)
bat_length_px = 110  # 棒頭延伸像素長度
frame_skip_step = 1  # 逐幀分析 (設為 1 精度最高)
target_width = 800  # 圖像縮放最大寬度

st.title("⚾ 棒球揮棒動作姿態與初速分析系統")
st.caption("基於 MediaPipe Pose Landmarker 之 AI 骨架偵測與慢動作回放系統")


# ==============================================================================
# 1. 輔助運算與影片剪輯函式
# ==============================================================================
def process_frame(frame, pose_landmarks, width, height, bat_length):
    """提取 MediaPipe 關鍵點並繪製骨架與延伸棒頭"""
    if not pose_landmarks:
        return None, None, None, None

    landmarks = pose_landmarks[0]

    def get_coords(idx):
        if idx < len(landmarks):
            lm = landmarks[idx]
            return int(lm.x * width), int(lm.y * height)
        return None

    # Pose 關鍵點索引: 11(左肩), 12(右肩), 15(左手腕), 16(右手腕)
    l_shoulder = get_coords(11)
    r_shoulder = get_coords(12)
    l_wrist = get_coords(15)
    r_wrist = get_coords(16)

    # 繪製肩膀與手部骨架
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

    # 估算棒頭位置
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

    shoulder_mid = (
        (l_shoulder[0] + r_shoulder[0]) / 2
        if (l_shoulder and r_shoulder)
        else (0, 0)
    )
    return shoulder_mid, wrist_center, bat_head, shoulder_mid


def save_swing_clip(frames, fps, width, height, output_path):
    """將揮棒片段寫入 WebM 影音檔"""
    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for f in frames:
        out.write(f)
    out.release()
    return output_path


# ==============================================================================
# 2. Streamlit 檔案上傳與 Session 狀態初始化
# ==============================================================================
uploaded_file = st.file_uploader(
    "📁 選擇或上傳揮棒影片", type=["mp4", "avi", "mov", "m4v"]
)

if uploaded_file is not None:
    # 建立檔案獨有識別碼，防範切換同名檔案時誤重置
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"

    if (
        "current_file_key" not in st.session_state
        or st.session_state.current_file_key != file_key
    ):
        st.session_state.current_file_key = file_key
        st.session_state.swing_events = []
        st.session_state.is_analyzed = False

    with st.sidebar:
        st.header("⚙️ 控制選項")
        if st.button("🔄 重新分析影片", use_container_width=True):
            st.session_state.is_analyzed = False
            st.session_state.swing_events = []
            st.rerun()

    # ==========================================================================
    # 3. 核心運算區 (僅在「未分析過」時跑 MediaPipe，解決選單切換當機問題)
    # ==========================================================================
    if not st.session_state.is_analyzed:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 比例縮放控制
        if orig_width > target_width:
            scale = target_width / float(orig_width)
            proc_width = target_width
            proc_height = int(orig_height * scale)
        else:
            scale = 1.0
            proc_width = orig_width
            proc_height = orig_height

        st.markdown("### 📹 全程動態分析預覽中...")
        st_frame = st.empty()
        progress_bar = st.progress(0)

        history_wrist = []
        current_swing_trajectory = []
        swing_state = 0  # 0: 準備, 1: 揮棒中, 2: 減速中, 3: 觸發結算
        swing_frames_data = []
        swing_raw_frames = []
        max_speed_in_swing = 0.0
        peak_frame_in_swing = 0
        cooldown_counter = 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        # 啟動 Pose Landmarker 任務
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

                if frame_skip_step > 1 and (frame_idx % frame_skip_step != 0):
                    continue

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

                start_trigger_speed = min_peak_speed * 0.4

                # 揮棒觸發邏輯判定
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
                            and len(swing_frames_data) > (15 // frame_skip_step)
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
                                cooldown_counter = int(
                                    (fps * 0.3) // frame_skip_step
                                )

                    # 畫黃色揮棒軌跡
                    if len(current_swing_trajectory) > 1:
                        pts = np.array(
                            current_swing_trajectory, np.int32
                        ).reshape((-1, 1, 2))
                        cv2.polylines(
                            annotated_frame, [pts], False, (0, 255, 255), 3
                        )

                    if swing_state in [1, 2]:
                        swing_raw_frames.append(annotated_frame.copy())

                    # 結算本次揮棒
                    if swing_state == 3:
                        launch_angle = 0.0
                        peak_sub_idx = [
                            i
                            for i, item in enumerate(swing_frames_data)
                            if item["frame"] == peak_frame_in_swing
                        ]
                        if peak_sub_idx:
                            p_idx = peak_sub_idx[0]
                            post_idx = min(
                                len(swing_frames_data) - 1, p_idx + 2
                            )
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
                            fps / frame_skip_step,
                            proc_width,
                            proc_height,
                            clip_filename,
                        )

                        # 將二進位檔傳入 Session，避開雲端檔案權限與死鎖瓶頸
                        with open(actual_clip_path, "rb") as vf:
                            video_bytes = vf.read()

                        detected_events.append({
                            "次數": f"第 {swing_num} 次揮棒",
                            "初速": f"{max_speed_in_swing:.1f} km/h",
                            "仰角": f"{launch_angle:.1f}°",
                            "耗時": f"{((len(swing_frames_data) * frame_skip_step) / fps):.2f} 秒",
                            "total_frames": len(swing_frames_data),
                            "video_bytes": video_bytes,
                            "detailed_df": detailed_df,
                        })

                        swing_state = 0
                        current_swing_trajectory = []
                        cooldown_counter = int((fps * 0.8) // frame_skip_step)

                # 即時渲染預覽
                if frame_idx % (frame_skip_step * 3) == 0:
                    frame_display = cv2.cvtColor(
                        annotated_frame, cv2.COLOR_BGR2RGB
                    )
                    st_frame.image(
                        frame_display, channels="RGB", use_container_width=True
                    )

        cap.release()
        st_frame.empty()
        progress_bar.empty()

        # 鎖定狀態並觸發 UI 更新
        st.session_state.swing_events = detected_events
        st.session_state.is_analyzed = True
        st.rerun()

    # ==========================================================================
    # 4. 揮棒結果展示區 (0 秒即時切換區)
    # ==========================================================================
    if st.session_state.swing_events:
        events = st.session_state.swing_events
        st.success(f"✅ 影片分析完成！一共偵測出 {len(events)} 次有效揮棒。")

        option_list = [e["次數"] for e in events]
        selected_swing_name = st.selectbox(
            "🎯 請選擇要查看與回放的揮棒次數：",
            options=option_list,
            index=len(option_list) - 1,
            key="swing_selector",
        )

        selected_event = next(
            (e for e in events if e["次數"] == selected_swing_name), events[0]
        )

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("⚡ 估算初速", selected_event["初速"])
        col2.metric("📐 預測仰角", selected_event["仰角"])
        col3.metric("⏱️ 揮棒耗時", selected_event["耗時"])
        col4.metric("🎞️ 記錄點數", f"{selected_event['total_frames']} 點")

        tab1, tab2 = st.tabs(["🎬 慢動作影片回放", "📊 該次揮棒獨立數據表"])

        with tab1:
            st.markdown(f"#### 🎬 {selected_event['次數']} 慢動作回放")
            st.video(selected_event["video_bytes"], format="video/webm")

        with tab2:
            st.markdown(
                f"#### 📋 {selected_event['次數']} - 逐幀速度與軌跡詳細數據表"
            )
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
                label=f"📥 下載 {selected_event['次數']} 數據明細 (CSV)",
                data=csv_data,
                file_name=f"{selected_event['次數']}_detail.csv",
                mime="text/csv",
            )

        st.markdown("---")
        with st.expander("📊 查看所有歷史揮棒數據彙整表"):
            df_all = pd.DataFrame(events)[
                ["次數", "初速", "仰角", "耗時", "total_frames"]
            ]
            df_all.columns = ["次數", "初速", "仰角", "耗時", "記錄點數"]
            st.dataframe(df_all, use_container_width=True)