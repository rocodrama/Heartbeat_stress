import cv2
import dlib
import numpy as np
from imutils import face_utils
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks
import time

# dlib 얼굴 감지 및 랜드마크 예측기 초기화
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# 웹캠 캡처 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# 실제 FPS 감지
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
if fps <= 0 or fps is None:  
    fps = 30  # 기본값 설정 (웹캠이 FPS 정보를 제공하지 않는 경우)

# 녹색 채널 신호와 타임스탬프를 저장할 버퍼 (약 10초 분량)
buffer_length = int(fps * 10)  # FPS 기반으로 10초 데이터 저장
green_buffer = deque(maxlen=buffer_length)
time_buffer = deque(maxlen=buffer_length)

# 밴드패스 필터 적용 함수
def bandpass_filter(data, lowcut=0.75, highcut=4, fs=30, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

start_time = time.time()

# 창 크기 조정 가능하도록 설정
cv2.namedWindow("Face, HR, HRV & Stress Measurement", cv2.WINDOW_NORMAL) 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) > 0:
        rect = rects[0]  # 첫 번째 얼굴 사용
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 얼굴 랜드마크 표시
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # 얼굴 윤곽선을 Convex Hull로 생성
        hull = cv2.convexHull(shape)

        # 마스크 생성 (배경 제거)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        # 녹색 채널에서 얼굴 내부 영역만 추출
        green_channel = frame[:, :, 1]  # 녹색 채널
        face_green = cv2.bitwise_and(green_channel, green_channel, mask=mask)

        # 평균 녹색 신호 계산 (배경 제외)
        mean_green = np.mean(face_green[mask == 255])
        current_time = time.time() - start_time
        green_buffer.append(mean_green)
        time_buffer.append(current_time)

        # 얼굴 윤곽선 표시
        cv2.polylines(frame, [hull], isClosed=True, color=(255, 0, 0), thickness=2)

    
    # 데이터가 충분히 쌓였을 때 심박수, HRV, 스트레스 지수 계산
    if len(green_buffer) == buffer_length:
        green_signal = np.array(green_buffer)
        t_signal = np.array(time_buffer)

        # 실제 FPS 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        # print(fps)
        if fps == 0 or fps is None:
            fps = 30  # 기본값 설정

        # 필터 적용
        filtered_signal = bandpass_filter(green_signal, fs=fps)

        # 피크 검출
        # 피크 감지
        expected_hr = 60  # 예상 평균 심박수
        expected_peak_distance = int(fps * 60 / expected_hr)  # 프레임 속도에 맞춰 조정

        # distance 값을 적절히 조정 (ex. 심박수가 40~120 BPM 사이에 맞게)
        peaks, _ = find_peaks(filtered_signal, distance=int(fps * 60 / 120), height=np.mean(filtered_signal) * 0.5)

        if len(peaks) > 1:
            # 심박수 계산
            duration = t_signal[-1] - t_signal[0]
            bpm = len(peaks) / duration * 60

            # NN 간격 계산
            nn_intervals = np.diff(t_signal[peaks])

            # IQR (Interquartile Range) 방법을 이용해 이상값 제거
            Q1 = np.percentile(nn_intervals, 25)
            Q3 = np.percentile(nn_intervals, 75)
            IQR = Q3 - Q1

            # 정상 범위 설정 (1.5 * IQR 범위 내)
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 이상값 제거
            filtered_nn_intervals = nn_intervals[(nn_intervals >= lower_bound) & (nn_intervals <= upper_bound)]

            # 이상값 제거 후 NN 간격이 남아있을 때만 계산
            if len(filtered_nn_intervals) > 1:    

                # SDNN 계산
                sdnn = np.std(nn_intervals)
                rmssd = np.sqrt(np.mean(np.square(np.diff(nn_intervals)))) if len(nn_intervals) > 1 else 0

                # 스트레스 지수 계산
                bin_width = 0.05
                bins = np.arange(0, np.max(nn_intervals) + bin_width, bin_width)
                counts, bin_edges = np.histogram(nn_intervals, bins=bins)
                max_bin_index = np.argmax(counts)
                AMo = (counts[max_bin_index] / len(nn_intervals)) * 100
                Mo = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
                MxDM = np.max(nn_intervals) - np.min(nn_intervals)
                # stress_index = AMo / (2 * Mo * MxDM) if Mo * MxDM != 0 else 0

                # 스트레스 지수 계산 시 보정값
                epsilon = 1e-6  # 너무 작은 값 방지
                # Mo가 너무 작으면 NN 간격 중앙값으로 대체
                Mo = Mo if Mo > 0 else np.median(nn_intervals)
                # MxDM 최소값 설정
                MxDM = np.max(nn_intervals) - np.min(nn_intervals)
                MxDM = MxDM if MxDM > 0.05 else 0.05  # 최소 50ms
                
                stress_index = AMo / (2 * Mo * max(MxDM, epsilon))

                # 스트레스 위험도 판별
                if stress_index <= 50:
                    stress_level = "Low"
                    stress_color = (0, 255, 0)  # 초록색
                elif stress_index <= 150:
                    stress_level = "Normal"
                    stress_color = (0, 255, 255)  # 노란색
                elif stress_index <= 300:
                    stress_level = "High"
                    stress_color = (0, 165, 255)  # 주황색
                else:
                    stress_level = "Very High"
                    stress_color = (0, 0, 255)  # 빨간색

                # 결과 출력
                cv2.putText(frame, f"HR: {bpm:.1f} BPM", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"SDNN: {sdnn:.3f} s", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"RMSSD: {rmssd:.3f} s", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"Stress Index: {stress_index:.3f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, stress_color, 2)
                cv2.putText(frame, f"Stress Level: {stress_level}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, stress_color, 2)

                # print(f"NN intervals (in seconds): {nn_intervals}")
                # print(f"Mo (Most frequent NN interval): {Mo}")
                # print(f"MxDM (Max NN - Min NN): {MxDM}")
                # print(f"AMo (Amplitude of Mode): {AMo}")

    else:
        cv2.putText(frame, "Measuring...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # # 영상 출력
    # # cv2.imshow("Face, HR, HRV & Stress Measurement", frame)

    # 화면 크기에 맞춰 비율 유지하면서 조정
    height, width = frame.shape[:2]
    window_height, window_width = 1080, 1920  # 원하는 윈도우 크기
    scale = min(window_width / width, window_height / height)
    resized_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))


    cv2.imshow("Face, HR, HRV & Stress Measurement", resized_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
        break

cap.release()
cv2.destroyAllWindows()
