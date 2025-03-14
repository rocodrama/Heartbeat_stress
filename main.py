# import the necessary packages
from imutils import face_utils
import dlib
import cv2
import numpy as np
import time
import math

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

# 심박수를 계산하는 함수 (얼굴의 색상 변화를 기반으로)
def estimate_heart_rate(roi):
    # ROI에서 평균 색상 계산 (대체 방법)
    avg_color = np.mean(roi, axis=(0, 1))
    heart_rate = avg_color[1] * 0.1  # 임시 방식 (보정이 필요)
    return heart_rate

# HRV 계산 함수
def calculate_hrv(rr_intervals):
    rr_intervals = np.array(rr_intervals)
    rr_intervals = rr_intervals[~np.isnan(rr_intervals)]  # NaN 값 제거
    if len(rr_intervals) > 1:
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        return sdnn, rmssd
    return None, None

# 얼굴을 추적하고 심박수를 계산하는 메인 루프
rr_intervals = []  # 심박수 간격 리스트
prev_heart_rate = None
start_time = time.time()

# 초기 sdnn, rmssd 값 정의 (없을 경우 에러 방지)
sdnn, rmssd = None, None

# 창 크기 한 번만 설정
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)  # 창 크기 조정 가능하도록 설정
cv2.resizeWindow("Output", 1920, 1080)  # 창 크기 설정

while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 얼굴 ROI (Region of Interest) 정의
        x, y, w, h = (rect.left(), rect.top(), rect.width(), rect.height())
        roi = image[y:y+h, x:x+w]
        
        # 심박수 추정
        heart_rate = estimate_heart_rate(roi)

        if prev_heart_rate is not None:
            # 심박수 간격 계산 (시간 차이 기반)
            rr_intervals.append(time.time() - start_time)

        prev_heart_rate = heart_rate
        start_time = time.time()

        # 얼굴 랜드마크를 이미지에 그리기
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    # HRV 계산
    if len(rr_intervals) > 1:
        sdnn, rmssd = calculate_hrv(rr_intervals)
        if sdnn is not None and rmssd is not None:
            cv2.putText(image, f"HRV (SDNN): {sdnn:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"HRV (RMSSD): {rmssd:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 스트레스 지수 추정 (HRV가 낮을수록 스트레스가 높음)
    if sdnn is not None and rmssd is not None:
        stress_index = (sdnn + rmssd) / 2
        cv2.putText(image, f"Stress Index: {stress_index:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # # show the output image with the face detections + facial landmarks
    # cv2.imshow("Output", image)

   # 비디오 프레임 크기 조정 (이미지를 창 크기에 맞게 크기 변경)
    image_resized = cv2.resize(image, (1920, 1080))  # 원하는 크기로 변경

    # 출력 이미지 크기 조정된 것을 보여줍니다.
    cv2.imshow("Output", image_resized)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
