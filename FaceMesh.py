import mediapipe as mp
import cv2
#
#
# mp_face_mesh = mp.solutions.face_mesh
# faceMesh = mp_face_mesh.FaceMesh()
# mp_draw = mp.solutions.drawing_utils
# cap = cv2.VideoCapture(0)
#
#
# while True:
#     ret, img = cap.read()
#
#     results = faceMesh.process(img)
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             mp_draw.draw_landmarks(img,face_landmarks,mp_face_mesh.FACEMESH_CONTOURS,
#                                    mp_draw.DrawingSpec((0,255,0),1,1))
#
#     print(results.multi_face_landmarks)
#     cv2.imshow("Face Mesh",img)
#     cv2.waitKey(1)

import time

mp_face_mesh = mp.solutions.face_mesh
faceMesh = mp_face_mesh.FaceMesh()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# 눈 깜빡임 감지를 위한 변수들
blink_counter = 0
total_blinks = 0
start_time = time.time()
last_blink_time = start_time

# 눈 랜드마크 인덱스
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]


#눈 작은사람은 눈꺼플 움직임 변화
def calculate_eyelid_movement(eye_landmarks, face_landmarks):
    p1 = face_landmarks.landmark[eye_landmarks[2]]
    p2 = face_landmarks.landmark[eye_landmarks[4]]
    return abs(p1.y - p2.y)

def calculate_ear(eye_landmarks, face_landmarks):
    # EAR(Eye Aspect Ratio) 계산 함수
    # 눈의 세로 길이와 가로 길이의 비율을 계산하여 눈의 열림/닫힘 상태를 판단
    p1 = face_landmarks.landmark[eye_landmarks[0]]
    p2 = face_landmarks.landmark[eye_landmarks[8]]
    p3 = face_landmarks.landmark[eye_landmarks[4]]
    p4 = face_landmarks.landmark[eye_landmarks[12]]

    ear = (abs(p2.y - p4.y) + abs(p1.y - p3.y)) / (2 * abs(p1.x - p3.x))
    return ear


while True:
    ret, img = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                   mp_draw.DrawingSpec((0, 255, 0), 1, 1))

            left_ear = calculate_ear(LEFT_EYE, face_landmarks)
            right_ear = calculate_ear(RIGHT_EYE, face_landmarks)
            ear = (left_ear + right_ear) / 2 #평균 EAR

            # 눈 깜빡임 감지 (EAR 임계값 조정 필요)0.2는 threshold
            if ear < 0.2 and time.time() - last_blink_time > 0.5:  # 최소 0.1초 간격
                blink_counter += 1
                total_blinks += 1
                last_blink_time = time.time()

            # # 메인 루프 내에서
            # left_movement = calculate_eyelid_movement(LEFT_EYE, face_landmarks)
            # right_movement = calculate_eyelid_movement(RIGHT_EYE, face_landmarks)
            # movement = (left_movement + right_movement) / 2
            #
            # if movement < 0.2:
            #     blink_counter += 1
            #     total_blinks += 1
            #     last_blink_time = time.time()

    # 경과 시간 계산
    elapsed_time = time.time() - start_time

    # 1분당 평균 깜빡임 횟수 계산
    blinks_per_minute = (total_blinks / elapsed_time) * 60

    # 화면에 정보 표시
    cv2.putText(img, f"Blinks: {blink_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Avg Blinks/Min: {blinks_per_minute:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Mesh", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

