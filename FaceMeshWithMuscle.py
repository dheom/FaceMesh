import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh
faceMesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Variables for blink detection
blink_counter = 0
total_blinks = 0
start_time = time.time()
last_blink_time = start_time

# Lists to store average values
avg_muscle_activations = []
avg_blinks_per_minute = []
time_stamps = []

# Eye landmarks indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Face muscles indices
FACE_MUSCLES = {
    "Left Eye Corner": [33, 133],
    "Right Eye Corner": [362, 263],
    "Left Mouth Corner": [61, 291],
    "Right Mouth Corner": [291, 321]
}


def calculate_ear(eye_landmarks, face_landmarks):
    # EAR(Eye Aspect Ratio) 계산 함수
    # 눈의 세로 길이와 가로 길이의 비율을 계산하여 눈의 열림/닫힘 상태를 판단
    p1 = face_landmarks.landmark[eye_landmarks[0]]
    p2 = face_landmarks.landmark[eye_landmarks[8]]
    p3 = face_landmarks.landmark[eye_landmarks[4]]
    p4 = face_landmarks.landmark[eye_landmarks[12]]

    ear = (abs(p2.y - p4.y) + abs(p1.y - p3.y)) / (2 * abs(p1.x - p3.x))
    return ear


def calculate_muscle_activity(landmarks, reference, face_size):
    current_distance = np.linalg.norm(np.array([landmarks[0].x, landmarks[0].y]) -
                                      np.array([landmarks[1].x, landmarks[1].y]))
    normalized_distance = current_distance / face_size
    normalized_reference = reference / face_size
    activity = abs((normalized_distance - normalized_reference) / normalized_reference) * 100
    return min(activity, 100)  # Cap activity at 100%


def calculate_relative_muscle_activity(landmarks, reference_distance):
    current_distance = np.linalg.norm(np.array([landmarks[0].x, landmarks[0].y]) -
                                      np.array([landmarks[1].x, landmarks[1].y]))
    activity = abs((current_distance - reference_distance) / reference_distance) * 100
    return min(activity, 100)  # Cap activity at 100%


def draw_egg_shape_guide(img):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    egg_width = int(width * 0.3)
    egg_height = int(height * 0.45)

    cv2.ellipse(img, center, (egg_width // 2, egg_height // 2),
                0, 0, 360, (0, 255, 0), 2)

    return img


def put_text_with_background(img, text, position, font, font_scale, text_color, bg_color):
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, 1)
    cv2.rectangle(img, position, (position[0] + text_width, position[1] + text_height), bg_color, -1)
    cv2.putText(img, text, (position[0], position[1] + text_height), font, font_scale, text_color, 1, cv2.LINE_AA)


reference_distances = {}

while True:
    ret, img = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(img_rgb)

    img = draw_egg_shape_guide(img)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_size = np.linalg.norm(
                np.array([face_landmarks.landmark[234].x, face_landmarks.landmark[234].y]) -
                np.array([face_landmarks.landmark[454].x, face_landmarks.landmark[454].y])
            )

            mp_draw.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                   mp_draw.DrawingSpec((0, 255, 0), 1, 1))

            left_ear = calculate_ear(LEFT_EYE, face_landmarks)
            right_ear = calculate_ear(RIGHT_EYE, face_landmarks)
            ear = (left_ear + right_ear) / 2

            if ear < 0.2 and time.time() - last_blink_time > 0.5:
                blink_counter += 1
                total_blinks += 1
                last_blink_time = time.time()

            total_activity = 0
            for muscle, landmark_indices in FACE_MUSCLES.items():
                lm1 = face_landmarks.landmark[landmark_indices[0]]
                lm2 = face_landmarks.landmark[landmark_indices[1]]

                if muscle not in reference_distances:
                    reference_distances[muscle] = np.linalg.norm(np.array([lm1.x, lm1.y]) -
                                                                 np.array([lm2.x, lm2.y]))

                activity = calculate_relative_muscle_activity([lm1, lm2], reference_distances[muscle])
                total_activity += activity

            avg_activity = total_activity / len(FACE_MUSCLES)

            avg_muscle_activations.append(avg_activity)

            text = f"Avg Muscle Activation: {avg_activity:.1f}%"
            put_text_with_background(img, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), (255, 255, 255))

    elapsed_time = time.time() - start_time
    blinks_per_minute = (total_blinks / elapsed_time) * 60

    avg_blinks_per_minute.append(blinks_per_minute)
    time_stamps.append(elapsed_time)

    put_text_with_background(img, f"Blinks: {blink_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                             (255, 255, 255))
    put_text_with_background(img, f"Avg Blinks/Min: {blinks_per_minute:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                             (0, 0, 0), (255, 255, 255))

    cv2.imshow("Face Mesh", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Ensure the lists have the same length
min_length = min(len(time_stamps), len(avg_muscle_activations), len(avg_blinks_per_minute))
time_stamps = time_stamps[:min_length]
avg_muscle_activations = avg_muscle_activations[:min_length]
avg_blinks_per_minute = avg_blinks_per_minute[:min_length]

# Calculate overall averages
overall_avg_muscle_activation = int(np.mean(avg_muscle_activations))
overall_avg_blinks_per_minute = int(np.mean(avg_blinks_per_minute))

# Print overall averages
print(f"Overall Average Muscle Activation: {overall_avg_muscle_activation}%")
print(f"Overall Average Blinks Per Minute: {overall_avg_blinks_per_minute}")

# Plotting the average values over time
plt.figure(figsize=(10, 5))

# Plotting muscle activation
plt.subplot(1, 2, 1)
plt.plot(time_stamps, avg_muscle_activations, label="Muscle Activation")
plt.xlabel("Time (seconds)")
plt.ylabel("Activation (%)")
plt.title("Average Muscle Activation Over Time")
plt.legend()

# Plotting blinks per minute
plt.subplot(1, 2, 2)
plt.plot(time_stamps, avg_blinks_per_minute, label="Blinks Per Minute", color='orange')
plt.xlabel("Time (seconds)")
plt.ylabel("Blinks/Min")
plt.title("Average Blinks Per Minute Over Time")
plt.legend()

plt.tight_layout()
plt.show()
