# import openvino as ov
# import os
#
# # 입력과 출력 경로 설정
# tflite_model_path = "C:\\Users\\USER\\PycharmProjects\\pythonProject\\.venv\\Lib\\site-packages\\mediapipe\\modules\\face_landmark\\face_landmark.tflite"
# ov_model_dir = "converted_model"
# ov_model_name = "face_landmark"
#
# # 모델 변환
# ov_model = ov.convert_model(tflite_model_path)
#
# # 변환된 모델 저장 디렉토리 생성
# os.makedirs(ov_model_dir, exist_ok=True)
#
# # .xml 파일 경로 설정
# ov_xml_path = os.path.join(ov_model_dir, ov_model_name + ".xml")
# # .bin 파일 경로 설정
# ov_bin_path = os.path.join(ov_model_dir, ov_model_name + ".bin")
#
# # 변환된 모델 저장
# ov.save_model(ov_model, ov_xml_path)
#
# print(f"Model {tflite_model_path} successfully converted and saved to {ov_xml_path}")

#파일 Openvino 최적화 성공

import numpy as np
import time
from openvino.runtime import Core

# OpenVINO 모델 경로
xml_path = "converted_model/face_landmark.xml"

# OpenVINO Core 초기화
ie = Core()
model = ie.read_model(model=xml_path)
compiled_model = ie.compile_model(model=model, device_name="CPU")
infer_request = compiled_model.create_infer_request()

# 입력 데이터 준비 (랜덤 데이터 예시, 모델의 입력 형식에 맞춤)
input_height = 192
input_width = 192
input_channels = 3
input_data = np.random.rand(1, input_height, input_width, input_channels).astype(np.float32)

# 입력 레이어 이름 가져오기
input_layer = compiled_model.input(0)

# 추론 실행 및 시간 측정
start_time = time.time()
infer_request.infer({input_layer: input_data})
inference_time = time.time() - start_time

print(f"OpenVINO 모델 추론 시간: {inference_time} 초")

import tensorflow as tf
import numpy as np
import time
import mediapipe as mp

# TensorFlow Lite 모델 경로
tflite_model_path = ".venv/Lib/site-packages/mediapipe/modules/face_landmark/face_landmark.tflite"

# TensorFlow Lite Interpreter 초기화
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# 입력 데이터 준비 (랜덤 데이터 예시, 모델의 입력 형식에 맞춤)
input_height = 192
input_width = 192
input_channels = 3
input_data = np.random.rand(1, input_height, input_width, input_channels).astype(np.float32)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], input_data)

# 추론 실행 및 시간 측정
start_time = time.time()
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
inference_time = time.time() - start_time

print(f"TensorFlow Lite 모델 추론 시간: {inference_time} 초")
