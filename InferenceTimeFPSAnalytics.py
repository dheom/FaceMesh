# 1. Inference Time (추론 시간)
# 지표: 평균 추론 시간, 지연 시간(Latency)
# 설명: 모델이 단일 입력에 대해 결과를 생성하는 데 걸리는 시간을 측정합니다. 다양한 입력 데이터 크기와 복잡성에 대해 측정할 수 있습니다.
# 2. Throughput (처리량)
# 지표: 초당 추론 횟수 (Inference per second, IPS), 초당 프레임 수 (Frames per second, FPS)
# 설명: 모델이 주어진 시간 동안 얼마나 많은 입력을 처리할 수 있는지를 측정합니다.
import numpy as np
import time
from openvino.runtime import Core

# OpenVINO 모델 경로
openvino_xml_path = "converted_model/face_landmark.xml"
openvino_bin_path = "converted_model/face_landmark.bin"

# OpenVINO Core 초기화
ie = Core()
model = ie.read_model(model=openvino_xml_path, weights=openvino_bin_path)
compiled_model = ie.compile_model(model=model, device_name="CPU")
infer_request = compiled_model.create_infer_request()

# 입력 데이터 준비 (랜덤 데이터 예시)
input_height = 192
input_width = 192
input_channels = 3
input_data = np.random.rand(1, input_height, input_width, input_channels).astype(np.float32)

# 추론 시간 측정
num_iterations = 10000
start_time = time.time()
for _ in range(num_iterations):
    infer_request.infer(inputs={compiled_model.input(0): input_data})
end_time = time.time()

average_inference_time = (end_time - start_time) / num_iterations
throughput = num_iterations / (end_time - start_time)

print(f"OpenVINO 모델 평균 추론 시간: {average_inference_time} 초")
print(f"OpenVINO 모델 처리량: {throughput} 추론/초")

import numpy as np
import time
import tensorflow as tf

# TensorFlow Lite 모델 경로
tflite_model_path = ".venv/Lib/site-packages/mediapipe/modules/face_landmark/face_landmark.tflite"

# TensorFlow Lite Interpreter 초기화
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 입력 데이터 준비 (랜덤 데이터 예시)
input_height = 192
input_width = 192
input_channels = 3
input_data = np.random.rand(1, input_height, input_width, input_channels).astype(np.float32)

# 추론 시간 측정
num_iterations = 10000
start_time = time.time()
for _ in range(num_iterations):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
end_time = time.time()

average_inference_time = (end_time - start_time) / num_iterations
throughput = num_iterations / (end_time - start_time)

print(f"TensorFlow Lite 모델 평균 추론 시간: {average_inference_time} 초")
print(f"TensorFlow Lite 모델 처리량: {throughput} 추론/초")
