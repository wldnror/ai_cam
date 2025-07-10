#!/usr/bin/env python3
from flask import Flask, Response, render_template
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2 import MappedArray
import cv2
import numpy as np
from functools import lru_cache

# — IMX500 모델 로드 & 인트린직스 가져오기 —
MODEL_PATH = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
imx500 = IMX500(MODEL_PATH)
intrinsics = imx500.network_intrinsics
intrinsics.update_with_defaults()

# — Flask & Picamera2 초기화 —
app = Flask(__name__)
picam2 = Picamera2(imx500.camera_num)

# — 카메라 설정: default preview configuration 사용 (post-processing 포함) —
config = picam2.create_preview_configuration(
    controls={"FrameRate": intrinsics.inference_rate},
    buffer_count=12
)  #  [oai_citation:0‡GitHub](https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_classification_demo.py?utm_source=chatgpt.com)
imx500.show_network_fw_progress_bar()
picam2.configure(config)
picam2.start()

# — 객체 탐지 결과 파싱 함수 —  [oai_citation:1‡GitHub](https://raw.githubusercontent.com/raspberrypi/picamera2/main/examples/imx500/imx500_object_detection_demo.py)
last_results = []
def parse_detections(metadata: dict):
    global last_results
    outputs = imx500.get_outputs(metadata, add_batch=True)
    if outputs is None:
        return last_results
    # outputs 구조: [boxes, scores, classes]
    boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]
    # 정규화 해제 & 좌표 순서 교정
    if intrinsics.bbox_normalization:
        boxes = boxes * imx500.get_input_size()[::-1]
    if intrinsics.bbox_order == "xy":
        boxes = boxes[:, [1,0,3,2]]
    last_results = []
    for box, score, cls in zip(boxes, scores, classes):
        if score < 0.55:  # threshold
            continue
        x0, y0, x1, y1 = box.astype(int)
        last_results.append((int(cls), float(score), (x0, y0, x1, y1)))
    return last_results

# — 웹 스트리밍용 프레임 생성기 — 
def gen_frames():
    while True:
        # 1) 메타데이터로부터 최신 탐지 결과 가져오기
        metadata = picam2.capture_metadata()
        detections = parse_detections(metadata)

        # 2) 원본 프레임 가져오기
        frame = picam2.capture_array("main")

        # 3) 탐지 결과를 프레임에 그리기
        for cls, score, (x0, y0, x1, y1) in detections:
            label = f"{intrinsics.labels[cls]} {score:.2f}"
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0,255,0), 2)
            cv2.putText(frame, label, (x0, y0-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # 4) JPEG 인코딩 후 반환
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
