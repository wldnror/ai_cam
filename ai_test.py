#!/usr/bin/env python3
"""
AI 카메라 실시간 스트리밍 (ONNX 기반 객체 인식 + 한글 라벨)

필요 환경:
  - Raspberry Pi OS Bookworm
  - CSI AI 카메라 연결
  - Python venv (권장)
    python3 -m venv venv_ai
    source venv_ai/bin/activate
    pip install flask numpy onnxruntime opencv-python-headless

사용법:
  python3 ai_stream_korean_onnx.py
  브라우저 접속 -> http://<PI_IP>:5000
"""
from flask import Flask, Response
import cv2
import onnxruntime as ort
import numpy as np

# 한글 라벨 매핑
LABEL_MAP = {
    "person": "사람",
    "bicycle": "자전거",
    "car": "자동차",
    "dog": "개",
    "cat": "고양이",
    # 필요한 클래스 추가
}

# 레이블 파일 로드 (영문 라벨 순서)
labels = []
with open('/usr/share/imx500-models/labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# ONNX 런타임 세션 초기화
session = ort.InferenceSession(
    '/usr/share/imx500-models/imx500_mobilenet_ssd.onnx',
    providers=['CPUExecutionProvider']
)

# GStreamer 카메라 파이프라인
gst_str = (
    'libcamerasrc ! '
    'video/x-raw,width=640,height=480,format=RGB888,framerate=30/1 ! '
    'videoconvert ! '
    'video/x-raw,format=BGR ! '
    'appsink drop=1 max-buffers=1 sync=false'
)
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

app = Flask(__name__)

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        H, W, _ = frame.shape

        # 전처리
        img = cv2.resize(frame, (300, 300))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
        inp = np.transpose(img, (2, 0, 1))[None, ...]

        # 추론
        outputs = session.run(None, {session.get_inputs()[0].name: inp})
        boxes, classes, scores = outputs[0][0], outputs[1][0], outputs[2][0]

        # 결과 그리기
        for box, cls, score in zip(boxes, classes, scores):
            if score < 0.5:
                continue
            x1, y1, x2, y2 = (box * np.array([W, H, W, H])).astype(int)
            label_en = labels[int(cls)]
            label_ko = LABEL_MAP.get(label_en, label_en)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label_ko} {score:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # JPEG 인코딩 후 스트리밍
        ret2, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes() if ret2 else b''
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return ('<html><body>'
            '<h1>AI 카메라 실시간 스트리밍 (ONNX)</h1>'
            '<img src="/video_feed" style="width:100%;max-width:640px;"/>'
            '</body></html>')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
