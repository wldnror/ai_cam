#!/usr/bin/env python3
from flask import Flask, Response, render_template, jsonify
from picamera2 import Picamera2
from picamera2.devices import IMX500
import cv2

# — 설정 —
MODEL_PATH = "/usr/share/imx500-models/" \
    "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
JSON_PIPELINE = "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json"

# — IMX500 & Intrinsics 로드 —
imx500 = IMX500(MODEL_PATH)
intrinsics = imx500.network_intrinsics
intrinsics.update_with_defaults()

# — Picamera2 구성 (on-camera post-processing 포함) —
picam2 = Picamera2(imx500.camera_num)
config = picam2.create_preview_configuration(
    main={"size": (640, 480)},
    controls={"FrameRate": intrinsics.inference_rate},
    post_process_file=JSON_PIPELINE
)
imx500.show_network_fw_progress_bar()  # 펌웨어 로드 진행 표시
picam2.configure(config)
picam2.start()

# — Flask 앱 정의 — 
app = Flask(__name__)

# 루트 페이지
@app.route('/')
def index():
    return render_template('index.html')

# stats 호출 방지용 더미 엔드포인트
@app.route('/stats')
def stats():
    return jsonify({})

# MJPEG 스트리밍 생성기
def gen_frames():
    while True:
        frame = picam2.capture_array("main")  # 이미 박스+라벨 오버레이됨
        # (디버그: 프레임 모양 확인용 로그)
        # print("frame shape:", frame.shape)
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
