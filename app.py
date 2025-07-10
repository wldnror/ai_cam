#!/usr/bin/env python3
from flask import Flask, Response, render_template
from picamera2 import Picamera2
from picamera2.devices import IMX500
import cv2

# 1) IMX500 RPK 모델 경로
MODEL_PATH = "/usr/share/imx500-models/" \
    "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"

# 2) IMX500 디바이스 인스턴스화
imx500 = IMX500(MODEL_PATH)
intrinsics = imx500.network_intrinsics

# 3) Picamera2 객체 생성
picam2 = Picamera2(imx500.camera_num)

# 4) 프리뷰용 config 생성
config = picam2.create_preview_configuration(
    main={"size": (640, 480)},
    controls={"FrameRate": intrinsics.inference_rate}
)

# 5) on-camera 파이프라인 지정 JSON 파일
config["post_process_file"] = "/usr/share/rpi-camera-assets/" \
    "imx500_mobilenet_ssd.json"

# 6) 네트워크 펌웨어 업로드 진행 표시 (완료될 때까지 블록)
imx500.show_network_fw_progress_bar()

# 7) 카메라 시작 (config를 인자로 전달)
picam2.start(config, show_preview=False)

# 8) Flask 앱 설정
app = Flask(__name__)

def gen_frames():
    while True:
        # 이미 바운딩 박스·라벨이 그려진 'main' 스트림 프레임 캡처
        frame = picam2.capture_array("main")
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
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
