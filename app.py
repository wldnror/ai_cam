#!/usr/bin/env python3
from flask import Flask, Response, render_template
from picamera2 import Picamera2
from picamera2.devices import IMX500
import cv2

# 1) IMX500용 RPK 모델 경로
MODEL_PATH = "/usr/share/imx500-models/" \
    "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"

# 2) IMX500 디바이스 로드
imx500 = IMX500(MODEL_PATH)

# 3) Picamera2 객체 생성 (카메라 번호 지정)
picam2 = Picamera2(imx500.camera_num)

# 4) 프리뷰용 config 생성
config = picam2.create_preview_configuration(
    main={"size": (640, 480)},
    controls={"FrameRate": imx500.network_intrinsics.inference_rate}
)

# 5) on-camera 파이프라인 지정 (JSON 파일)
config["post_process_file"] = "/usr/share/rpi-camera-assets/" \
    "imx500_mobilenet_ssd.json"
#    └─ rpicam-apps의 MobileNet-SSD 파이프라인 설정 파일 

# 6) config 적용 및 카메라 시작
picam2.configure(config)
imx500.show_network_fw_progress_bar()   # 펌웨어 로딩 진행 표시
picam2.start()

# 7) Flask 앱 설정
app = Flask(__name__)

def gen_frames():
    while True:
        # 이미 박스·라벨이 그려진 'main' 스트림 프레임을 가져옴
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
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
