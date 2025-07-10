#!/usr/bin/env python3
from flask import Flask, Response, render_template
from picamera2 import Picamera2
from picamera2.devices import IMX500
import cv2

# 1) IMX500 모델 로드
MODEL_PATH = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
imx500 = IMX500(MODEL_PATH)

# 2) Picamera2 객체 생성
picam2 = Picamera2(imx500.camera_num)

# 3) preview configuration 생성
config = picam2.create_preview_configuration(
    main={"size": (640, 480)},
    controls={"FrameRate": imx500.network_intrinsics.inference_rate}
)

# 4) on-camera 파이프라인 지정 (MobileNet-SSD + draw 단계 포함)
config["post_process_file"] = "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json"

# 5) 구성 적용 및 카메라 시작
picam2.configure(config)
imx500.show_network_fw_progress_bar()   # (선택) 로딩 상태 출력
picam2.start()

# 6) Flask 앱 정의
app = Flask(__name__)

def gen_frames():
    while True:
        # 이미 박스와 레이블이 그려진 'main' 스트림 프레임 획득
        frame = picam2.capture_array("main")
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        # multipart로 인코딩
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
    # 시스템 파이썬(가상환경 비활성화 상태)에서 실행하세요!
    app.run(host='0.0.0.0', port=5000, threaded=True)
