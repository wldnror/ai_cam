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

# 3) 네트워크 펌웨어를 먼저 로드 (업로드 완료 전까지 기다림)
#    show_progress=True로 하면 터미널에 진행률이 표시됩니다.
imx500.load_network_firmware(show_progress=True)

# 4) Picamera2 객체 생성 (올바른 camera_num 지정)
picam2 = Picamera2(imx500.camera_num)

# 5) 프리뷰용 config 생성
config = picam2.create_preview_configuration(
    main={"size": (640, 480)},
    controls={"FrameRate": imx500.network_intrinsics.inference_rate}
)

# 6) on-camera 파이프라인 설정 JSON 지정
config["post_process_file"] = "/usr/share/rpi-camera-assets/" \
    "imx500_mobilenet_ssd.json"

# 7) config 적용 후 카메라 시작
picam2.configure(config)
picam2.start()   # 펌웨어 업로드가 완료된 상태이므로 오류 없이 시작됩니다

# 8) Flask 앱 설정
app = Flask(__name__)

def gen_frames():
    while True:
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
