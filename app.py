#!/usr/bin/env python3
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
from picamera2.devices import IMX500
import cv2

# — 1) IMX500 모델 로드 —
MODEL_PATH = "/usr/share/imx500-models/" \
    "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
imx500 = IMX500(MODEL_PATH)

# — 2) Picamera2 객체 생성 (IMX500 카메라 번호 지정) —
picam2 = Picamera2(imx500.camera_num)

# — 3) Preview(lores) + on-sensor 추론 설정 —
config = picam2.create_preview_configuration(
    lores={                  # Viewfinder 해상도
        "size": (1280, 720)
    },
    main={                    # (선택) 고해상도 캡처용
        "size": (1280, 720)
    },
    controls={
        # IMX500의 추론 속도에 맞춰 프레임 속도 설정
        "FrameRate": imx500.network_intrinsics.inference_rate
    }
)
# on-sensor MobileNet-SSD + draw JSON 지정
config["post_process_file"] = "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json"

# — 4) 설정 적용 & 카메라 시작 —
picam2.configure(config)
picam2.start()

# — 5) Flask 앱 정의 —
app = Flask(__name__)
HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>AI 카메라 스트리밍</title>
  <style>body{margin:0;text-align:center;}img{width:100%;height:auto;}</style>
</head>
<body>
  <h1>On-sensor 객체 감지 스트림</h1>
  <img src="{{ url_for('video_feed') }}" alt="Video stream"/>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

def gen_frames():
    """lores 스트림(박스+레이블 포함)을 JPEG로 인코딩해 MJPEG 스트림으로 전송"""
    while True:
        frame = picam2.capture_array("lores")  # on-sensor 뷰파인더 결과
        ret, buf = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    # 시스템 Python3 또는 venv에서 실행
    app.run(host="0.0.0.0", port=5000, threaded=True)
