#!/usr/bin/env python3
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
from picamera2.devices import IMX500
import cv2

# — 1) IMX500 모델 로드 —
MODEL_PATH = "/usr/share/imx500-models/" \
    "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
imx500 = IMX500(MODEL_PATH)

# — 2) Picamera2 객체 생성 —
picam2 = Picamera2(imx500.camera_num)

# — 3) lores 스트림에만 on-sensor 파이프라인 붙이기 —
config = picam2.create_preview_configuration(
    lores={                  # 뷰파인더용 스트림
        "size": (1280, 720),
        "post_process_file": "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json"
    },
    main={                    # (필요하다면) 고해상도 캡처용
        "size": (1280, 720)
    },
    controls={
        # 센서 추론 속도에 맞춰
        "FrameRate": imx500.network_intrinsics.inference_rate
    }
)

picam2.configure(config)
picam2.start()

# — 4) Flask 앱 정의 —
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
  <img src="{{ url_for('video_feed') }}" alt="Video feed"/>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

def gen_frames():
    """lores 스트림(viewfinder)에 이미 그려진 박스+레이블을 가져와 MJPEG으로 스트리밍"""
    while True:
        # 이제 'lores'에는 on-sensor가 그려준 결과가 그대로 들어옵니다!
        frame = picam2.capture_array("lores")
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
