#!/usr/bin/env python3
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import cv2

# 1) Picamera2 초기화
picam2 = Picamera2()

# 2) on-sensor 파이프라인을 lores 스트림에 적용할 config 생성
config = picam2.create_preview_configuration(
    lores={                  # 뷰파인더 해상도
        "size": (1280, 720)
    },
    main={                    # (필요시) 고해상도 캡처용
        "size": (1280, 720)
    },
    controls={
        # inference_rate를 맞춰주면 프레임 드랍 없이 inference
        "FrameRate": picam2.device.network_intrinsics.inference_rate
    }
)
# on-sensor SSD+draw JSON 지정
config["post_process_file"] = "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json"

# 3) 설정 적용 및 카메라 시작
picam2.configure(config)
picam2.start()

# 4) Flask 앱 정의
app = Flask(__name__)
HTML = """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>AI 카메라 스트리밍</title>
<style>body{margin:0;text-align:center;}img{width:100%;}</style>
</head>
<body>
  <h1>On-sensor 객체 감지 스트림</h1>
  <img src="{{ url_for('video_feed') }}" />
</body>
</html>
"""
@app.route("/")
def index():
    return render_template_string(HTML)

def gen_frames():
    """lores 스트림을 JPEG로 인코딩해 MJPEG으로 Yield"""
    while True:
        # 'lores'에는 센서가 그리고 있는 박스+레이블이 포함되어 있음
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
    # 시스템 Python3 / venv에서 실행
    app.run(host="0.0.0.0", port=5000, threaded=True)
