#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")  # Python 경고 억제

import os
import sys
import queue
from picamera2 import Picamera2
from flask import Flask, Response

# 0) 화면 절전/DPMS 비활성화 (X 환경일 때만)
try:
    if os.environ.get('DISPLAY'):
        os.system('setterm -blank 0 -powerdown 0 -powersave off')
        os.system('xset s off; xset s noblank; xset -dpms')
        print("⏱️ 화면 절전/스크린세이버 비활성화 완료")
except Exception:
    pass

# 1) CSI 카메라 초기화 (Picamera2 사용, MJPEG 포맷으로 설정)
try:
    picam2 = Picamera2()
    # MJPEG 형식으로 직접 JPEG 스트림을 받을 수 있음
    config = picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "MJPEG"},
        lores={"size": (640, 360)},
        buffer_count=2
    )
    picam2.configure(config)
    picam2.start()
    # 워밍업 캡처
    for _ in range(3):
        picam2.capture_buffer("main")
    print(">>> Using CSI camera module (MJPEG)")
except Exception as e:
    print(f"[ERROR] CSI 카메라 초기화 실패: {e}")
    sys.exit(1)

# 2) Flask 앱 설정
app = Flask(__name__)

# 3) 순수 CSI 카메라 MJPEG 스트리밍

def generate():
    boundary = b'--frame\r\n'
    header = b'Content-Type: image/jpeg\r\n\r\n'
    while True:
        # capture_buffer 로 바로 JPEG 프레임 바이트를 가져옴
        buf = picam2.capture_buffer("main")
        yield boundary + header + buf + b'\r\n'

@app.route('/')
def index():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
