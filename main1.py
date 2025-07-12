#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")  # Python 경고 억제

import os
import sys
import cv2
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

# 1) CSI 카메라 초기화: Preview Configuration (BGR888 포맷)
try:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()
    # 워밍업 프레임
    for _ in range(5): picam2.capture_array("main")
    print(">>> Using CSI camera preview configuration (BGR888)")
except Exception as e:
    print(f"[ERROR] CSI 카메라 초기화 실패: {e}")
    sys.exit(1)

# 2) Flask 앱 설정
app = Flask(__name__)

# 3) MJPEG 스트림 엔드포인트
@app.route('/stream')
def stream():
    boundary = b'--frame\r\n'
    header = b'Content-Type: image/jpeg\r\n\r\n'
    def generate():
        while True:
            # RGB888 ndarray 반환
            frame = picam2.capture_array("main")
            # OpenCV에 BGR 형식으로 전달
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ret, jpg = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue
            data = jpg.tobytes()
            yield boundary + header + data + b'\r\n'
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 4) 간단 HTML 페이지 제공
@app.route('/')
def index():
    return (
        '<html><head><title>CSI Camera Stream</title></head>'
        '<body><h1>CSI Camera MJPEG Stream</h1>'
        '<img src="/stream" style="width:100%; height:auto; background:black;" />'
        '</body></html>'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
