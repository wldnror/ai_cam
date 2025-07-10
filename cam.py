#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")  # 경고 억제

import os
import cv2
from flask import Flask, Response, render_template

# 화면 절전/DPMS 비활성화 (X가 있을 때만)
try:
    if os.environ.get('DISPLAY'):
        os.system('setterm -blank 0 -powerdown 0 -powersave off')
        os.system('xset s off; xset s noblank; xset -dpms')
        print("⏱️ 화면 절전/스크린세이버 비활성화 완료")
except Exception:
    pass

# CSI 카메라 또는 USB 카메라 선택
class CSICamera:
    def __init__(self):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (1280, 720)}, lores={"size": (640, 360)}, buffer_count=2
        )
        self.picam2.configure(config)
        self.picam2.start()
        for _ in range(3):
            self.picam2.capture_array("main")

    def read(self):
        return True, self.picam2.capture_array("main")

class USBCamera:
    def __init__(self):
        self.cap = None
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(5): cap.read()
                self.cap = cap
                break
        if self.cap is None:
            raise RuntimeError("사용 가능한 USB 웹캠을 찾을 수 없습니다.")

    def read(self):
        return self.cap.read()

# 카메라 초기화
try:
    camera = CSICamera()
    print(">>> Using CSI camera module")
except Exception as e:
    print(f"[ERROR] CSI 카메라 초기화 실패: {e}")
    camera = USBCamera()
    print(">>> Using USB webcam")

# Flask 앱 설정
app = Flask(__name__)

def generate():
    while True:
        ret, frame = camera.read()
        if not ret:
            continue
        # JPEG 인코딩
        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_bytes = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # index.html 파일에 <img src="/video_feed"> 추가

@app.route('/video_feed')
def video_feed():
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
