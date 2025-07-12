#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")  # Python 경고 억제

import os
import sys
import queue
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from flask import Flask, Response

# 0) 화면 절전/DPMS 비활성화 (X 환경일 때만)
try:
    if os.environ.get('DISPLAY'):
        os.system('setterm -blank 0 -powerdown 0 -powersave off')
        os.system('xset s off; xset s noblank; xset -dpms')
        print("⏱️ 화면 절전/스크린세이버 비활성화 완료")
except Exception:
    pass

# 1) CSI 카메라 초기화 및 MJPEG 인코더 설정
try:
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "YUV420"},
        lores={"size": (640, 360)},
        buffer_count=2
    )
    picam2.configure(config)
    frame_queue = queue.Queue(maxsize=1)

    class FrameWriter:
        """파일객체 인터페이스로 MJPEG 프레임을 버퍼에 푸시"""
        def write(self, buf):
            if not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(buf)

    encoder = MJPEGEncoder()
    # MJPEG 스트림 녹화 시작: encoder와 FrameWriter를 positional 인수로 전달
    picam2.start_recording(encoder, FrameWriter())
    print(">>> Using CSI camera MJPEG stream")
except Exception as e:
    print(f"[ERROR] CSI 카메라 초기화 실패: {e}")
    sys.exit(1)

# 2) Flask 앱 설정
app = Flask(__name__)

# MJPEG 스트림 생성기
def generate():
    boundary = b'--frame\r\n'
    header = b'Content-Type: image/jpeg\r\n\r\n'
    while True:
        buf = frame_queue.get()
        yield boundary + header + buf + b'\r\n'

@app.route('/')
def index():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
