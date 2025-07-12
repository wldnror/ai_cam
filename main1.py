#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")  # Python 경고 억제

import os
import sys
import io
import queue
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from flask import Flask, Response

# 0) 화면 절전/DPMS 비활성화 (X 환경일 때만)
try:
    if os.environ.get('DISPLAY'):
        os.system('setterm -blank 0 -powerdown 0 -powersave off')
        os.system('xset s off; xset s noblank; xset -dpms')
        print("⏱️ 화면 절전/스크린세이버 비활성화 완료")
except Exception:
    pass

# 1) CSI 카메라 초기화 및 MJPEG 스트림 생성 (JpegEncoder 사용)
try:
    picam2 = Picamera2()
    # RGB888 형식으로 받아 PIL 인코딩에 적합
    config = picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "RGB888"},
        lores={"size": (640, 360)}
    )
    picam2.configure(config)
    picam2.start()
    # 워밍업 프레임
    for _ in range(3): picam2.capture_array("main")
    print(">>> Using CSI camera module with JpegEncoder")

    frame_queue = queue.Queue(maxsize=1)
    class FrameWriter:
        """파일객체 인터페이스: JpegEncoder가 쓴 MJPEG 프레임을 버퍼에 저장"""
        def write(self, buf):
            # buf는 완성된 JPEG 프레임(바이트)
            if not frame_queue.empty():
                try: frame_queue.get_nowait()
                except queue.Empty: pass
            frame_queue.put(buf)

    # JPEG 인코더(q=80)로 MJPEG 스트림 생성
    encoder = JpegEncoder(q=80)
    picam2.start_recording(encoder, FrameWriter())
except Exception as e:
    print(f"[ERROR] CSI 카메라 초기화 실패: {e}")
    sys.exit(1)

# 2) Flask 앱 설정
app = Flask(__name__)

def generate():
    # multipart MJPEG 응답 바운더리
    boundary = b'--frame\r\n'
    header = b'Content-Type: image/jpeg\r\n\r\n'
    while True:
        buf = frame_queue.get()  # MJPEG 프레임 데이터
        yield boundary + header + buf + b'\r\n'

@app.route('/')
def index():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
