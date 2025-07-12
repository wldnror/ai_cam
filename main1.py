#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")  # Python 경고 억제

import os
import sys
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

# 1) CSI 카메라 초기화 (YUV420 포맷으로 설정)
try:
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "YUV420"},
        lores={"size": (640, 360)},
        buffer_count=2
    )
    picam2.configure(config)
    picam2.start()
    # 워밍업 캡처
    for _ in range(3): picam2.capture_array("main")
    print(">>> Using CSI camera module (YUV420)")
except Exception as e:
    print(f"[ERROR] CSI 카메라 초기화 실패: {e}")
    sys.exit(1)

# 2) 프레임 저장용 큐
frame_queue = queue.Queue(maxsize=1)

# 3) FrameWriter 클래스 정의
class FrameWriter:
    """JpegEncoder가 생성한 JPEG 프레임을 큐에 저장"""
    def write(self, buf):
        try:
            data = buf.tobytes() if hasattr(buf, 'tobytes') else bytes(buf)
            if not frame_queue.empty():
                frame_queue.get_nowait()
            frame_queue.put(data)
        except Exception:
            pass

# 4) JpegEncoder로 녹화 시작
encoder = JpegEncoder(q=80)
picam2.start_recording(encoder, FrameWriter())

# 5) Flask 앱 설정
app = Flask(__name__)

# 6) MJPEG 스트림 생성기
def generate():
    boundary = b'--frame\r\n'
    header = b'Content-Type: image/jpeg\r\n\r\n'
    while True:
        buf = frame_queue.get()
        yield boundary + header + buf + b'\r\n'

# 7) HTML 페이지 제공
@app.route('/')
def index():
    return (
        '<html><head><title>CSI Camera Stream</title></head>'
        '<body><h1>CSI Camera MJPEG Stream</h1>'
        '<img src="/stream" style="width:100%;" />'
        '</body></html>'
    )

# 8) MJPEG 스트림 엔드포인트
@app.route('/stream')
def stream():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
