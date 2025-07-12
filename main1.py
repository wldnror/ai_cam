#!/usr/bin/env python3
from picamera import PiCamera
from flask import Flask, Response
import io
import time

# ──────────────────────────────────────────────────────────────────────────────
# 1) 카메라 초기화
camera = PiCamera()
camera.resolution = (1280, 720)
camera.framerate = 24
# 워밍업
time.sleep(2)

# ──────────────────────────────────────────────────────────────────────────────
# 2) MJPEG 스트림 생성기
def gen_frames():
    stream = io.BytesIO()
    # use_video_port=True 로 빠른 버퍼링
    for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
        stream.seek(0)
        frame = stream.read()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )
        stream.seek(0)
        stream.truncate()

# ──────────────────────────────────────────────────────────────────────────────
# 3) Flask 앱 설정
app = Flask(__name__)

@app.route('/')
def index():
    return (
        "<html><body>"
        "<h1>Camera Stream</h1>"
        "<img src='/video_feed' />"
        "</body></html>"
    )

@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    # 0.0.0.0 으로 바인딩해 네트워크 상에서 접근 가능하도록
    app.run(host='0.0.0.0', port=5000)
