#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

import os
import time
import threading
import queue
import subprocess
import sys

import cv2
import torch
import psutil
from flask import Flask, Response, render_template, jsonify, request

# 0) 화면 절전/DPMS 비활성화(기존 그대로)

# 1) PyTorch 최적화(기존 그대로)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True).eval()

# 2) 카메라 인터페이스 정의
class VideoFileCamera:
    """테스트용: 비디오 파일을 OpenCV로 제어"""
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"비디오 파일을 열 수 없습니다: {path}")
    def read(self):
        ret, frame = self.cap.read()
        # 파일 끝에 도달하면 다시 처음부터 순환 재생(선택사항)
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

# (기존) CSI/USB 카메라 클래스는 그대로 두어도 됩니다.

# ▶ 여기부터 변경: 파일 경로 인자를 주면 VideoFileCamera 사용
if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
    video_path = sys.argv[1]
    camera = VideoFileCamera(video_path)
    print(f">>> Using video file: {video_path}")
else:
    try:
        from picamera2 import Picamera2
        camera = CSICamera()
        print(">>> Using CSI camera module")
    except Exception:
        camera = USBCamera()
        print(">>> Using USB webcam")

# 3) 백그라운드 프레임 처리 스레드 + 큐 (이하 기존 코드 그대로)
frame_queue = queue.Queue(maxsize=1)

def capture_and_process():
    fps = 10
    interval = 1.0 / fps
    target_size = (320, 320)
    skip_interval = 2
    frame_count = 0
    last = time.time()
    last_results = None

    while True:
        now = time.time()
        sleep = interval - (now - last)
        if sleep > 0:
            time.sleep(sleep)
        last = time.time()

        ret, frame = camera.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % skip_interval == 0:
            with torch.no_grad():
                small = cv2.resize(frame, target_size)
                last_results = model(small)

        if last_results is None:
            continue

        h_ratio = frame.shape[0] / target_size[1]
        w_ratio = frame.shape[1] / target_size[0]
        for *box, conf, cls in last_results.xyxy[0]:
            x1, y1, x2, y2 = map(int, (
                box[0] * w_ratio,
                box[1] * h_ratio,
                box[2] * w_ratio,
                box[3] * h_ratio
            ))
            label = last_results.names[int(cls)]
            if label in ('person', 'car'):
                color = (0,0,255) if label=='person' else (255,0,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        data = buf.tobytes()

        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(data)

threading.Thread(target=capture_and_process, daemon=True).start()

# 4) Flask 앱 & 스트리밍 + 통계 엔드포인트 (이하 기존 코드 그대로)
app = Flask(__name__)
def generate():
    while True:
        frame = frame_queue.get()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    resp = Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    direct_passthrough=True)
    resp.headers.update({
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    })
    return resp

@app.route('/stats')
def stats():
    # …(기존 stats 엔드포인트)…

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
