#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")  # 모든 파이썬 경고 억제

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

# 0) 화면 절전/DPMS 비활성화
try:
    if os.environ.get('DISPLAY'):
        os.system('setterm -blank 0 -powerdown 0 -powersave off')
        os.system('xset s off; xset s noblank; xset -dpms')
        print("⏱️ 화면 절전/스크린세이버 비활성화 완료")
except Exception:
    pass

# 1) PyTorch 스레드 & 추론 모드 최적화
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.eval()

# 2) CSI 카메라 모듈 정의 (Picamera2)
class CSICamera:
    def __init__(self):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        # BGR 이미지 직접 출력
        config = self.picam2.create_video_configuration(
            main = {"size": (1280, 720), "format": "BGR888"},
            lores = {"size": (640, 360)},
            buffer_count = 2
        )
        self.picam2.configure(config)
        self.picam2.start()
        for _ in range(3):
            self.picam2.capture_array("main")

    def read(self):
        # BGR888 포맷이므로 그대로 사용
        return True, self.picam2.capture_array("main")

try:
    camera = CSICamera()
    print(">>> Using CSI camera module")
except Exception as e:
    print(f"[ERROR] CSI 카메라 초기화 실패: {e}")
    sys.exit(1)

# 3) 백그라운드 프레임 처리
frame_queue = queue.Queue(maxsize=1)

def capture_and_process():
    fps = 10
    interval = 1.0 / fps
    target_size = (320, 320)
    skip_interval = 2
    frame_count = 0
    last_results = None
    last = time.time()

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

        # 박스 그리기
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
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        data = buf.tobytes()

        if not frame_queue.empty():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put(data)

threading.Thread(target=capture_and_process, daemon=True).start()

# 4) Flask 앱
app = Flask(__name__)

def generate():
    while True:
        frame = frame_queue.get()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    resp = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers['Cache-Control']='no-cache, no-store, must-revalidate'
    resp.headers['Pragma']='no-cache'
    resp.headers['Expires']='0'
    return resp

@app.route('/stats')
def stats():
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory().percent
    temp = None
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            temp = float(f.read())/1000.0
    except: pass
    try:
        out = subprocess.check_output(['iwconfig','wlan0'], stderr=subprocess.DEVNULL).decode()
        signal = int([p for p in out.split() if p.startswith('level=')][0].split('=')[1])
    except: signal=None
    return jsonify({
        'cpu_percent':cpu,
        'memory_percent':mem,
        'temperature_c':temp,
        'wifi_signal_dbm':signal
    })

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)
