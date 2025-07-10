#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")  # 모든 파이썬 경고 억제

import logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
werkzeug_log = logging.getLogger('werkzeug')
werkzeug_log.setLevel(logging.ERROR)

import os
import sys
import time
import threading
import queue
import subprocess

import cv2
import torch
import psutil
from flask import Flask, Response, render_template, jsonify, request

# ──────────────────────────────────────────────────────────────────────────────
# 0) 화면 절전/DPMS 비활성화
try:
    if os.environ.get('DISPLAY'):
        os.system('setterm -blank 0 -powerdown 0 -powersave off')
        os.system('xset s off; xset s noblank; xset -dpms')
        print("⏱️ 화면 절전/스크린세이버 비활성화 완료")
except:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# 1) 모델 로드
# (1) 캐시 폴더 삭제하여 깨끗하게
#    $ rm -rf ~/.cache/torch/hub/ultralytics_yolov5_master
# (2) force_reload=True 로 강제 재로딩
model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5n',
    pretrained=True,
    force_reload=True
)
model.eval()
print(f"YOLOv5 loaded on {model.device}")

# ──────────────────────────────────────────────────────────────────────────────
# 2) CSI 카메라 정의 (Picamera2 + libcamera)
class CSICamera:
    def __init__(self):
        try:
            from picamera2 import Picamera2
        except ModuleNotFoundError:
            sys.exit(
                "ERROR: picamera2/libcamera 모듈을 찾을 수 없습니다.\n"
                "sudo apt install -y python3-picamera2 libcamera-apps\n"
                "혹은 venv 생성 시 --system-site-packages 옵션을 사용하세요."
            )
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main         = {"format": "XRGB8888", "size": (1280, 720)},
            lores        = {"format": "XRGB8888", "size": (640, 360)},
            buffer_count = 2
        )
        self.picam2.configure(config)
        self.picam2.start()
        for _ in range(3):
            self.picam2.capture_array("main")

    def read(self):
        return True, self.picam2.capture_array("main")

# CSI 카메라만 강제 사용
camera = CSICamera()
print(">>> Using CSI camera")

# ──────────────────────────────────────────────────────────────────────────────
# 3) 백그라운드 처리
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
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put(data)

threading.Thread(target=capture_and_process, daemon=True).start()

# ──────────────────────────────────────────────────────────────────────────────
# 4) Flask 앱
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
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/stats')
def stats():
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    temp = None
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            temp = float(f.read()) / 1000.0
    except:
        pass
    signal = None
    try:
        out = subprocess.check_output(['iwconfig','wlan0'], stderr=subprocess.DEVNULL).decode()
        for part in out.split():
            if part.startswith('level='):
                signal = int(part.split('=')[1])
    except:
        pass
    return jsonify({
        'cpu_percent': cpu,
        'memory_percent': mem.percent,
        'temperature_c': temp,
        'wifi_signal_dbm': signal
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
