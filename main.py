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
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# 1) YOLOv5 모델 로드 (로컬 클론 + DetectMultiBackend + AutoShape)
YOLOROOT = os.path.expanduser('~/yolov5')
if not os.path.isdir(YOLOROOT):
    print(f"Cloning YOLOv5 repo to {YOLOROOT}...")
    subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git', YOLOROOT], check=True)
sys.path.insert(0, YOLOROOT)
from models.common import DetectMultiBackend, AutoShape
from utils.torch_utils import select_device

device = select_device('cpu')
WEIGHTS = os.path.join(YOLOROOT, 'yolov5n.pt')
if not os.path.exists(WEIGHTS):
    print(f"Downloading weights to {WEIGHTS}...")
    torch.hub.download_url_to_file(
        'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt', WEIGHTS
    )
backend = DetectMultiBackend(WEIGHTS, device=device, fuse=True)
backend.model.eval()
model = AutoShape(backend.model)

# confidence threshold 설정 (기본 0.25 → 필요시 조정)
model.conf = 0.25

# ──────────────────────────────────────────────────────────────────────────────
# 2) 카메라 클래스 정의
class CSICamera:
    """CSI 카메라를 Picamera2로 제어"""
    def __init__(self):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        cfg = self.picam2.create_video_configuration(
            main={"size": (1280, 720)}, lores={"size": (640, 360)}, buffer_count=2
        )
        self.picam2.configure(cfg)
        self.picam2.start()
        for _ in range(3):
            self.picam2.capture_array("main")

    def read(self):
        rgb = self.picam2.capture_array("main")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return True, bgr

class USBCamera:
    """USB 웹캠을 OpenCV로 제어"""
    def __init__(self):
        self.cap = None
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(5):
                    cap.read()
                self.cap = cap
                break
        if not self.cap:
            raise RuntimeError("사용 가능한 USB 웹캠이 없습니다.")

    def read(self):
        return self.cap.read()

# CSI 모듈 시도, 실패 시 USB
try:
    camera = CSICamera()
    print(">>> Using CSI camera module")
except Exception as e:
    print(f"[ERROR] CSI init failed: {e}")
    camera = USBCamera()
    print(">>> Using USB webcam")

# ──────────────────────────────────────────────────────────────────────────────
# 3) 백그라운드 프레임 처리 스레드
frame_queue = queue.Queue(maxsize=1)

def capture_and_process():
    fps = 10
    interval = 1.0 / fps
    target_size = 320  # AutoShape 인풋 크기

    while True:
        start = time.time()
        ret, frame = camera.read()
        if not ret:
            continue

        # 1) 추론 & 렌더링
        with torch.no_grad():
            results = model(frame, size=target_size)
            results.render()  # frame 위에 박스와 라벨을 그림

        # 2) 결과 가져오기 (RGB → BGR)
        annotated = results.imgs[0]
        frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        # 3) JPEG 인코딩 & 큐에 저장
        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        data = buf.tobytes()
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(data)

        # 4) FPS 유지
        elapsed = time.time() - start
        sleep = interval - elapsed
        if sleep > 0:
            time.sleep(sleep)

threading.Thread(target=capture_and_process, daemon=True).start()

# ──────────────────────────────────────────────────────────────────────────────
# 4) Flask 앱 및 엔드포인트
app = Flask(__name__)

def generate():
    while True:
        frame = frame_queue.get()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def get_network_signal(interface='wlan0'):
    try:
        out = subprocess.check_output(['iwconfig', interface], stderr=subprocess.DEVNULL).decode()
        for part in out.split():
            if part.startswith('level='):
                return int(part.split('=')[1])
    except Exception:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    resp = Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers.update({
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    })
    return resp

@app.route('/stats')
def stats():
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    temp = None
    try:
        temp = float(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000.0
    except Exception:
        pass
    sig = get_network_signal('wlan0')
    return jsonify(
        camera=1,
        cpu_percent=cpu,
        memory_percent=mem.percent,
        temperature_c=temp,
        wifi_signal_dbm=sig
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
