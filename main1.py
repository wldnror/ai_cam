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
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, Response, render_template, jsonify, request

# ----------------------------------------
# 0) 한글 폰트 설치 확인 및 로드
FONT_PATH = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if not os.path.exists(FONT_PATH):
    try:
        print("한글 폰트가 없어 설치를 시도합니다...")
        subprocess.run(['sudo', 'apt-get', 'update'], check=True)
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'fonts-nanum'], check=True)
    except Exception as e:
        print(f"폰트 설치 실패: {e}")
try:
    font = ImageFont.truetype(FONT_PATH, 24)
except Exception:
    font = ImageFont.load_default()
    print("한글 폰트를 로드하지 못해 기본 폰트를 사용합니다.")

# ----------------------------------------
# 1) 화면 절전/DPMS 비활성화
try:
    if os.environ.get('DISPLAY'):
        os.system('setterm -blank 0 -powerdown 0 -powersave off')
        os.system('xset s off; xset s noblank; xset -dpms')
        print("⏱️ 화면 절전/스크린세이버 비활성화 완료")
except Exception:
    pass

# ----------------------------------------
# 2) PyTorch 스레드 수 & YOLOv5 모델 로드
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

YOLOROOT = os.path.expanduser('~/yolov5')
if not os.path.isdir(YOLOROOT):
    subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git', YOLOROOT], check=True)
sys.path.insert(0, YOLOROOT)
from models.common import DetectMultiBackend, AutoShape
from utils.torch_utils import select_device

device = select_device('cpu')
MODEL_NAME = "yolov5n.pt"
backend = None
model = None

def load_model(weights_name):
    global backend, model
    weights_path = os.path.join(YOLOROOT, weights_name)
    if not os.path.exists(weights_path):
        torch.hub.download_url_to_file(
            f'https://github.com/ultralytics/yolov5/releases/download/v7.0/{weights_name}',
            weights_path
        )
    backend = DetectMultiBackend(weights_path, device=device, fuse=True)
    backend.model.eval()
    model = AutoShape(backend.model)
    print(f"[MODEL] Loaded {weights_name}")

load_model(MODEL_NAME)

# 한글 레이블 매핑
label_map = {'person': '사람', 'car': '자동차'}

# ----------------------------------------
# 3) 카메라 클래스 정의
class CSICamera:
    def __init__(self):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        cfg = self.picam2.create_video_configuration(
            main={"size": (720, 480)}, lores={"size": (640, 360)}, buffer_count=6
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
    def __init__(self):
        self.cap = None
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
                for _ in range(5):
                    cap.read()
                self.cap = cap
                break
        if not self.cap:
            raise RuntimeError("사용 가능한 USB 웹캠이 없습니다.")

    def read(self):
        return self.cap.read()

try:
    camera = CSICamera()
    print(">>> Using CSI camera module")
except Exception as e:
    print(f"[ERROR] CSI init failed: {e}")
    camera = USBCamera()
    print(">>> Using USB webcam")

# ----------------------------------------
# 4) 트래커 생성 함수 탐색
# CSRT 우선, 없으면 KCF 사용, legacy fallback 포함

def _find_tracker_ctor():
    names = [
        'TrackerCSRT_create',
        'legacy.TrackerCSRT_create',
        'TrackerKCF_create',
        'legacy.TrackerKCF_create'
    ]
    for name in names:
        parts = name.split('.')
        mod = cv2
        for p in parts[:-1]:
            mod = getattr(mod, p, None)
            if mod is None:
                break
        if mod and hasattr(mod, parts[-1]):
            return getattr(mod, parts[-1])
    return None

TrackerCreate = _find_tracker_ctor()
if TrackerCreate is None:
    print("[WARN] Compatible tracker not found. Tracking disabled; 매 프레임 YOLO로만 검출합니다.")
    TRACKING_ENABLED = False
else:
    TRACKING_ENABLED = True

# ----------------------------------------
# 6) Flask 앱 및 엔드포인트 설정
app = Flask(__name__)

def generate():
    while True:
        frame = frame_queue.get()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    resp = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers.update({'Cache-Control':'no-cache, no-store, must-revalidate',
                         'Pragma':'no-cache','Expires':'0'})
    return resp

@app.route('/stats')
def stats():
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory().percent
    temp = None
    try:
        temp = float(open('/sys/class/thermal/thermal_zone0/temp').read())/1000.0
    except:
        pass
    try:
        out = subprocess.check_output(['iwconfig','wlan0'], stderr=subprocess.DEVNULL).decode()
        sig = int([p.split('=')[1] for p in out.split() if p.startswith('level=')][0])
    except:
        sig = None
    with fps_lock:
        fps = round(current_fps,1)
    return jsonify(camera=1, cpu_percent=cpu, memory_percent=mem,
                   temperature_c=temp, wifi_signal_dbm=sig, fps=fps)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
