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

# 단일 모델 로드 설정
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

# 시작 시 모델 로드
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
# 4) 객체 검출 + 트래킹 기반 프레임 처리
frame_queue = queue.Queue(maxsize=3)
current_fps = 0.0
fps_lock = threading.Lock()

def create_tracker():
    # 경량 트래커로 MOSSE 사용 (빠름). 정확도가 더 필요하면 CSRT로 변경하세요.
    return cv2.TrackerMOSSE_create()

def capture_and_track():
    global current_fps
    fps = 10
    interval = 1.0 / fps
    target_size = 270

    detection_interval = 5  # N 프레임마다 모델 검출 수행
    frame_count = 0
    trackers = []  # (tracker, label) 리스트

    while True:
        start = time.time()
        ret, frame = camera.read()
        if not ret:
            continue

        frame_count += 1
        boxes = []

        if frame_count % detection_interval == 0:
            trackers.clear()
            with torch.no_grad():
                results = model(frame, size=target_size)
            for *box, conf, cls in results.xyxy[0]:
                label = results.names[int(cls)]
                if label not in label_map:
                    continue
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2, label, float(conf)))
                tracker = create_tracker()
                tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                trackers.append((tracker, label))
        else:
            new_boxes = []
            for tracker, label in trackers:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = map(int, bbox)
                    new_boxes.append((x, y, x+w, y+h, label, None))
            boxes = new_boxes

        for x1, y1, x2, y2, label, conf in boxes:
            color = (255, 0, 0) if label == 'person' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label_map[label]} {conf*100:.1f}%" if conf is not None else f"{label_map[label]} (추적됨)"
            img_pil = Image.fromarray(frame[:, :, ::-1])
            draw = ImageDraw.Draw(img_pil)
            size = draw.textsize(text, font=font)
            draw.rectangle([x1, y1-size[1]-4, x1+size[0]+4, y1], fill=(0,0,0))
            draw.text((x1+2, y1-size[1]-2), text, font=font, fill=(255,255,255))
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not frame_queue.empty():
            frame_queue.get_nowait()
        frame_queue.put(buf.tobytes())

        elapsed = time.time() - start
        instant_fps = 1.0/elapsed if elapsed>0 else 0.0
        with fps_lock:
            current_fps = instant_fps

        time.sleep(max(0, interval-elapsed))

threading.Thread(target=capture_and_track, daemon=True).start()

# ----------------------------------------
# 5) Flask 앱 및 엔드포인트
app = Flask(__name__)

def generate():
    while True:
        frame = frame_queue.get()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
... (이하 동일)
