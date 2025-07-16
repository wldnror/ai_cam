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
from concurrent.futures import ThreadPoolExecutor

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
import multiprocessing
n_cpu = multiprocessing.cpu_count()
torch.set_num_threads(n_cpu)
torch.set_num_interop_threads(n_cpu)

YOLOROOT = os.path.expanduser('~/yolov5')
if not os.path.isdir(YOLOROOT):
    subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git', YOLOROOT], check=True)
sys.path.insert(0, YOLOROOT)
from models.common import DetectMultiBackend, AutoShape
from utils.torch_utils import select_device

device = select_device('cpu')

MODEL_NAME = "yolov5s.pt"
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

label_map = {'person': '사람', 'car': '자동차'}

# ----------------------------------------
# 3) 카메라 클래스 정의
class CSICamera:
    def __init__(self):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        cfg = self.picam2.create_video_configuration(
            main={"size": (1280, 720)}, lores={"size": (360, 360)}, buffer_count=6
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
    constructors = [
        ('cv2', 'TrackerMOSSE_create'),
        ('cv2.legacy', 'TrackerMOSSE_create'),
        ('cv2.legacy', 'TrackerCSRT_create'),
        ('cv2', 'TrackerCSRT_create'),
        ('cv2.legacy', 'TrackerKCF_create'),
        ('cv2', 'TrackerKCF_create'),
        ('cv2.legacy', 'TrackerMIL_create'),
        ('cv2', 'TrackerMIL_create')
    ]
    for module_name, func_name in constructors:
        try:
            module = cv2
            if module_name == 'cv2.legacy' and hasattr(cv2, 'legacy'):
                module = cv2.legacy
            tracker_fn = getattr(module, func_name, None)
            if tracker_fn:
                return tracker_fn()
        except Exception:
            continue
    raise RuntimeError("사용 가능한 트래커를 찾을 수 없습니다.")

def capture_and_track():
    global current_fps
    target_size = 360
    detection_interval = 5
    frame_count = 0
    trackers = []
    executor = ThreadPoolExecutor(max_workers=1)

    while True:
        start = time.time()
        ret, frame = camera.read()
        if not ret:
            continue

        frame_count += 1
        full_frame = frame
        boxes = []

        if frame_count % detection_interval == 0:
            # clear trackers
            trackers.clear()

            # resize & async detect
            small = cv2.resize(full_frame, (target_size, target_size))
            future = executor.submit(lambda img: model(img, size=target_size), small)
            results = future.result()

            for *box, conf, cls in results.xyxy[0]:
                label = results.names[int(cls)]
                if label not in label_map:
                    continue
                x1, y1, x2, y2 = box
                scale_x = full_frame.shape[1] / target_size
                scale_y = full_frame.shape[0] / target_size
                x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)

                # init tracker on full_frame
                trk = create_tracker()
                trk.init(full_frame, (x1, y1, x2-x1, y2-y1))
                trackers.append((trk, label))
                boxes.append((x1, y1, x2, y2, label, float(conf)))

        else:
            # tracking only
            new_boxes = []
            for trk, label in trackers:
                success, bbox = trk.update(full_frame)
                if not success:
                    continue
                x, y, w, h = map(int, bbox)
                new_boxes.append((x, y, x+w, y+h, label, None))
            boxes = new_boxes

        # draw boxes
        for x1, y1, x2, y2, label, conf in boxes:
            color = (255,0,0) if label=='person' else (0,0,255)
            cv2.rectangle(full_frame, (x1,y1), (x2,y2), color, 2)
            text = (
                f"{label_map[label]} {conf*100:.1f}%"
                if conf is not None else f"{label_map[label]} (추적됨)"
            )
            img_pil = Image.fromarray(full_frame[:,:,::-1])
            draw = ImageDraw.Draw(img_pil)
            w_txt, h_txt = draw.textsize(text, font=font)
            draw.rectangle([x1, y1-h_txt-4, x1+w_txt+4, y1], fill=(0,0,0))
            draw.text((x1+2, y1-h_txt-2), text, font=font, fill=(255,255,255))
            full_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # encode & enqueue
        _, buf = cv2.imencode('.jpg', full_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not frame_queue.empty():
            frame_queue.get_nowait()
        frame_queue.put(buf.tobytes())

        # update fps
        elapsed = time.time() - start
        with fps_lock:
            current_fps = 1.0/elapsed if elapsed>0 else 0.0

        # minimize sleep
        time.sleep(0.001)

threading.Thread(target=capture_and_track, daemon=True).start()

# ----------------------------------------
# 5) Flask 앱 및 엔드포인트
app = Flask(__name__)

def generate():
    while True:
        frame = frame_queue.get()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    resp = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers.update({
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    })
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
        fps = round(current_fps, 1)
    return jsonify(
        camera=1,
        cpu_percent=cpu,
        memory_percent=mem,
        temperature_c=temp,
        wifi_signal_dbm=sig,
        fps=fps
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
