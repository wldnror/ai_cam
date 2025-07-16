#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")  # 모든 파이썬 경고 억제

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
from flask import Flask, Response, render_template, jsonify, request
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# ----------------------------------------
# 0) CPU 스레드 및 BLAS 환경 변수 설정
n_cpu = multiprocessing.cpu_count()
# GIL 컨텐션 방지용으로 절반 정도만 OpenMP/BLAS에 할당
os.environ["OMP_NUM_THREADS"] = str(n_cpu//2)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cpu//2)
os.environ["MKL_NUM_THREADS"] = str(n_cpu//2)
torch.set_num_threads(n_cpu)
torch.set_num_interop_threads(n_cpu)

# ----------------------------------------
# 1) 한글 폰트 설치 확인 및 로드 (PIL은 더 이상 drawing에 사용하지 않지만 로딩해둡니다)
from PIL import ImageFont
FONT_PATH = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if not os.path.exists(FONT_PATH):
    try:
        subprocess.run(['sudo', 'apt-get', 'update'], check=True)
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'fonts-nanum'], check=True)
    except Exception as e:
        print(f"폰트 설치 실패: {e}")
try:
    font = ImageFont.truetype(FONT_PATH, 24)
except Exception:
    from PIL import ImageFont
    font = ImageFont.load_default()

# ----------------------------------------
# 2) 화면 절전/DPMS 비활성화
try:
    if os.environ.get('DISPLAY'):
        os.system('setterm -blank 0 -powerdown 0 -powersave off')
        os.system('xset s off; xset s noblank; xset -dpms')
        print("⏱️ 화면 절전/스크린세이버 비활성화 완료")
except Exception:
    pass

# ----------------------------------------
# 3) YOLOv5 모델 로드
YOLOROOT = os.path.expanduser('~/yolov5')
if not os.path.isdir(YOLOROOT):
    subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git', YOLOROOT], check=True)
sys.path.insert(0, YOLOROOT)
from models.common import DetectMultiBackend, AutoShape
from utils.torch_utils import select_device

device = select_device('cpu')  # 혹은 'cuda' 사용 가능
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
# 4) 카메라 클래스 정의 (변경 없음)
class CSICamera:
    def __init__(self):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        cfg = self.picam2.create_video_configuration(
            main={"size": (720, 480)}, lores={"size": (400, 400)}, buffer_count=6
        )
        self.picam2.configure(cfg)
        self.picam2.start()
        for _ in range(3):
            self.picam2.capture_array("main")
    def read(self):
        rgb = self.picam2.capture_array("main")
        return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

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
# 5) 비동기 추론 함수 정의
def async_inference(img, size):
    return model(img, size=size)

# ----------------------------------------
# 6) 프레임 처리 파이프라인
frame_queue = queue.Queue(maxsize=8)
current_fps = 0.0
fps_lock = threading.Lock()

def create_tracker():
    for module_name, func_name in [
        ('cv2','TrackerMOSSE_create'),('cv2.legacy','TrackerMOSSE_create'),
        ('cv2.legacy','TrackerCSRT_create'),('cv2','TrackerCSRT_create'),
        ('cv2.legacy','TrackerKCF_create'),('cv2','TrackerKCF_create'),
        ('cv2.legacy','TrackerMIL_create'),('cv2','TrackerMIL_create')
    ]:
        try:
            module = cv2.legacy if module_name=='cv2.legacy' and hasattr(cv2,'legacy') else cv2
            fn = getattr(module, func_name, None)
            if fn:
                return fn()
        except:
            pass
    raise RuntimeError("사용 가능한 트래커가 없습니다.")

def capture_and_track():
    global current_fps
    target_size = 360
    detection_interval = 10  # 매 10프레임마다 검출
    frame_count = 0
    trackers = []
    executor = ProcessPoolExecutor(max_workers=2)

    while True:
        start = time.time()
        ret, full_frame = camera.read()
        if not ret:
            continue

        frame_count += 1
        boxes = []

        if frame_count % detection_interval == 0:
            trackers.clear()
            small = cv2.resize(full_frame, (target_size, target_size))
            future = executor.submit(async_inference, small, target_size)
            results = future.result()

            for *box, conf, cls in results.xyxy[0]:
                label = results.names[int(cls)]
                if label not in label_map:
                    continue
                x1, y1, x2, y2 = map(int, [
                    box[0] * full_frame.shape[1] / target_size,
                    box[1] * full_frame.shape[0] / target_size,
                    box[2] * full_frame.shape[1] / target_size,
                    box[3] * full_frame.shape[0] / target_size
                ])
                trk = create_tracker()
                trk.init(full_frame, (x1, y1, x2-x1, y2-y1))
                trackers.append((trk, label))
                boxes.append((x1, y1, x2, y2, label, float(conf)))
        else:
            for trk, label in trackers:
                success, bbox = trk.update(full_frame)
                if not success:
                    continue
                x, y, w, h = map(int, bbox)
                boxes.append((x, y, x+w, y+h, label, None))

        # OpenCV 순수 그리기
        for x1, y1, x2, y2, label, conf in boxes:
            color = (255,0,0) if label=='person' else (0,0,255)
            cv2.rectangle(full_frame, (x1,y1), (x2,y2), color, 2)
            text = (f"{label_map[label]} {conf*100:.1f}%"
                    if conf is not None else f"{label_map[label]} (추적됨)")
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(full_frame,
                          (x1, y1-th-4), (x1+tw+4, y1),
                          (0,0,0), -1)
            cv2.putText(full_frame, text,
                        (x1+2, y1-2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 1)

        # JPEG 인코딩 & 큐에 저장
        _, buf = cv2.imencode('.jpg', full_frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 40])
        if not frame_queue.empty():
            frame_queue.get_nowait()
        frame_queue.put(buf.tobytes())

        # FPS 계산
        elapsed = time.time() - start
        with fps_lock:
            current_fps = 1.0/elapsed if elapsed>0 else 0.0

        time.sleep(0.001)

threading.Thread(target=capture_and_track, daemon=True).start()

# ----------------------------------------
# 7) Flask 앱
app = Flask(__name__)

def generate():
    while True:
        frame = frame_queue.get()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame + b'\r\n')

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
    mem = psutil.virtual_memory().percent
    temp = None
    try:
        temp = float(open(
            '/sys/class/thermal/thermal_zone0/temp').read())/1000.0
    except:
        pass
    try:
        out = subprocess.check_output(
            ['iwconfig','wlan0'], stderr=subprocess.DEVNULL
        ).decode()
        sig = int([p.split('=')[1]
                   for p in out.split() if p.startswith('level=')][0])
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
