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

# ──────────────────────────────────────────────────────────────────────────────
# 0) 한글 폰트 설치/로드
FONT_PATH = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if not os.path.exists(FONT_PATH):
    try:
        print("한글 폰트가 없어 설치 시도...")
        subprocess.run(['sudo','apt-get','update'], check=True)
        subprocess.run(['sudo','apt-get','install','-y','fonts-nanum'], check=True)
    except Exception as e:
        print(f"폰트 설치 실패: {e}")
try:
    font = ImageFont.truetype(FONT_PATH, 24)
except:
    font = ImageFont.load_default()
    print("한글 폰트 로드 실패, 기본 폰트 사용")

# ──────────────────────────────────────────────────────────────────────────────
# 1) 화면 절전/DPMS 비활성화
try:
    if os.environ.get('DISPLAY'):
        os.system('setterm -blank 0 -powerdown 0 -powersave off')
        os.system('xset s off; xset s noblank; xset -dpms')
        print("화면 절전 및 DPMS 비활성화 완료")
except:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# 2) PyTorch 스레드 & YOLOv5 모델 로드 (단일 모델)
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

YOLOROOT = os.path.expanduser('~/yolov5')
if not os.path.isdir(YOLOROOT):
    subprocess.run(['git','clone','https://github.com/ultralytics/yolov5.git', YOLOROOT], check=True)
sys.path.insert(0, YOLOROOT)
from models.common import DetectMultiBackend, AutoShape
from utils.torch_utils import select_device

device = select_device('cpu')

# 단일 모델 로드 함수
def load_model(weights):
    global backend, model
    path = os.path.join(YOLOROOT, weights)
    if not os.path.exists(path):
        torch.hub.download_url_to_file(
            f'https://github.com/ultralytics/yolov5/releases/download/v7.0/{weights}',
            path
        )
    backend = DetectMultiBackend(path, device=device, fuse=True)
    backend.model.eval()
    model = AutoShape(backend.model)
    print(f"[MODEL] Loaded {weights}")

# 원하는 가중치 파일만 로드 (예: yolov5m.pt)
load_model("yolov5m.pt")

# ──────────────────────────────────────────────────────────────────────────────
label_map = {'person':'사람','car':'자동차'}

# ──────────────────────────────────────────────────────────────────────────────
class CSICamera:
    def __init__(self):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        cfg = self.picam2.create_video_configuration(
            main={"size":(1280,720)},
            lores={"size":(640,360)},
            buffer_count=6
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
                cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,4)
                for _ in range(5): cap.read()
                self.cap = cap; break
        if not self.cap:
            raise RuntimeError("웹캠 없음")
    def read(self):
        return self.cap.read()

try:
    camera = CSICamera(); print(">>> CSI camera")
except Exception as e:
    print(f"[ERROR] CSI init failed: {e}")
    camera = USBCamera(); print(">>> USB camera")

frame_queue = queue.Queue(maxsize=3)

# 추적(Tracking) 준비
Tracker_create = None
for name in ("CSRT","KCF","MOSSE"):
    f = f"Tracker{name}_create"
    if hasattr(cv2, f):
        Tracker_create = getattr(cv2, f)
        print(f"Using cv2.{f}")
        break
    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, f):
        Tracker_create = getattr(cv2.legacy, f)
        print(f"Using cv2.legacy.{f}")
        break

use_tracking = Tracker_create is not None
if not use_tracking:
    print("경고: Tracker가 없어 매 프레임 감지-only 모드로 동작합니다.")

trackers = []      # (tracker, id, label, conf)
next_id = 0
REDTECT_INTERVAL = 30
frame_count = 0
DETECT_CONF_THRESH = 0.4

def capture_and_process():
    global trackers, next_id, frame_count

    fps = 15
    interval = 1.0/fps
    target_size = 320

    while True:
        start = time.time()
        ret, frame = camera.read()
        if not ret: continue
        frame_count += 1

        # -- 감지 or 추적 결정 --
        do_detect = (not use_tracking) or (frame_count % REDTECT_INTERVAL == 1) or (not trackers)

        boxes_to_draw = []
        if do_detect:
            # YOLO 감지
            with torch.no_grad():
                results = model(frame, size=target_size)
            dets = []
            for *box, conf, cls in results.xyxy[0]:
                if float(conf) < DETECT_CONF_THRESH: continue
                lbl = results.names[int(cls)]
                if lbl not in label_map: continue
                x1,y1,x2,y2 = map(int, box)
                dets.append((x1,y1,x2-x1,y2-y1,lbl,float(conf)))

            # Tracker 초기화
            if use_tracking:
                trackers = []
                for x,y,w,h,lbl,conf in dets:
                    trk = Tracker_create()
                    trk.init(frame,(x,y,w,h))
                    trackers.append((trk, next_id, lbl, conf))
                    next_id += 1

            boxes_to_draw = [(x,y,x+w,y+h,lbl,conf) for x,y,w,h,lbl,conf in dets]

        else:
            # 추적 단계
            new_trk = []
            for trk, tid, lbl, _ in trackers:
                ok, box = trk.update(frame)
                if not ok: continue
                x,y,w,h = map(int, box)
                new_trk.append((trk, tid, lbl, None))
                boxes_to_draw.append((x,y,x+w,y+h,lbl,None))
            trackers[:] = new_trk

            # 중간 재감지 (옵션)
            if use_tracking and frame_count % REDTECT_INTERVAL == 0:
                with torch.no_grad():
                    r2 = model(frame, size=target_size)
                for *box, conf, cls in r2.xyxy[0]:
                    if float(conf) < DETECT_CONF_THRESH: continue
                    lbl = results.names[int(cls)]
                    if lbl not in label_map: continue
                    x1,y1,x2,y2 = map(int, box)
                    trk = Tracker_create()
                    trk.init(frame,(x1,y1,x2-x1,y2-y1))
                    trackers.append((trk, next_id, lbl, float(conf)))
                    boxes_to_draw.append((x1,y1,x2,y2,lbl,float(conf)))
                    next_id += 1

        # -- 시각화 --
        for x1,y1,x2,y2,lbl,conf in boxes_to_draw:
            color = (255,0,0) if lbl=='person' else (0,0,255)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            txt = f"{label_map[lbl]} {conf*100:.1f}%" if conf else label_map[lbl]
            cv2.putText(frame,txt,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        # JPEG 인코딩 & 큐 업로드
        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY),60])
        if not frame_queue.empty():
            frame_queue.get_nowait()
        frame_queue.put(buf.tobytes())

        elapsed = time.time()-start
        time.sleep(max(0, interval-elapsed))

threading.Thread(target=capture_and_process, daemon=True).start()

# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
def generate():
    while True:
        f = frame_queue.get()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+f+b'\r\n')

@app.route('/')
def index(): return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    resp = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers.update({'Cache-Control':'no-cache','Pragma':'no-cache','Expires':'0'})
    return resp

@app.route('/stats')
def stats():
    c = psutil.cpu_percent(interval=0.5)
    m = psutil.virtual_memory().percent
    t = None
    try: t = float(open('/sys/class/thermal/thermal_zone0/temp').read())/1000.0
    except: pass
    try:
        out = subprocess.check_output(['iwconfig','wlan0'],stderr=subprocess.DEVNULL).decode()
        w = int([p.split('=')[1] for p in out.split() if p.startswith('level=')][0])
    except: w=None
    return jsonify(camera=1, cpu_percent=c, memory_percent=m, temperature_c=t, wifi_signal_dbm=w)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)
