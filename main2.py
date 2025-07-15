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
import glob

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
# 1) YOLOv5 모델 로드 (DetectMultiBackend + AutoShape)
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

# ──────────────────────────────────────────────────────────────────────────────
# 2) USB 캠 정의 및 재시작 로직
class USBCamera:
    def __init__(self):
        self.cap = None
        self.open()

    def open(self):
        # /dev/video*에서 사용 가능한 디바이스 검색
        device_paths = sorted(glob.glob('/dev/video*'))
        for dev in device_paths:
            cap = cv2.VideoCapture(dev)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(5):
                    cap.read()
                self.cap = cap
                print(f"[INFO] Opened camera device: {dev}")
                return
        raise RuntimeError("사용 가능한 USB 웹캠이 없습니다.")

    def restart(self):
        try:
            if self.cap:
                self.cap.release()
        except:
            pass
        time.sleep(1)
        print("[INFO] Restarting USB camera...")
        self.open()
        print("[INFO] USB camera reinitialized.")

    def read(self):
        return self.cap.read()

# 초기 카메라 생성
try:
    camera = USBCamera()
except Exception as e:
    print(f"[ERROR] 초기 카메라 열기 실패: {e}")
    sys.exit(1)
print(">>> Using USB webcam only")

# ──────────────────────────────────────────────────────────────────────────────
# 3) 큐 및 쓰레드 구조
raw_queue = queue.Queue(maxsize=2)
frame_queue = queue.Queue(maxsize=2)

_last_frame_time = time.time()

def capture_loop():
    global _last_frame_time, camera
    while True:
        try:
            ret, frame = camera.read()
        except Exception as e:
            print(f"[ERROR] Camera read exception: {e}")
            camera.restart()
            continue

        now = time.time()
        if ret:
            _last_frame_time = now
            if not raw_queue.full():
                raw_queue.put(frame)
        else:
            if now - _last_frame_time > 2.0:
                print("[WARN] No frames received for 2s, attempting camera restart")
                camera.restart()
                _last_frame_time = now
            time.sleep(0.01)


def inference_loop():
    fps = 10
    interval = 1.0 / fps
    target_size = 320
    skip_interval = 2
    count = 0
    last_log = time.time()
    infer_count = 0

    while True:
        start = time.time()
        try:
            frame = raw_queue.get(timeout=1)
        except queue.Empty:
            print("[WARN] raw_queue empty, skipping inference")
            continue

        count += 1
        if count % skip_interval == 0:
            try:
                inf_start = time.time()
                with torch.no_grad():
                    results = model(frame, size=target_size)
                inf_time = (time.time() - inf_start) * 1000
                infer_count += 1

                if infer_count % 10 == 0:
                    now = time.time()
                    real_fps = 10 / (now - last_log)
                    print(f"[INFO] Inference FPS: {real_fps:.2f}, Avg latency: {inf_time:.1f} ms")
                    last_log = now

                for *box, conf, cls in results.xyxy[0]:
                    x1, y1, x2, y2 = map(int, box)
                    label = results.names[int(cls)]
                    if label in ('person', 'car'):
                        color = (0,0,255) if label=='person' else (255,0,0)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except Exception as e:
                print(f"[ERROR] Inference exception: {e}")

        try:
            _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY),80])
            if not frame_queue.full():
                frame_queue.put(buf.tobytes())
        except Exception as e:
            print(f"[ERROR] JPEG encoding exception: {e}")

        elapsed = time.time() - start
        if elapsed < interval:
            time.sleep(interval - elapsed)

threading.Thread(target=capture_loop, daemon=True).start()
threading.Thread(target=inference_loop, daemon=True).start()

# ──────────────────────────────────────────────────────────────────────────────
# 4) Flask 앱 및 엔드포인트
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
    resp = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers.update({'Cache-Control':'no-cache, no-store, must-revalidate', 'Pragma':'no-cache','Expires':'0'})
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
    sig = None
    try:
        out = subprocess.check_output(['iwconfig','wlan0'], stderr=subprocess.DEVNULL).decode()
        for p in out.split():
            if p.startswith('level='):
                sig = int(p.split('=')[1])
    except:
        pass
    return jsonify(camera=1, cpu_percent=cpu, memory_percent=mem, temperature_c=temp, wifi_signal_dbm=sig)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
