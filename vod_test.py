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

import cv2
import numpy as np
import torch
import psutil
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, Response, render_template, jsonify, request

# 0) 화면 절전/DPMS 비활성화
try:
    if os.environ.get('DISPLAY'):
        os.system('setterm -blank 0 -powerdown 0 -powersave off')
        os.system('xset s off; xset s noblank; xset -dpms')
        print("⏱️ 화면 절전/스크린세이버 비활성화 완료")
except Exception:
    pass

# 1) PyTorch 최적화
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True).eval()

# 2) 한국어 레이블 매핑 및 폰트 설정
label_map = { 'person': '사람', 'car': '자동차' }

# 시도할 한글 폰트 경로 목록
font_paths = [
    '/usr/share/fonts/truetype/noto/NotoSansCJKkr-Regular.otf',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
]

font = None
for path in font_paths:
    if os.path.isfile(path):
        try:
            font = ImageFont.truetype(path, 24)
            print(f"✅ 폰트 로드: {path}")
            break
        except OSError:
            continue
if font is None:
    print("⚠️ 사용할 한글 폰트를 찾지 못했습니다. 기본 폰트로 대체합니다.")
    font = ImageFont.load_default()

# 3) 카메라 인터페이스 정의
class CSICamera:
    def __init__(self):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (1280, 720)},
            lores={"size": (640, 360)},
            buffer_count=2
        )
        self.picam2.configure(config)
        self.picam2.start()
        for _ in range(3):
            self.picam2.capture_array("main")
    def read(self):
        return True, self.picam2.capture_array("main")

class USBCamera:
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
        if self.cap is None:
            raise RuntimeError("사용 가능한 USB 웹캠을 찾을 수 없습니다.")
    def read(self):
        return self.cap.read()

class VideoFileCamera:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"비디오 파일을 열 수 없습니다: {path}")
    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

# 4) 카메라 선택: Desktop/1.mp4 우선
video_path = os.path.expanduser('~/Desktop/1.mp4')
if os.path.isfile(video_path):
    camera = VideoFileCamera(video_path)
    print(f">>> 비디오 파일 사용: {video_path}")
else:
    try:
        camera = CSICamera()
        print(">>> CSI 카메라 모듈 사용")
    except:
        camera = USBCamera()
        print(">>> USB 웹캠 사용")

# 5) 프레임 처리 및 큐
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

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        h_ratio = frame.shape[0] / target_size[1]
        w_ratio = frame.shape[1] / target_size[0]

        for *box, conf, cls in last_results.xyxy[0]:
            # confidence 30% 이하인 경우 표시하지 않음
            if conf < 0.30:
                continue

            x1, y1, x2, y2 = map(int, (
                box[0] * w_ratio,
                box[1] * h_ratio,
                box[2] * w_ratio,
                box[3] * h_ratio
            ))
            label_en = last_results.names[int(cls)]
            if label_en in label_map:
                label_ko = label_map[label_en]
                color = (255, 0, 0) if label_en == 'car' else (0, 0, 255)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1, y1 - 30), label_ko, font=font, fill=color)

        frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        data = buf.tobytes()
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(data)

threading.Thread(target=capture_and_process, daemon=True).start()

# 6) Flask 앱
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
    except:
        return None

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
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    temp = None
    try:
        temp = float(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000.0
    except:
        pass
    signal = get_network_signal('wlan0')
    return jsonify({
        'cpu_percent': cpu,
        'memory_percent': mem.percent,
        'temperature_c': temp,
        'wifi_signal_dbm': signal
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
