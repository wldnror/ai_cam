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
import pickle
import atexit

import cv2
import numpy as np
import torch
import psutil
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, Response, render_template, jsonify, request

# 캐시 파일 경로
CACHE_PATH = '/home/user/cache.pkl'

# 0) 캐시 로드: 재부팅 후에도 사용
try:
    with open(CACHE_PATH, 'rb') as f:
        detection_cache, repeat_count = pickle.load(f)
        print("✅ 캐시 로드 완료")
except Exception:
    detection_cache = {}
    repeat_count     = {}

# 종료 직전 캐시 저장
@atexit.register
def save_on_exit():
    try:
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump((detection_cache, repeat_count), f)
        print("💾 종료 직전 캐시 저장 완료")
    except Exception as e:
        print(f"⚠️ 캐시 저장 실패: {e}")

# 1) PyTorch 모델 로드 (4단계)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
model_fast   = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True).eval()
model_refine = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).eval()
model_heavy  = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True).eval()

# 2) 캐시, 재생 횟수, 단계 설정
STAGE_CONFIG = {
    1: {'size': (160, 90),   'model': model_fast,   'thresh': 0.80},  # 초저해상도
    2: {'size': (320, 180),  'model': model_fast,   'thresh': 0.65},  # 저해상도
    3: {'size': (640, 360),  'model': model_refine, 'thresh': 0.50},  # 중간해상도
    4: {'size': (1280, 720), 'model': model_heavy,  'thresh': 0.50},  # 원본 해상도
}
MAX_STAGE = 4
skip_interval = 2

detection_cache = detection_cache
repeat_count     = repeat_count

# 3) 레이블 매핑 및 폰트 설정
label_map = {'person': '사람', 'car': '자동차'}
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

# 4) 카메라 인터페이스 정의
class CSICamera:
    def __init__(self):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (1280, 720)}, lores={"size": (640, 360)}, buffer_count=2
        )
        self.picam2.configure(config)
        self.picam2.start()
        for _ in range(3): self.picam2.capture_array("main")
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
                for _ in range(5): cap.read()
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

# 5) 카메라 선택
video_path = os.path.expanduser('~/Desktop/1.mp4')
if os.path.isfile(video_path):
    camera = VideoFileCamera(video_path)
    use_file = True
    print(f">>> 비디오 파일 사용: {video_path}")
else:
    try:
        camera = CSICamera()
        use_file = False
        print(">>> CSI 카메라 모듈 사용")
    except:
        camera = USBCamera()
        use_file = False
        print(">>> USB 웹캠 사용")

# 6) 주기적 캐시 저장 스레드
threading.Thread(target=lambda: (time.sleep(60), pickle.dump((detection_cache, repeat_count), open(CACHE_PATH,'wb')), print("💾 주기적 캐시 저장 완료")), daemon=True).start()

# 7) 프레임 처리 및 큐
frame_queue = queue.Queue(maxsize=1)

def capture_and_process():
    fps = 10
    interval = 1.0 / fps
    frame_count = 0
    last_results = None

    while True:
        start = time.time()
        ret, frame = camera.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % skip_interval != 0 and last_results is not None:
            results, infer_size = last_results
        else:
            frame_idx = int(camera.cap.get(cv2.CAP_PROP_POS_FRAMES)) if use_file else frame_count
            repeat_count[frame_idx] = repeat_count.get(frame_idx, 0) + 1
            desired_stage = min(repeat_count[frame_idx], MAX_STAGE)

            cached = detection_cache.get(frame_idx)
            use_cached = False
            if cached:
                _, _, cached_stage, cached_conf = cached
                if cached_stage >= desired_stage:
                    if desired_stage == MAX_STAGE or cached_conf >= STAGE_CONFIG[desired_stage]['thresh']:
                        use_cached = True
            if use_cached:
                results, infer_size, _, _ = cached
            else:
                cfg = STAGE_CONFIG[desired_stage]
                inp = cv2.resize(frame, cfg['size'])
                with torch.no_grad():
                    res = cfg['model'](inp)
                confs = res.xyxy[0][:,4]
                max_conf = confs.max().item() if confs.numel()>0 else 0.0
                results = res
                infer_size = cfg['size']
                detection_cache[frame_idx] = (results, infer_size, desired_stage, max_conf)

        last_results = (results, infer_size)

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        h_ratio = frame.shape[0] / infer_size[1]
        w_ratio = frame.shape[1] / infer_size[0]
        for *box, conf, cls in results.xyxy[0]:
            if conf < 0.20: continue
            x1, y1, x2, y2 = map(int, (box[0]*w_ratio, box[1]*h_ratio, box[2]*w_ratio, box[3]*h_ratio))
            label_en = results.names[int(cls)]
            label_ko = label_map.get(label_en)
            if label_ko:
                text = f"{label_ko} {conf.item()*100:.1f}%"
                color = (255,0,0) if label_en=='car' else (0,0,255)
                draw.rectangle([x1,y1,x2,y2], outline=color, width=2)
                draw.text((x1, y1-30), text, font=font, fill=color)

        frame_out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', frame_out, [int(cv2.IMWRITE_JPEG_QUALITY),80])
        data = buf.tobytes()
        if not frame_queue.empty():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put(data)

        elapsed = time.time() - start
        time.sleep(max(0, interval - elapsed))

threading.Thread(target=capture_and_process, daemon=True).start()

# 8) Flask 앱
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
    resp = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame', direct_passthrough=True)
    resp.headers.update({'Cache-Control':'no-cache, no-store, must-revalidate','Pragma':'no-cache','Expires':'0'})
    return resp

@app.route('/stats')
def stats():
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    temp = None
    try: temp = float(open('/sys/class/thermal/thermal_zone0/temp').read())/1000.0
    except: pass
    signal = None
    try:
        out = subprocess.check_output(['iwconfig','wlan0'], stderr=subprocess.DEVNULL).decode()
        for part in out.split():
            if part.startswith('level='): signal=int(part.split('=')[1])
    except: pass
    return jsonify({'cpu_percent':cpu,'memory_percent':mem.percent,'temperature_c':temp,'wifi_signal_dbm':signal})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
