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
import shelve

import cv2
import numpy as np
import torch
import psutil
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, Response, render_template, jsonify

# ─────────────────────────────────────────────────────────────────────────────
# 1) 디스크 기반 캐시(shelve) 열기 (결코 삭제하지 않음)
CACHE_SHELF = '/home/user/cache_shelf.db'
REPEAT_SHELF = '/home/user/repeat_shelf.db'
disk_cache   = shelve.open(CACHE_SHELF, writeback=False)
disk_repeat  = shelve.open(REPEAT_SHELF, writeback=False)

# 2) PyTorch 모델 로드 (4단계)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
model_fast   = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True).eval()
model_refine = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).eval()
model_heavy  = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True).eval()

# 3) 단계별 설정
STAGE_CONFIG = {
    1: {'size': (160,  90),  'model': model_fast,   'thresh': 0.80},
    2: {'size': (320, 180),  'model': model_fast,   'thresh': 0.65},
    3: {'size': (640, 360),  'model': model_refine, 'thresh': 0.50},
    4: {'size': (1280,720),  'model': model_heavy,  'thresh': 0.50},
}
MAX_STAGE = 4
skip_interval = 2

# 4) 레이블 및 폰트 설정
label_map = {'person': '사람', 'car': '자동차'}
font_paths = [
    '/usr/share/fonts/truetype/noto/NotoSansCJKkr-Regular.otf',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
]
font = None
for p in font_paths:
    if os.path.isfile(p):
        try:
            font = ImageFont.truetype(p, 24)
            print(f"✅ 폰트 로드: {p}")
            break
        except OSError:
            pass
if font is None:
    print("⚠️ 한글 폰트 미발견, 기본 폰트 사용")
    font = ImageFont.load_default()

# 5) 카메라 인터페이스 정의
class CSICamera:
    def __init__(self):
        from picamera2 import Picamera2
        picam2 = Picamera2()
        cfg = picam2.create_video_configuration(
            main={"size": (1280,720)}, lores={"size": (640,360)}, buffer_count=2
        )
        picam2.configure(cfg)
        picam2.start()
        for _ in range(3):
            picam2.capture_array("main")
        self.picam2 = picam2
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
        if not self.cap:
            raise RuntimeError("USB 카메라를 찾을 수 없습니다.")
    def read(self):
        return self.cap.read()

class VideoFileCamera:
    def __init__(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"파일 열기 실패: {path}")
        self.cap = cap
    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

# 6) 카메라 선택 및 초기 진행 상황
video_path = os.path.expanduser('~/Desktop/1.mp4')
if os.path.isfile(video_path):
    camera, use_file = VideoFileCamera(video_path), True
    total_frames = int(camera.cap.get(cv2.CAP_PROP_FRAME_COUNT))
else:
    try:
        camera, use_file = CSICamera(), False
    except:
        camera, use_file = USBCamera(), False

print(f">>> {'비디오 파일' if use_file else '라이브 카메라'} 사용 중")
if use_file:
    recorded = len(disk_repeat)
    print(f"진행: 총 {total_frames}프레임 중 {recorded}개 기록, 남은 {total_frames-recorded}개")

# 7) 메모리 캐시(LRU 인덱스만 유지, 디스크는 보존)
from collections import OrderedDict
MEM_CACHE_MAX = 1000
mem_keys = OrderedDict()

# 8) 프레임 처리 스레드
frame_queue = queue.Queue(maxsize=1)

def capture_and_process():
    fps, interval = 10, 1.0/10
    frame_counter = 0
    last = None

    while True:
        t0 = time.time()
        ret, frame = camera.read()
        if not ret:
            continue

        frame_counter += 1
        # skip logic
        if frame_counter % skip_interval != 0 and last:
            results, infer_size = last
        else:
            key = str(int(camera.cap.get(cv2.CAP_PROP_POS_FRAMES))) if use_file else str(frame_counter)

            # 1) repeat count 업데이트
            rc = disk_repeat.get(key, 0) + 1
            disk_repeat[key] = rc
            stage = min(rc, MAX_STAGE)

            # 2) cache lookup
            if key in mem_keys:
                results, infer_size, cs, conf = disk_cache[key]
            else:
                if key in disk_cache:
                    results, infer_size, cs, conf = disk_cache[key]
                else:
                    cfg = STAGE_CONFIG[stage]
                    inp = cv2.resize(frame, cfg['size'])
                    with torch.no_grad(): res = cfg['model'](inp)
                    confs = res.xyxy[0][:,4]
                    conf = confs.max().item() if confs.numel() else 0.0
                    results, infer_size, cs, conf = res, cfg['size'], stage, conf
                    try:
                        disk_cache[key] = (results, infer_size, stage, conf)
                    except Exception as e:
                        print(f"⚠️ 디스크 캐시 저장 실패: {e}")
                # 조건 미달 시 재추론
                if cs < stage or (stage < MAX_STAGE and conf < STAGE_CONFIG[stage]['thresh']):
                    cfg = STAGE_CONFIG[stage]
                    inp = cv2.resize(frame, cfg['size'])
                    with torch.no_grad(): res = cfg['model'](inp)
                    confs = res.xyxy[0][:,4]
                    conf = confs.max().item() if confs.numel() else 0.0
                    results, infer_size, cs, conf = res, cfg['size'], stage, conf
                    try:
                        disk_cache[key] = (results, infer_size, stage, conf)
                    except Exception as e:
                        print(f"⚠️ 디스크 캐시 저장 실패: {e}")

            # 3) LRU 인덱스 업데이트
            mem_keys[key] = None
            mem_keys.move_to_end(key)
            if len(mem_keys) > MEM_CACHE_MAX:
                old, _ = mem_keys.popitem(last=False)
            last = (results, infer_size)

        # 그리기 및 인코딩
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        h_ratio = frame.shape[0] / infer_size[1]
        w_ratio = frame.shape[1] / infer_size[0]
        for *b, cf, cl in results.xyxy[0]:
            if cf < 0.20:
                continue
            x1, y1, x2, y2 = map(int, (b[0]*w_ratio, b[1]*h_ratio, b[2]*w_ratio, b[3]*h_ratio))
            kn = results.names[int(cl)]
            ko = label_map.get(kn)
            if ko:
                txt = f"{ko} {cf.item()*100:.1f}%"
                col = (255, 0, 0) if kn == 'car' else (0, 0, 255)
                draw.rectangle([x1, y1, x2, y2], outline=col, width=2)
                draw.text((x1, y1-30), txt, font=font, fill=col)

        buf = cv2.imencode('.jpg', cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1]
        frame_queue.put(buf.tobytes())

        time.sleep(max(0, interval - (time.time() - t0)))

threading.Thread(target=capture_and_process, daemon=True).start()

# 9) Flask 앱
app = Flask(__name__)

def generate():
    while True:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_queue.get() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    resp = Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    direct_passthrough=True)
    resp.headers.update({'Cache-Control': 'no-cache', 'Pragma': 'no-cache', 'Expires': '0'})
    return resp

@app.route('/stats')
def stats():
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory().percent
    temp = None
    try:
        temp = float(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000.0
    except:
        pass
    signal = None
    try:
        out = subprocess.check_output(['iwconfig', 'wlan0'], stderr=subprocess.DEVNULL).decode()
        for part in out.split():
            if part.startswith('level='):
                signal = int(part.split('=')[1])
    except:
        pass
    return jsonify(cpu_percent=cpu, memory_percent=mem, temperature_c=temp, wifi_signal_dbm=signal)

@app.route('/progress')
def progress():
    tot = int(camera.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if use_file else None
    rec = len(disk_repeat)
    rem = tot - rec if tot is not None else None
    return jsonify(total_frames=tot, recorded_frames=rec, remaining_frames=rem)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)    app.run(host='0.0.0.0', port=5000)
