#!/usr/bin/env python3
import warnings, logging, os, sys, time, threading, queue, subprocess
import cv2, torch, psutil, numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, Response, render_template, jsonify

# ────────── 0) SORT 소스 경로 추가 ──────────
sys.path.append(os.path.join(os.path.dirname(__file__), 'sort'))
from sort import Sort

# ────────── 1) 한글 폰트 로드 ──────────
FONT_PATH = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if not os.path.exists(FONT_PATH):
    subprocess.run(['sudo','apt-get','update'], check=True)
    subprocess.run(['sudo','apt-get','install','-y','fonts-nanum'], check=True)
try:
    font = ImageFont.truetype(FONT_PATH, 24)
except:
    font = ImageFont.load_default()

# ────────── 2) 화면 절전 해제 ──────────
if os.environ.get('DISPLAY'):
    os.system('setterm -blank 0 -powerdown 0 -powersave off')
    os.system('xset s off; xset s noblank; xset -dpms')

# ────────── 3) YOLOv5 로드 ──────────
torch.set_num_threads(8); torch.set_num_interop_threads(8)
YOLOROOT = os.path.expanduser('~/yolov5')
if not os.path.isdir(YOLOROOT):
    subprocess.run(['git','clone','https://github.com/ultralytics/yolov5.git', YOLOROOT], check=True)
sys.path.insert(0, YOLOROOT)
from models.common import DetectMultiBackend, AutoShape
from utils.torch_utils import select_device

device = select_device('cpu')
backend = DetectMultiBackend(os.path.join(YOLOROOT,'yolov5n.pt'), device=device, fuse=True)
backend.model.eval()
model = AutoShape(backend.model)
label_map = {'person':'사람','car':'자동차'}

# ────────── 4) 카메라 추상화 ──────────
class CSICamera:
    def __init__(self):
        from picamera2 import Picamera2
        self.pic = Picamera2()
        cfg = self.pic.create_video_configuration(main={"size":(720,480)})
        self.pic.configure(cfg); self.pic.start()
        for _ in range(3): self.pic.capture_array("main")
    def read(self):
        rgb = self.pic.capture_array("main")
        return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

class USBCamera:
    def __init__(self):
        self.cap = next((cv2.VideoCapture(i) for i in range(5) if cv2.VideoCapture(i).isOpened()), None)
        if not self.cap: raise RuntimeError("No USB camera")
    def read(self):
        return self.cap.read()

try:
    camera = CSICamera()
    print(">>> CSI camera")
except:
    camera = USBCamera()
    print(">>> USB camera")

# ────────── 5) SORT 트래커 초기화 ──────────
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
frame_q = queue.Queue(maxsize=3)
fps_lock = threading.Lock(); current_fps = 0.0

def capture_and_process():
    global current_fps
    fps_interval = 1/10
    while True:
        t0 = time.time()
        ret, frame = camera.read()
        if not ret: continue

        # YOLO 검출
        res = model(frame, size=270)
        dets = []
        for *box, conf, cls in res.xyxy[0]:
            lbl = res.names[int(cls)]
            if lbl not in label_map: continue
            x1,y1,x2,y2 = map(int, box)
            dets.append([x1,y1,x2,y2,float(conf)])

        # SORT 업데이트
        dets_np = np.array(dets) if dets else np.empty((0,5))
        tracks = tracker.update(dets_np)

        # 박스 그리기
        for x1,y1,x2,y2,tid in tracks.astype(int):
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame, f"ID{tid}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)

        # 인코딩 & 큐
        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY),60])
        if not frame_q.empty(): frame_q.get_nowait()
        frame_q.put(buf.tobytes())

        # FPS 계산
        with fps_lock:
            elapsed = time.time()-t0
            current_fps = 1/elapsed if elapsed>0 else 0
        time.sleep(max(0, fps_interval - elapsed))

threading.Thread(target=capture_and_process, daemon=True).start()

# ────────── 6) Flask 서버 ──────────
app = Flask(__name__)

def gen():
    while True:
        f = frame_q.get()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+f+b'\r\n')

@app.route('/video_feed')
def video_feed():
    resp = Response(gen(),
                    mimetype='multipart/x-mixed-replace;boundary=frame')
    resp.headers.update({'Cache-Control':'no-cache'})
    return resp

@app.route('/stats')
def stats():
    cpu = psutil.cpu_percent(0.5)
    mem = psutil.virtual_memory().percent
    try:
        temp = float(open('/sys/class/thermal/thermal_zone0/temp').read())/1000
    except: temp = None
    with fps_lock: fpsv = round(current_fps,1)
    return jsonify(camera=1, cpu_percent=cpu,
                   memory_percent=mem,
                   temperature_c=temp, fps=fpsv)

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)
