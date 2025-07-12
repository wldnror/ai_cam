#!/usr/bin/env python3
import os
# ──────────────────────────────────────────────────────────────────────────────
# 0) OpenMP/BLAS 쓰레드풀 크기 설정 (전체 CPU 활용)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import warnings
warnings.filterwarnings("ignore")  # 모든 파이썬 경고 억제

import logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)  # yolov5 허브 로깅 억제
logging.getLogger('torch').setLevel(logging.WARNING)
werkzeug_log = logging.getLogger('werkzeug')  # Flask 요청 로그 억제
werkzeug_log.setLevel(logging.ERROR)

import os
import time
import threading
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
import psutil
from flask import Flask, Response, render_template, jsonify, request

# ──────────────────────────────────────────────────────────────────────────────
# 0) 화면 절전/DPMS 비활성화 (X가 있을 때만)
try:
    if os.environ.get('DISPLAY'):
        os.system('setterm -blank 0 -powerdown 0 -powersave off')
        os.system('xset s off; xset s noblank; xset -dpms')
        print("⏱️ 화면 절전/스크린세이버 비활성화 완료")
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# 1) PyTorch 스레드 & 추론 모드 최적화
torch.set_num_threads(4)
torch.set_num_interop_threads(4)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.eval()

# ──────────────────────────────────────────────────────────────────────────────
# 2) 카메라 인터페이스 정의
class CSICamera:
    """Raspberry Pi CSI 카메라 모듈을 Picamera2로 제어"""
    def __init__(self):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main         = {"size": (1280, 720)},
            lores        = {"size": (640, 360)},
            buffer_count = 2
        )
        self.picam2.configure(config)
        self.picam2.start()
        # 워밍업 프레임 드랍
        self.picam2.capture_array("main")

    def read(self):
        return True, self.picam2.capture_array("main")


class USBCamera:
    """USB 웹캠을 OpenCV로 제어 — MJPEG, 버퍼 확대, 초기 플러시"""
    def __init__(self):
        self.cap = None
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)    # 드라이버 레벨 버퍼 4프레임
                for _ in range(5):
                    cap.read()
                self.cap = cap
                break
        if self.cap is None:
            raise RuntimeError("사용 가능한 USB 웹캠을 찾을 수 없습니다.")

    def read(self):
        return self.cap.read()


# CSI 우선, USB 대체
try:
    camera = CSICamera()
    print(">>> Using CSI camera module")
except Exception as e:
    print(f"[WARN] CSI init failed: {e}\n    Falling back to USB")
    camera = USBCamera()
    print(">>> Using USB webcam")


# ──────────────────────────────────────────────────────────────────────────────
# 3) 백그라운드 프레임 처리 스레드 + 큐
frame_queue = queue.Queue(maxsize=3)        # 버퍼 3프레임으로 확대
executor    = ThreadPoolExecutor(max_workers=2)

def inference_task(frame, target_size):
    small = cv2.resize(frame, target_size)
    with torch.no_grad():
        return model(small)

def capture_and_process():
    fps           = 10
    interval      = 1.0 / fps
    target_size   = (320, 320)
    skip_interval = 2
    frame_count   = 0
    last_results  = None
    future        = None

    while True:
        start_time = time.time()

        ret, frame = camera.read()
        if not ret:
            continue

        frame_count += 1
        # 1) N번째 프레임마다 비동기 추론 제출
        if frame_count % skip_interval == 0:
            future = executor.submit(inference_task, frame, target_size)

        # 2) 완료된 추론 결과 가져오기
        if future and future.done():
            last_results = future.result()
            future = None

        if last_results is None:
            # 아직 추론 결과가 없으면 다음 루프
            continue

        # 3) 박스 그리기
        h_ratio = frame.shape[0] / target_size[1]
        w_ratio = frame.shape[1] / target_size[0]
        for *box, conf, cls in last_results.xyxy[0]:
            x1, y1, x2, y2 = map(int, (
                box[0] * w_ratio,
                box[1] * h_ratio,
                box[2] * w_ratio,
                box[3] * h_ratio
            ))
            label = last_results.names[int(cls)]
            if label in ('person', 'car'):
                color = (0,0,255) if label=='person' else (255,0,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 4) JPEG 인코딩
        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        data = buf.tobytes()

        # 5) 큐에 최신 프레임만 유지
        try:
            frame_queue.put_nowait(data)
        except queue.Full:
            # 가득 차 있으면 가장 오래된 프레임 버리고 넣기
            frame_queue.get_nowait()
            frame_queue.put_nowait(data)

        # 6) 프레임 간격 보정
        elapsed = time.time() - start_time
        sleep = interval - elapsed
        if sleep > 0:
            time.sleep(sleep)

threading.Thread(target=capture_and_process, daemon=True).start()


# ──────────────────────────────────────────────────────────────────────────────
# 4) Flask 앱 & 스트리밍 + 통계 엔드포인트
app = Flask(__name__)

def generate():
    while True:
        frame = frame_queue.get()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame +
               b'\r\n')

def get_network_signal(interface='wlan0'):
    try:
        out = subprocess.check_output(['iwconfig', interface], stderr=subprocess.DEVNULL).decode()
        for part in out.split():
            if part.startswith('level='):
                return int(part.split('=')[1])
    except Exception:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    resp = Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    direct_passthrough=True)
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/stats')
def stats():
    cam_no = request.args.get('cam', default=1, type=int)
    cpu    = psutil.cpu_percent(interval=0.5)
    mem    = psutil.virtual_memory().percent
    temp   = None
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            temp = float(f.read()) / 1000.0
    except Exception:
        pass
    signal = get_network_signal('wlan0')
    return jsonify({
        'camera':         cam_no,
        'cpu_percent':    cpu,
        'memory_percent': mem,
        'temperature_c':  temp,
        'wifi_signal_dbm': signal
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
