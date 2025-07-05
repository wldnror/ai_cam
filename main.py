#!/usr/bin/env python3
import os
import time
import threading
import queue
import subprocess

import cv2
import torch
import psutil
from flask import Flask, Response, render_template, jsonify, request

# ──────────────────────────────────────────────────────────────────────────────
# 0) 실행 중 화면 꺼짐·절전 모드 방지 (콘솔 전용이라면 아예 불필요)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# 1) PyTorch 최적화 설정
# ──────────────────────────────────────────────────────────────────────────────
# (a) 스레드 수 제한 → 컨텍스트 스위치 오버헤드 감소
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# (b) 모델 로드 후 eval 모드 & no_grad 사용
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.eval()
# ──────────────────────────────────────────────────────────────────────────────

# 2) 카메라 인터페이스 통일 (변경 없음)
class CSICamera:
    from picamera2 import Picamera2
    def __init__(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main          = {"size": (1280, 720)},
            lores         = {"size": (640, 360)},
            buffer_count  = 2
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
                cap.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc(*'MJPG'))
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

try:
    camera = CSICamera()
    print(">>> Using CSI camera module")
except Exception:
    camera = USBCamera()
    print(">>> Using USB webcam")

# 3) 백그라운드 프레임 처리 스레드 + 큐
frame_queue = queue.Queue(maxsize=1)

def capture_and_process():
    # 해상도↓, FPS↓ 적용
    target_resolution = (320, 320)
    fps = 10
    interval = 1.0 / fps
    last = time.time()

    frame_count = 0
    skip_interval = 2   # 2프레임에 한 번씩 추론

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
        # 스킵 전략: skip_interval에 맞춰서만 추론
        if frame_count % skip_interval == 0:
            # resize once per inference
            small = cv2.resize(frame, target_resolution)
            with torch.no_grad():
                results = model(small)

        # 마지막 results를 계속 재사용
        # 박스 그리기 (원본 해상도에 맞춰 스케일)
        h_ratio = frame.shape[0] / target_resolution[1]
        w_ratio = frame.shape[1] / target_resolution[0]
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, (
                box[0] * w_ratio,
                box[1] * h_ratio,
                box[2] * w_ratio,
                box[3] * h_ratio
            ))
            label = results.names[int(cls)]
            if label in ('person', 'car'):
                color = (0,0,255) if label=='person' else (255,0,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # JPEG 인코딩 (필요하면 hardware encoder로 대체 고려)
        _, buf = cv2.imencode('.jpg', frame,
                              [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        data = buf.tobytes()

        # 최신 프레임만 유지
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(data)

threading.Thread(target=capture_and_process,
                 daemon=True).start()

# 4) Flask 앱 & 스트리밍 + 통계 엔드포인트 (변경 없음)
app = Flask(__name__)

def generate():
    while True:
        frame = frame_queue.get()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame + b'\r\n'
        )

def get_network_signal(interface='wlan0'):
    try:
        out = subprocess.check_output(['iwconfig', interface],
                                      stderr=subprocess.DEVNULL).decode()
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
    resp.headers.update({
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    })
    return resp

@app.route('/stats')
def stats():
    cam_no = request.args.get('cam', default=1, type=int)
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    temp = None
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            temp = float(f.read()) / 1000.0
    except Exception:
        pass
    signal = get_network_signal('wlan0')
    return jsonify({
        'camera':         cam_no,
        'cpu_percent':    cpu,
        'memory_percent': mem.percent,
        'temperature_c':  temp,
        'wifi_signal_dbm': signal
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
