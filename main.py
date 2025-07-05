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
# 0) 실행 중 화면 꺼짐·절전 모드 방지
# 콘솔(blank) + X11(DPMS) 모두 비활성화
try:
    os.system('setterm -blank 0 -powerdown 0 -powersave off')
    os.environ.setdefault('DISPLAY', ':0')
    os.system('xset s off')
    os.system('xset s noblank')
    os.system('xset -dpms')
    print("⏱️ 화면 절전/블랭킹 기능 비활성화 완료")
except Exception as e:
    print("⚠️ 전원/스크린세이버 비활성화 중 오류:", e)
# ──────────────────────────────────────────────────────────────────────────────

# 1) 모델 로드 (한 번만) – yolov5n 으로 가볍게
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# 2) 카메라 인터페이스 통일
class CSICamera:
    """Raspberry Pi CSI 카메라 모듈을 Picamera2로 제어"""
    def __init__(self):
        from picamera2 import Picamera2
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
    """USB 웹캠을 OpenCV로 제어 — MJPEG, 버퍼 최소화, 초기 플러시"""
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

# 3) 카메라 선택: CSI 우선, 없으면 USB
try:
    camera = CSICamera()
    print(">>> Using CSI camera module")
except Exception:
    camera = USBCamera()
    print(">>> Using USB webcam")

# 4) 백그라운드 프레임 처리 스레드 + 큐
frame_queue = queue.Queue(maxsize=1)

def capture_and_process():
    fps = 15
    interval = 1.0 / fps
    last = time.time()

    while True:
        now = time.time()
        sleep = interval - (now - last)
        if sleep > 0:
            time.sleep(sleep)
        last = time.time()

        ret, frame = camera.read()
        if not ret:
            continue

        # YOLOv5 인퍼런스
        small = cv2.resize(frame, (640, 640))
        results = model(small)

        # 박스 그리기
        h_ratio = frame.shape[0] / 640
        w_ratio = frame.shape[1] / 640
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

        # JPEG 인코딩
        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        data = buf.tobytes()

        # 큐에 최신 프레임만 유지
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(data)

threading.Thread(target=capture_and_process, daemon=True).start()

# 5) Flask 앱 & 스트리밍 + 통계 엔드포인트
app = Flask(__name__)

def generate():
    while True:
        frame = frame_queue.get()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

def get_network_signal(interface='wlan0'):
    """iwconfig 파싱하여 dBm 리턴"""
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
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
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
