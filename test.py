#!/usr/bin/env python3
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

import cv2
import torch
import psutil
from flask import Flask, Response, render_template, jsonify, request

# ──────────────────────────────────────────────────────────────────────────────
# AI 카메라(IMX500) 자동 사용 여부 결정
use_ai_camera = False
try:
    from picamera2 import Picamera2, MappedArray
    from picamera2.devices import IMX500
    # 사전 학습된 RPK 모델 경로
    model_path = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
    imx500 = IMX500(model_path)
    intrinsics = imx500.network_intrinsics
    use_ai_camera = True
    print("⏱️ Using Raspberry Pi AI Camera with on-sensor acceleration")
except Exception as e:
    print("⚠️ AI Camera unavailable, falling back to CPU mode:", e)

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
# 1) PyTorch 스레드 & 추론 모드 최적화 (CPU 모드일 때만)
if not use_ai_camera:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    model.eval()

# ──────────────────────────────────────────────────────────────────────────────
# 2) 카메라 초기화
if use_ai_camera:
    # AI 카메라 전용 Picamera2 설정
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720)},
        lores={"size": (640, 360)},
        controls={"FrameRate": intrinsics.inference_rate},
        buffer_count=4
    )
    # 펌웨어 로딩 진행바 표시
    imx500.show_network_fw_progress_bar()
    picam2.configure(config)
    picam2.start()
else:
    # 기존 CSI 또는 USB 카메라
    class CSICamera:
        from picamera2 import Picamera2 as _PC2
        def __init__(self):
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            cfg = self.picam2.create_video_configuration(
                main={"size": (1280, 720)},
                lores={"size": (640, 360)},
                buffer_count=2
            )
            self.picam2.configure(cfg)
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
                    for _ in range(5): cap.read()
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

# ──────────────────────────────────────────────────────────────────────────────
# 3) 백그라운드 프레임 처리 스레드 + 큐
frame_queue = queue.Queue(maxsize=1)

def capture_and_process():
    # CPU 모드 시 프레임 스킵 등의 설정
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
        if sleep > 0: time.sleep(sleep)
        last = time.time()

        # 이미지 획득
        if use_ai_camera:
            metadata = picam2.capture_metadata()
            frame = metadata['main']
            # On-sensor 추론 결과 획득
            outputs = imx500.get_outputs(metadata, add_batch=True)
            if outputs:
                boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]
                for (y1, x1, y2, x2), conf, cls in zip(boxes, scores, classes):
                    if conf < 0.5: continue
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    label = intrinsics.labels[int(cls)]
                    color = (0,0,255) if label=='person' else (255,0,0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            ret, frame = camera.read()
            if not ret: continue
            frame_count += 1
            if frame_count % skip_interval == 0:
                with torch.no_grad():
                    small = cv2.resize(frame, target_size)
                    last_results = model(small)
            if last_results:
                h_ratio = frame.shape[0] / target_size[1]
                w_ratio = frame.shape[1] / target_size[0]
                for *box, conf, cls in last_results.xyxy[0]:
                    x1, y1, x2, y2 = map(int, (box[0]*w_ratio, box[1]*h_ratio, box[2]*w_ratio, box[3]*h_ratio))
                    label = last_results.names[int(cls)]
                    if label in ('person','car'):
                        color = (0,0,255) if label=='person' else (255,0,0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # JPEG 인코딩 및 큐에 입력
        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        data = buf.tobytes()
        if not frame_queue.empty():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put(data)

threading.Thread(target=capture_and_process, daemon=True).start()

# ──────────────────────────────────────────────────────────────────────────────
# 4) Flask 앱 & 스트리밍 + 통계 엔드포인트
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
        'camera': cam_no,
        'cpu_percent': cpu,
        'memory_percent': mem.percent,
        'temperature_c': temp,
        'wifi_signal_dbm': signal
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
