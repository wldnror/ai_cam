#!/usr/bin/env python3
import cv2
import psutil
import subprocess
from flask import Flask, Response, render_template, jsonify

# CSI 카메라 (Picamera2) 지원 여부 확인
try:
    from picamera2 import Picamera2
    has_picam = True
except ImportError:
    has_picam = False

# 카메라 초기화
if has_picam:
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (1280, 720)})
    picam2.configure(config)
    picam2.start()
else:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError('카메라를 시작할 수 없습니다.')

app = Flask(__name__)

def generate():
    while True:
        if has_picam:
            # Picamera2에서 프레임 읽기 (RGB -> BGR 변환)
            rgb = picam2.capture_array()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                continue
        # JPEG 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    resp = Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
    resp.headers.update({
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    })
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
    try:
        out = subprocess.check_output(['iwconfig', 'wlan0'], stderr=subprocess.DEVNULL).decode()
        sig = int([p.split('=')[1] for p in out.split() if p.startswith('level=')][0])
    except:
        sig = None
    return jsonify(
        camera=1,
        cpu_percent=cpu,
        memory_percent=mem,
        temperature_c=temp,
        wifi_signal_dbm=sig
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
