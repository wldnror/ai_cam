#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import os
import time
import threading
import queue
import subprocess

import numpy as np
import mss
import cv2
import psutil
from flask import Flask, Response, render_template, jsonify

# ──────────────────────────────────────────────────────────────────────────────
# 화면 절전/DPMS 비활성화
if os.environ.get('DISPLAY'):
    os.system('setterm -blank 0 -powerdown 0 -powersave off')
    os.system('xset s off; xset s noblank; xset -dpms')

# ──────────────────────────────────────────────────────────────────────────────
# 1) ScreenCamera: X11 화면 캡처
class ScreenCamera:
    def __init__(self, width=1280, height=720):
        self.sct = mss.mss()
        mon = self.sct.monitors[1]
        self.crop = {
            "top": mon["top"],
            "left": mon["left"],
            "width": width,
            "height": height
        }

    def read(self):
        img = self.sct.grab(self.crop)
        frame = np.array(img)[:, :, :3]  # BGRA → BGR
        return True, frame

# 항상 ScreenCamera 사용
camera = ScreenCamera()
print(">>> Using Screen capture via mss")

# ──────────────────────────────────────────────────────────────────────────────
# 2) 백그라운드 캡처 + 큐
frame_queue = queue.Queue(maxsize=1)

def capture_loop(fps=10):
    interval = 1.0 / fps
    while True:
        ret, frame = camera.read()
        if not ret:
            continue
        time.sleep(interval)
        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        data = buf.tobytes()
        if not frame_queue.empty():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put(data)

threading.Thread(target=capture_loop, daemon=True).start()

# ──────────────────────────────────────────────────────────────────────────────
# 3) Flask 앱 & 라우트
app = Flask(__name__)

def gen_frames():
    while True:
        frame = frame_queue.get()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # templates/index.html 에 <img src="/video_feed"> 만 있으면 됩니다.
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/stats')
def stats():
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory().percent
    temp = None
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            temp = float(f.read()) / 1000.0
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
    return jsonify({
        'cpu_percent': cpu,
        'memory_percent': mem,
        'temperature_c': temp,
        'wifi_signal_dbm': signal
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
