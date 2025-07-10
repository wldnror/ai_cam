#!/usr/bin/env python3
import os
import glob
import time
import threading
import queue
import subprocess
import psutil
from flask import Flask, Response, render_template, jsonify

# ──────────────────────────────────────────────────────────────────────────────
# 절전/DPMS 방지
try:
    os.system('setterm -blank 0 -powerdown 0 -powersave off')
except:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# 1) FBDevCamera: /dev/fb* 자동 탐색 → ffmpeg MJPEG pipe
class FBDevCamera:
    def __init__(self, width=1280, height=720, fps=10):
        # 사용 가능한 fb 장치 찾기
        fbs = glob.glob('/dev/fb*')
        if not fbs:
            raise RuntimeError("No framebuffer device found")
        # HDMI 쪽이 fb1 이면 fb1을 우선, 아니면 첫 번째
        fbdev = '/dev/fb1' if '/dev/fb1' in fbs else fbs[0]
        print(f">>> Opening framebuffer {fbdev}")

        cmd = [
            'ffmpeg',
            '-f', 'fbdev',
            '-framerate', str(fps),
            '-video_size', f'{width}x{height}',
            '-i', fbdev,
            '-vf', f'scale={width}:{height}',
            '-q:v', '5',
            '-f', 'mjpeg',
            'pipe:1'
        ]
        # bufsize=0 로 하면 chunk read 과실 줄어듭니다
        self.proc = subprocess.Popen(cmd,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL,
                                     bufsize=0)
        self.buffer = b''

    def read(self):
        # stdout에서 JPEG SOI/EOI로 자르기
        while True:
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                return False, None
            self.buffer += chunk
            start = self.buffer.find(b'\xff\xd8')
            end   = self.buffer.find(b'\xff\xd9')
            if start != -1 and end != -1 and end > start:
                jpg = self.buffer[start:end+2]
                self.buffer = self.buffer[end+2:]
                return True, jpg  # 이미 JPEG 바이트

# ──────────────────────────────────────────────────────────────────────────────
# camera 객체는 오직 FBDevCamera
camera = FBDevCamera(width=1280, height=720, fps=10)

# ──────────────────────────────────────────────────────────────────────────────
# 2) 캡처 스레드 + 큐
frame_queue = queue.Queue(maxsize=1)

def capture_loop():
    while True:
        ret, jpg = camera.read()
        if not ret:
            continue
        # 이미 JPEG이므로 바로 큐에 넣기
        if not frame_queue.empty():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put(jpg)

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
            temp = float(f.read())/1000.0
    except:
        pass
    signal = None
    try:
        out = subprocess.check_output(['iwconfig','wlan0'],
                                      stderr=subprocess.DEVNULL).decode()
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

if __name__=='__main__':
    # gevent가 설치되어 있다면
    # from gevent.pywsgi import WSGIServer
    # WSGIServer(('0.0.0.0',5000), app).serve_forever()

    # 기본 Flask
    app.run(host='0.0.0.0', port=5000)
