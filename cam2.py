#!/usr/bin/env python3
import os
import time
import threading
import queue
import subprocess
import psutil
import cv2
from flask import Flask, Response, render_template, jsonify

# ──────────────────────────────────────────────────────────────────────────────
# framebuffer 절전/DPMS 무시 (콘솔에서는 동작 안 해도 무시)
try:
    os.system('setterm -blank 0 -powerdown 0 -powersave off')
except:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# 1) FBDevCamera: /dev/fb0 → ffmpeg → MJPEG pipe
class FBDevCamera:
    def __init__(self, width=1280, height=720, fps=10):
        cmd = [
            'ffmpeg',
            '-f', 'fbdev',
            '-framerate', str(fps),
            '-video_size', f'{width}x{height}',
            '-i', '/dev/fb0',
            '-vf', f'scale={width}:{height}',
            '-q:v', '5',
            '-f', 'mjpeg',
            'pipe:1'
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        self.buffer = b''

    def read(self):
        # stdout 에서 JPEG 단위로 읽기
        while True:
            chunk = self.proc.stdout.read(1024)
            if not chunk:
                return False, None
            self.buffer += chunk
            start = self.buffer.find(b'\xff\xd8')  # SOI
            end   = self.buffer.find(b'\xff\xd9')  # EOI
            if start != -1 and end != -1 and end > start:
                jpg = self.buffer[start:end+2]
                self.buffer = self.buffer[end+2:]
                # 디코딩 확인 (선택)
                img = cv2.imdecode(
                    np.frombuffer(jpg, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )
                return True, img

# ──────────────────────────────────────────────────────────────────────────────
# 카메라 객체: FBDevCamera 만 사용
camera = FBDevCamera(width=1280, height=720, fps=10)
print(">>> Using framebuffer camera via ffmpeg")

# ──────────────────────────────────────────────────────────────────────────────
# 2) 백그라운드 캡처 + 큐
frame_queue = queue.Queue(maxsize=1)

def capture_loop():
    while True:
        ret, frame = camera.read()
        if not ret:
            continue
        # JPEG 재인코딩 (품질 80)
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
    # gevent 사용 시
    # from gevent.pywsgi import WSGIServer
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever()

    # 일반 Flask
    app.run(host='0.0.0.0', port=5000)
