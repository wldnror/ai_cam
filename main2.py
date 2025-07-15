#!/usr/bin/env python3
import cv2
import numpy as np
import psutil
import subprocess
from flask import Flask, Response, render_template, jsonify

# Picamera2 사용 가능 여부 확인
try:
    from picamera2 import Picamera2, Preview
    has_picam = True
except ImportError:
    has_picam = False

def apply_grayworld_wb(img: np.ndarray) -> np.ndarray:
    # GrayWorld 화이트밸런스
    b, g, r = img[...,0], img[...,1], img[...,2]
    b_avg, g_avg, r_avg = b.mean(), g.mean(), r.mean()
    gray = (b_avg + g_avg + r_avg) / 3
    kb, kg, kr = gray/b_avg, gray/g_avg, gray/r_avg
    img[...,0] = np.clip(b * kb, 0, 255)
    img[...,1] = np.clip(g * kg, 0, 255)
    img[...,2] = np.clip(r * kr, 0, 255)
    return img.astype(np.uint8)

def apply_gamma(img: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    # 감마 보정: 밝기 증폭
    inv = 1.0 / gamma
    table = np.array([((i/255.0) ** inv) * 255 for i in range(256)]).astype('uint8')
    return cv2.LUT(img, table)

# 카메라 초기화
if has_picam:
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})
    picam2.configure(config)

    # AWB 끄고 수동 게인·노출 ↑
    picam2.set_controls({
        "AwbEnable": 0,
        "ColourGains": (1.2, 1.1),
        "AnalogueGain": 2.0,        # 게인 2배
        "ExposureTime": 30000       # 노출 30 ms
    })

    try:
        picam2.start_preview(Preview.DRM)
    except RuntimeError:
        pass
    picam2.start()

else:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 5000)
    # 노출·게인 수동 조절
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1=manual
    cap.set(cv2.CAP_PROP_EXPOSURE, -4)      # 값 높일수록 밝아짐 (카메라마다 범위 다름)
    cap.set(cv2.CAP_PROP_GAIN, 4)           # 게인 값 (카메라마다 max 다름)

    if not cap.isOpened():
        raise RuntimeError("카메라를 시작할 수 없습니다.")

app = Flask(__name__)

def generate():
    while True:
        if has_picam:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                continue

        # 1) GrayWorld WB
        frame = apply_grayworld_wb(frame)
        # 2) 감마 보정
        frame = apply_gamma(frame, gamma=1.8)

        # JPEG 인코딩 & 전송
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

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
        'Cache-Control':'no-cache, no-store, must-revalidate',
        'Pragma':'no-cache',
        'Expires':'0'
    })
    return resp

@app.route('/stats')
def stats():
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory().percent
    try:
        temp = float(open('/sys/class/thermal/thermal_zone0/temp').read())/1000.0
    except:
        temp = None
    try:
        out = subprocess.check_output(['iwconfig','wlan0'],stderr=subprocess.DEVNULL).decode()
        sig = int([p.split('=')[1] for p in out.split() if p.startswith('level=')][0])
    except:
        sig = None
    return jsonify(camera=1, cpu_percent=cpu, memory_percent=mem, temperature_c=temp, wifi_signal_dbm=sig)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
