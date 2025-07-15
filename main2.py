#!/usr/bin/env python3
import cv2
import numpy as np
import psutil
import subprocess
from flask import Flask, Response, render_template, jsonify

# Picamera2 사용 가능 여부
try:
    from picamera2 import Picamera2, Preview
    has_picam = True
except ImportError:
    has_picam = False

def apply_grayworld_wb(img: np.ndarray) -> np.ndarray:
    """
    GrayWorld 알고리즘 기반 화이트밸런스 보정
    """
    # 채널별 평균 계산
    b_avg, g_avg, r_avg = img[...,0].mean(), img[...,1].mean(), img[...,2].mean()
    # 전체 평균
    gray_avg = (b_avg + g_avg + r_avg) / 3
    # 채널별 스케일
    kb, kg, kr = gray_avg/b_avg, gray_avg/g_avg, gray_avg/r_avg
    # 보정
    img[...,0] = np.clip(img[...,0]*kb, 0, 255)
    img[...,1] = np.clip(img[...,1]*kg, 0, 255)
    img[...,2] = np.clip(img[...,2]*kr, 0, 255)
    return img.astype(np.uint8)

# 카메라 초기화
if has_picam:
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2.configure(config)
    
    # -- AWB 및 게인 설정 --
    picam2.set_controls({
        "AwbEnable": 0,            # 자동 AWB 끄기
        "ColourGains": (1.2, 1.1), # R 게인, B 게인 (환경에 맞춰 조정)
        "AnalogueGain": 1.0,       # ISO 유사 설정
        "ExposureTime": 10000      # 노출 시간 (마이크로초)
    })
    # DRM 프리뷰 (선택)
    try:
        picam2.start_preview(Preview.DRM)
    except RuntimeError:
        pass
    picam2.start()

else:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # 자동 WB 끄고 색온도 수동 설정
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 5000)
    # 노출/게인도 가능하면 수동 조절
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1=수동
    # cap.set(cv2.CAP_PROP_EXPOSURE, -6)     # OpenCV scale
    if not cap.isOpened():
        raise RuntimeError("카메라를 시작할 수 없습니다.")

app = Flask(__name__)

def generate():
    while True:
        if has_picam:
            # RGB888 배열
            frame = picam2.capture_array()
            # BGR로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                continue

        # 소프트웨어 화이트밸런스 보정 (GrayWorld)
        frame = apply_grayworld_wb(frame)

        # JPEG 인코딩
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        jpg_bytes = buf.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')

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
