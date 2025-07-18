#!/usr/bin/env python3
import cv2
import psutil
import subprocess
from flask import Flask, Response, render_template, jsonify

# Picamera2와 DRM 프리뷰용 Preview 클래스 가져오기
try:
    from picamera2 import Picamera2, Preview
    has_picam = True
except ImportError:
    has_picam = False

# 카메라 초기화
if has_picam:
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2.configure(config)

    # 화이트밸런스 예시: 자동 AWB 켜기
    picam2.set_controls({"AwbEnable": 1})
    # 수동 게인으로 바꾸고 싶으면 아래처럼 사용
    # picam2.set_controls({
    #     "AwbEnable": 0,
    #     "ColourGains": (1.4, 1.2)
    # })

    # DRM 프리뷰(선택)
    try:
        picam2.start_preview(Preview.DRM)
    except RuntimeError:
        pass

    picam2.start()

else:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 자동 화이트밸런스 켜기
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    # 수동으로 색온도 조절하고 싶으면:
    # cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    # cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)

    if not cap.isOpened():
        raise RuntimeError('카메라를 시작할 수 없습니다.')

app = Flask(__name__)

def generate():
    while True:
        if has_picam:
            # Picamera2가 반환하는 배열을 그대로 사용
            frame = picam2.capture_array()
            # 웹에서 색이 뒤집혀 보이면 아래 주석 해제
            # frame = frame[..., ::-1]

        else:
            ret, frame = cap.read()
            if not ret:
                continue
            # VideoCapture가 BGR로 주는 걸 RGB로 바꾸고 싶으면 주석 해제
            # frame = frame[..., ::-1]

        # JPEG 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        # MJPEG 스트림 전송
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
    # 캐시 무효화
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
    # 디버그 끄고 배포용으로 실행
    app.run(host='0.0.0.0', port=5000)
