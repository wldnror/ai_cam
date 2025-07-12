#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")  # Python 경고 억제

import os
import sys
import io
import numpy as np
import subprocess
from picamera2 import Picamera2
from flask import Flask, Response
from PIL import Image

# 0) 화면 절전/DPMS 비활성화 (X 환경일 때만)
try:
    if os.environ.get('DISPLAY'):
        os.system('setterm -blank 0 -powerdown 0 -powersave off')
        os.system('xset s off; xset s noblank; xset -dpms')
        print("⏱️ 화면 절전/스크린세이버 비활성화 완료")
except Exception:
    pass

# 1) CSI 카메라 초기화 (Picamera2 사용)
try:
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "RGB888"},
        lores={"size": (640, 360)},
        buffer_count=2
    )
    picam2.configure(config)
    picam2.start()
    # 워밍업 프레임
    for _ in range(3):
        picam2.capture_array("main")
    print(">>> Using CSI camera module")
except Exception as e:
    print(f"[ERROR] CSI 카메라 초기화 실패: {e}")
    sys.exit(1)

# 2) Flask 앱 설정
app = Flask(__name__)

# 순수 CSI 카메라 화면 스트리밍 (채널 순서 보정 포함)
def generate():
    while True:
        frame = picam2.capture_array("main")  # 기본 RGB888 출력
        # 만약 색상이 뒤바뀌어 보이면, BGR로 반환된 경우일 수 있음
        # frame = frame[..., ::-1]  # BGR→RGB 변환 (필요 시 주석 해제)

        # PIL로 JPEG 인코딩
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        data = buf.getvalue()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

@app.route('/')
def index():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
