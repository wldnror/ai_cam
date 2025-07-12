#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")  # Python 경고 억제

import os
import sys
import io
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

# 1) CSI 카메라 초기화 (Picamera2 사용, Preview configuration으로 출력)
try:
    picam2 = Picamera2()
    # preview 모드: 기본 BGR888 형식, 실시간 디스플레이/스트리밍용
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720)},
    )
    picam2.configure(config)
    picam2.start()
    # 워밍업 프레임
    for _ in range(3):
        picam2.capture_array("main")
    print(">>> Using CSI camera preview configuration")
except Exception as e:
    print(f"[ERROR] CSI 카메라 초기화 실패: {e}")
    sys.exit(1)

# 2) Flask 앱 설정
app = Flask(__name__)

# 3) 순수 CSI 카메라 화면 스트리밍 (BGR888 그대로 JPEG으로 인코딩)
def generate():
    while True:
        # create_preview_configuration은 BGR888로 반환
        frame = picam2.capture_array("main")
        # frame은 BGR이므로, PIL로 인식하려면 RGB로 변환
        frame = frame[..., ::-1]
        img = Image.fromarray(frame, 'RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.getvalue() + b'\r\n')

@app.route('/')
def index():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
