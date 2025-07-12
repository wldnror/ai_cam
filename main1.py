#!/usr/bin/env python3
import cv2
from flask import Flask, Response

# ──────────────────────────────────────────────────────────────────────────────
# 1) USB 웹캠 초기화
#    (필요하다면 인덱스를 0→1→2… 바꿔주세요)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("USB 웹캠을 열 수 없습니다")

# 해상도 설정 (선택)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ──────────────────────────────────────────────────────────────────────────────
# 2) 프레임 제너레이터
def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # JPEG 인코딩
        success, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            continue
        jpg = buf.tobytes()

        # multipart 스트림으로 출력
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
        )

# ──────────────────────────────────────────────────────────────────────────────
# 3) Flask 앱 & 라우트
app = Flask(__name__)

@app.route('/')
def index():
    # 간단한 HTML 뷰어
    return (
        "<html><body>"
        "<h1>USB Webcam Stream</h1>"
        "<img src='/video_feed' style='width:100%;max-width:1280px;'/>"
        "</body></html>"
    )

@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    # 네트워크상 접근 가능하도록 0.0.0.0 바인딩
    app.run(host='0.0.0.0', port=5000)
