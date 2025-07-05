# main.py

import time
import cv2
import torch
from flask import Flask, Response, render_template

# 1) 모델 로드 (한 번만)
#    필요에 따라 'yolov5n'으로 바꿔 연산량 더 줄일 수 있습니다.
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def get_camera():
    """첫 번째 연결 가능한 카메라를 MJPEG 모드, 1280×720 해상도로 설정하여 반환."""
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # MJPEG 모드로 전송 대역폭↓
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            # 캡처 해상도 설정
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
    raise RuntimeError("카메라를 찾을 수 없습니다.")

cap = get_camera()
app = Flask(__name__)

def generate():
    """프레임을 읽어 YOLO 추론 → 박스 그리기 → JPEG로 인코딩 → 스트리밍"""
    fps = 15
    interval = 1.0 / fps
    last_time = time.time()

    while True:
        # FPS 제한
        now = time.time()
        elapsed = now - last_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
        last_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # 2) 인퍼런스용으로 640×640 리사이즈
        small = cv2.resize(frame, (640, 640))
        results = model(small)  # 내부적으로 small을 모델 입력으로 사용

        # 3) 결과 박스 좌표를 원본 크기로 스케일링
        h_ratio = frame.shape[0] / 640
        w_ratio = frame.shape[1] / 640
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, (
                box[0] * w_ratio,
                box[1] * h_ratio,
                box[2] * w_ratio,
                box[3] * h_ratio
            ))
            label = results.names[int(cls)]
            # 사람=빨강, 차=파랑, 기타=초록
            color = (
                (0, 0, 255) if label == 'person'
                else (255, 0, 0) if label == 'car'
                else (0, 255, 0)
            )
            if label in ['person', 'car']:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

        # 4) JPEG 품질 80으로 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes + b'\r\n'
        )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # direct_passthrough=True 로 Flask가 버퍼 없이 바로 전송
    res = Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        direct_passthrough=True
    )
    # 브라우저 캐시 방지
    res.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    res.headers['Pragma']        = 'no-cache'
    res.headers['Expires']       = '0'
    return res

if __name__ == '__main__':
    # 프로덕션용은 Gunicorn + eventlet/gevent 사용 권장
    app.run(host='0.0.0.0', port=5000)
