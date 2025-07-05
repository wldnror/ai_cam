# main.py

import time
import threading
import queue

import cv2
import torch
from flask import Flask, Response, render_template

# 1) 모델 로드 (한 번만) – 원하는 모델로 변경 가능
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# 2) 카메라 인터페이스 통일
class CSICamera:
    def __init__(self):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (1280, 720)},
            lores={"size": (640, 360)}
        )
        self.picam2.configure(config)
        self.picam2.start()

    def read(self):
        frame = self.picam2.capture_array("main")
        return True, frame

class USBCamera:
    def __init__(self):
        self.cap = None
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # MJPEG 모드, 1280×720
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap = cap
                break
        if self.cap is None:
            raise RuntimeError("사용 가능한 USB 웹캠을 찾을 수 없습니다.")

    def read(self):
        return self.cap.read()

# 3) 카메라 선택: CSI 우선 → 없으면 USB
try:
    camera = CSICamera()
    print(">>> Using CSI camera module")
except Exception:
    camera = USBCamera()
    print(">>> Using USB webcam")

# 4) 프레임 처리 스레드 & 큐
frame_queue = queue.Queue(maxsize=1)

def capture_and_process():
    fps = 15
    interval = 1.0 / fps
    last_time = time.time()

    while True:
        now = time.time()
        elapsed = now - last_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
        last_time = time.time()

        ret, frame = camera.read()
        if not ret:
            continue  # 읽기 실패 시 다음 루프

        # 인퍼런스: 640×640 리사이즈
        small = cv2.resize(frame, (640, 640))
        results = model(small)

        # 박스 좌표 스케일링 후 그리기
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
            if label in ['person', 'car']:
                color = (0,0,255) if label=='person' else (255,0,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # JPEG 인코딩 (품질 80)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame_bytes = buffer.tobytes()

        # 큐에 최신 프레임만 유지
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame_bytes)

# 데몬 스레드로 시작
thread = threading.Thread(target=capture_and_process, daemon=True)
thread.start()

# 5) Flask 앱
app = Flask(__name__)

def generate():
    while True:
        frame_bytes = frame_queue.get()
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
    res = Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        direct_passthrough=True
    )
    res.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    res.headers['Pragma']        = 'no-cache'
    res.headers['Expires']       = '0'
    return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
