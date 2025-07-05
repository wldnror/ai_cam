# main.py

import os
import time
import threading
import queue
import cv2
import torch
from flask import Flask, Response, render_template
from flask_socketio import SocketIO

# ──────────────────────────────────────────────────────────────────────────────
# 0) 화면 꺼짐·절전 모드 방지
try:
    os.system('setterm -blank 0 -powerdown 0 -powersave off')
    os.environ.setdefault('DISPLAY', ':0')
    os.system('xset s off')
    os.system('xset s noblank')
    os.system('xset -dpms')
    print("⏱️ 화면 절전/블랭킹 기능 비활성화 완료")
except Exception as e:
    print("⚠️ 전원/스크린세이버 비활성화 중 오류:", e)
# ──────────────────────────────────────────────────────────────────────────────

# 1) 모델 로드 (YOLOv5n)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# 2) 카메라 인터페이스 통일 (CSI 우선, 없으면 USB)
class CSICamera:
    def __init__(self):
        from picamera2 import Picamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (1280, 720)}, lores={"size": (640, 360)}, buffer_count=2
        )
        self.picam2.configure(config); self.picam2.start()
        for _ in range(3): self.picam2.capture_array("main")
    def read(self):
        return True, self.picam2.capture_array("main")

class USBCamera:
    def __init__(self):
        self.cap = None
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(5): cap.read()
                self.cap = cap; break
        if self.cap is None:
            raise RuntimeError("사용 가능한 USB 웹캠을 찾을 수 없습니다.")
    def read(self):
        return self.cap.read()

try:
    camera = CSICamera(); print(">>> Using CSI camera module")
except Exception:
    camera = USBCamera(); print(">>> Using USB webcam")

# 3) 프레임 처리 및 스트리밍 준비
frame_queue = queue.Queue(maxsize=1)
fps_counter = {"last_time": time.time(), "frames": 0, "fps": 0.0}

def capture_and_process():
    target_fps = 15
    interval = 1.0 / target_fps

    while True:
        start = time.time()
        ret, frame = camera.read()
        if not ret:
            continue

        # YOLOv5 inference
        small = cv2.resize(frame, (640, 640))
        results = model(small)

        # draw boxes
        h_ratio, w_ratio = frame.shape[0]/640, frame.shape[1]/640
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, [box[0]*w_ratio, box[1]*h_ratio, box[2]*w_ratio, box[3]*h_ratio])
            label = results.names[int(cls)]
            if label in ['person','car']:
                color = (0,0,255) if label=='person' else (255,0,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        # encode JPEG
        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        data = buf.tobytes()
        if not frame_queue.empty():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put(data)

        # FPS 계산
        fps_counter["frames"] += 1
        now = time.time()
        if now - fps_counter["last_time"] >= 1.0:
            fps_counter["fps"] = fps_counter["frames"] / (now - fps_counter["last_time"])
            fps_counter["frames"] = 0
            fps_counter["last_time"] = now

        # sleep to maintain target fps
        elapsed = time.time() - start
        if elapsed < interval:
            time.sleep(interval - elapsed)

# 백그라운드에서 프레임 생성
threading.Thread(target=capture_and_process, daemon=True).start()

# 4) Flask + SocketIO 앱 설정
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def generate_mjpeg():
    while True:
        frame = frame_queue.get()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    res = Response(generate_mjpeg(),
                   mimetype='multipart/x-mixed-replace; boundary=frame',
                   direct_passthrough=True)
    res.headers.update({
        'Cache-Control':'no-cache, no-store, must-revalidate',
        'Pragma':'no-cache','Expires':'0'
    })
    return res

# 5) 메트릭 전송용 WebSocket 채널
def get_rssi():
    # 예시: 실제 환경에 맞게 RPi 무선 모듈에서 RSSI 읽기
    # return int(os.popen("iwconfig wlan0").read().split("Signal level=")[1].split(" ")[0][:-3])
    return -40  # TODO: 구현

def metrics_broadcaster():
    while True:
        socketio.emit('metrics', {
            'rssi': get_rssi(),
            'fps': round(fps_counter["fps"], 1),
            'timestamp': time.time()
        })
        socketio.sleep(1)  # eventlet 사용 시

socketio.start_background_task(metrics_broadcaster)

if __name__ == '__main__':
    # eventlet이 필요합니다: pip install eventlet
    socketio.run(app, host='0.0.0.0', port=5000)
