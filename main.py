import cv2
import numpy as np
from flask import Flask, Response, render_template
import glob

# ——— 1) 카메라 열기 ———
def open_camera():
    for dev in sorted(glob.glob('/dev/video*')):
        cap = cv2.VideoCapture(dev)
        if cap.isOpened():
            print(f"[INFO] using camera: {dev}")
            return cap
    raise RuntimeError("No camera device found.")

cap = open_camera()

# ——— 2) DNN 초기화 ———
prototxt = "/home/pi/models/MobileNetSSD_deploy.prototxt"
model    = "/home/pi/models/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

CLASSES = [
    "background","aeroplane","bicycle","bird","boat",
    "bottle","bus","car","cat","chair","cow","diningtable",
    "dog","horse","motorbike","person","pottedplant",
    "sheep","sofa","train","tvmonitor"
]

# ——— 3) Flask 앱 세팅 ———
app = Flask(__name__)

@app.route('/')
def index():
    # 단순한 HTML: <img src="/video_feed">
    return render_template('index.html')

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # DNN 전처리
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # 탐지 결과 순회
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue

            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            if label not in ("person", "car"):
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            color = (0,0,255) if label=="person" else (255,0,0)  # BGR
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}",
                        (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # JPEG로 인코딩 후 바이트 스트림 생성
        ret2, jpeg = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 0.0.0.0:5000 으로 외부 접속 허용
    app.run(host='0.0.0.0', port=5000, threaded=True)
