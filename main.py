import cv2
import numpy as np
from flask import Flask, Response
import glob

# Optional screen capture
try:
    from mss import mss
    have_mss = True
except ImportError:
    have_mss = False

# ——— 1) 카메라 열기 또는 화면 캡처 준비 ———
def init_video_source():
    # 카메라 탐색
    for dev in sorted(glob.glob('/dev/video*')):
        cap = cv2.VideoCapture(dev)
        if cap.isOpened():
            print(f"[INFO] using camera: {dev}")
            return 'camera', cap
    # 카메라 없으면 화면 캡처
    if have_mss:
        print("[INFO] no camera found, using screen capture")
        sct = mss()
        monitor = sct.monitors[0]
        return 'screen', (sct, monitor)
    raise RuntimeError("No camera device found and screen capture not available.")

video_type, video_obj = init_video_source()

# ——— 2) DNN 초기화 ———
prototxt = "models/MobileNetSSD_deploy.prototxt"
model    = "models/MobileNetSSD_deploy.caffemodel"
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
    html = '''<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Camera/Screen Stream</title>
    <style>
      body { text-align: center; font-family: sans-serif; }
      img { max-width: 100%; height: auto; }
    </style>
  </head>
  <body>
    <h2>Raspberry Pi AI Stream ({source})</h2>
    <img src="/video_feed" alt="Stream">
  </body>
</html>'''.format(source=video_type)
    return Response(html, mimetype='text/html')


def generate_frames():
    while True:
        # 프레임 획득
        if video_type == 'camera':
            ret, frame = video_obj.read()
            if not ret:
                break
        else:
            # 화면 캡처
            sct, monitor = video_obj
            img = np.array(sct.grab(monitor))[:, :, :3]
            frame = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

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

            color = (0,0,255) if label=="person" else (255,0,0)
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
    app.run(host='0.0.0.0', port=5000, threaded=True)
