# server.py
import cv2
import torch
from flask import Flask, Response, render_template

# 모델 로드 (한 번만)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def get_camera():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
    raise RuntimeError("카메라를 찾을 수 없습니다.")

cap = get_camera()
app = Flask(__name__)

def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 추론
        results = model(frame)
        for *box, conf, cls in results.xyxy[0]:
            label = results.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)
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

        # JPEG 인코딩
        _, buffer = cv2.imencode('.jpg', frame)
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
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    # 디버그 모드 꺼도 되고, host/port 필요에 맞게 조정
    app.run(host='0.0.0.0', port=5000)
