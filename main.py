import cv2
import torch
from flask import Flask, Response
import sys
import os

# YOLOv5 경로 추가
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.augmentations import letterbox

# YOLO 모델 로드
device = 'cpu'
model = DetectMultiBackend('yolov5s.pt', device=device)
model.eval()

# 클래스 이름
names = model.names

# 카메라 자동 탐지
def get_camera():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
    raise RuntimeError("카메라를 찾을 수 없습니다.")

cap = get_camera()
app = Flask(__name__)

# 영상 스트리밍 생성기
def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 전처리
        img = letterbox(frame, 640, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 추론
        pred = model(img)
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        # 결과 처리
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    label = names[int(cls)]
                    color = (0, 0, 255) if label == 'person' else (255, 0, 0) if label == 'car' else (0, 255, 0)
                    if label in ['person', 'car']:
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # JPEG 인코딩
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
