#!/usr/bin/env python3
from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np

# 한글 라벨 매핑 (필요에 따라 클래스 추가)
LABEL_MAP = {
    "person": "사람",
    "bicycle": "자전거",
    "car": "자동차",
    "dog": "개",
    "cat": "고양이",
    # TODO: 다른 클래스 매핑 추가
}

# AI 후처리 JSON 모델 파일 경로
MODEL_JSON = "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json"

# Picamera2 설정: AI 카메라의 후처리 파이프라인 적용
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)},
    controls={"PostProcessFile": MODEL_JSON}
)
picam2.configure(config)
picam2.start()

app = Flask(__name__)

def gen_frames():
    while True:
        # 후처리된 프레임과 메타데이터 캡처
        frame = picam2.capture_array()  # RGB 포맷
        metadata = picam2.capture_metadata()
        # OpenCV는 BGR 포맷 사용
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 메타데이터에서 검출 결과 가져오기
        results = metadata.get("PostProcess/0")
        if results:
            boxes = results.get("boxes", [])
            classes = results.get("classes", [])
            scores = results.get("scores", [])
            label_names = results.get("label_names", [])
            h, w, _ = frame.shape
            for box, cls, score in zip(boxes, classes, scores):
                if score < 0.5:
                    continue
                x1, y1, x2, y2 = (box * np.array([w, h, w, h])).astype(int)
                label_en = label_names[int(cls)]
                label_ko = LABEL_MAP.get(label_en, label_en)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label_ko} {score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # JPEG 인코딩 후 스트리밍
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return """
    <html><body>
    <h1>AI 카메라 실시간 스트리밍</h1>
    <img src="/video_feed" style="width:100%;max-width:640px;" />
    </body></html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
