ㅂㅂㅂ#!/usr/bin/env python3
from flask import Flask, Response, render_template
from picamera2 import Picamera2
import cv2

# Flask 앱 초기화
app = Flask(__name__)

# Picamera2(V2 등 일반 CSI 카메라) 초기화
picam2 = Picamera2()
# RGB888 포맷으로 받아오도록 설정
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

def gen_frames():
    while True:
        # RGB 배열 가져오기
        frame_rgb = picam2.capture_array("main")
        # OpenCV는 BGR을 쓰므로 변환
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # JPEG 인코딩
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
