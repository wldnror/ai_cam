# ai_stream_picamera2.py 예시
from flask import Flask, Response
from picamera2 import Picamera2, Preview
import cv2

app = Flask(__name__)

def gen_frames():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    while True:
        frame = picam2.capture_array()            # NumPy 배열(RGB)
        # ★ 여기에 AI 후처리(객체 인식 등)를 Picamera2의 post_callback으로 넣어도 됩니다.
        # OpenCV는 BGR 순서이니 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html><body>
    <h1>AI Camera Live</h1>
    <img src="/video_feed" style="width:100%;max-width:640px"/>
    </body></html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
