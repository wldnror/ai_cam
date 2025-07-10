# ai_stream.py
from flask import Flask, Response
import cv2

app = Flask(__name__)

def gen_frames():
    cap = cv2.VideoCapture(0)             # CSI 카메라: device 0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        success, frame = cap.read()       # 프레임 읽기
        if not success:
            break
        # ★여기에 rpicam-hello --headless와 연동하거나
        #   Picamera2 API로 후처리(객체인식/포즈추정) 적용 가능
        ret, buf = cv2.imencode('.jpg', frame)
        frame_bytes = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # 브라우저에서 바로 영상이 보이도록 간단한 HTML
    return """
    <html><body>
    <h1>AI Camera Live</h1>
    <img src="/video_feed" style="width:100%;max-width:640px"/>
    </body></html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
