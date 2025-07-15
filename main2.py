#!/usr/bin/env python3
import cv2
from flask import Flask, Response

app = Flask(__name__)


def generate():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('카메라를 시작할 수 없습니다.')
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/')
def index():
    return '''
    <html>
      <head><title>Camera Stream</title></head>
      <body>
        <h1>Camera Stream</h1>
        <img src="/video_feed" />
      </body>
    </html>
    '''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
