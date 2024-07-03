from flask import Flask, Response
from picamera2 import Picamera2, Preview
import io
import cv2  # Make sure to import OpenCV
from flask_basicauth import BasicAuth

app = Flask(__name__)

# Set the basic auth username and password
app.config['BASIC_AUTH_USERNAME'] = 'admin'
app.config['BASIC_AUTH_PASSWORD'] = 'password'

basic_auth = BasicAuth(app)

def generate_frames():
    with Picamera2() as camera:
        camera.configure(camera.create_preview_configuration(main={"size": (640, 480)}))
        camera.start()
        while True:
            frame = camera.capture_array()
            # Convert from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_data = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/video_feed')
@basic_auth.required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "Video streaming server is running"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
