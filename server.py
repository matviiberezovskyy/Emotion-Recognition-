from flask import Flask, request, render_template, send_file, Response
from werkzeug.utils import secure_filename
from flasgger import Swagger, swag_from
import os
import io

import numpy as np
from PIL import Image
import cv2

from swagger import image_endpoint_form, video_endpoint_form
from detection import YoloDetection, gen_frames

# Flask setup
emotion_app = Flask(__name__)
emotion_app.config['UPLOAD_FOLDER'] = './uploads/'
emotion_app.config['SWAGGER'] = {
    'title': "Facial Emotion Identifier",
    'uiversion': 3,
    'version': "1.0.0",
    'description': "API for analyzing facial expressions in images and live video streams"
}

swagger_ui = Swagger(emotion_app)

# Initialize detection engine
emotion_analyzer = YoloDetection(model_name="em_det_yolo_v1.pt")


@emotion_app.route('/')
def home_view():
    """Render demo UI"""
    return render_template('demo_page.html')


@emotion_app.route('/image-detection', methods=['POST'])
@swag_from(image_endpoint_form)
def process_uploaded_image():
    """
    Handle single image uploads and apply emotion recognition
    """
    if 'image' not in request.files:
        return 'Image file not found in request', 400

    uploaded_file = request.files['image']
    if uploaded_file.filename == '':
        return 'No file selected for upload', 400

    if uploaded_file:
        # Save the uploaded image securely
        safe_name = secure_filename(uploaded_file.filename)
        saved_path = os.path.join(emotion_app.config['UPLOAD_FOLDER'], safe_name)
        uploaded_file.save(saved_path)

        # Load, preprocess, and analyze the image
        loaded_image = Image.open(saved_path).convert("RGB")
        frame_array = np.array(loaded_image)
        resized_frame = cv2.resize(frame_array, (512, 512))
        annotated = emotion_analyzer.detect_from_image(resized_frame, conf=0.3, skip_classes=[])
        result_image = Image.fromarray(annotated)

        # Return result as image response
        output_stream = io.BytesIO()
        result_image.save(output_stream, format="PNG")
        output_stream.seek(0)

        # Clean up uploaded file
        os.remove(saved_path)

        return send_file(output_stream, mimetype='image/png')


@emotion_app.route('/video-feed', methods=['GET'])
@swag_from(video_endpoint_form)
def stream_video_feed():
    """
    Serve MJPEG stream with real-time emotion prediction
    """
    return Response(gen_frames(emotion_analyzer), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    emotion_app.run(host='0.0.0.0', port=8000)
