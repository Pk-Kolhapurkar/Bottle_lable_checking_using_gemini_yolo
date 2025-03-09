


import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import base64
import os
import gradio as gr
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Set up Google API Key (Avoid hardcoding in production)
os.environ["GOOGLE_API_KEY"] = "Your_api_key"  # Replace with your actual API Key

# ✅ Initialize the Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ✅ Load the YOLO model
yolo_model = YOLO("best.pt")  # Ensure "best.pt" is in the working directory
names = yolo_model.names  # Class names from the YOLO model

def encode_image_to_base64(image):
    """Encodes an image to a base64 string."""
    _, img_buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(img_buffer).decode('utf-8')

def analyze_image_with_gemini(image):
    """Sends an image to Gemini AI for analysis."""
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        return "Error: Invalid image."
    
    image_data = encode_image_to_base64(image)
    message = HumanMessage(content=[
        {"type": "text", "text": """
        Analyze this image and determine if the label is present on the bottle.
        Return the result strictly in a structured table format:
        
        | Label Present | Damage |
        |--------------|--------|
        | Yes/No       | Yes/No |
        """},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}, "description": "Detected product"}
    ])
    
    try:
        response = gemini_model.invoke([message])
        return response.content
    except Exception as e:
        return f"Error processing image: {e}"

def process_video(video_path):
    """Processes the uploaded video frame by frame using YOLO and Gemini AI."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video file."
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video_path = "output.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    vertical_center = width // 2
    analyzed_objects = {}  # Dictionary to store analyzed objects

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = yolo_model.track(frame, persist=True)

        if results and results[0].boxes is not None and results[0].boxes.xyxy is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) // 2

                # Draw detection box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'ID: {track_id}', (x2, y2), 1, 1)
                cvzone.putTextRect(frame, f'{names[class_id]}', (x1, y1), 1, 1)

                # If object is near vertical center and hasn't been analyzed yet
                if abs(center_x - vertical_center) < 10 and track_id not in analyzed_objects:
                    crop = frame[y1:y2, x1:x2]
                    response = analyze_image_with_gemini(crop)

                    # Store analyzed object to prevent duplicate analysis
                    analyzed_objects[track_id] = response

                    # Log response and display on frame
                    print(f"Object {track_id}: {response}")
                    cvzone.putTextRect(frame, response, (x1, y1 - 10), 1, 1, colorT=(255, 255, 255), colorR=(0, 0, 255))
        
        out.write(frame)

    cap.release()
    out.release()

    return output_video_path

def gradio_interface(video_path):
    """Handles Gradio video input and processes it."""
    if video_path is None:
        return "Error: No video uploaded."
    return process_video(video_path)

# ✅ Gradio UI setup
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.File(type="filepath", label="Upload Video"),  # Accepts video files
    outputs=gr.Video(label="Processed Video"),  # Outputs processed video
    title="YOLO + Gemini AI Video Analysis",
    description="Upload a video to detect objects and analyze them using Gemini AI.",
)

if __name__ == "__main__":
    iface.launch(share=True)


