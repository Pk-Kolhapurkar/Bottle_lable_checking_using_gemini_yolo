import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import base64
import os
import gradio as gr
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Set up Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDT0y1kJqgGKiOYiYFMXc-2kTgV_WLbOpA"#os.getenv("GOOGLE_API_KEY")

# ✅ Initialize the Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ✅ Load the YOLO model
yolo_model = YOLO("/content/Bottle_lable_checking_using_gemini_yolo/best.pt")  
names = yolo_model.names  

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
        return "Error: Could not open video file.", ""

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video_path = "output.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    vertical_center = width // 2
    analyzed_objects = {}  
    log_messages = []

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

                # ✅ Apply bounding box only after the bottle reaches the left half of the frame
                if center_x > vertical_center:
                    continue  # Skip drawing before it crosses the center to the left side
                
                # Draw detection box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'ID: {track_id}', (x2, y2), 1, 1)
                cvzone.putTextRect(frame, f'{names[class_id]}', (x1, y1), 1, 1)

                # ✅ Ensure label (analysis result) remains visible after detection
                if track_id not in analyzed_objects:
                    crop = frame[y1:y2, x1:x2]
                    response = analyze_image_with_gemini(crop)
                    analyzed_objects[track_id] = response

                    log_messages.append(f"Object {track_id}: {response}")  # ✅ Add log
                    print(f"Object {track_id}: {response}")  # ✅ Print log for debugging

                # 🛠️ Keep analysis text on screen for each analyzed object
                if track_id in analyzed_objects:
                    response_text = analyzed_objects[track_id]
                    text_x = 50  # Left side
                    text_y = height // 2  # Middle of the frame
                    cvzone.putTextRect(frame, response_text, (text_x, text_y), 2, 2, colorT=(255, 255, 255), colorR=(0, 0, 255))

        out.write(frame)

    cap.release()
    out.release()

    return output_video_path, "\n".join(log_messages)  # ✅ Return logs along with the processed video

def gradio_interface(video_path):
    """Handles Gradio video input and processes it."""
    if video_path is None:
        return "Error: No video uploaded.", ""
    
    return process_video(video_path)

# ✅ Sample video file
sample_video_path = "/content/Bottle_lable_checking_using_gemini_yolo/vid4.mp4"  # Make sure this file is available in the working directory

# ✅ Gradio UI setup with sample video
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.File(value=sample_video_path, type="filepath", label="Upload Video (Sample Included)"),
    outputs=[
        gr.Video(label="Processed Video"),
        gr.Textbox(label="Processing Logs", lines=10, interactive=False)
    ],
    title="YOLO + Gemini AI Video Analysis",
    description="Upload a video to detect objects and analyze them using Gemini AI.\nA sample video is preloaded for quick testing.",
)

if __name__ == "__main__":
    iface.launch(share=True)
