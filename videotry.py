import numpy as np
from inference_sdk import InferenceHTTPClient
from inference import get_model
import supervision as sv
import cv2

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="censored"
)

# load the video
video_file = "istockphoto-842308772-640_adpp_is.mp4"
cap = cv2.VideoCapture(video_file)

# create output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# process every 5 frames of the video
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 1 == 0:
        # infer on the current frame
        result = CLIENT.infer(frame, model_id="smart-vehicle-detection/4")

        # load the results into the supervision Detections api
        detections = sv.Detections.from_inference(result)

        # create supervision annotators
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # annotate the frame with our inference results
        annotated_frame = bounding_box_annotator.annotate(
            scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections)

        # write the annotated frame to the output video
        out.write(annotated_frame)

    frame_count += 1

# release the video capture and writer
cap.release()
out.release()
