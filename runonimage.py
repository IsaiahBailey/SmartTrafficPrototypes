
import numpy as np
from inference_sdk import InferenceHTTPClient

from inference import get_model
# import supervision to visualize our results
import supervision as sv
# import cv2 to helo load our image
import cv2
# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="itatZcph2BZkzZWWLVMp"
)

# infer on a local image


result = CLIENT.infer("20240422_150411.jpg", model_id="smart-vehicle-detection/4")

image_file = "MicrosoftTeams.image.jpg"
image = cv2.imread(image_file)

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(result)

# create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)

