
#%%
import threading
import cv2
import numpy as np
import tensorflow as tf
import time
import threading
from djitellopy import tello
#%%
# Load the saved object detection model
model = tf.saved_model.load('faster_rcnn_resnet50_v1_640x640_coco17_tpu-8')

# Load the COCO class names
with open('coco_labels.txt', 'r') as f:
    class_names = [line.strip() for line in f]
#%%
# Initialize the Tello drone
tello = tello.Tello()
tello.connect()
tello.streamoff()
tello.streamon()

# Define a function for processing the drone's video stream
def process_video_stream():
    while True:
        # Read the drone's video stream
        frame = tello.get_frame_read().frame
        
        # Convert the color of the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize the frame to match the input size of the object detection model
        resized_frame = cv2.resize(rgb_frame, (640, 640))
        
        # Convert the resized frame to a tensor
        input_tensor = tf.convert_to_tensor(resized_frame)
        
        # Add a batch dimension to the tensor
        input_tensor = input_tensor[tf.newaxis, ...]
        
        # Run the object detection model on the input tensor
        detections = model(input_tensor)
        
        # Extract the results from the detections tensor
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        
        # Filter out detections with a score below a certain threshold
        threshold = 0.7
        filtered_indices = np.where(scores > threshold)[0]
        filtered_boxes = boxes[filtered_indices]
        filtered_scores = scores[filtered_indices]
        filtered_classes = classes[filtered_indices]
        
        # Apply non-maximum suppression to remove redundant detections
        nms_threshold = 0.5
        selected_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=len(filtered_boxes), iou_threshold=nms_threshold)
        selected_boxes = tf.gather(filtered_boxes, selected_indices).numpy()
        selected_scores = tf.gather(filtered_scores, selected_indices).numpy()
        selected_classes = tf.gather(filtered_classes, selected_indices).numpy()
        
        # Draw bounding boxes around the selected detections
        for box, score, class_id in zip(selected_boxes, selected_scores, selected_classes):
            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * 640)
            xmax = int(xmax * 640)
            ymin = int(ymin * 640)
            ymax = int(ymax * 640)
            if class_id - 1 < len(class_names):
                class_name = class_names[class_id - 1]
            else:
                class_name = 'Unknown'
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Get the centroid of the bounding box
            centroid_x = int((xmin + xmax) / 2)
            centroid_y = int((ymin + ymax) / 2)
            
            # Draw a circle at the centroid
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
        
        # Display the processed frame
        cv2.imshow('Object Detection', frame)
        
        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Wait for a short time before processing the next frame
        time.sleep(0.1)

# Define a function for starting the video stream and processing it in a separate thread
def start_video_stream():
    # Start the video stream
    tello.streamon()
    
    # Create a thread for processing the video stream
    video_thread = threading.Thread(target=process_video_stream)
    
    # Start the thread
    video_thread.start()

# Call the function to start the video stream and process it in a separate thread
start_video_stream()


# Take off
tello.takeoff()
time.sleep(5)

# Move the drone forward
tello.move_forward(50)
time.sleep(5)

# Move the drone up
tello.move_up(50)
time.sleep(5)

# Move the drone to the right
tello.move_right(50)
time.sleep(5)

# Move the drone down
tello.move_down(50)
time.sleep(5)

# Move the drone to the left
tello.move_left(50)
time.sleep(5)

# Land the drone
tello.land()

