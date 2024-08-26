import time
import cv2
from PIL import Image
import torchvision
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
from datetime import datetime
import argparse
from utility import *
from deep_sort_realtime.deepsort_tracker import DeepSort
# from module.segmentation_package.interpreter_segm import SegmentationInference
from ts_segmentor_helper_functions import read_image
from ts_segmentor_helper_functions import convert_output_to_mask_and_polygons
from ts_segmentor_helper_functions import map_coordinates_to_original
from ts_segmentor_helper_functions import resize_frame
from ts_segmentor_helper_functions import draw_bounding_box_with_id
from ts_segmentor_helper_functions import do_paste_mask
from ts_segmentor_helper_functions import draw_tracker_points

from ts_segmentor_helper_functions import csv_writer

import numpy as np
import sys

video_path = '/media/youcef/Elements/SSF/RoboB_OakD/ObjectDetection/Videos/2023/01/30/2023-01-30_12:54:18.425_Fish1.avi'
output_directory = video_path.split('/')[-1].replace('.avi', '')
print(output_directory)
text = TextHelper()
delta = 5

# Check if GPU is available and set device
device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device_gpu)

# Class names corresponding to the output classes
class_names = ['fully visible', 'partially visible']

# Define the image classifier model
class ImageClassifier(nn.Module):
  def __init__(self):
    super(ImageClassifier, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(96, 128, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.dropout = nn.Dropout(0.5)
    self.fc1 = nn.Linear(128 * 16 * 16, 512)
    self.fc2 = nn.Linear(512, 2) # 2 classes: fully visible or partially visible

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = self.pool(F.relu(self.conv4(x)))
    x = x.view(-1, 128 * 16 * 16) # Adjusted size after pooling
    x = self.dropout(F.relu(self.fc1(x)))
    x = self.fc2(x)
    return x

# Load the saved model parameters
model_path = 'model_bounding.pth'
model = ImageClassifier().to(device_gpu)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode


# Define the transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def preprocess_image(image):
    # If `image` is a PIL image, convert it to a NumPy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if the image is not already
    if len(image.shape) == 3:
        binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    padding = image.shape[0] // 100 * 5
    
    # Apply binary thresholding
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    
    # Detect contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        if w >= h:
            size = w
        else:
            size = h

        roi = np.zeros((size + padding * 2, size + padding * 2, 3), dtype=np.uint8)
        x_offset = (roi.shape[1] - w) // 2
        y_offset = (roi.shape[0] - h) // 2
        roi[y_offset:y_offset+h, x_offset:x_offset+w] = image[y:y+h, x:x+w]
    else:
        roi = image
    
    # Convert NumPy array to PIL image
    roi = Image.fromarray(roi)
    
    # Convert grayscale to RGB
    roi = roi.convert('RGB')
    
    # Apply transformations
    roi = transform(roi)
    
    # Add batch dimension
    roi = roi.unsqueeze(0)
    
    # Move to the appropriate device
    return roi.to(device_gpu)

# Function to predict the class of an image
def predict_image(image_path, model, class_names):
    image = preprocess_image(image_path)
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Function to create a directory if it does not exist
def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to load an image and convert it to binary format
def load_image(image):
  # Load the image in grayscale mode
  # Apply a binary threshold to convert the image to binary format
  _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
  return binary_image

# Function to remove noise from the binary image
def remove_noise(binary_image):
    # Define a kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)
        
    # Apply morphological operations
    # Opening operation (erosion followed by dilation) to remove small noise
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
        
    # Closing operation (dilation followed by erosion) to close small holes inside objects
    cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, kernel, iterations=2)

    return cleaned_image

# Function to detect contours in the binary image
def detect_contours(image):
  contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  return contours

# Function to rotate the image to a horizontal position
def horizontal_state_tuning(image):
    contours = detect_contours(image)

    # Assuming the largest contour is the one we are interested in
    contour = max(contours, key=cv2.contourArea)

    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)

    # Get the angle of the major axis of the ellipse
    angle = ellipse[2]
                
    # Get the center of the image
    image_center = (image.shape[1] // 2, image.shape[0] // 2)

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle - 90, 1.0)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return rotated_image

# Function to map a point in the rotated image back to the original image
def get_original_point(image, point):
    contours = detect_contours(image)

    # Assuming the largest contour is the one we are interested in
    contour = max(contours, key=cv2.contourArea)

    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)

    # Get the angle of the major axis of the ellipse
    angle = ellipse[2]
                
    # Get the center of the image
    image_center = (image.shape[1] // 2, image.shape[0] // 2)

    # Calculate the inverse rotation matrix
    inverse_rotation_matrix = cv2.getRotationMatrix2D(image_center, 90 - angle, 1.0)

    # Add a third row to the rotation matrix to make it 3x3 for affine transformation
    inverse_rotation_matrix = np.vstack([inverse_rotation_matrix, [0, 0, 1]])

    # Transform the tail point back to the original image
    point_rotated_homogeneous = np.array([point[0], point[1], 1]).T
    point_original = np.dot(inverse_rotation_matrix, point_rotated_homogeneous)
    point_original = (int(point_original[0]), int(point_original[1]))

    return point_original

# Function to detect key points (nose, tail, top fin) on the fish
def detect_key_point(image):
    contours = detect_contours(image)
    contour = max(contours, key=cv2.contourArea)
        
    # Find extreme points
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])

    # Divide the fish into segments
    segment_width = math.ceil((rightmost[0] - leftmost[0]) / 10 * 4)  # Divide into 10 segments
    segment_image = image.copy()
    segment_image[0:segment_image.shape[0], leftmost[0] + segment_width:segment_image.shape[1]] = 0

    segment_width1 = segment_width // 10 # Divide into 10 segments
    tail_position = 'right'
    segment_area = []
    segment_id = 0
    for i in range(10):
        x1 = leftmost[0] + segment_width - (i + 1) * segment_width1
        x2 = x1 + segment_width1

        segment = segment_image[:, min(x1, x2):max(x1, x2)]
        segment_contours, _ = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if segment_contours:
            segment_contour = max(segment_contours, key=cv2.contourArea)
            segment_area.append(cv2.contourArea(segment_contour))
        else:
            segment_area.append(0)
    segment_area.reverse()

    index = find_inflection_point(segment_area)
    if index != -1:
        segment_id = index

    body_image = image.copy()
    tail_image = image.copy()
    width = math.ceil((rightmost[0] - leftmost[0]) / 10)
    if segment_id != 0:
        tail_position = 'left'
        body_image[0:body_image.shape[0], 0:leftmost[0] + width * 3] = 0
        tail_image[0:tail_image.shape[0], leftmost[0] + width * 4:tail_image.shape[1]] = 0
    else:
        body_image[0:body_image.shape[0], leftmost[0] + width * 7:tail_image.shape[1]] = 0
        tail_image[0:tail_image.shape[0], 0:leftmost[0] + width * 6] = 0

    nose_point, top_fin_point = detect_nose_and_top_fin(body_image, tail_position)
    tail_point = detect_tail_middle(tail_image, tail_position)

    return nose_point, top_fin_point, tail_point

# Function to detect the middle of the tail
def detect_tail_middle(image, position):
    contours = detect_contours(image)
    contour = max(contours, key=cv2.contourArea)
        
    # Find extreme points
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])

    # Divide the fish into segments
    segment_width = (rightmost[0] - leftmost[0]) // 10  # Divide into 10 segments
    tail_segment = None
    segment_area_min = float('inf')
   
    for i in range(8):  # We will have 10 segments to analyze
        if position == 'left':
            x1 = rightmost[0] - (i + 1) * segment_width
            x2 = x1 + segment_width
        else:
            x1 = leftmost[0] + i * segment_width
            x2 = x1 + segment_width

        segment = image[:, min(x1, x2):max(x1, x2)]
        segment_contours, _ = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if segment_contours:
            segment_contour = max(segment_contours, key=cv2.contourArea)
            segment_area = cv2.contourArea(segment_contour)
            if segment_area < segment_area_min:
                segment_area_min = segment_area
                tail_segment = (min(x1, x2), max(x1, x2))

    if tail_segment is not None:
        tail_x_coord = (tail_segment[0] + tail_segment[1]) // 2
    else:
        tail_x_coord = (leftmost[0] + rightmost[0]) // 2

    line_eq = (float('inf'), tail_x_coord)
    tail_coords = find_intersections(contour, line_eq)
    if len(tail_coords) == 1:
        tail_coords.append(tail_coords[0])

    tail_y_coord = int((tail_coords[0][1] + tail_coords[1][1]) / len(tail_coords))
    
    return (tail_x_coord, tail_y_coord)

# Function to detect the start point of the top fin
def detect_nose_and_top_fin(image, position):
    if position ==  'left':
        image = cv2.flip(image, 1)
    
    contours = detect_contours(image)
    contour = max(contours, key=cv2.contourArea)
        
    # Find extreme points
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    
    nose_x_coord = leftmost[0]
    nose_point = find_y_coordinates(contour, nose_x_coord)[0]

    start_point_of_top_fin = tuple(contour[contour[:, :, 1].argmin()][0])

    if position == 'left':
        nose_point = (image.shape[1] - nose_point[0], nose_point[1])
        start_point_of_top_fin = (image.shape[1] - start_point_of_top_fin[0], start_point_of_top_fin[1])

    return nose_point, start_point_of_top_fin
    
# Function to find y-coordinates corresponding to an x-coordinate in the contour
def find_y_coordinates(contour, x_coord):
    y_coords = []
    x_coords = []
    for point in contour:
        if x_coord - 3 <= point[0][0] <= x_coord + 3:
            y_coords.append(point[0][1])
            x_coords.append(point[0][0])

    if y_coords:
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Find indices of min and max values
        min_index = y_coords.index(min_y)
        max_index = y_coords.index(max_y)

        min_x = x_coords[min_index]
        max_x = x_coords[max_index]

        return [(min_x, min_y), (max_x, max_y)]
    else:
        return []

# Function to calculate the equation of the line passing through two points
def get_line_equation(p1, p2):
    """Calculate the slope (m) and y-intercept (b) of the line passing through points p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2

    if x1 != x2:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
    else:
        m = float('inf')
        b = y1

    return m, b

# Function to calculate the equation of the line perpendicular to a given line passing through a point
def get_perpendicular_line(p, line_eq):
    """Calculate the equation of the line perpendicular to the given line passing through point p."""
    m1, _ = line_eq
    if m1 != 0:
        m2 = -1 / m1
        b2 = p[1] - m2 * p[0]
        return m2, b2
    elif m1 == 0:
        # Vertical line through point p
        return float('inf'), p[0]
    else:
        return float('inf'), 0

# Function to find the intersections of a line with the contour
def find_intersections(contour, line_parms):
    intersections = []
    n = len(contour)

    if line_parms[0] == float('inf'):
        # Vertical line case
        x_intersect = line_parms[1]
        for i in range(n - 1):
            x1, y1 = contour[i][0]
            x2, y2 = contour[i + 1][0]
            
            # Check if the segment crosses the vertical line
            if (x1 <= x_intersect <= x2) or (x2 <= x_intersect <= x1):
                # Calculate the intersection
                if x2 != x1:
                    m = (y2 - y1) / (x2 - x1)
                    c = y1 - m * x1
                    y = m * x_intersect + c
                    intersections.append((int(x_intersect), int(y)))
                else:
                    # Handle vertical segment case
                    intersections.append((int(x1), int(y1)))
    else:
        for i in range(n - 1):
            x1, y1 = contour[i][0]
            x2, y2 = contour[i + 1][0]

            # Line equation at endpoints
            y_line1 = line_parms[0] * x1 + line_parms[1]
            y_line2 = line_parms[0] * x2 + line_parms[1]

            # Check if segment crosses the line
            if (y1 >= y_line1 and y2 <= y_line2) or (y1 <= y_line1 and y2 >= y_line2):
                # Calculate the intersection
                if x2 != x1:
                    m = (y2 - y1) / (x2 - x1)
                    c = y1 - m * x1
                    x = (line_parms[1] - c) / (m - line_parms[0])
                    y = line_parms[0] * x + line_parms[1]
                    intersections.append((int(x), int(y)))
                else:
                    intersections.append((x1, y1))
    
    if not intersections:
        return []
    
    y_values = [point[1] for point in intersections]
    min_y = min(y_values)
    max_y = max(y_values)
    min_y_point = [point for point in intersections if point[1] == min_y][0]
    max_y_point = [point for point in intersections if point[1] == max_y][0]
    
    point = [min_y_point, max_y_point]
    point = remove_duplicates(point)

    return point

def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def find_inflection_point(prices):
    increase_count = 0
    
    for i in range(2, len(prices) - 2):
        if prices[i - 2] - 2 >= prices[i-1] >= prices[i] and prices[i - 1] >= prices[i] < prices[i + 1]:
            increase_count += 1
            return i

    return -1  # Return -1 if no such index is found

def put_text_on_image(img_draw, xc, yc, x_mm, y_mm ,z_mm):

   text.rectangle(img_draw, (xc - delta, yc - delta), (xc + delta, yc + delta))
   text.putText(img_draw, "X: " + ("{:.1f}m".format(x_mm / 1000) if not math.isnan(x_mm) else "--"),
                                 (xc + 10, yc + 20))
   text.putText(img_draw, "Y: " + ("{:.1f}m".format(y_mm / 1000) if not math.isnan(y_mm) else "--"),
                                 (xc + 10, yc + 35))
   text.putText(img_draw, "Z: " + ("{:.1f}m".format(z_mm / 1000) if not math.isnan(z_mm) else "--"),
                                 (xc + 10, yc + 50))


def main():
 
    make_directory(output_directory)
    data_writing = csv_writer(csv_path='test.csv')
    model_path = 'torchScript_segmentor_model_july.ts'

    # video_path = '../Seal.mp4'
    output_folder_path = 'Processed_Results'
    print('start')
    alpha = 1
    fps = 30

    seg_ts_model = torch.jit.load(model_path)
    seg_ts_model.eval()

    seg_ts_model.to(device_gpu)
    time_per_frame = []
    tracker = DeepSort(max_age=5, max_iou_distance=0.5)


    
    # Connect to device and start pipeline

    length_data = {}  # length data dicL: {key is track id: [[1st list bbox width, height], key_missed_count]}
    track_centers_data = {}  # TrackPoint
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    try:
                
        while True:
            ret,frame_orig = cap.read()

            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            #orig_draw = frame_orig.copy()
            orig_height, orig_width, _ = frame_orig.shape
            frame = resize_frame(frame_orig)
            img_draw = frame.copy()
            height_f, width_f, _ = img_draw.shape
            start_time = time.time()
            scale_x = int(orig_width / width_f)
            scale_y = int(orig_height / height_f)
            input_np = read_image(frame)
            input_ts = torch.as_tensor(input_np.astype("float32").transpose(2, 0, 1))

            outputs_ts = seg_ts_model(input_ts)
            # masks = outputs_ts[2]
            time_per_frame.append(round(time.time() - start_time, 2) * 1000)
            print("==>>>> Inference time by TorchScript models has taken {} ms".format(
                round((time.time() - start_time) * 1000, 2)))

            print('*************************************************************************')

            masks = outputs_ts[2]
            # image_processed = post_process_outputs(outputs_ts, input_np)
            bounding_boxes = []
            for ind in range(len(outputs_ts[0])):
                x1, y1, x2, y2 = int(outputs_ts[0][ind][0]), int(outputs_ts[0][ind][1]), int(
                    outputs_ts[0][ind][2]), int(
                    outputs_ts[0][ind][3])
                crop_mask = img_draw[y1:y2, x1:x2]

                img_w, img_h = x2 - x1, y2 - y1
                print('img_w =',img_w)
                print('img_h =',img_h)
                mask = masks[ind, None, :, :]
                black_image_mask_shape = np.zeros((int(img_h),int(img_w),3),dtype=np.uint8)
                mask_black_image = np.zeros((int(height_f), int(width_f), 3), dtype=np.uint8)  # Create a black background image
                mask = mask.cpu()
                masks_chunk = do_paste_mask(mask, img_h, img_w)

                np_mask = masks_chunk.numpy()
                np_mask1 = masks_chunk.numpy()
                np_mask = np.where(np_mask > 0.5, 255, 0)

                np_mask1 = np_mask1.squeeze(axis=(0, 1))  # Remove singleton dimensions
                np_mask = np_mask.squeeze(axis=(0, 1))  # Remove singleton dimensions
                np_mask1 = np_mask1.astype(np.uint8)  # Ensure the mask is of type uint8
                np_mask = np_mask.astype(np.uint8)  # Ensure the mask is of type uint8
                reddish_tint_color = np.array([0, 0, 255], dtype=np.uint8)  # [B, G, R]

                alpha = 0.5

                crop_mask[np_mask > 0] = crop_mask[np_mask > 0] * (alpha) + reddish_tint_color * alpha
                img_draw[y1:y2, x1:x2] = crop_mask
                black_image_mask_shape[np_mask > 0] =[255, 255, 255]
                print('mask_black_image = ', mask_black_image.shape)
                mask_black_image[y1:y2, x1:x2] = black_image_mask_shape
                #binary_image = mask_black_image
                #mask_black_image = cv2.cvtColor(mask_black_image, cv2.COLOR_GRAY2BGR)
                
                img_draw = cv2.resize(img_draw, (orig_width, orig_height))
                mask_black_image = cv2.resize(mask_black_image, (orig_width, orig_height))

                start_time = time.time()
                predicted_class = predict_image(mask_black_image, model, class_names)
                end_time = time.time()
                prediction_time = end_time - start_time

                print(f'The image is {predicted_class}')
                print(f'Prediction Time: {prediction_time:.6f} seconds')
                mask_black_image_color = mask_black_image.copy()
                mask_black_image = cv2.cvtColor(mask_black_image, cv2.COLOR_BGR2GRAY)
                binary_image = load_image(mask_black_image)
                
                if predicted_class == 'fully visible':
                    filename = 'mask_'+str(frame_num)+'.jpg'
                    frame_num = frame_num + 1
                    rotated_image = horizontal_state_tuning(binary_image)
                    cleaned_image = remove_noise(rotated_image)
                    image = rotated_image.copy()
                    
                    contours = detect_contours(cleaned_image)
                    if not contours:
                        continue
                    
                    nose_point, top_fin_point, tail_point = detect_key_point(cleaned_image)
                    original_nose_point = get_original_point(binary_image, nose_point)
                    original_tail_point = get_original_point(binary_image, tail_point)
                    original_top_fin_point = get_original_point(binary_image, top_fin_point)
                   
                    # calculate the length of fish
                    length = int(math.sqrt(math.pow(original_nose_point[1] - original_tail_point[1], 2) + math.pow(original_nose_point[0] - original_tail_point[0], 2)))

                    # check the top fin point
                    if (original_nose_point[0] + original_tail_point[0]) / 2 - length / 20 >= original_top_fin_point[0] or original_top_fin_point[0] >= (original_nose_point[0] + original_tail_point[0]) / 2 + length / 20:
                        continue                
                   
                    contours = detect_contours(binary_image)
                    contour = max(contours, key=cv2.contourArea)

                    # Get the line equation connecting the nose and tail
                    line_eq = get_line_equation(original_nose_point, original_tail_point)

                    # Get the equation of the perpendicular line from the dorsal start point
                    perp_line_eq = get_perpendicular_line(original_top_fin_point, line_eq)

                    # Find the intersection of the perpendicular line with the contour
                    intersections = find_intersections(contour, perp_line_eq)
                    if intersections:
                        if intersections.__len__() > 1:
                            height = math.sqrt(math.pow(intersections[0][1] - intersections[1][1], 2) + math.pow(intersections[0][0] - intersections[1][0], 2))
                        else:
                            height = math.sqrt(math.pow(intersections[0][1] - original_top_fin_point[1], 2) + math.pow(intersections[0][0] - original_top_fin_point[0], 2))

                    # check the length and height ration            
                    if 5.5 <= length / height or length / height <= 2.5:
                        continue

                    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
                    cv2.circle(binary_image, original_nose_point, 3, (0, 0, 255), -1)
                    cv2.circle(binary_image, (original_tail_point[0], original_tail_point[1]), 3, (255, 0, 0), -1)
                    cv2.line(binary_image, original_nose_point, original_tail_point, (255, 0, 255), 2)

                    # Draw the vertical line from the known point to the intersection point on the second line
                    if intersections:
                        if intersections.__len__() > 1:
                            cv2.circle(binary_image, (intersections[0][0], intersections[0][1]), 3, (0, 255, 0), -1)
                            cv2.circle(binary_image, (intersections[1][0], intersections[1][1]), 3, (0, 255, 0), -1)
                            cv2.line(binary_image, intersections[0], intersections[1], (0, 255, 255), 2)
                        else:
                            cv2.circle(binary_image, original_top_fin_point, 3, (0, 255, 0), -1)
                            cv2.circle(binary_image, intersections[0], 3, (0, 255, 0), -1)
                            cv2.line(binary_image, intersections[0], original_top_fin_point, (0, 255, 255), 2)

                    cv2.imwrite(os.path.join(output_directory, filename), binary_image)
            
            if True:
                image = cv2.resize(frame_orig, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR)
                cv2.imshow('image_processed', image)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        
        cv2.destroyAllWindows()
        cap.release()
    finally:
        # Stop streaming
        print('')

if __name__ == "__main__":
    main()
