import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time

# Define directories
base_dir = 'mask'
output_dir = 'output'

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Class names corresponding to the output classes
class_names = ['fully visible', 'partially visible']

# Define the image classifier model
class ImageClassifier(nn.Module):
  def __init__(self):
    super(ImageClassifier, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.fc1 = nn.Linear(256 * 16 * 16, 512)
    self.fc2 = nn.Linear(512, 2) # 2 classes: fully visible or partially visible

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = self.pool(F.relu(self.conv4(x)))
    x = x.view(-1, 256 * 16 * 16) # Adjusted size after pooling
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# Load the saved model parameters
model_path = 'model.pth'
model = ImageClassifier().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = Image.fromarray(image)  # Convert NumPy array to PIL image
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

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
def load_image(image_path):
  # Load the image in grayscale mode
  image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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
    # Assume the largest contour is the fish
    contour = max(contours, key=cv2.contourArea)
        
    # Find extreme points
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])

    # Divide the fish into segments
    segment_width = abs(rightmost[0] - leftmost[0]) // 10  # Divide into 10 segments
    tail_segment = None
    tail_position = 'left'
    segment_area_min = float('inf')
    segment_id = 0

    for i in range(1, 9):  # Analyze 9 segments
        x1 = leftmost[0] + i * segment_width
        x2 = x1 + segment_width

        segment = image[:, min(x1, x2):max(x1, x2)]
            
        # Find contours in the segment
        segment_contours, _ = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Assume the largest contour is the fish
        segment_contour = max(segment_contours, key=cv2.contourArea)
        segment_area = cv2.contourArea(segment_contour)
        if segment_area < segment_area_min:
            segment_area_min = segment_area
            segment_id = i
            tail_segment = (min(x1, x2), max(x1, x2)) 

    head_image = image.copy()
    body_image = image.copy()
    tail_image = image.copy()

    if tail_segment:
        if segment_id < 5:
            head_image[0:head_image.shape[0], 0:leftmost[0] + segment_width * 8] = 0
            body_image[0:body_image.shape[0], 0:leftmost[0] + segment_width * 4] = 0
            body_image[0:body_image.shape[0], leftmost[0] + segment_width * 8:body_image.shape[1]] = 0
            tail_image[0:tail_image.shape[0], tail_segment[1]:tail_image.shape[1]] = 0
        else:
            head_image[0:head_image.shape[0], leftmost[0] + segment_width * 2:head_image.shape[1]] = 0
            body_image[0:body_image.shape[0], 0:leftmost[0] + segment_width * 2] = 0
            body_image[0:body_image.shape[0], leftmost[0] + segment_width * 6:body_image.shape[1]] = 0
            tail_image[0:tail_image.shape[0], 0:tail_segment[0]] = 0
            tail_position = 'right'
            
    nose_point = detect_nose(head_image, tail_position)
    tail_point = detect_tail_middle(tail_image, tail_position)
    top_fin_point = detect_top_fin(body_image, tail_position)

    return tail_point, nose_point, top_fin_point

# Function to detect the nose of the fish
def detect_nose(image, position):
    contours = detect_contours(image)

    # Assume the largest contour is the fish
    contour = max(contours, key=cv2.contourArea)
        
    # Find extreme points
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])

    if position == 'left':
        nose_x_coord = rightmost[0]
    else:
        nose_x_coord = leftmost[0]

    nose_coords = find_y_coordinates(contour, nose_x_coord)

    return (nose_coords[0][0], nose_coords[0][1])

# Function to detect the middle of the tail
def detect_tail_middle(image, position):
    contours = detect_contours(image)

    # Assume the largest contour is the fish
    contour = max(contours, key=cv2.contourArea)
        
    # Find extreme points
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])

    # Divide the fish into segments
    segment_width = abs(rightmost[0] - leftmost[0]) // 10  # Divide into 10 segments
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
            
        # Find contours in the segment
        segment_contours, _ = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if segment_contours:
            # Assume the largest contour is the fish
            segment_contour = max(segment_contours, key=cv2.contourArea)
            segment_area = cv2.contourArea(segment_contour)

            if segment_area < segment_area_min:
                segment_area_min = segment_area
                tail_segment = (min(x1, x2), max(x1, x2))

    if tail_segment is not None:
        tail_x_coord = (tail_segment[0] + tail_segment[1]) // 2
    else:
        tail_x_coord = (leftmost[0] + rightmost[0]) // 2
    tail_coords = find_y_coordinates(contour, tail_x_coord)
    tail_x_coord = int((tail_coords[0][0] + tail_coords[1][0]) / len(tail_coords))
    tail_y_coord = int((tail_coords[0][1] + tail_coords[1][1]) / len(tail_coords))
    
    return (tail_x_coord, tail_y_coord)

# Function to detect the start point of the top fin
def detect_top_fin(image, position):
    contours = detect_contours(image)

    # Assume the largest contour is the fish
    contour = max(contours, key=cv2.contourArea)

    if position ==  'left':
        image = cv2.flip(image, 1)
        contours = detect_contours(image)

        # Assume the largest contour is the fish
        contour = max(contours, key=cv2.contourArea)
    
    # Compute differences with step size of 5
    start = 0
    step = 1
    half_len = contour.shape[0] // 2

    # Select elements with the specified step size
    x_coords = contour[start:half_len:step, 0, 0]
    y_coords = contour[start:half_len:step, 0, 1]

    # Compute dx and dy using the step size
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)

    # Initialize the threshold value
    threshold_value = 4
    index = None

    # Define the loop to decrement threshold_value until a valid index is found or threshold_value reaches -1
    while threshold_value >= 0 and index is None:
        # Find the indices where x is greater than the threshold value and y is 0
        indices = [i for i, (x_val, y_val) in enumerate(zip(dx, dy)) if abs(x_val) >= threshold_value and y_val == 0]
        
        # Further filter these indices to only include those greater than 9
        filtered_indices = [i for i in indices if i > 9]
        
        if filtered_indices:
            index = filtered_indices[0]
        elif indices:
            index = indices[0]
        else:
            # Decrement the threshold value
            threshold_value -= 1
    
    # Get the start point of the top fin
    start_point_of_top_fin = tuple(contour[index][0])

    if position == 'left':
        start_point_of_top_fin = (image.shape[1] - start_point_of_top_fin[0], start_point_of_top_fin[1])

    return start_point_of_top_fin

# Function to find y-coordinates corresponding to an x-coordinate in the contour
def find_y_coordinates(contour, x_coord):
    y_coords = []
    x_coords = []
    for point in contour:
        if x_coord - 7 <= point[0][0] <= x_coord + 7:
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
    
    return intersections

if __name__ == "__main__":
    make_directory(output_dir)
    
    images = os.listdir(base_dir)
    for i in range(len(images)):
        filename = images[i]
        
        print(filename)
        if filename.endswith('.png') or filename.endswith('.jpg'):
            # Example usage
            image_path = os.path.join(base_dir, filename)

            start_time = time.time()
            predicted_class = predict_image(image_path, model, class_names)
            end_time = time.time()
            prediction_time = end_time - start_time

            print(f'The image is {predicted_class}')
            print(f'Prediction Time: {prediction_time:.6f} seconds')

            binary_image = load_image(image_path)
            cleaned_image = remove_noise(binary_image)

            if predicted_class == 'fully visible':
                rotated_image = horizontal_state_tuning(cleaned_image)
                image = rotated_image.copy()
                
                tail_point, nose_point, top_fin_point = detect_key_point(rotated_image)
                original_nose_point = get_original_point(cleaned_image, nose_point)
                original_tail_point = get_original_point(cleaned_image, tail_point)
                original_top_fin_point = get_original_point(cleaned_image, top_fin_point)
                
                contours = detect_contours(cleaned_image)
                contour = max(contours, key=cv2.contourArea)

                cleaned_image = cv2.cvtColor(cleaned_image, cv2.COLOR_GRAY2BGR)
                cv2.circle(cleaned_image, (original_nose_point[0], original_nose_point[1]), 3, (0, 0, 255), -1)
                cv2.circle(cleaned_image, (original_tail_point[0], original_tail_point[1]), 3, (255, 0, 0), -1)
                cv2.line(cleaned_image, original_nose_point, original_tail_point, (255, 0, 255), 2)
                
                # Get the line equation connecting the nose and tail
                line_eq = get_line_equation(original_nose_point, original_tail_point)

                # Get the equation of the perpendicular line from the dorsal start point
                perp_line_eq = get_perpendicular_line(original_top_fin_point, line_eq)

                # Find the intersection of the perpendicular line with the contour
                intersections = find_intersections(contour, perp_line_eq)

                # Draw the vertical line from the known point to the intersection point on the second line
                if intersections.__len__() > 1:
                    cv2.circle(cleaned_image, (intersections[0][0], intersections[0][1]), 3, (0, 255, 0), -1)
                    cv2.circle(cleaned_image, (intersections[1][0], intersections[1][1]), 3, (0, 255, 0), -1)
                    cv2.line(cleaned_image, intersections[0], intersections[1], (0, 255, 255), 2)
                else:
                    cv2.circle(cleaned_image, original_top_fin_point, 3, (0, 255, 0), -1)
                    cv2.circle(cleaned_image, intersections[0], 3, (0, 255, 0), -1)
                    cv2.line(cleaned_image, intersections[0], original_top_fin_point, (0, 255, 255), 2)

                cv2.imwrite(os.path.join(output_dir, filename), cleaned_image)

            cv2.putText(cleaned_image, predicted_class, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(filename, cleaned_image)

            # while True:
            c = cv2.waitKey(1)
            if c == 27:
                break
            cv2.destroyAllWindows()
    