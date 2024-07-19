import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2

# Define directories
base_dir = 'D:/mywork/fish detect/image'

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Class names corresponding to the output classes
class_names = ['fully visible', 'partially visible']

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)  # Adjusted size after pooling
        self.fc2 = nn.Linear(512, 2)  # 2 classes: fully visible or partially visible

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)  # Adjusted size after pooling
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

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def white_pixels_count(binary_image, filename):
    row_directory = 'row'
    col_directory = 'col'
    make_directory(row_directory)
    make_directory(col_directory)

    # Count the number of white pixels in each row
    row_counts = np.sum(binary_image == 255, axis=1)

    # Count the number of white pixels in each column
    column_counts = np.sum(binary_image == 255, axis=0)

    # Save the result images
    row_output_path = os.path.join(row_directory, filename)  # Replace with your desired save path
    column_output_path = os.path.join(col_directory, filename)  # Replace with your desired save path

    # Plot row-wise white pixel counts
    plt.figure(figsize=(10, 4))
    plt.plot(row_counts, color='blue')
    plt.title('Row-wise White Pixel Counts')
    plt.xlabel('Row Index')
    plt.ylabel('White Pixel Count')
    plt.grid()
    plt.savefig(row_output_path)
    plt.show()

    # Plot column-wise white pixel counts
    plt.figure(figsize=(10, 4))
    plt.plot(column_counts, color='green')
    plt.title('Column-wise White Pixel Counts')
    plt.xlabel('Column Index')
    plt.ylabel('White Pixel Count')
    plt.grid()
    plt.savefig(column_output_path)
    plt.show()

def load_image(image_path):
  # Load the image in grayscale mode
  image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  # Apply a binary threshold to convert the image to binary format
  _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
  return binary_image

def remove_noise(binary_image):
  # Define a kernel for morphological operations
  kernel = np.ones((3, 3), np.uint8)
    
  # Apply morphological operations
  # Opening operation (erosion followed by dilation) to remove small noise
  cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    
  # Closing operation (dilation followed by erosion) to close small holes inside objects
  cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    
  return cleaned_image

def detect_contours(image):
  # Find contours in the binary image
  contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  return contours

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

    for i in range(1, 9):  # We will have 10 segments to analyze
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

        # Draw the segment boundaries
        # cv2.rectangle(output_image, (min(x1, x2), 0), (max(x1, x2), output_image.shape[0]), (255, 0, 0), 1)

    head_image = image.copy()
    body_image = image.copy()
    tail_image = image.copy()

    # Highlight the tail segment
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
        # Assume the largest contour is the fish
        segment_contour = max(segment_contours, key=cv2.contourArea)
        segment_area = cv2.contourArea(segment_contour)

        if segment_area < segment_area_min:
            segment_area_min = segment_area
            tail_segment = (min(x1, x2), max(x1, x2))

    tail_x_coord = (tail_segment[0] + tail_segment[1]) // 2
    tail_coords = find_y_coordinates(contour, tail_x_coord)
    tail_x_coord = int((tail_coords[0][0] + tail_coords[1][0]) / len(tail_coords))
    tail_y_coord = int((tail_coords[0][1] + tail_coords[1][1]) / len(tail_coords))
    
    return (tail_x_coord, tail_y_coord)

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
    step = 2
    half_len = contour.shape[0] // 2

    # Select elements with the specified step size
    x_coords = contour[start:half_len:step, 0, 0]
    y_coords = contour[start:half_len:step, 0, 1]

    # Compute dx and dy using the step size
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)

    # Compute gradients
    gradients = np.arctan2(dy, dx) * 180 / np.pi
   
    gradient_changes = np.abs(np.diff(gradients))
    max_change_index = np.argmax(gradient_changes)

    # Get the start point of the top fin
    start_point_of_top_fin = tuple(contour[max_change_index][0])

    if position == 'left':
        start_point_of_top_fin = (image.shape[1] - start_point_of_top_fin[0], start_point_of_top_fin[1])

    return start_point_of_top_fin

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

def get_line_equation(p1, p2):
    """Calculate the slope (m) and y-intercept (b) of the line passing through points p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    return m, b

def get_perpendicular_line(p, line_eq):
    """Calculate the equation of the line perpendicular to the given line passing through point p."""
    m1, _ = line_eq
    if m1 != 0:
        m2 = -1 / m1
    else:
        m2 = float('inf')  # Vertical line
    b2 = p[1] - m2 * p[0]

    return m2, b2

def find_intersections(contour, line_parms):
    intersections = []
    n = len(contour)

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
                x = (c - line_parms[1]) / (line_parms[0] - m)
                y = line_parms[0] * x + line_parms[1]
                intersections.append((int(x), int(y)))
    
    return intersections

if __name__ == "__main__":
    for filename in os.listdir(base_dir):
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

                # Assume the largest contour is the fish
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
                intersection_point = None
                min_distance = float('inf')
                min_y = 0

                intersections = find_intersections(contour, perp_line_eq)
                
                # print(f"original_nose_point: {original_nose_point}")
                # print(f"original_tail_point: {original_tail_point}")
                # print(f"original_top_fin_point: {original_top_fin_point}")
                # print(f"intersection_point: {intersection_point}")
                # Draw the vertical line from the known point to the intersection point on the second line
                cv2.line(cleaned_image, intersections[0], intersections[1], (0, 255, 255), 2)
                cv2.circle(cleaned_image, (intersections[0][0], intersections[0][1]), 3, (0, 255, 0), -1)
                cv2.circle(cleaned_image, (intersections[1][0], intersections[1][1]), 3, (0, 255, 0), -1)
                           
            cv2.putText(cleaned_image, predicted_class, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(filename, cleaned_image)

            # white_pixels_count(binary_image, filename)

            # while True:
            c = cv2.waitKey(500)
            if c == 27:
                break
            cv2.destroyAllWindows()
    