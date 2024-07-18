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
base_dir = 'D:/mywork/fish_detect/FullFishDataset/Input'

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

def detect_contours(binary_image):
  # Find contours in the binary image
  contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  return contours

def remove_tail(binary_image):
    contours = detect_contours(binary_image=binary_image)

    # Assume the largest contour is the fish
    contour = max(contours, key=cv2.contourArea)
        
    # Find extreme points
    leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
    rightmost = tuple(contour[contour[:, :, 0].argmax()][0])

    # Determine if the tail is on the left or right
    tail_on_left = True if leftmost[0] < rightmost[0] else False

    # Set the direction for segmentation
    if tail_on_left:
        start_x, end_x = leftmost[0], rightmost[0]
    else:
        start_x, end_x = rightmost[0], leftmost[0]

    # Divide the fish into segments
    segment_width = abs(end_x - start_x) // 10  # Divide into 10 segments
    tail_segment = None
    segment_area_min = float('inf')
    segment_id = 0

    for i in range(1, 9):  # We will have 10 segments to analyze
        if tail_on_left:
            x1 = start_x + i * segment_width
            x2 = x1 + segment_width
        else:
            x1 = start_x - i * segment_width
            x2 = x1 - segment_width

        segment = binary_image[:, min(x1, x2):max(x1, x2)]
            
        # Find contours in the segment
        segment_contours, _ = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segment_area = cv2.contourArea(segment_contours[0])
        if segment_area < segment_area_min:
            segment_area_min = segment_area
            segment_id = i
            tail_segment = (min(x1, x2), max(x1, x2)) 

        # Draw the segment boundaries
        # cv2.rectangle(binary_image, (min(x1, x2), 0), (max(x1, x2), binary_image.shape[0]), (255, 0, 0), 1)

    # Highlight the tail segment
    if tail_segment:
        if segment_id < 5:
            binary_image[0:binary_image.shape[0], 0:tail_segment[1]-segment_width] = 0
            start_x_coord = start_x + segment_id * segment_width
            end_x_coord = end_x
        else:
            binary_image[0:binary_image.shape[0], tail_segment[0]+segment_width:binary_image.shape[1]] = 0
            start_x_coord = start_x
            end_x_coord = start_x + (segment_id + 1) * segment_width

    coords = []
    start_y_coords = find_y_coordinates(contour, start_x_coord)
    start_y_coord = int(sum(start_y_coords) / len(start_y_coords))
    coords.append([start_x_coord, start_y_coord])
    cv2.circle(binary_image, (start_x_coord, start_y_coord), 5, (127), -1)
    end_y_coords = find_y_coordinates(contour, end_x_coord)
    end_y_coord = int(sum(end_y_coords) / len(end_y_coords))
    coords.append([end_x_coord, end_y_coord])
    cv2.circle(binary_image, (end_x_coord, end_y_coord), 5, (127), -1)
    mid_x_coord = int((start_x_coord+end_x_coord)/2)
    mid_y_coords = find_y_coordinates(contour, mid_x_coord)
    for y in mid_y_coords:
        coords.append([mid_x_coord, y])
        cv2.circle(binary_image, (mid_x_coord, y), 5, (127), -1)
    cv2.line(binary_image, (start_x_coord, start_y_coord), (end_x_coord, end_y_coord), (127), 2)

    return coords

def find_y_coordinates(contour, x_coord):
    y_coords = []
    for point in contour:
        if x_coord - 10 <= point[0][0] <= x_coord + 10:
            y_coords.append(point[0][1])

    if y_coords:
      min_y = min(y_coords)
      max_y = max(y_coords)
      return [min_y, max_y]
    else:
      return []

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

            cv2.imshow('input', binary_image)

            if predicted_class == 'fully visible':
                cleaned_image = remove_noise(binary_image)
                coords = remove_tail(cleaned_image)
                output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
                print(coords)
                if coords:
                    for coord in coords:
                        cv2.circle(output_image, (coord[0], coord[1]), 5, (0, 0, 255), -1)
                    cv2.line(output_image, (coords[0][0], coords[0][1]), (coords[1][0], coords[1][1]), (0, 255, 0), 2)
                    cv2.line(output_image, (coords[2][0], coords[2][1]), (coords[3][0], coords[3][1]), (0, 255, 0), 2)
                cv2.imshow('output', output_image)
            
            # white_pixels_count(binary_image, filename)

            while True:
                c = cv2.waitKey(100)
                if c == 27:
                    break
            cv2.destroyAllWindows()
    