import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Define directories
base_dir = 'D:/mywork/fish_detect/images'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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
        self.fc2 = nn.Linear(512, 2)  # 2 classes: fully visible or partially visible

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Model, loss function, optimizer
model_path = 'D:/mywork/fish detect/final/model_bounding.pth'
model = ImageClassifier().to(device)
model.load_state_dict(torch.load(model_path))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
# optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, weight_decay=1e-4, centered=False)
# optimizer = optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-4)
# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4, nesterov=True)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Initialize weights
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)

model.apply(weights_init)

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

# Function to extract ROI with padding
def extract_roi_with_padding(image):
    padding = image.shape[0] // 100 * 5
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
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
        return roi
    return image  # Return the original image if no contours are found

# Custom dataset class to preprocess images with bounding box extraction
class CustomROIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        for label, subdir in enumerate(['fully', 'partially']):
            subdir_path = os.path.join(root_dir, subdir)
            for filename in os.listdir(subdir_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    self.image_files.append(os.path.join(subdir_path, filename))
                    self.labels.append(label)

        print(f"Found {len(self.image_files)} images in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = cv2.imread(img_name)
        roi = extract_roi_with_padding(image)
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        label = self.labels[idx]
        if self.transform:
            roi = self.transform(roi_pil)
        return roi, label

# Verify dataset directories and file counts
print(f"Training directory: {train_dir}, Validation directory: {validation_dir}")
print(f"Training files: {sum([len(files) for r, d, files in os.walk(train_dir)])}, Validation files: {sum([len(files) for r, d, files in os.walk(validation_dir)])}")

# Define data augmentation transformations
data_augmentation_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Define validation transformations
validation_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load the datasets with the custom dataset class
train_dataset = CustomROIDataset(root_dir=train_dir, transform=data_augmentation_transforms)
val_dataset = CustomROIDataset(root_dir=validation_dir, transform=validation_transforms)

# Debug print to ensure datasets are loaded
print(f"Number of training samples: {len(train_dataset)}, Number of validation samples: {len(val_dataset)}")

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Transfer to GPU
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Transfer to GPU
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    print(f"Validation Loss: {val_loss/len(val_loader)}")

    # Save the model
    model_path = 'model_bounding' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
