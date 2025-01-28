import cv2
import numpy as np
import torch
from torchvision.models import ResNet18_Weights
import torchvision.models as models
import torchvision.transforms as transforms
from datetime import datetime
from PIL import Image
import os


# Load a pretrained CNN (e.g., ResNet) for feature extraction
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Updated argument for weights
model.fc = torch.nn.Identity()  # Remove the classification layer for embeddings
model.eval()

# Define transformations for the RGB and depth frames
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_rgbembeddings(video_path, model):
    """Extract frame embeddings from an RGB video."""
    cap = cv2.VideoCapture(video_path)
    embeddings = []
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame and pass through the model
        input_frame = transform(frame).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            embedding = model(input_frame).numpy().flatten()  # Get the embedding
            embeddings.append(embedding)
        
        # Get the timestamp (frame position in seconds)
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

    cap.release()
    return np.array(embeddings), np.array(timestamps)



def extract_depthembeddings(video_path, model):
    """Extract embeddings from a depth video."""
    cap = cv2.VideoCapture(video_path)
    embeddings = []
    timestamps = []

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return np.array([]), np.array([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check the shape of the frame to ensure it's being read
        # print(f"Frame shape: {frame.shape}")  # Check this in the console

        # Normalize depth values (assuming 16-bit depth, scale to [0, 255])
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Check the normalized frame shape
        # print(f"Normalized frame shape: {frame.shape}")

        # Convert single-channel depth image to 3 channels by stacking
        frame = np.expand_dims(frame, axis=-1)  # Shape becomes (H, W, 1)
        frame = np.concatenate([frame] * 3, axis=-1)  # Now shape is (H, W, 3)

        # Check the final shape after conversion
        # print(f"Final frame shape after conversion: {frame.shape}")

        print(f"Initial frame shape: {frame.shape}")
        print(f"Number of dimensions: {frame.ndim}")
        if frame.ndim == 3:
            print(f"Number of channels: {frame.shape[2]}")

        if len(frame.shape) == 2:  # Single-channel (grayscale)
            frame = np.expand_dims(frame, axis=-1)  # Add a channel dimension
            frame = np.concatenate([frame] * 3, axis=-1)  # Convert to 3 channels
            print(f"Converted to 3 channels: {frame.shape}")

        # Ensure it's in the correct shape (H, W, 3) and is a 3D array (height, width, channels)
        if frame.ndim == 3 and frame.shape[2] == 3:
            input_frame = transform(frame).unsqueeze(0)  # Transform and add batch dimension
            print(f"Input frame shape after transformation: {input_frame.shape}")
        else:
            continue  # Skip if the frame is not in the expected shape

        with torch.no_grad():
            embedding = model(input_frame)
            print(f"Embedding shape: {embedding.shape}")
            embedding = embedding.numpy().flatten()  # Flatten the embedding
            embeddings.append(embedding)

        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

    cap.release()

    # Check if embeddings were actually extracted
    if len(embeddings) == 0:
        print("No embeddings were extracted from the depth video.")

    return np.array(embeddings), np.array(timestamps)



# Example usage
rgb_embeddings, rgb_timestamps = extract_rgbembeddings('trial1/videos_1/rgb_1.avi', model)
depth_embeddings, depth_timestamps = extract_depthembeddings('trial1/videos_1/depth_1.avi', model)

def save_embeddings_to_npz(embeddings, timestamps, output_path):
    """Save embeddings and timestamps to a compressed .npz file."""
    np.savez_compressed(output_path, embeddings=embeddings, timestamps=timestamps)

# Example usage after extracting embeddings:
save_embeddings_to_npz(rgb_embeddings, rgb_timestamps, "rgb_embeddings.npz")
save_embeddings_to_npz(depth_embeddings, depth_timestamps, "depth_embeddings.npz")



# data = np.load("rgb_embeddings.npz")
# embeddings = data["embeddings"]
# timestamps = data["timestamps"]