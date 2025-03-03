import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def compute_features(file_path):
    with open(file_path, 'rb') as file:
        byte_content = file.read()
        byte_counter = Counter(byte_content)
        histogram = [byte_counter.get(i, 0) for i in range(256)]
        total_bytes = sum(histogram)
        
        if total_bytes > 0:
            normalized_histogram = np.array(histogram) / total_bytes
            entropy = -np.sum(normalized_histogram[normalized_histogram > 0] * np.log2(normalized_histogram[normalized_histogram > 0]))
        else:
            normalized_histogram = np.zeros(256)  # Return an array of zeros if the sum is zero
            entropy = 0
        
        feature_vector = np.append(normalized_histogram, entropy)
        return feature_vector

def process_directory(directory_path):
    features = []
    labels = []
    label_map = {}  # To store folder names and corresponding labels
    label_index = 0

    for root, dirs, files in os.walk(directory_path):
        for dir in dirs:
            current_dir_path = os.path.join(root, dir)
            for file in os.listdir(current_dir_path):
                file_path = os.path.join(current_dir_path, file)
                if file_path.lower().endswith('.pdf'):  # Check if the file is a PDF
                    try:
                        feature_vector = compute_features(file_path)
                        features.append(feature_vector)
                        if dir not in label_map:
                            label_map[dir] = label_index
                            label_index += 1
                        labels.append(label_map[dir])
                    except Exception as e:
                        print(f"Failed to process {file_path}: {e}")

    return np.array(features), np.array(labels), label_map

def oversample_classes(features, labels, label_map, target_counts):
    # Find indices of each class to oversample
    for label, count in target_counts.items():
        label_index = label_map[label]
        indices = np.where(labels == label_index)[0]
        current_count = len(indices)
        if current_count < count:
            # Oversample the class
            oversampled_indices = resample(indices, replace=True, n_samples=count - current_count)
            features = np.concatenate([features, features[oversampled_indices]])
            labels = np.concatenate([labels, labels[oversampled_indices]])

    return features, labels

# Processing the directory containing files
data_directory = r"C:\Users\moizz\Downloads\Tools"
features, labels, label_map = process_directory(data_directory)

# Define target oversampling counts for specific classes
target_counts = {list(label_map.keys())[0]: 5000, list(label_map.keys())[1]: 5000, list(label_map.keys())[2]: 5000, list(label_map.keys())[4]: 5000, list(label_map.keys())[5]: 5000, list(label_map.keys())[7]: 5000}  # Adjust these based on actual class names if necessary  # Use actual class names from your directory

# Oversample specific classes
features, labels = oversample_classes(features, labels, label_map, target_counts)

print("Features shape:", features.shape)
print("Labels:", labels)
print("Label map:", label_map)