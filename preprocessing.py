import os
import cv2
import numpy as np

# Preprocessing functions

def morphological_operations(original_image, operation='open', kernel_size=4, iterations=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'open':
        return cv2.morphologyEx(original_image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    else:
        raise ValueError("Operation not recognized. Choose from ['open'].")

def apply_mask(original_image, mask):
    return cv2.bitwise_and(original_image, original_image, mask=mask)

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=0.3, tileGridSize=(8,8))
    return clahe.apply(image)

def apply_median_filter(image):
    return cv2.medianBlur(image, 5)

def unsharp_mask(image):
    blurred = cv2.GaussianBlur(image, (2,2), 10.0)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

# New function to preprocess images directly in a specified folder

def preprocess_images_in_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Iterate through each image in the folder
    for img_file in os.listdir(folder_path):
        if img_file.endswith('.png'):
            image_path = os.path.join(folder_path, img_file)
            save_path = os.path.join(output_folder, img_file)
            
            # Read, preprocess, and save each image
            original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(original_image, (1024, 1024))  # Resize to 1024x1024
            opened_image = morphological_operations(resized_image)
            masked_image = apply_mask(resized_image, opened_image)
            clahe_processed = apply_clahe(masked_image)
            median_processed = apply_median_filter(clahe_processed)
            final_image = unsharp_mask(median_processed)
            cv2.imwrite(save_path, final_image)

# Specify the source folder and destination folder for preprocessed images
src_folder = "mPC/Prior"
dest_folder = "mPC/Final"

# Preprocess the images
preprocess_images_in_folder(src_folder, dest_folder)

print("Preprocessing completed for all images in the folder.")
