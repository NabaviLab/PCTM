#cropped image
import cv2
import os

def crop_breast_from_image(image_path):
    # Load the image
    img = cv2.imread(image_path, 0)
    
    # Set a threshold value (this might need tuning, or you can try adaptive thresholding)
    ret, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)

    # Find the contours in the image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort the contours by area (largest area first) and keep the largest one; 
    # this is assuming the largest contour will be the breast
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    # Get bounding rectangle for the largest contour
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop the image to this bounding rectangle
    cropped_img = img[y:y+h, x:x+w]

    return cropped_img

input_folder = "part_1"
output_folder = "cropped"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".tif") or filename.endswith(".png") or filename.endswith(".jpg"):
        # Construct the input and output paths
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Crop the breast from the image and save the result
        cropped_img = crop_breast_from_image(input_path)
        cv2.imwrite(output_path, cropped_img)
