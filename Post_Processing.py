import cv2
import numpy as np
import os


def adjust_brightness_contrast(image, brightness_adjust=0.0, contrast_adjust=1.0):
  """
  Adjusts brightness and contrast of an image for improved post-processing of bright images.

  Args:
      image: The input image as a NumPy array.
      brightness_adjust: A float value to adjust image brightness (positive for brighter, negative for darker).
      contrast_adjust: A float value to adjust image contrast (higher values for more contrast).

  Returns:
      The brightness and contrast adjusted image as a NumPy array.
  """

  # Convert image to HSV color space for better control over brightness
  hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  # Adjust brightness by modifying the Value (V) channel
  v_channel = hsv_image[:, :, 2].astype(np.float32)  # Extract V channel
  mean_intensity = cv2.mean(image)[0]  # Assuming grayscale for simplicity (consider converting to grayscale if needed)

  # Adaptive brightness adjustment for extremely white images
  if mean_intensity > 230:  # Adjust threshold as needed
    brightness_adjust = - (mean_intensity - 230) / 50.0  # Adjust scaling factor

  v_channel += brightness_adjust  # Adjust brightness
  v_channel = np.clip(v_channel, 0, 255)  # Clip values to valid range (0-255)
  hsv_image[:, :, 2] = v_channel.astype(np.uint8)  # Update V channel

  # Adjust contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Improved parameters
  bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)  # Convert back to BGR
  new_image = clahe.apply(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY))  # CLAHE on grayscale

  # Ensure images have the same size before combining
  new_image = cv2.resize(new_image, dsize=(bgr_image.shape[1], bgr_image.shape[0]), interpolation=cv2.INTER_AREA)

  # Check and convert color spaces if necessary
  if len(bgr_image.shape) == 3 and len(new_image.shape) == 2:
      new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR

  # Combine adjusted brightness and CLAHE-enhanced contrast
  combined_image = cv2.addWeighted(bgr_image, 1.0 - contrast_adjust, new_image, contrast_adjust, 0)

  return combined_image


# Define the image folder path (replace with yours)
image_folder = r'C:\AlliedVision\Python_Files\Chirag_Files\Image_collection'

# Define the desired output folder path (replace with yours)
output_folder = r'C:\AlliedVision\Python_Files\Chirag_Files\Image Output'  # Create this folder if it doesn't exist

# Load images and handle potential errors
try:
  image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.tiff', '.png'))]

except Exception as e:
  print("Error accessing image folder:", e)
  exit()

# Process each image in the folder
for i in range(len(image_paths)):
  #image_path = image_paths[i]  # Use the original image paths
  image_path = os.path.join(image_folder, str(i + 1) + ".tiff")
  #print(image_path)
  try:
    # Load the image
    image = cv2.imread(image_path)
    #print(image_path)
    if image is None:
      print(f"Error: Could not read image from {image_path}")
      continue  # Skip to next image if loading fails

    # Adjust brightness and contrast
    adjusted_image = adjust_brightness_contrast(image, brightness_adjust=-0.2, contrast_adjust=1.2)

    # Define a unique output filename based on the original filename
    output_filename = os.path.join(output_folder, os.path.basename(image_path))

    # Save the adjusted image
    cv2.imwrite(output_filename, adjusted_image)
    print(f"Successfully saved adjusted image: {output_filename}")
  except Exception as e:
    print(f"Error processing image {image_path}:", e)

# No need to display original image since saving is the focus
# cv2.waitKey(0)
# cv2.destroyAllWindows()