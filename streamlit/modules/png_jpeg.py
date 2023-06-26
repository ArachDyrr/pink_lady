import os
from PIL import Image

input_folder = './streamlit/batch1'
output_folder = './streamlit/batch1'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over the files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # Open the PNG image
        png_path = os.path.join(input_folder, filename)
        png_image = Image.open(png_path)

        # Convert the image to RGB mode if it's not already in RGB
        if png_image.mode != 'RGB':
            png_image = png_image.convert('RGB')

        # Create the output JPEG filename by replacing the extension
        jpeg_filename = os.path.splitext(filename)[0] + '.jpeg'
        jpeg_path = os.path.join(output_folder, jpeg_filename)

        # Save the image as JPEG
        png_image.save(jpeg_path, 'JPEG')

        print(f"Converted {filename} to {jpeg_filename}")

print("Conversion completed.")
