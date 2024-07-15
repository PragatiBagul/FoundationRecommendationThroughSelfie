import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
import face_recognition
from colormap import rgb2hex, hex2rgb
from matplotlib.patches import Rectangle
import webbrowser
import json

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

# Calculate Euclidean distance between two RGB values
def euclidean_distance(rgb1, rgb2):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))

def crop_face(image_file):
    image_content = image_file

    # face_locations is a tuple that contains 4 pixels corresponding to the corners of the face
    # in the form of top, right, bottom, left
    #     image = face_recognition.load_image_file(image_file)
    face_locations = face_recognition.face_locations(np.array(image_file))

    # if there is no face in the image, return none
    # else, use PIL library to crop your image according to the 4 pixels specified in face_lcoations
    if len(face_locations) > 1:
        return None
    # if there are multiple faces in the image, return none, so that we can process this
    # as an invalid image
    elif len(face_locations) == 0:
        return None
    else:
        left = face_locations[0][3]
        top = face_locations[0][0]
        right = face_locations[0][1]
        bottom = face_locations[0][2]

        cropped_face = image_content.crop((left, top, right, bottom))

        return cropped_face

uploaded_file = st.file_uploader("Choose a image file", type="jpg")

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    pil_image = crop_face(image_pil)
    open_cv_image = np.array(pil_image)

    # Convert RGB to BGR
    image_original = open_cv_image[:, :, ::-1].copy()
    image = image_original

    # Adjust the brightness and contrast
    # Adjusts the brightness by adding 10 to each pixel value
    brightness = 27
    # Adjusts the contrast by scaling the pixel values by 2.3
    contrast = 2.3
    image = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness)

    # Check if the image is loaded properly
    if image is None:
        print("Error: Could not load image.")
    else:
        # Convert the image from BGR to RGB (cv2 uses BGR by default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the image using matplotlib
        plt.imshow(image_rgb)
        plt.axis('off')  # Hide the axis
        plt.show()

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Display the HSV image
        plt.subplot(1, 2, 2)
        plt.imshow(image_hsv)
        plt.title('HSV Image')
        plt.axis('off')

        plt.show()

        # Define the lower and upper thresholds for blue color in HSV
        lower_blue = np.array([0, 48, 80])
        upper_blue = np.array([20, 255, 255])

        # Create a mask with the specified thresholds
        mask = cv2.inRange(image_hsv, lower_blue, upper_blue)

        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=mask)

        # Convert images from BGR to RGB for displaying with matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # Display the original image, mask, and result image
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(result_rgb)
        plt.title('Result Image')
        plt.axis('off')

        plt.show()

        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image from BGR to HSV
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper thresholds for skin tone in HSV
        lower_skin = np.array([0, 30, 60])
        upper_skin = np.array([20, 150, 255])

        # Create a mask with the specified thresholds
        mask = cv2.inRange(image_hsv, lower_skin, upper_skin)

        # Apply the mask to the original image
        skin = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

        # Calculate the mean RGB values for the skin tone
        skin_masked = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
        mean_rgb = cv2.mean(skin_masked, mask=mask)[:3]  # Ignore the alpha channel

        # Display the original image, mask, and skin tone image
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(skin)
        plt.title('Skin Tone Image')
        plt.axis('off')

        plt.show()

        # Print the average RGB skin tone value
        print("Average RGB skin tone value:", mean_rgb)

        hex_code = rgb2hex(int(mean_rgb[0]),int(mean_rgb[1]),int(mean_rgb[2]))

        # Define the RGB color values (normalized to range [0, 1])
        red = mean_rgb[0]/255
        green = mean_rgb[1]/255
        blue = mean_rgb[2]/255

        # Create a 1x1 figure
        fig, ax = plt.subplots(figsize=(3, 3))

        # Plot a color patch with the specified RGB values
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=(red, green, blue)))

        # Set plot limits and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        # Show the plot
        plt.title(f'RGB: ({red}, {green}, {blue})')
        plt.show()

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        image_rgb = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
        # Display the image in the first subplot
        axes[0].imshow(image_rgb)
        axes[0].axis('off')  # Hide the axis
        # Create a color patch plot in the second subplot
        axes[1].axis('off')  # Hide the axis for the second subplot

        # Plot a color patch with the specified RGB values
        axes[1].add_patch(plt.Rectangle((0, 0), 1, 1, color=(red, green, blue)))

        # Set plot limits and labels
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        # Show the plot
        plt.title(f'RGB: ({red}, {green}, {blue})')
        st.pyplot(plt.gcf())

        st.text(hex_code)

        df = pd.read_csv('./allShades.csv')

        # Convert sample hex code to RGB
        sample_rgb = hex_to_rgb(hex_code)

        # Add a column with the Euclidean distance to the sample RGB
        df['distance'] = df['hex'].apply(lambda x: euclidean_distance(hex_to_rgb(x), sample_rgb))

        # Sort the DataFrame by distance and select the top 10 closest hex codes
        top_10_closest = df.nsmallest(10, 'distance')

        # Get the indices of the top 10 closest hex codes
        indices_top_10_closest = top_10_closest.index.tolist()

        print(indices_top_10_closest)


        # List of indices to display
        indices = indices_top_10_closest

        # Create a figure and set of subplots with 5x2 grid
        fig, axes = plt.subplots(5, 2, figsize=(10, 15))

        # Flatten the axes array for easier iteration
        axes = axes.flatten()

        # Plot each hex color and add annotations
        for i, (ax, idx) in enumerate(zip(axes, indices)):
            color = df.loc[idx, 'hex']
            brand_name = df.loc[idx, 'brand']
            product_name = df.loc[idx, 'product']
            product_url = df.loc[idx, 'url']

            # Add a colored square
            ax.add_patch(Rectangle((0.1, 0.2), 0.8, 0.6, color=color, transform=ax.transAxes, clip_on=False))

            ax.axis('off')  # Hide the axes
            ax.text(0.5, 0.8, brand_name, ha='center', va='center', fontsize=12, color='black', transform=ax.transAxes)
            ax.text(0.5, 0.6, product_name, ha='center', va='center', fontsize=10, color='black', transform=ax.transAxes)

            # Adding a clickable link
            ax.text(0.5, 0.4, 'Link', ha='center', va='center', fontsize=10, color='blue', transform=ax.transAxes,
                    url=product_url)

        # Hide any unused subplots
        for j in range(len(indices), len(axes)):
            axes[j].axis('off')

        # Function to open URLs
        def on_click(event):
            for ax in event.canvas.figure.axes:
                for text in ax.texts:
                    if text.contains(event)[0]:
                        url = text.get_url()
                        if url:
                            webbrowser.open(url)

        # Enable interactive mode for clickable link
        fig.canvas.mpl_connect('button_press_event', on_click)

        # Adjust layout and display the plot
        plt.tight_layout()
        st.pyplot(plt.gcf())

