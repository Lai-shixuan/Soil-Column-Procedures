import numpy as np
import cv2


def watershed(image):
    def generate_random_colors(num_colors):
        return np.random.randint(0, 255, size=(num_colors, 3))

    # Step 2: Preprocess the image
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Step 3: Thresholding
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 4: Finding sure background and foreground areas
    # Noise removal (optional)
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=1)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.01*dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # cv2.imshow('unknown Image', unknown)

    # Step 5: Applying the Watershed Algorithm
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply the watershed algorithm
    markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)

    # Step 6: Identify the largest area (background)
    unique, counts = np.unique(markers, return_counts=True)
    background_marker = unique[np.argmax(counts[1:]) + 1]  # Ignore the first count (background)

    # Step 7: Colorize the markers
    # Generate random colors for each marker
    num_markers = np.max(markers)
    colors = generate_random_colors(num_markers + 1)
    # colors[background_marker] = [0, 0, 0]  # Set background color to black
    colors[background_marker] = [255, 255, 255]  # Set background color to black

    # Create an image to display the colors
    output = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

    for marker in range(num_markers + 1):
        # output[markers == marker] = colors[marker]
        if marker != background_marker:
            output[markers == marker] = [0, 0, 0]
        if marker == background_marker:
            output[markers == marker] = [255, 255, 255]

    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    return output

    # # Step 8: Display the Result
    # cv2.imshow('Segmented Image', output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # # Step 9: Save the Result
    # cv2.imwrite('color_segmented_image.jpg', output)
