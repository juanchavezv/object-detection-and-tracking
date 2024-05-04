"""
This Code 

python object-detection-tracking.py --video_file dogs_in_patio4.mp4  --dog_selection_picture arlo-identifier-real5.jpg --frame_resize_percentage 100
"""

import cv2
import argparse
import numpy as np

#x_min,y_min,x_max,y_max=38400,2160,0,0

def parse_user_input():
    parser = argparse.ArgumentParser(prog='Object Detection and Tracking', 
                                    description='Identify the object asigned by the user', 
                                    epilog='Juan Carlos Chávez Villarreal - 2024')
    parser.add_argument('-vp',
                        '--video_file',
                        type=str,
                        required=True,
                        help="Please type the path to the video that is being analyced")
    
    parser.add_argument('-dog',
                        '--dog_selection_picture',
                        type=str,
                        required=True,
                        help="Please type the the path to the picture with the dog we want to follow")
    
    parser.add_argument('--frame_resize_percentage', 
                        type=int, 
                        help='Rescale the video frames, e.g., 20 if scaled to 20%')
    
    args = parser.parse_args()
    return args    

def rescale_frame(frame, percentage):
    """
    rescales the frame to the size the user indicates

    Input:
        frame: video frame being shown
    
    Return:
        resized frame:
    
    """
    # Resize current frame
    width = int(frame.shape[1] * percentage / 100)
    height = int(frame.shape[0] * percentage / 100)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame

def resize_image(img: np.ndarray) -> np.ndarray:
    """
    Resize the image to a specified scale for better visualization.

    Args:
        img (np.ndarray): Image to resize.

    Returns:
        np.ndarray: Resized image.
    """
    width = int(img.shape[1] * 1)
    height = int(img.shape[0] * 1)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img

def feature_extraction(frame):
    """
    Extract SIFT features from the image.

    Args:
        frame (np.ndarray): Frame data in which to find keypoints.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the keypoints and descriptors for the image.
    """
    # Convert the current frame from BGR to HSV
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    # Create a mask to exclude the white color
    white_mask = cv2.inRange(frame_HSV, lower_white, upper_white)

    # Apply the mask to the original image
    bitwise_AND = cv2.bitwise_and(frame, frame, mask=white_mask)
    
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour based on the contour area
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a blank mask the same size as the original
        largest_contour_mask = np.zeros_like(white_mask)

        # Draw the largest contour onto the mask
        cv2.drawContours(largest_contour_mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)

        # Apply this new mask to the original image to isolate the area of the largest contour
        masked_frame = cv2.bitwise_and(frame, frame, mask=largest_contour_mask)

        draw_object(frame, largest_contour)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(masked_frame, None)
    return keypoints, descriptors

def draw_object(frame, largest_contour):
    # Calculate de position to draw a rectange around the dog's identifier
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Draw a red rectangle around the dog's identifier
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    #cv2.circle(frame,())


def match_features(descriptors_1: np.ndarray, descriptors_2: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Match SIFT features using the FLANN based matcher.

    Args:
        descriptors_1 (np.ndarray): Descriptors from the dog's image.
        descriptors_2 (np.ndarray): Descriptors from the frame.

    Returns:
        tuple[np.ndarray, list]: A tuple containing the matches and the mask for good matches.
    """
    # Define FLANN-based matcher parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # Initialize FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Use knnMatch to find matches
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply Lowe's ratio test
    matches_mask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            matches_mask[i] = [1, 0]
    
    return matches, matches_mask

def draw_keypoints(img):
    """
    Draw keypoints on the image.

    Args:
        img (np.ndarray): Image on which to draw keypoints.

    Returns:
        np.ndarray: Image with keypoints drawn.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray, None)
    img_with_kp = cv2.drawKeypoints(gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img_with_kp

def draw_matches(img_1: np.ndarray, img_2: np.ndarray, matches: np.ndarray, mask: list) -> np.ndarray:
    """
    Draw matches between two images.

    Args:
        img_1 (np.ndarray): First image.
        img_2 (np.ndarray): Second image.
        matches (np.ndarray): Matched features.
        mask (list): Mask for good matches.

    Returns:
        np.ndarray: Image with matches drawn.
    """
    img = img_2["image"]
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=mask, flags=2)
    #img = q(img_1["image"], img_1["features"]["kp"], img_2["image"], img_2["features"]["kp"], matches, output_img, **draw_params)
    img = cv2.drawKeypoints(img_2["image"], img_2["features"]["kp"], None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    return img

def close_windows(cap:cv2.VideoCapture)->None:
    
    # Destroy all visualisation windows
    cv2.destroyAllWindows()

    # Destroy 'VideoCapture' object
    cap.release()

def pipeline():
    # Parse user's input data
    user_input = parse_user_input()

    # Create dictionaries to contain image data
    dog  =  {"image": "", "features": {"kp": "", "descriptors": ""}}
    dogs_frame = {"image": "", "features": {"kp": "", "descriptors": ""}}

    # Create a video capture object
    cap = cv2.VideoCapture(user_input.video_file)

    # Main loop
    while cap.isOpened():

        # Read current frame
        ret, frame = cap.read()

        # Check if the image was correctly captured
        if not ret:
            print("ERROR! - current frame could not be read")
            break

        # Resize current frame
        dog_pic = cv2.imread(user_input.dog_selection_picture)

        # Load images
        dog["image"] = dog_pic
        dogs_frame["image"] = frame

        # Extract img features
        dog["features"]["kp"], dog["features"]["descriptors"] = feature_extraction(dog["image"])
        dogs_frame["features"]["kp"], dogs_frame["features"]["descriptors"] = feature_extraction(dogs_frame["image"])

        # Match features
        matches, matches_mask = match_features(dog["features"]["descriptors"], dogs_frame["features"]["descriptors"])
        
        # Draw matches
        img_with_matches = draw_matches(dog, dogs_frame, matches, matches_mask)
        cv2.namedWindow("video with matches", cv2.WINDOW_NORMAL)
        cv2.imshow("video with matches",img_with_matches)

        # The program finishes if the key 'q' is pressed
        key = cv2.waitKey(5)
        if key == ord('q') or key == 27:
            print("Programm finished!")
            break

if __name__ == "__main__":
    pipeline()