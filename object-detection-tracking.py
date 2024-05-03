"""
This Code 

python object-detection-tracking.py --video_file dogs_in_patio.mp4  --dog_selection_picture caspian.jpg --frame_resize_percentage 30
"""

import cv2
import argparse
import numpy as np

x_min,y_min,x_max,y_max=38400,2160,0,0

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
    width = int(img.shape[1] * 0.2)
    height = int(img.shape[0] * 0.2)
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

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

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
    search_params = dict(checks=150)

    # Initialize FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Use knnMatch to find matches
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply Lowe's ratio test
    matches_mask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
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
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=mask, flags=cv2.DrawMatchesFlags_DEFAULT)
    img = cv2.drawMatchesKnn(img_1["image"], img_1["features"]["kp"], img_2["image"], img_2["features"]["kp"], matches, None, **draw_params)
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
        frame = rescale_frame(frame, user_input.frame_resize_percentage)
        dog_pic = resize_image(cv2.imread(user_input.dog_selection_picture))

        # Load images
        dog["image"] = dog_pic
        dogs_frame["image"] = frame

        # Extract img features
        dog["features"]["kp"], dog["features"]["descriptors"] = feature_extraction(dog["image"])
        dogs_frame["features"]["kp"], dogs_frame["features"]["descriptors"] = feature_extraction(dogs_frame["image"])

        #dog_keypoints = draw_keypoints(dog)
        #dogs_frame_keypoints = draw_keypoints(dogs_frame)

        # Match features
        matches, matches_mask = match_features(dog["features"]["descriptors"], dogs_frame["features"]["descriptors"])

        # Draw matches
        img_with_matches = draw_matches(dog, dogs_frame, matches, matches_mask)
        #cv2.imshow("Dog Keypoints",dog_keypoints)
        #cv2.imshow("Video Keypoints",dogs_frame_keypoints)
        cv2.imshow("video with matches",img_with_matches)

        # The program finishes if the key 'q' is pressed
        key = cv2.waitKey(5)
        if key == ord('q') or key == 27:
            print("Programm finished!")
            break

if __name__ == "__main__":
    pipeline()