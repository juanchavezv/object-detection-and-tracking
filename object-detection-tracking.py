"""
This Code  

python object-detection-tracking.py --video_file dogs_in_patio4.mp4  --dog_selection_picture arlo-identifier-real5.jpg --frame_resize_percentage 100
"""

import cv2
import argparse
import numpy as np

cnt_left_2_right = 0
cnt_right_2_left = 0
pos_tm1 = -1

#x_min,y_min,x_max,y_max=38400,2160,0,0

def parse_user_input():
    """
    Parse the command-line arguments provided by the user.

    Returns:
        tuple[str, str]: A tuple containing the path to the object image and the input video.
    """
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
    
    args = parser.parse_args()
    return args

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

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(masked_frame, None)
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

def extract_good_keypoints(keypoints, matches, matches_mask):
    """

    """
    good_keypoints = []

    for i, (m, n) in enumerate(matches):
        if matches_mask[i][0]:
            good_keypoints.append(keypoints[m.trainIdx])

    return good_keypoints

def draw_characteristics( frame, keypoints):
    """
    Calculate the center point of all keypoints.

    Args:
        keypoints (list of cv2.KeyPoint): List of keypoints.

    Returns:
        tuple: x, y coordinates of the center point.
    """
    if not keypoints:
        return (0, 0)
    
    frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    # Extracting the x and y coordinates of the keypoints
    coordinates = np.array([(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints])
    
    # Calculating the mean of the coordinates
    center = np.mean(coordinates, axis=0)
    center_int = (int(center[0]), int(center[1]))
    frame =  cv2.circle(frame, center_int, 5, (0, 0, 255), -1)

    # Find the leftmost point (minimum x-coordinate)
    leftmost = coordinates[np.argmin(coordinates[:, 0])]

    # Find the rightmost point (maximum x-coordinate)
    rightmost = coordinates[np.argmax(coordinates[:, 0])]

    # Find the uppermost point (minimum y-coordinate)
    uppermost = coordinates[np.argmin(coordinates[:, 1])]

    # Find the lowermost point (maximum y-coordinate)
    lowermost = coordinates[np.argmax(coordinates[:, 1])]

    # Drawing the rectangle
    frame = cv2.rectangle(frame, (leftmost[0], uppermost[1]), (rightmost[0], lowermost[1]), (0, 0, 255), 2)  
    return center_int, frame

def frame_position(width, center):
    global cnt_left_2_right, cnt_right_2_left, pos_tm1
    pos_t0 = center[0]
    if pos_tm1 != -1:  # Check if pos_tm1 has been initialized
        if pos_t0 >= int(width / 2) and pos_tm1 < int(width / 2):
            cnt_left_2_right += 1
        elif pos_t0 <= int(width / 2) and pos_tm1 > int(width / 2):
            cnt_right_2_left += 1
    pos_tm1 = pos_t0  # Update pos_tm1 regardless of the condition

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

        height, width = dogs_frame["image"].shape[:2]

        # Extract img features
        dog["features"]["kp"], dog["features"]["descriptors"] = feature_extraction(dog["image"])
        dogs_frame["features"]["kp"], dogs_frame["features"]["descriptors"] = feature_extraction(dogs_frame["image"])

        # Match features
        matches, matches_mask = match_features(dog["features"]["descriptors"], dogs_frame["features"]["descriptors"])

        dogs_frame["features"]["kp"] = extract_good_keypoints(dogs_frame["features"]["kp"], matches, matches_mask)

        center_point, dogs_frame["image"] = draw_characteristics(dogs_frame["image"],dogs_frame["features"]["kp"])
        
        cv2.line(dogs_frame["image"],(int(width/2),0),(int(width/2),height),(255, 0, 0),2)
        frame_position(width, center_point)

        txt_left_2_right = f"Dog crossing the reference line from left to right: {cnt_left_2_right}"
        txt_right_2_left = f"Dog crossing the reference line from right to left: {cnt_right_2_left}"
        cv2.putText(dogs_frame["image"], txt_left_2_right, (15, height - 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)
        cv2.putText(dogs_frame["image"], txt_left_2_right, (15, height - 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(dogs_frame["image"], txt_right_2_left, (15, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)
        cv2.putText(dogs_frame["image"], txt_right_2_left, (15, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.namedWindow("video with matches", cv2.WINDOW_NORMAL)
        cv2.imshow("video with matches",dogs_frame["image"])

        # The program finishes if the key 'q' is pressed
        key = cv2.waitKey(5)
        if key == ord('q') or key == 27:
            print("Programm finished!")
            break
 
if __name__ == "__main__":
    pipeline()