import cv2
import numpy as np
import mediapipe as mp
import os

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def resize_image(image_path, size=(256, 256)):
    """ Resize image to a fixed size. """
    image = cv2.imread(image_path)
    image = cv2.resize(image, size)
    return image

def segment_person(image):
    """ Dummy function for person segmentation. (Replace with a proper model) """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return mask

def get_pose_keypoints(image):
    """ Extract keypoints using MediaPipe Pose. """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    if results.pose_landmarks:
        keypoints = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
        return keypoints
    return None

# Example usage
if __name__ == "__main__":
    img_path = "../dataset/persons/person1.jpg"
    image = resize_image(img_path)
    keypoints = get_pose_keypoints(image)

    if keypoints:
        print("Pose Keypoints:", keypoints)
    else:
        print("Pose detection failed!")
