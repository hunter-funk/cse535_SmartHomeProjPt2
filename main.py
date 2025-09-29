# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf
import csv

## import the handfeature extractor class
from handshape_feature_extractor import HandShapeFeatureExtractor
from frameextractor import frameExtractor
extractor = HandShapeFeatureExtractor().get_instance()

training_vects = []
test_vects = []

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

def process_all_training_videos(directory, extractor):
    features = {}
    frames_dir = os.path.join(directory, "train_frames")
    os.makedirs(frames_dir, exist_ok=True)
    i = 0

    for video in os.listdir(directory):
        # only process video files
        if not video.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            print(f"Skipping non-video file: {video}")
            continue

        video_path = os.path.join(directory, video)
        print(f"Processing training video: {video_path}")

        frameExtractor(video_path, frames_dir, i)

        frame_path = os.path.join(frames_dir, f"{i+1:05d}.png")
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Warning: Failed to read extracted frame {frame_path}")
            continue

        feature_vec = extractor.extract_feature(img)
        feature_vec = normalize(feature_vec)
        features[video] = feature_vec
        print(f"For video {video_path}, test frame is {frame_path}")
        i += 1
    print(features)
    return features


# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video


def process_all_testing_videos(directory, extractor):
    features = {}
    frames_dir = os.path.join(directory, "test_frames")
    os.makedirs(frames_dir, exist_ok=True)
    i = 0

    for video in os.listdir(directory):
        # only process video files
        if not video.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            print(f"Skipping non-video file: {video}")
            continue

        video_path = os.path.join(directory, video)
        # print(f"Processing test video: {video_path}")

        frameExtractor(video_path, frames_dir, i)

        frame_path = os.path.join(frames_dir, f"{i+1:05d}.png")
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Warning: Failed to read extracted frame {frame_path}")
            continue

        feature_vec = extractor.extract_feature(img)
        feature_vec = normalize(feature_vec)
        features[video] = feature_vec
        i += 1
    # print(features)
    return features

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
def cosine_similarity(vect1, vect2):
    vect1 = vect1.flatten()
    vect2 = vect2.flatten()
    vect1 = vect1 / (np.linalg.norm(vect1) + 1e-10)
    vect2 = vect2 / (np.linalg.norm(vect2) + 1e-10)
    return np.dot(vect1, vect2)

def get_gesture_name(filename):
    name = os.path.splitext(os.path.basename(filename))[0]  # remove extension
    if name.startswith("H-"):
        name = name[2:]  # remove "H-"

    # Map exact names to formatted labels
    mapping = {
        "DecreaseFanSpeed": "Decrease Fan Speed",
        "FanOff": "FanOff",
        "FanOn": "FanOn",
        "IncreaseFanSpeed": "Increase Fan Speed",
        "LightOff": "LightOff",
        "LightOn": "LightOn",
        "SetThermo": "SetThermo"
    }

    if name in mapping:
        return mapping[name]

    # Otherwise it's a digit 0–9
    return name


def compare_and_save_results(train_features, test_features, output_csv="Results.csv"):
    # Mapping gestures to numeric labels
    gesture_to_label = {
        "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
        "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "Decrease Fan Speed": 10,
        "FanOff": 11,
        "FanOn": 12,
        "Increase Fan Speed": 13,
        "LightOff": 14,
        "LightOn": 15,
        "SetThermo": 16
    }

    output_labels = []

    for test_name, test_vec in test_features.items():

        class_scores = {}
        best_match, best_score = None, -1
        #print(f"\n=== Comparing {test_name} ===")
        for train_name, train_vec in train_features.items():
            gesture_name = get_gesture_name(train_name)
            score = cosine_similarity(test_vec, train_vec)
            # #print(f"  vs {train_name}: {score:.4f}")
            if gesture_name not in class_scores:
                class_scores[gesture_name] = []
            class_scores[gesture_name].append(score)

        avg_scores = {g: np.mean(scores) for g, scores in class_scores.items()}

        best_gesture = max(avg_scores, key=avg_scores.get)       
        label = gesture_to_label.get(best_gesture, -1)  # -1 if something unexpected
        output_labels.append(label)

    # Save single-column CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for label in output_labels:
            writer.writerow([label])

    print(f"Results saved to {output_csv}")



if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    train_dir = os.path.join(base_dir, "traindata")
    test_dir = os.path.join(base_dir, "test")

    train_features = process_all_training_videos(train_dir, extractor)
    test_features = process_all_testing_videos(test_dir, extractor)

    compare_and_save_results(train_features, test_features)