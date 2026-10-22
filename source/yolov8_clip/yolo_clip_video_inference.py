#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO + CLIP Combined Video Inference Script

This script runs inference on a video file using YOLO for detection and CLIP for re-ranking/classification.
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

try:
    import clip
except ImportError:
    raise ImportError("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")

def crop_with_context(img: np.ndarray, box: tuple, expand: float = 1.8, min_side: int = 64) -> np.ndarray:
    """Crop a region from the image with context padding."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
    bw, bh = (x2-x1), (y2-y1)
    side = max(bw, bh)
    side = max(side*expand, min_side)
    nx1 = int(round(cx - side/2))
    ny1 = int(round(cy - side/2))
    nx2 = int(round(cx + side/2))
    ny2 = int(round(cy + side/2))
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(w-1, nx2)
    ny2 = min(h-1, ny2)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return img[ny1:ny2, nx1:nx2, :]

def run_yolo_clip_video_inference(video_path, yolo_model_path, output_path, custom_classes=None, conf_threshold=0.5, clip_conf_threshold=0.1):
    """
    Run YOLO detection + CLIP re-ranking on video.

    Args:
        video_path (str): Path to input video file
        yolo_model_path (str): Path to YOLO model weights
        output_path (str): Path for output video with detections
        custom_classes (list): Custom class names for CLIP to focus on
        conf_threshold (float): Confidence threshold for YOLO detection
        clip_conf_threshold (float): Confidence threshold for CLIP re-ranking
    """
    # Default classes if none provided
    if custom_classes is None:
        custom_classes = [
            "tennis ball", "tennis racket", "person playing tennis", 
            "tennis court", "player", "sports equipment", "ball"
        ]
    
    # Check if video and model files exist
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not os.path.exists(yolo_model_path):
        raise FileNotFoundError(f"YOLO model file not found: {yolo_model_path}")
    
    # Load YOLO model
    print(f"Loading YOLO model: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)
    
    # Load CLIP model
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device=device)
    
    # Encode the class names for CLIP
    text_descriptions = [f"a photo of {c}" for c in custom_classes]
    text_tokens = clip.tokenize(text_descriptions).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")
    print(f"Detecting with YOLO and re-ranking with CLIP for classes: {custom_classes}")

    # Define output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    print("Starting YOLO + CLIP video inference...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Run YOLO detection
        results = yolo_model(frame, conf=conf_threshold, verbose=False)
        
        # Process detections and apply CLIP re-ranking
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                scores = result.boxes.conf.cpu().numpy()  # Get confidence scores
                classes = result.boxes.cls.cpu().numpy()  # Get class IDs
                
                # For each detection, crop the region and run CLIP classification
                for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Crop the region from the frame
                    crop_img = crop_with_context(frame, (x1, y1, x2, y2))
                    if crop_img is not None:
                        # Convert crop to PIL and preprocess for CLIP
                        pil_crop = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                        image_input = clip_preprocess(pil_crop).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            # Encode the image crop
                            image_features = clip_model.encode_image(image_input)
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            
                            # Calculate similarity with text descriptions
                            logits_per_image = (image_features @ text_features.T).softmax(dim=-1)
                            clip_probs = logits_per_image.cpu().numpy()[0]
                        
                        # Find the class with highest CLIP probability
                        clip_max_idx = np.argmax(clip_probs)
                        clip_max_prob = clip_probs[clip_max_idx]
                        
                        # Draw bounding box and labels if CLIP confidence is above threshold
                        if clip_max_prob > clip_conf_threshold:
                            # Draw the bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw CLIP-based label
                            clip_label = f"{custom_classes[clip_max_idx]}: {clip_max_prob:.2f}"
                            cv2.putText(frame, clip_label, (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # Optionally, also show the original YOLO detection
                            yolo_label = f"YOLO: {score:.2f}"
                            cv2.putText(frame, yolo_label, (x1, y1 - 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Write the annotated frame to output video
        out.write(frame)

        # Print progress
        if frame_count % 30 == 0:  # Print every 30 frames
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")

    # Release resources
    cap.release()
    out.release()

    print(f"YOLO + CLIP inference completed! Output saved to: {output_path}")
    print(f"Processed {frame_count} frames total")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="YOLO + CLIP Combined Video Inference")
    parser.add_argument("--video", type=str, default="/home/peter/PycharmProjects/Analogical-Reasoning-main/Analogical/tennis.mp4",
                        help="Path to input video file")
    parser.add_argument("--model", type=str, default="/home/peter/PycharmProjects/Analogical-Reasoning-main/Analogical/runs/detect/train81/weights/best.pt",
                        help="Path to YOLO model weights")
    parser.add_argument("--output", type=str, default="yolo_clip_tennis_output.mp4",
                        help="Path for output video with detections")
    parser.add_argument("--classes", nargs="+", default=[
        "tennis ball", "tennis racket", "person playing tennis", 
        "tennis court", "player", "sports equipment", "ball"
    ], help="Classes to focus CLIP on")
    parser.add_argument("--yolo-conf", type=float, default=0.5,
                        help="YOLO confidence threshold for detection")
    parser.add_argument("--clip-conf", type=float, default=0.1,
                        help="CLIP confidence threshold for re-ranking")

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run YOLO + CLIP video inference
    run_yolo_clip_video_inference(
        video_path=args.video,
        yolo_model_path=args.model,
        output_path=args.output,
        custom_classes=args.classes,
        conf_threshold=args.yolo_conf,
        clip_conf_threshold=args.clip_conf
    )


if __name__ == "__main__":
    main()