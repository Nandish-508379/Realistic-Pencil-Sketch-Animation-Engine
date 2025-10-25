"""
=============================================================================
Project ID   : 006/2025
Project Name : Dynamic Pencil Sketch Animation & Color Transition Engine
Author       : M NANDISH
Release Year : 2025
License      : MIT License
Description  : High-performance computer vision pipeline using OpenCV & NumPy
               that transforms raster images into realistic pencil sketches,
               simulates organic human sketching stroke-by-stroke, and smoothly
               transitions back into vibrant color photography with optional
               video rendering.
=============================================================================
"""

import cv2
import numpy as np
import sys
import os
import argparse

# --- DEFAULT CONFIGURATION SETTINGS (PROJECT 006/2025) ---
DEFAULT_IMAGE_NAME = "sample_test.png"

# Speed controls for the drawing animation (smaller delay = faster drawing)
DRAW_FRAMES = 300
DRAW_DELAY_MS = 10

# Speed controls for the color transition
COLOR_STEPS = 100
COLOR_DELAY_MS = 30

# Video Recording Configuration
RECORD_VIDEO = True
VIDEO_OUTPUT_PATH = "sketch_animation.mp4"
# --------------------------------------------------------


def create_sketch(img: np.ndarray) -> np.ndarray:
    """
    Transforms an input BGR image into a pencil sketch using the Color Dodge
    blending technique with inverted Gaussian blur.

    Pipeline:
      1. Grayscale Conversion: Extracts intensity information.
      2. Bitwise Inversion: Inverts luminance values.
      3. Gaussian Smoothing: Computes high-pass frequency response.
      4. Color Dodge Division: Amplifies high-contrast edge boundaries.
    """
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Invert the grayscale image
    inv_gray = cv2.bitwise_not(gray)
    
    # Step 3: Apply Gaussian blur to inverted image (21x21 kernel)
    blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
    
    # Step 4: Blend grayscale with blurred inverse using Color Dodge division
    # Formula: (gray / (255 - blur)) * 256
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    
    # Convert single-channel sketch back to 3-channel BGR for downstream blending
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def animate_sketch(
    img_path: str,
    draw_frames: int = DRAW_FRAMES,
    draw_delay_ms: int = DRAW_DELAY_MS,
    color_steps: int = COLOR_STEPS,
    color_delay_ms: int = COLOR_DELAY_MS,
    record_video: bool = RECORD_VIDEO,
    video_output_path: str = VIDEO_OUTPUT_PATH
) -> None:
    """
    Executes the progressive sketching animation, dynamic stroke sorting with
    Gaussian noise, and alpha-blended color emergence.
    """
    print(f"[Project 006/2025] Loading source image from: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"[Error] Could not read image at: {img_path}")
        print("Please verify the file exists and is a valid image format.")
        return

    # Resize image if it exceeds screen viewing boundaries while preserving aspect ratio
    max_dim = 800
    h, w = img.shape[:2]
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    out_video = None
    if record_video:
        print(f"[Project 006/2025] Video recording enabled: {video_output_path}")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        # Windows Media Foundation backend fallback handling
        try:
            out_video = cv2.VideoWriter(video_output_path, cv2.CAP_MSMF, fourcc, 30.0, (w, h))
            if not out_video.isOpened():
                # Fallback to default backend if MSMF fails
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_video = cv2.VideoWriter(video_output_path, fourcc, 30.0, (w, h))
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(video_output_path, fourcc, 30.0, (w, h))

    def write_frames(frame: np.ndarray, ms_delay: int):
        if out_video and out_video.isOpened():
            if ms_delay >= 100:
                frames_to_write = max(1, int((ms_delay / 1000.0) * 30))
                for _ in range(frames_to_write):
                    out_video.write(frame)
            else:
                out_video.write(frame)

    print("[Project 006/2025] Generating pencil sketch representation...")
    sketch = create_sketch(img)
    sketch_save_path = "sketch_output.png"
    cv2.imwrite(sketch_save_path, sketch)
    print(f"[Project 006/2025] Saved high-resolution sketch to: {sketch_save_path}")

    # Blank white canvas initialized to match image dimensions
    canvas = np.ones_like(img) * 255

    window_name = "Project 006/2025 - Sketch Animation Engine (Press ESC to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, canvas)
    write_frames(canvas, 1000)
    cv2.waitKey(1000)  # Initial pause before sketch strokes begin

    print("[Project 006/2025] Synthesizing human sketching strokes...")
    gray_sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
    
    # Identify non-white sketch pixels (intensity < 240 indicates pencil graphite)
    dark_pixels = np.argwhere(gray_sketch < 240)

    # Stochastic Stroke Ordering:
    # Adding Gaussian noise to Y-coordinates mimics hand tremor and organic stroke sequence
    noise = np.random.normal(0, h // 10, len(dark_pixels))
    sort_keys = dark_pixels[:, 0] + noise
    sorted_indices = np.argsort(sort_keys)
    dark_pixels = dark_pixels[sorted_indices]

    # Calculate batch size for each progressive frame
    batch_size = max(1, len(dark_pixels) // draw_frames)

    # Progressive Drawing Loop
    for i in range(0, len(dark_pixels), batch_size):
        batch = dark_pixels[i : i + batch_size]
        for y, x in batch:
            canvas[y, x] = sketch[y, x]

        cv2.imshow(window_name, canvas)
        write_frames(canvas, draw_delay_ms)
        if cv2.waitKey(draw_delay_ms) & 0xFF == 27:
            print("[Project 006/2025] Animation interrupted by user.")
            break

    # Display completed pencil sketch
    canvas = sketch.copy()
    cv2.imshow(window_name, canvas)
    write_frames(canvas, 1000)
    cv2.waitKey(1000)

    print("[Project 006/2025] Blending colors via linear alpha interpolation...")
    # Smooth Color Emergence Loop
    for i in range(color_steps + 1):
        alpha = i / color_steps
        # I_blended = alpha * I_original + (1 - alpha) * I_sketch
        blended = cv2.addWeighted(img, alpha, sketch, 1.0 - alpha, 0.0)
        cv2.imshow(window_name, blended)
        write_frames(blended, color_delay_ms)
        if cv2.waitKey(color_delay_ms) & 0xFF == 27:
            print("[Project 006/2025] Color transition interrupted by user.")
            break

    # Display final original photograph
    cv2.imshow(window_name, img)
    write_frames(img, 2000)
    print("[Project 006/2025] Animation complete! Press any key to exit.")

    if out_video and out_video.isOpened():
        out_video.release()
        print(f"[Project 006/2025] Video successfully rendered to: {video_output_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resolve_image_path(user_input: str = None) -> str:
    """Finds an appropriate image from arguments or directory defaults."""
    if user_input and os.path.exists(user_input):
        return os.path.abspath(user_input)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Preferred defaults
    candidates = [
        DEFAULT_IMAGE_NAME,
        "sample.png",
        "sample.jpg",
        "sample.jpeg",
        "sample_test.jpg"
    ]
    for candidate in candidates:
        full_path = os.path.join(script_dir, candidate)
        if os.path.exists(full_path):
            return full_path

    # Fallback to any image file found in script directory
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    for fname in os.listdir(script_dir):
        if fname.lower().endswith(valid_exts):
            return os.path.join(script_dir, fname)

    return os.path.join(script_dir, DEFAULT_IMAGE_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project 006/2025: Dynamic Pencil Sketch Animation & Color Transition Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--image",
        type=str,
        default=None,
        help="Path to input image file (optional, defaults to local sample image)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DRAW_FRAMES,
        help="Number of animation frames for sketching"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=COLOR_STEPS,
        help="Number of interpolation steps for color transition"
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable automatic MP4 video export"
    )
    parser.add_argument(
        "-o", "--output-video",
        type=str,
        default=VIDEO_OUTPUT_PATH,
        help="Destination path for output MP4 video"
    )

    args = parser.parse_args()
    target_image = resolve_image_path(args.image)

    animate_sketch(
        img_path=target_image,
        draw_frames=args.frames,
        color_steps=args.steps,
        record_video=not args.no_record,
        video_output_path=args.output_video
    )
