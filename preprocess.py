import cv2
import numpy as np
import argparse
import os


def preprocess_sketch(
    input_path, output_path, size=512, denoise=True, adaptive_thresh=False, pad=True
):
    """
    Load a sketch image and prepare it for ControlNet conditioning.

    Args:
        input_path: path to raw sketch image
        output_path: path to save processed sketch
        size: target square size (ControlNet default 512)
        denoise: apply denoising (useful for hand-drawn/scanned sketches)
        adaptive_thresh: apply adaptive thresholding to clean up lines
    """
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {input_path}")

    if denoise:
        img = cv2.fastNlMeansDenoising(img, h=10)

    if adaptive_thresh:
        img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=7,
        )

    if pad:
        # Resize with aspect-ratio-preserving pad to square
        h, w = img.shape
        scale = size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        y_off = (size - new_h) // 2
        x_off = (size - new_w) // 2
        top, bottom = y_off, size - new_h - y_off
        left, right = x_off, size - new_w - x_off
        canvas = cv2.copyMakeBorder(
            resized, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE
        )
    else:
        # Direct resize, no padding (avoids edge artifacts from padding)
        canvas = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    # Convert back to 3-channel for ControlNet input
    output_img = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Processed Sketch", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(output_path, output_img)
    print(f"Saved processed sketch: {output_path}")

    return output_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input sketch")
    parser.add_argument("--output", required=True, help="Path to save processed sketch")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--adaptive_thresh", action="store_true", help="Apply adaptive thresholding"
    )
    parser.add_argument("--denoise", action="store_true", help="Apply denoising")
    parser.add_argument(
        "--no_pad",
        action="store_true",
        help="Direct resize instead of aspect-preserving pad",
    )
    args = parser.parse_args()

    preprocess_sketch(
        args.input,
        args.output,
        size=args.size,
        denoise=args.denoise,
        adaptive_thresh=args.adaptive_thresh,
        pad=not args.no_pad,
    )
