#!/usr/bin/env python3
"""
Batch-sort images into folders based on RF-DETR object detections.

Requires:
    pip install rfdetr inference-models[trt10] tqdm

Pass the TensorRT .trt engine to --model.
Example:
    python processor_rfdeter.py -i ~/Pictures -o ~/Pictures_Sorted \
        -m ~/rfdetr_trt_model/rfdetr-large_fp16.trt --conf 0.40
--mode resize (baseline), --mode tile --tile-threshold 0 (all tiled), --mode both --max-edge 4096
"""

import os
import sys
import shutil
import argparse
import cv2
import supervision as sv
import csv
import time
from datetime import datetime
from PIL import Image, UnidentifiedImageError
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from rfdetr import RFDETRInference
MODEL_PATH = "~/rfdetr_trt_model/rfdetr-large_fp16.trt"
#MODEL_PATH = "~/rfdetr_trt_model/rfdetr-medium_fp16.trt"



# --- CONFIGURATION ---


DEFAULT_SOURCE = "~/Pictures"
#DEFAULT_SOURCE = "~/bus"
DEFAULT_OUTPUT = "~/Picturez"
DEFAULT_CONF = 0.40

# If multiple classes are detected, the first matching class here wins.
PRIORITY = [
    "person", "car", "bus", "truck",
    "motorcycle", "bicycle", "train", "boat",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="dsbatch",
        description="Batch-sort images using RF-DETR with TensorRT.",
    )
    parser.add_argument(
        "--input", "-i", default=DEFAULT_SOURCE,
        help=f"source directory to scan (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help=f"destination root for sorted folders (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--conf", "-c", type=float, default=DEFAULT_CONF,
        help=f"detection confidence threshold (default: {DEFAULT_CONF})",
    )
    parser.add_argument(
        "--model", "-m", default=MODEL_PATH,
        help=f"path to the RF-DETR model (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print routing decisions without copying files",
    )
    parser.add_argument(
        "--mode", choices=["resize", "tile", "both"], default="both",
        help="resize: single-pass (default); tile: native-resolution tiles; "
         "both: downscale to --max-edge then tile",
    )
    parser.add_argument("--tile-size", type=int, default=704,
        help="tile dimensions in pixels (default: 704, matches RF-DETR Large)")
    parser.add_argument("--tile-overlap", type=int, default=100,
        help="tile overlap in pixels (default: 100)")
    parser.add_argument("--max-edge", type=int, default=4096,
        help="max long edge before tiling in 'both' mode (default: 4096)")
    parser.add_argument("--tile-threshold", type=int, default=1000,
        help="skip tiling below this long edge (default: 1000)")
    return parser.parse_args()


def pick_primary(detected):
    """Choose the highest-priority detected class."""
    for class_name in PRIORITY:
        if class_name in detected:
            return class_name
    return sorted(detected)[0]

def log_failure(log_path, stage, img_path, reason):
    p = Path(log_path)
    new = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if new:
            w.writerow(["timestamp", "stage", "path", "reason"])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            stage,
            str(img_path),
            reason,
        ])

def is_within(path, parent):
    """True when path is parent itself or is below parent."""
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def make_destination(img_path, source_dir, output_root, target_folder):
    """Preserve subdirectories and disambiguate same-named files from different folders."""
    rel_path = os.path.relpath(img_path, source_dir)
    base_name = os.path.basename(img_path)
    relative_parent = os.path.dirname(rel_path)

    if relative_parent:
        dir_prefix = relative_parent.replace(os.sep, "__")
        stem, extension = os.path.splitext(base_name)
        new_filename = f"{stem}__{dir_prefix}{extension}"
    else:
        new_filename = base_name

    return os.path.join(target_folder, relative_parent, new_filename)


Image.MAX_IMAGE_PIXELS = 100_000_000  # ~100 MP; PIL raises above this

def validate_image(path, max_bytes=50 * 1024 * 1024, max_pixels=100_000_000):
    """Cheap pre-flight check. Returns (ok: bool, reason: str)."""
    p = Path(path)
    try:
        if not p.is_file():
            return False, "not a regular file"
        size = p.stat().st_size
        if size == 0:
            return False, "empty file"
        if size > max_bytes:
            return False, f"file too large ({size / 1e6:.1f} MB)"
        with Image.open(p) as im:
            w, h = im.size
            if w <= 0 or h <= 0:
                return False, f"bad dimensions {w}x{h}"
            if w * h > max_pixels:
                return False, f"decompression bomb ({w}x{h})"
            im.verify()
        return True, "ok"
    except Image.DecompressionBombError as e:
        return False, f"decompression bomb: {e}"
    except UnidentifiedImageError:
        return False, "not a recognisable image"
    except (OSError, ValueError, SyntaxError) as e:
        return False, f"unreadable: {e}"


def detect_with_mode(predictor, image_path, conf_thresh, args):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("cv2.imread returned None")

    long_edge = max(img.shape[:2])

    # NOTE: This function must be indented inside detect_with_mode
    def predict_rgb(arr):
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return predictor.predict(
            rgb,
            threshold=conf_thresh,
            include_source_image=False
        )

    if args.mode == "resize" or long_edge < args.tile_threshold:
        result = predict_rgb(img)
    else:
        if args.mode == "both" and long_edge > args.max_edge:
            scale = args.max_edge / long_edge
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        slicer = sv.InferenceSlicer(
            callback=predict_rgb,
            slice_wh=(args.tile_size, args.tile_size),
            overlap_wh=(args.tile_overlap, args.tile_overlap),
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
        )
        result = slicer(img)

    # Extract class names meeting the confidence threshold
    detected = set()
    if result is not None and result.data is not None and "class_name" in result.data:
        for name, conf in zip(result.data["class_name"], result.confidence):
            if float(conf) >= conf_thresh:
                detected.add(str(name))
    return detected



def main():
    args = parse_args()
    source_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(args.input)))
    output_root = os.path.abspath(os.path.expandvars(os.path.expanduser(args.output)))
    model_path = os.path.abspath(os.path.expandvars(os.path.expanduser(args.model)))

    if not 0.0 <= args.conf <= 1.0:
        print("Error: --conf must be between 0.0 and 1.0.", file=sys.stderr)
        return 2
    if not os.path.isdir(source_dir):
        print(f"Error: source directory does not exist: {source_dir}", file=sys.stderr)
        return 1
    if not os.path.isfile(model_path):
        print(f"Error: TensorRT engine not found: {model_path}", file=sys.stderr)
        return 1

    print(f"--> Loading TensorRT engine: {model_path}")
    try:
        predictor = RFDETRInference(model_path)
    except Exception as exc:
        print(f"Error: could not load TensorRT engine: {exc}", file=sys.stderr)
        return 1
    print(f"--> Scanning images in {source_dir}...")
    all_image_paths = []

    for root, dirs, files in os.walk(source_dir):
        dirs[:] = [
            directory for directory in dirs
            if not is_within(os.path.join(root, directory), output_root)
        ]
        for filename in files:
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                all_image_paths.append(os.path.join(root, filename))

    all_image_paths.sort()
    print(f"--> Found {len(all_image_paths)} images to analyze and sort.")

    if not all_image_paths:
        return 0
    class_counts = Counter()
    last_update = time.time()
    if not args.dry_run:
        os.makedirs(output_root, exist_ok=True)
        os.makedirs(os.path.join(output_root, "empty_or_unknown"), exist_ok=True)
    LOG_PATH = os.path.join(output_root, "failures.csv")

    pbar = tqdm(all_image_paths, desc="Processing Images", unit="img")
    for img_path in pbar:
        filename = os.path.basename(img_path)

        ok, why = validate_image(img_path)
        if not ok:
            print(f"\n[skip] {img_path}: {why}", file=sys.stderr)
            log_failure(LOG_PATH, "validate", img_path, why)
            continue

        try:
            detected_objects = detect_with_mode(
                predictor=predictor,
                image_path=img_path,
                conf_thresh=args.conf,
                args=args,
            )
        except Exception as exc:
            print(f"\n[skip] {filename}: inference failed: {exc}", file=sys.stderr)
            log_failure(LOG_PATH, "inference", img_path, str(exc))
            continue

        if detected_objects:
            primary_class = pick_primary(detected_objects)
            target_folder = os.path.join(output_root, primary_class)
            class_counts[primary_class] += 1
        else:
            target_folder = os.path.join(output_root, "empty_or_unknown")
            class_counts["empty_or_unknown"] += 1

        dest_path = make_destination(img_path, source_dir, output_root, target_folder)

        if not args.dry_run:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            try:
                shutil.copy2(img_path, dest_path)
            except Exception as exc:
                print(f"[copy-fail] {filename}: {exc}", file=sys.stderr)
                log_failure(LOG_PATH, "copy", img_path, str(exc))

        # Update the progress bar text (throttled to every 0.5s)
        if time.time() - last_update > 0.5:
            top_classes = ", ".join(
                f"{name}:{count}" for name, count in class_counts.most_common(3)
            )
            pbar.set_postfix_str(top_classes)
            last_update = time.time()

    print("\n-- Class summary --")
    for name, count in class_counts.most_common():
        print(f"  {name:20s} {count}")

    print(f"\nAll done! Your files are in: {output_root}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
