import argparse
import csv
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_size(s: str) -> Tuple[int, int]:
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise ValueError("size must be like 640x640 or 1024x385")
    return int(parts[0]), int(parts[1])


def add_noise(img, noise_type: str, noise_level: float):
    if noise_type == "none" or noise_level <= 0:
        return img
    if noise_type == "gaussian":
        sigma = max(0.0, noise_level)
        gauss = cv2.randn(img.copy(), 0, sigma)
        noisy = cv2.add(img, gauss)
        return noisy
    if noise_type == "saltpepper":
        amount = min(max(noise_level, 0.0), 1.0)
        out = img.copy()
        num = int(amount * img.shape[0] * img.shape[1])
        # salt
        coords = (
            cv2.randu(img[..., :1], 0, 1),
        )  # dummy to seed
        for _ in range(num // 2):
            x = int(cv2.randu(img[..., :1], 0, 1)[0][0][0] * img.shape[1])
            y = int(cv2.randu(img[..., :1], 0, 1)[0][0][0] * img.shape[0])
            out[y, x] = 255
        for _ in range(num // 2):
            x = int(cv2.randu(img[..., :1], 0, 1)[0][0][0] * img.shape[1])
            y = int(cv2.randu(img[..., :1], 0, 1)[0][0][0] * img.shape[0])
            out[y, x] = 0
        return out
    return img


def compress_image(img, compress_type: str, quality: int):
    if compress_type == "none":
        return img
    if compress_type == "jpeg":
        quality = max(1, min(quality, 100))
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return img
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return dec
    if compress_type == "png":
        level = max(0, min(quality // 10, 9))
        ok, enc = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, level])
        if not ok:
            return img
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return dec
    return img


def resize_adaptive(img, target_w: int, target_h: int, keep_aspect: bool):
    if not keep_aspect:
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 255, dtype=img.dtype)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def process_sequence(seq_dir: Path, out_root: Path, size: Tuple[int, int], keep_aspect: bool,
                     noise_type: str, noise_level: float, compress_type: str, quality: int,
                     label_root: Path | None):
    images_dir = seq_dir / "images"
    manifest_path = out_root / seq_dir.name / "manifest.csv"
    out_images_dir = out_root / seq_dir.name / "images"
    out_labels_dir = out_root / seq_dir.name / "labels"
    ensure_dir(out_images_dir)
    ensure_dir(out_labels_dir)
    ensure_dir(manifest_path.parent)
    target_w, target_h = size

    rows = []
    for fname in sorted(images_dir.glob("*.png")):
        img = cv2.imread(str(fname))
        if img is None:
            continue
        img_p = add_noise(img, noise_type, noise_level)
        img_c = compress_image(img_p, compress_type, quality)
        img_r = resize_adaptive(img_c, target_w, target_h, keep_aspect)
        out_name = fname.name
        cv2.imwrite(str(out_images_dir / out_name), img_r)

        label_path = None
        if label_root:
            # Try YOLO txt or semantic PNG by mirroring structure
            yolo = label_root / seq_dir.name / (fname.stem + ".txt")
            sem = label_root / seq_dir.name / (fname.stem + ".png")
            if yolo.exists():
                label_path = yolo
                # copy as-is for traceability; resizing handled in downstream if needed
                try:
                    with open(yolo, "rb") as fsrc, open(out_labels_dir / yolo.name, "wb") as fdst:
                        fdst.write(fsrc.read())
                except Exception:
                    pass
            elif sem.exists():
                label_path = sem
                try:
                    mask = cv2.imread(str(sem), cv2.IMREAD_UNCHANGED)
                    if mask is not None:
                        mask_r = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                        cv2.imwrite(str(out_labels_dir / sem.name), mask_r)
                except Exception:
                    pass

        rows.append({
            "seq": seq_dir.name,
            "fname": fname.name,
            "out_path": str(out_images_dir / out_name),
            "label_path": str(out_labels_dir / Path(label_path).name) if label_path else "",
            "noise_type": noise_type,
            "noise_level": noise_level,
            "compress_type": compress_type,
            "quality": quality,
            "keep_aspect": int(keep_aspect),
            "target_w": target_w,
            "target_h": target_h,
        })

    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "seq","fname","out_path","label_path","noise_type","noise_level","compress_type","quality","keep_aspect","target_w","target_h"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser("Preprocess images with traceability manifests")
    p.add_argument("--rgb_root", required=True, help="Root of extracted RGB sequences (contains <SEQ>/images)")
    p.add_argument("--out_root", required=True, help="Output root for preprocessed images and manifests")
    p.add_argument("--size", required=True, help="Target size WxH, e.g. 640x640 or 1024x385")
    p.add_argument("--keep_aspect", action="store_true", help="Pad to keep aspect ratio (letterbox)")
    p.add_argument("--noise_type", default="none", choices=["none","gaussian","saltpepper"], help="Noise type")
    p.add_argument("--noise_level", type=float, default=0.0, help="Noise intensity (sigma or fraction)")
    p.add_argument("--compress_type", default="none", choices=["none","jpeg","png"], help="Compression type")
    p.add_argument("--quality", type=int, default=90, help="Compression quality (JPEG 1-100; PNG 0-9 via buckets)")
    p.add_argument("--label_root", default="", help="Optional label root to mirror (YOLO txt or semantic PNG)")

    a = p.parse_args()
    size = parse_size(a.size)
    rgb_root = Path(a.rgb_root)
    out_root = Path(a.out_root)
    label_root = Path(a.label_root) if a.label_root else None

    for seq in sorted(rgb_root.glob("*")):
        if not (seq / "images").exists():
            continue
        process_sequence(seq, out_root, size, a.keep_aspect, a.noise_type, a.noise_level, a.compress_type, a.quality, label_root)


if __name__ == "__main__":
    main()
