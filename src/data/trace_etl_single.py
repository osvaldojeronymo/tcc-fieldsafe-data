import os
import csv
import argparse
from datetime import datetime

import cv2
import numpy as np


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def letterbox(img: np.ndarray, size=(640, 640)):
    h, w = img.shape[:2]
    target_w, target_h = size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas, {
        "orig_w": w, "orig_h": h,
        "new_w": new_w, "new_h": new_h,
        "x_off": x_off, "y_off": y_off,
        "target_w": target_w, "target_h": target_h,
        "scale": scale,
    }


def add_gaussian_noise(img: np.ndarray, sigma: float):
    if sigma <= 0:
        return img.copy(), {"noise_sigma": 0.0}
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy, {"noise_sigma": sigma}


def compress_jpeg(img: np.ndarray, quality: int, out_path: str):
    ok = cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return ok, {"jpeg_quality": int(quality)}


def main():
    ap = argparse.ArgumentParser(description="Traçar ETL & Degradação Controlada para uma imagem")
    ap.add_argument("image", help="Caminho da imagem RGB de entrada")
    ap.add_argument("outdir", help="Diretório de saída para artefatos")
    ap.add_argument("seq", help="Nome da sequência (para rastreabilidade)")
    ap.add_argument("image_id", help="ID lógico da imagem (para nomeação)")
    ap.add_argument("label_path", nargs="?", default="", help="Caminho do rótulo (opcional)")
    ap.add_argument("--size", default="640x640", help="Tamanho alvo, ex.: 640x640")
    ap.add_argument("--noise", type=float, default=5.0, help="Sigma do ruído gaussiano")
    ap.add_argument("--jpeg", type=int, default=85, help="Qualidade JPEG para compressão")
    ap.add_argument("--annotate", action="store_true", help="Adicionar rótulos de estágio na visualização")
    args = ap.parse_args()

    if "x" in args.size:
        tw, th = [int(x) for x in args.size.split("x")]
    else:
        tw, th = 640, 640

    ensure_dir(args.outdir)
    img = cv2.imread(args.image)
    assert img is not None, f"Não foi possível ler {args.image}"

    # ET: resize com letterbox (mantendo aspecto)
    lb_img, lb_info = letterbox(img, (tw, th))
    lb_path = os.path.join(args.outdir, f"{args.image_id}_letterbox.png")
    cv2.imwrite(lb_path, lb_img)

    # Degradação controlada: ruído + compressão
    noisy_img, noise_info = add_gaussian_noise(lb_img, args.noise)
    noisy_path = os.path.join(args.outdir, f"{args.image_id}_noisy.png")
    cv2.imwrite(noisy_path, noisy_img)

    jpeg_path = os.path.join(args.outdir, f"{args.image_id}_compressed.jpg")
    ok, jpeg_info = compress_jpeg(noisy_img, args.jpeg, jpeg_path)
    assert ok, "Falha ao salvar JPEG"

    # Visualização lado a lado (3 estágios): letterbox, noisy, compressed
    comp_bgr = cv2.imread(jpeg_path)
    if comp_bgr is None:
        comp_bgr = noisy_img.copy()
    # Garantir mesmo tamanho (target_w x target_h)
    lb_vis = cv2.resize(lb_img, (lb_info["target_w"], lb_info["target_h"]))
    noisy_vis = cv2.resize(noisy_img, (lb_info["target_w"], lb_info["target_h"]))
    comp_vis = cv2.resize(comp_bgr, (lb_info["target_w"], lb_info["target_h"]))
    if args.annotate:
        def put_label(img, text):
            overlay = img.copy()
            cv2.rectangle(overlay, (0,0), (lb_info["target_w"], 30), (255,255,255), -1)
            img = overlay
            cv2.putText(img, text, (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            return img
        lb_vis = put_label(lb_vis, f"ETL: letterbox {lb_info['target_w']}x{lb_info['target_h']} (scale={lb_info['scale']:.3f})")
        noisy_vis = put_label(noisy_vis, f"Ruído gaussiano sigma={noise_info['noise_sigma']}")
        comp_vis = put_label(comp_vis, f"JPEG qualidade={jpeg_info['jpeg_quality']}")
    side_by_side = cv2.hconcat([lb_vis, noisy_vis, comp_vis])
    sbs_path = os.path.join(args.outdir, f"{args.image_id}_stages_side_by_side.png")
    cv2.imwrite(sbs_path, side_by_side)

    # Manifesto didático por imagem
    manifest_path = os.path.join(args.outdir, f"{args.image_id}_trace_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "timestamp", "seq", "image_id", "src_path",
            "letterbox_path", "noisy_path", "jpeg_path", "stages_side_by_side_path",
            "label_path", "orig_w", "orig_h", "new_w", "new_h",
            "x_off", "y_off", "target_w", "target_h", "scale",
            "noise_sigma", "jpeg_quality"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seq": args.seq,
            "image_id": args.image_id,
            "src_path": os.path.abspath(args.image),
            "letterbox_path": os.path.abspath(lb_path),
            "noisy_path": os.path.abspath(noisy_path),
            "jpeg_path": os.path.abspath(jpeg_path),
            "stages_side_by_side_path": os.path.abspath(sbs_path),
            "label_path": os.path.abspath(args.label_path) if args.label_path else "",
            **lb_info,
            **noise_info,
            **jpeg_info,
        })

    print(f"Artefatos gerados em: {args.outdir}")
    print(f"Manifesto: {manifest_path}")
    print(f"Visualização 3 estágios: {sbs_path}")


if __name__ == "__main__":
    main()
