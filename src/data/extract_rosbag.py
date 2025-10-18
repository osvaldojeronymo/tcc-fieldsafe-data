#!/usr/bin/env python3
"""
Extrai frames RGB de arquivos ROS .bag, gerando PNGs + CSV (frames_manifest.csv).
Suporta sensor_msgs/Image e sensor_msgs/CompressedImage. Permite amostragem (--every_n).

Saída:
  <outdir>/
    images/  (PNGs)
    meta/
    frames_manifest.csv   (cols: stamp_ns,filename,width,height)
"""

import os, csv, re, argparse
import cv2
import numpy as np

# ROS
try:
    import rosbag
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, CompressedImage
except ImportError as e:
    print("\n[ERRO] Precisa de rosbag + cv_bridge (ROS Noetic).")
    raise

def sanitize_topic(topic: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', topic.strip('/'))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="Caminho para o arquivo .bag")
    ap.add_argument("--topic", required=True, help="Tópico RGB (ex.: /Multisense/left/image_rect_color)")
    ap.add_argument("--outdir", required=True, help="Diretório de saída")
    ap.add_argument("--every_n", type=int, default=1, help="Salvar 1 a cada N imagens (amostragem)")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = sem limite")
    args = ap.parse_args()

    bag_path = args.bag
    topic = args.topic
    outdir = args.outdir

    ensure_dir(outdir)
    imgdir = os.path.join(outdir, "images"); ensure_dir(imgdir)
    metadir = os.path.join(outdir, "meta"); ensure_dir(metadir)

    bagbase = os.path.splitext(os.path.basename(bag_path))[0]
    topic_s = sanitize_topic(topic)

    csv_path = os.path.join(outdir, "frames_manifest.csv")
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=["stamp_ns","filename","width","height"])
        writer.writeheader()

        bridge = CvBridge()
        saved = 0
        i = 0

        with rosbag.Bag(bag_path, "r") as bag:
            for _, msg, t in bag.read_messages(topics=[topic]):
                i += 1
                if args.every_n > 1 and (i % args.every_n != 0):
                    continue

                # Decodifica - usar hasattr em vez de isinstance devido a tipos dinâmicos do rosbag
                img = None

                # sensor_msgs/Image
                if all(hasattr(msg, a) for a in ("encoding", "data", "width", "height")):
                    try:
                        img = bridge.imgmsg_to_cv2(msg)  # não força bgr8
                        # se vier gray, converte pra BGR pra padronizar downstream
                        if img is not None and len(img.shape) == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    except Exception:
                        img = None

                # sensor_msgs/CompressedImage
                elif all(hasattr(msg, a) for a in ("format", "data")):
                    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                    dec = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if dec is not None:
                        img = dec

                # Se não reconheceu, segue pra próxima
                if img is None:
                    continue

                h, w = img.shape[:2]
                stamp_ns = int(t.to_nsec())
                fname = f"rgb_{stamp_ns}.png"
                cv2.imwrite(os.path.join(imgdir, fname), img)
                writer.writerow({"stamp_ns": stamp_ns, "filename": f"images/{fname}", "width": w, "height": h})
                saved += 1

                if args.max_frames and saved >= args.max_frames:
                    break

    # salva um txt com metadados leves
    with open(os.path.join(metadir, "source.txt"), "w") as f:
        f.write(f"bag={bagbase}\ntopic={topic}\nevery_n={args.every_n}\n")

    print(f"[OK] {saved} frames -> {csv_path}")

if __name__ == "__main__":
    main()
