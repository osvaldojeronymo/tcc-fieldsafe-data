import sys
from pathlib import Path

try:
    from rosbags.rosbag1 import Reader
except Exception as e:
    print("Erro: precisa do pacote 'rosbags' para detectar tópicos.")
    raise


def is_image_connection(conn) -> bool:
    # Suporta formatos ROS1 ('sensor_msgs/Image') e ROS2 ('sensor_msgs/msg/Image').
    mt = conn.msgtype
    return (
        mt in {"sensor_msgs/Image", "sensor_msgs/CompressedImage"}
        or mt.endswith("/Image")
        or mt.endswith("/CompressedImage")
    )


def score_topic(name: str) -> int:
    n = name.lower()
    score = 0
    # Heurística simples: priorizar termos típicos de webcam USB
    for kw in ("usb", "webcam", "logitech", "c920", "camera", "image_raw", "image_color"):
        if kw in n:
            score += 1
    # Penalizar tópicos de stereo/multisense
    for bad in ("multisense", "stereo", "left", "right"):
        if bad in n:
            score -= 1
    return score


def detect_webcam_topic(bag_path: Path):
    with Reader(bag_path) as reader:
        image_topics = [conn for conn in reader.connections if is_image_connection(conn)]
        if not image_topics:
            return None, []
        scored = sorted(((score_topic(c.topic), c.topic, c.msgtype) for c in image_topics), reverse=True)
        best = scored[0][1]
        return best, [(t, mt) for _, t, mt in scored]


def main():
    # Usage: python detect_webcam_topic.py <bag> [--quiet]
    if len(sys.argv) < 2:
        print("Uso: python detect_webcam_topic.py <caminho_para_arquivo.bag> [--quiet]")
        sys.exit(2)
    bag = Path(sys.argv[1])
    quiet = (len(sys.argv) > 2 and sys.argv[2] == "--quiet")
    if not bag.exists():
        print(f"Arquivo .bag não encontrado: {bag}")
        sys.exit(2)
    best, ranked = detect_webcam_topic(bag)
    if best is None:
        if quiet:
            print("")
        else:
            print("Nenhum tópico de imagem encontrado.")
        sys.exit(1)
    if quiet:
        print(best)
    else:
        print("Tópicos de imagem encontrados (ordenados por relevância):")
        for t, mt in ranked:
            print(f"- {t}  ({mt})")
        print(f"\nSugerido como webcam: {best}")


if __name__ == "__main__":
    main()
