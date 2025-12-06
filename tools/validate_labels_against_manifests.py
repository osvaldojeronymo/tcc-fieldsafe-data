import argparse
import csv
from pathlib import Path


def validate(root: Path):
    reports = []
    for manifest in root.glob('*/manifest.csv'):
        seq = manifest.parent.name
        with open(manifest, newline='') as f:
            r = csv.DictReader(f)
            rows = list(r)
        total = len(rows)
        missing = []
        for row in rows:
            lp = row.get('label_path', '')
            if not lp or not Path(lp).exists():
                missing.append(row.get('fname', ''))
        reports.append((seq, total, len(missing), missing))
    return reports


def main():
    p = argparse.ArgumentParser('Validate presence of labels referenced by manifests')
    p.add_argument('--preprocessed_root', required=True, help='Root folder containing <SEQ>/manifest.csv')
    args = p.parse_args()
    root = Path(args.preprocessed_root)
    reps = validate(root)
    for seq, total, nmiss, missing in reps:
        print(f'{seq}: total={total}, missing_labels={nmiss}')
        if missing:
            for fn in missing[:10]:
                print(f'  - missing: {fn}')


if __name__ == '__main__':
    main()
