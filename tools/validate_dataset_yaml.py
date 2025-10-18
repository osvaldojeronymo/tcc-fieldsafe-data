#!/usr/bin/env python3
"""
Validador simples para verificar se os caminhos definidos em dataset.yml existem
e estão consistentes com a estrutura esperada.
"""

import os, sys, yaml

def expand_vars(s, env):
    if not isinstance(s, str):
        return s
    out = s
    for k, v in env.items():
        out = out.replace("${" + k + "}", v)
    # Expande expressões encadeadas como ${paths.train_images}
    while "${" in out:
        start = out.find("${")
        end = out.find("}", start)
        key = out[start + 2:end]
        if key in env:
            out = out[:start] + env[key] + out[end + 1:]
        else:
            break
    return out

def main(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = {}
    if "base_dir" in cfg:
        env["base_dir"] = cfg["base_dir"]

    if "paths" in cfg:
        for k, v in cfg["paths"].items():
            env[f"paths.{k}"] = expand_vars(v, env)

    keys = [
        "base_dir",
        "paths.split_dir",
        "paths.train_images", "paths.val_images", "paths.test_images",
        "paths.train_labels", "paths.val_labels", "paths.test_labels",
    ]
    resolved = {k: expand_vars("${" + k + "}", env) for k in keys}

    print("🔍 Caminhos resolvidos:")
    for k, v in resolved.items():
        print(f" - {k}: {v}")

    ok = True
    for k, v in resolved.items():
        if v and os.path.exists(v):
            continue
        else:
            print(f"⚠️  Caminho não encontrado: {v}")
            ok = False

    if ok:
        print("\n✅ Tudo certo! Estrutura parece consistente.")
    else:
        print("\n⚠️ Ajuste os caminhos no dataset.yml ou crie as pastas antes de seguir.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("uso: python tools/validate_dataset_yaml.py experiments/configs/dataset.yml")
        sys.exit(1)
    main(sys.argv[1])
