"""Repair the trust_remote_code module cache for jinaai/jina-embeddings-v3.

The VPS often downloads `jinaai/xlm-roberta-flash-implementation` partially
(flaky network), leaving transformers_modules missing files (mlp.py, mha.py,
...). This downloads every file with per-file retries until the full set is
verified present in the exact cache location transformers expects.

Usage:
    HF_ENDPOINT=https://hf-mirror.com python scripts/fetch_flash_modules.py
"""
import os
import sys
import time

import requests

REPO = "jinaai/xlm-roberta-flash-implementation"
COMMIT = "845308d0fd72a8406a3e378450e1a09522790419"
SANITIZED = "xlm_hyphen_roberta_hyphen_flash_hyphen_implementation"

FILES = [
    "block.py", "configuration_xlm_roberta.py", "convert_roberta_weights_to_flash.py",
    "embedding.py", "mha.py", "mlp.py", "modeling_lora.py", "modeling_xlm_roberta.py",
    "rotary.py", "stochastic_depth.py", "xlm_padding.py",
]


def main() -> None:
    endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    target = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "modules", "transformers_modules",
        "jinaai", SANITIZED, COMMIT,
    )
    os.makedirs(target, exist_ok=True)

    ok = True
    for f in FILES:
        path = os.path.join(target, f)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"  {f}: already present", flush=True)
            continue
        url = f"{endpoint}/{REPO}/resolve/{COMMIT}/{f}"
        for attempt in range(10):
            try:
                resp = requests.get(url, timeout=120)
                if resp.status_code == 200 and len(resp.content) > 0:
                    tmp = path + ".part"
                    with open(tmp, "wb") as fh:
                        fh.write(resp.content)
                    os.replace(tmp, path)
                    print(f"  {f}: OK ({len(resp.content)} bytes)", flush=True)
                    break
                print(f"  {f}: HTTP {resp.status_code}, retrying ...", flush=True)
            except requests.RequestException as e:
                print(f"  {f}: {e}, retrying ...", flush=True)
            time.sleep(10)
        else:
            ok = False
            print(f"  {f}: FAILED after retries", flush=True)

    missing = sorted(f for f in FILES if not os.path.exists(os.path.join(target, f)))
    if missing:
        print(f"ERROR: still missing {missing}", flush=True)
        sys.exit(1)
    print(f"OK — all {len(FILES)} files verified at {target}", flush=True)


if __name__ == "__main__":
    main()
