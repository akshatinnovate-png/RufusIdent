"""Download and verify the face models (about 300 MB, one time).

    python setup_models.py
"""
import hashlib
import os
import sys
import urllib.request
import zipfile

from config import load_config

BUFFALO = ("https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
           "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f")
FILES = {
    # name: (url or member of buffalo_l.zip, sha256)
    "det_10g.onnx": ("buffalo_l.zip", "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"),
    "w600k_r50.onnx": ("buffalo_l.zip", "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"),
    "MiniFASNetV2.onnx": ("https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx",
                          "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907"),
    "MiniFASNetV1SE.onnx": ("https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV1SE.onnx",
                            "ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676"),
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: str):
    print(f"Downloading {url.rsplit('/', 1)[-1]} …")
    def hook(blocks, bs, total):
        if total > 0:
            print(f"\r  {min(blocks * bs, total) / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    urllib.request.urlretrieve(url, dest + ".part", hook)
    print()
    os.replace(dest + ".part", dest)


def main():
    d = load_config().models_dir
    os.makedirs(d, exist_ok=True)
    need = [n for n, (_, h) in FILES.items() if not (os.path.exists(os.path.join(d, n)) and sha256(os.path.join(d, n)) == h)]
    if not need:
        print(f"All models present and verified in {d}")
        return
    if any(FILES[n][0] == "buffalo_l.zip" for n in need):
        z = os.path.join(d, "buffalo_l.zip")
        if not (os.path.exists(z) and sha256(z) == BUFFALO[1]):
            download(BUFFALO[0], z)
            if sha256(z) != BUFFALO[1]:
                sys.exit("buffalo_l.zip checksum mismatch: delete it and try again.")
        with zipfile.ZipFile(z) as zf:
            for n in need:
                if FILES[n][0] == "buffalo_l.zip":
                    member = next(m for m in zf.namelist() if m.endswith(n))
                    with zf.open(member) as src, open(os.path.join(d, n), "wb") as dst:
                        dst.write(src.read())
        os.remove(z)
    for n in need:
        src, h = FILES[n]
        if src != "buffalo_l.zip":
            download(src, os.path.join(d, n))
        if sha256(os.path.join(d, n)) != h:
            sys.exit(f"{n}: checksum mismatch. Delete it and run again.")
        print(f"  ✓ {n}")
    print(f"Models ready in {d}")


if __name__ == "__main__":
    main()
