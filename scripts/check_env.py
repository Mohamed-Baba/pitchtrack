"""Check the environment before anything else.

    python scripts/check_env.py            # versions, GPU, sm_120 support
    python scripts/check_env.py --yolo     # also downloads and runs yolo26n.pt once
"""
import argparse
import platform
import time

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", action="store_true", help="load yolo26n.pt and time one prediction")
    args = ap.parse_args()
    print(f"python {platform.python_version()} | numpy {np.__version__} | opencv {cv2.__version__}")
    try:
        import torch
    except ImportError:
        print("torch: NOT INSTALLED -- install the CUDA build first (README)")
        return
    print(f"torch {torch.__version__} | CUDA build {torch.version.cuda} | cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        arch = torch.cuda.get_arch_list()
        print(f"GPU: {name} | capability sm_{cap[0]}{cap[1]} | compiled archs: {arch}")
        if f"sm_{cap[0]}{cap[1]}" not in arch:
            print("WARNING: this torch build has no kernels for your GPU: reinstall a build that lists it")
        x = torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize()
        tic = time.perf_counter()
        for _ in range(10):
            x @ x
        torch.cuda.synchronize()
        print(f"matmul 4096^2 x10: {1000 * (time.perf_counter() - tic):.1f} ms")
    if args.yolo:
        from ultralytics import YOLO, __version__

        model = YOLO("yolo26n.pt")
        img = np.zeros((1080, 1920, 3), np.uint8)
        device = 0 if torch.cuda.is_available() else "cpu"
        model.predict(img, imgsz=1280, device=device, verbose=False)
        tic = time.perf_counter()
        model.predict(img, imgsz=1280, device=device, verbose=False)
        print(f"ultralytics {__version__}: yolo26n at 1280 in {1000 * (time.perf_counter() - tic):.1f} ms")


if __name__ == "__main__":
    main()
