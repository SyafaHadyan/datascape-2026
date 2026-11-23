"""Resize every listed image once to the letterboxed square size and store them in one
uint8 .npy file, so training reads small arrays instead of decoding the full-size photos."""
import argparse
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from img_utils import LetterboxResize

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")


def load_one(args):
    path, size = args
    with Image.open(path) as im:
        return np.asarray(LetterboxResize(size)(im.convert("RGB")), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--procs", type=int, default=4)
    args = parser.parse_args()

    paths = Path(args.paths_file).read_text().splitlines()
    tmp_out = args.out + ".partial.npy"
    out = np.lib.format.open_memmap(tmp_out, mode="w+", dtype=np.uint8, shape=(len(paths), args.size, args.size, 3))
    with Pool(args.procs) as pool:
        for i, arr in enumerate(pool.imap(load_one, [(p, args.size) for p in paths], chunksize=32)):
            out[i] = arr
            if i % 2000 == 0:
                print(f"  cached {i}/{len(paths)}", flush=True)
    out.flush()
    del out
    Path(tmp_out).rename(args.out)  # renamed last, so a half-written cache is never picked up
    print(f"cached {len(paths)} images to {args.out}", flush=True)


if __name__ == "__main__":
    main()
