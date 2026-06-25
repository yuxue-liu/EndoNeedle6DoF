"""Concatenate the first N frames of one mp4 (pairs 1-3, salvaged from the crashed
combined run) with all frames of another (the re-run pair 4) into one finalized
mp4. No ffmpeg: a single cv2 re-encode pass that also writes the moov atom.
"""
import argparse
import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--head', required=True, help='source 1 (take first --head-frames)')
    p.add_argument('--head-frames', type=int, required=True)
    p.add_argument('--tail', required=True, help='source 2 (take all frames)')
    p.add_argument('--output', required=True)
    p.add_argument('--fps', type=float, default=30.0)
    args = p.parse_args()

    c1 = cv2.VideoCapture(args.head)
    w = int(c1.get(3)); h = int(c1.get(4))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
    n = 0
    for _ in range(args.head_frames):
        ok, fr = c1.read()
        if not ok:
            break
        writer.write(fr); n += 1
    c1.release()
    print(f'[concat] head: wrote {n} frames')

    c2 = cv2.VideoCapture(args.tail)
    m = 0
    while True:
        ok, fr = c2.read()
        if not ok:
            break
        if (fr.shape[1], fr.shape[0]) != (w, h):
            fr = cv2.resize(fr, (w, h))
        writer.write(fr); m += 1
    c2.release()
    writer.release()
    print(f'[concat] tail: wrote {m} frames')
    print(f'[concat] total {n + m} -> {args.output}')


if __name__ == '__main__':
    main()
