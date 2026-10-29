"""Contact sheet of evenly spaced frames from an mp4 for reviewer agents. Usage: python3 scripts/video_contact_sheet.py in.mp4 out.png [--n 8] [--cols 4] [--width 600]"""
import argparse
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("out")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--cols", type=int, default=4)
ap.add_argument("--width", type=int, default=600)
a = ap.parse_args()
nb = int(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", a.video]).decode().strip())
step = max(1, nb // a.n)
rows = (a.n + a.cols - 1) // a.cols
vf = f"select='not(mod(n\\,{step}))',scale={a.width}:-1,tile={a.cols}x{rows}"
subprocess.check_call(["ffmpeg", "-y", "-v", "error", "-i", a.video, "-vf", vf, "-frames:v", "1", a.out])
print("wrote", a.out, "frames", nb, "step", step)
