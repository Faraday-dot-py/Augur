#!/bin/bash
# usage: scripts/render_readme_gif_previews.sh   (run from the repo root; local CPU, streamed by ffmpeg)
set -euo pipefail
src=videos/youtube_upload
out=docs/design/assets/readme
mkdir -p "$out"

render() {
  local in="$1" name="$2" start="$3" len="$4" fps="$5" width="$6"
  [ -f "$in" ] || { echo "missing $in" >&2; exit 1; }
  ffmpeg -v error -y -ss "$start" -t "$len" -i "$in" \
    -vf "fps=$fps,scale=$width:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" \
    -loop 0 "$out/$name.gif"
  ls -l "$out/$name.gif"
}

render "$src/1 billion bodies falling into a black-hole-like core (learned gravity model).mp4" black-hole-1e9-lensed-preview 0 12.5 8 480
render "$src/10 billion bodies - gravity collapse at three zoom levels.mp4" black-hole-1e10-three-zoom-preview 0 8.4 8 560
render "$src/A star, 8 planets, a moon and a belt - exact vs learned.mp4" solar-system-exact-vs-learned-preview 0 20 6 480
render "$src/Planetesimal disk around a star - reference simulation and a failed learned toy.mp4" planetesimal-disk-preview 0 20 6 480
