// Small canvas line plots: the error and energy traces.
function setup(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr; canvas.height = h * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

// Two aligned traces (model, truth) over the last `hist.length` ticks.
export function drawEnergy(canvas, modelHist, truthHist) {
  const { ctx, w, h } = setup(canvas);
  const all = modelHist.concat(truthHist).filter(Number.isFinite);
  if (!all.length) return;
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
  const x0 = 4, x1 = w - 4, y0 = h - 4, y1 = 4;
  const n = Math.max(modelHist.length, truthHist.length, 2);
  const px = (i) => x0 + (x1 - x0) * i / (n - 1);
  const py = (y) => y1 + (hi - y) / (hi - lo) * (y0 - y1);
  const draw = (hist, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < hist.length; i++) {
      const x = px(i), y = py(hist[i]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  };
  draw(truthHist, "#ff9a3c");
  draw(modelHist, "#6fc3ff");
}

// Single trace on a log-free axis from 0, for the mean position error.
export function drawSeries(canvas, hist, color) {
  const { ctx, w, h } = setup(canvas);
  if (hist.length < 2) return;
  const hi = Math.max(...hist, 1e-6);
  const x0 = 4, x1 = w - 4, y0 = h - 4, y1 = 4;
  ctx.strokeStyle = "#56627a"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y0); ctx.stroke();
  ctx.strokeStyle = color; ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < hist.length; i++) {
    const x = x0 + (x1 - x0) * i / (hist.length - 1), y = y0 - hist[i] / hi * (y0 - y1);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
}
