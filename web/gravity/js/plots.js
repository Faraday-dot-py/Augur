// Small canvas line plots: the learned force curve and the energy trace.
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

// f(log d) / (d^2+1) sampled over d in [dMin, dMax], plus markers for the
// current pair distances (edges: array of d values).
export function drawForceCurve(canvas, forceMlp, dMin, dMax, edges) {
  const { ctx, w, h } = setup(canvas);
  const n = 96;
  const ys = new Float64Array(n);
  let maxAbs = 1e-9;
  for (let i = 0; i < n; i++) {
    const d = dMin * Math.pow(dMax / dMin, i / (n - 1));
    const y = forceMlp.eval(Math.log(d)) / (d * d + 1);
    ys[i] = y;
    maxAbs = Math.max(maxAbs, Math.abs(y));
  }
  const x0 = 4, x1 = w - 4, y0 = h - 4, y1 = 4, mid = (y0 + y1) / 2;
  const px = (i) => x0 + (x1 - x0) * i / (n - 1);
  const py = (y) => mid - (y / maxAbs) * (mid - y1 - 2);
  ctx.strokeStyle = "#56627a"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x0, mid); ctx.lineTo(x1, mid); ctx.stroke();
  ctx.strokeStyle = "#6fc3ff"; ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) { const x = px(i), y = py(ys[i]); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
  ctx.stroke();
  if (edges && edges.length) {
    ctx.fillStyle = "#ff9a3c";
    for (const d of edges) {
      if (d < dMin || d > dMax) continue;
      const i = (Math.log(d / dMin) / Math.log(dMax / dMin)) * (n - 1);
      const y = forceMlp.eval(Math.log(d)) / (d * d + 1);
      ctx.beginPath(); ctx.arc(px(i), py(y), 2.5, 0, Math.PI * 2); ctx.fill();
    }
  }
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
