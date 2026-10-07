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
export function drawForceCurve(canvas, forceMlp, dMin, dMax, edges, trueLaw) {
  const { ctx, w, h } = setup(canvas);
  const n = 96;
  const ys = new Float64Array(n), ts = new Float64Array(n);
  let maxAbs = 1e-9;
  for (let i = 0; i < n; i++) {
    const d = dMin * Math.pow(dMax / dMin, i / (n - 1));
    const y = forceMlp.eval(Math.log(d)) / (d * d + 1);
    ys[i] = y;
    ts[i] = trueLaw ? trueLaw(d) : 0;
    maxAbs = Math.max(maxAbs, Math.abs(y), Math.abs(ts[i]));
  }
  const x0 = 4, x1 = w - 4, y0 = h - 4, y1 = 4, mid = (y0 + y1) / 2;
  const px = (i) => x0 + (x1 - x0) * i / (n - 1);
  const py = (y) => mid - (y / maxAbs) * (mid - y1 - 2);
  ctx.strokeStyle = "#56627a"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x0, mid); ctx.lineTo(x1, mid); ctx.stroke();
  ctx.strokeStyle = "#6fc3ff"; ctx.lineWidth = trueLaw ? 3 : 1.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) { const x = px(i), y = py(ys[i]); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
  ctx.stroke();
  if (trueLaw) {
    ctx.strokeStyle = "#ff9a3c"; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]);
    ctx.beginPath();
    for (let i = 0; i < n; i++) { const x = px(i), y = py(ts[i]); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (edges && edges.length) {
    ctx.fillStyle = "#e4eaf5";
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

// The traced body (centre) with its drawn neighbours at their true relative positions and the force each
// exerts on it (arrow length is signed-log of the magnitude); tr is CentralNet.step's trace.
export function drawHood(canvas, tr) {
  const { ctx, w, h } = setup(canvas);
  if (!tr || !tr.s1.edges.length) return;
  const cx = w / 2, cy = h / 2, es = tr.s1.edges;
  const reach = Math.max(...es.map((e) => e.dist), 1e-6);
  const k = Math.min(cx, cy) * 0.78 / reach;
  ctx.font = "10px ui-monospace, monospace";
  ctx.textBaseline = "middle";
  for (const e of es) {
    const x = cx + e.rel[0] * k, y = cy + e.rel[1] * k;
    ctx.strokeStyle = "rgba(111,195,255,0.25)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
    ctx.fillStyle = "#8390a8";
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillText("#" + e.src, x + 6, y);
    const m = Math.hypot(e.force[0], e.force[1]);
    if (m < 1e-12) continue;
    const len = 10 + 34 * Math.min(1, Math.log1p(m / 1e-3) / Math.log1p(300));
    ctx.strokeStyle = "#e4eaf5"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + (e.force[0] / m) * len, cy + (e.force[1] / m) * len); ctx.stroke();
  }
  ctx.fillStyle = "#6fc3ff";
  ctx.beginPath(); ctx.arc(cx, cy, 4.5, 0, Math.PI * 2); ctx.fill();
}
