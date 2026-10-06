// the four cloud-in-cell corners of one body: [ix, iy, weight] per corner, cells outside the grid dropped
export function bodyCorners(x, y, G, extent) {
  const h = extent / G, half = extent / 2;
  const ux = (x + half) / h - 0.5, uy = (y + half) / h - 0.5;
  const ix = Math.floor(ux), iy = Math.floor(uy);
  const fx = ux - ix, fy = uy - iy;
  const out = [];
  for (let oy = 0; oy < 2; oy++) {
    for (let ox = 0; ox < 2; ox++) {
      const cx = ix + ox, cy = iy + oy;
      if (cx < 0 || cx >= G || cy < 0 || cy >= G) continue;
      out.push([cx, cy, (ox ? fx : 1 - fx) * (oy ? fy : 1 - fy)]);
    }
  }
  return out;
}
