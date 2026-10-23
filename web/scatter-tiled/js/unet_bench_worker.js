import { TileNet, parseWeights } from "./tiled_model.js";

let net;
onmessage = ({ data }) => {
  if (data.init) { net = new TileNet(parseWeights(data.init.manifest, data.init.buf)); postMessage({ ready: true }); return; }
  const t0 = performance.now();
  for (const x of data.xs) net.unet(x);
  postMessage({ ms: performance.now() - t0 });
};
