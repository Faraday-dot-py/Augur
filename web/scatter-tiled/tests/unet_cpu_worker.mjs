import { parentPort, workerData } from "node:worker_threads";
import { TileNet, parseWeights } from "../js/tiled_model.js";

const net = new TileNet(parseWeights(workerData.manifest, workerData.buf));
parentPort.on("message", ({ xs }) => {
  const t0 = performance.now();
  for (const x of xs) net.unet(x);
  parentPort.postMessage({ ms: performance.now() - t0 });
});
