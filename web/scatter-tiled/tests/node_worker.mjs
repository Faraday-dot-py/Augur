import { parentPort, workerData } from "node:worker_threads";
import { TileSet, parseWeights } from "../js/tiled_model.js";

const set = new TileSet(parseWeights(workerData.manifest, workerData.buf));
parentPort.on("message", ({ id, msg }) => {
  const [res] = set.handle(msg);
  parentPort.postMessage({ id, res });
});
