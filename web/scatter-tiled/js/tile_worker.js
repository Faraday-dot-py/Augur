import { TileSet, parseWeights } from "./tiled_model.js";

let set;

onmessage = (e) => {
  const m = e.data;
  try {
    if (m.init) {
      set = new TileSet(parseWeights(m.init.manifest, m.init.buf));
      postMessage({ id: m.id, res: {} });
      return;
    }
    const [res, tr] = set.handle(m.msg);
    postMessage({ id: m.id, res }, tr);
  } catch (err) {
    postMessage({ id: m.id, error: err.message });
  }
};
