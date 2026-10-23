// Opt-in pool that runs the finish phase's U-Net for all of its tiles as one WebGPU batch (UNetGPU); everything else stays on the CPU TileSet.
// The finish message is run twice: a recording pass collects the U-Net inputs (field restored afterwards), the GPU batch runs, a replay pass uses its outputs.
// Same call(i, msg) interface as LocalPool, so TiledDriver is used unchanged. Falls back to the plain CPU TileSet when no GPU was created.
import { TileSet, GT } from "./tiled_model.js";
import { UNetGPU } from "./unet_gpu.js";

export class GpuPool {
  constructor(weights, gpu) {
    this.set = new TileSet(weights); this.size = 1; this.gpu = gpu; this.last = null;
  }

  static async create(weights, manifest, buf, T, gpu) {
    const u = await UNetGPU.tryCreate(manifest, buf, T, gpu);
    return new GpuPool(weights, u);
  }

  async call(i, m) {
    const set = this.set;
    if (m.cmd !== "finish" || !this.gpu) return set.handle(m)[0];
    const net = set.net, orig = net.unet, GG = GT * GT, T = this.gpu.T, xs = [], saved = new Map(set.field);
    net.unet = (x) => { xs.push(Float32Array.from(x)); return new Float32Array(GG); };
    try { set.finish({ ...m, trace: false, unet: -1 }); } finally { net.unet = orig; }
    set.field = saved;
    const x = new Float32Array(T * 6 * GG);
    xs.forEach((a, k) => x.set(a, k * 6 * GG));
    const timing = {};
    const out = await this.gpu.run(x, timing);
    this.last = timing;
    let k = 0;
    net.unet = () => out.slice(k * GG, ++k * GG);
    try { return set.finish(m)[0]; } finally { net.unet = orig; }
  }
}
