// WebGPU port of the tile U-Net forward (ScatterNet.unet in ../../scatter/js/scatter_model.js), batched over T tiles in one command buffer.
// Opt-in: UNetGPU.tryCreate returns null when WebGPU or an adapter is missing, and the caller keeps using the CPU path.
// Layout: activations are float32 [tile][channel][y][x]; weights are the whole weights.bin, addressed by the manifest float offsets.

const CONV = /* wgsl */ `
struct P { cinA: u32, cinB: u32, cout: u32, H: u32, W: u32, up: u32, gelu: u32, wOff: u32, bOff: u32, ng: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read> A: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;

fn erf_(x0: f32) -> f32 {
  let x = abs(x0);
  let t = 1.0 / (1.0 + 0.5 * x);
  let y = 1.0 - t * exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return select(-y, y, x0 >= 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
  let H = p.H; let W = p.W;
  if (g.x >= W || g.y >= H) { return; }
  let tile = wg.z / p.ng; let c0 = (wg.z % p.ng) * 8u;
  let cin = p.cinA + p.cinB;
  var Ha = H; var Wa = W;
  if (p.up == 1u) { Ha = H / 2u; Wa = W / 2u; }
  let baseA = tile * p.cinA * Ha * Wa;
  let baseB = tile * p.cinB * H * W;
  var acc: array<f32, 8>;
  for (var k = 0u; k < 8u; k++) { acc[k] = select(0.0, wt[p.bOff + c0 + k], c0 + k < p.cout); }
  for (var ci = 0u; ci < cin; ci++) {
    for (var ky = 0u; ky < 3u; ky++) {
      let yy = i32(g.y) + i32(ky) - 1;
      if (yy < 0 || yy >= i32(H)) { continue; }
      for (var kx = 0u; kx < 3u; kx++) {
        let xx = i32(g.x) + i32(kx) - 1;
        if (xx < 0 || xx >= i32(W)) { continue; }
        var v: f32;
        if (ci < p.cinA) {
          var sy = u32(yy); var sx = u32(xx);
          if (p.up == 1u) { sy = sy >> 1u; sx = sx >> 1u; }
          v = A[baseA + (ci * Ha + sy) * Wa + sx];
        } else {
          v = B[baseB + ((ci - p.cinA) * H + u32(yy)) * W + u32(xx)];
        }
        let t = ky * 3u + kx;
        for (var k = 0u; k < 8u; k++) {
          if (c0 + k < p.cout) { acc[k] += wt[p.wOff + ((c0 + k) * cin + ci) * 9u + t] * v; }
        }
      }
    }
  }
  for (var k = 0u; k < 8u; k++) {
    if (c0 + k < p.cout) {
      var r = acc[k];
      if (p.gelu == 1u) { r = 0.5 * r * (1.0 + erf_(r * 0.70710678118654752)); }
      O[((tile * p.cout + c0 + k) * H + g.y) * W + g.x] = r;
    }
  }
}`;

const POOL = /* wgsl */ `
struct P { C: u32, H: u32, W: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> I: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let h2 = p.H / 2u; let w2 = p.W / 2u;
  if (g.x >= w2 || g.y >= h2) { return; }
  let i = g.z * p.H * p.W + 2u * g.y * p.W + 2u * g.x;
  O[g.z * h2 * w2 + g.y * w2 + g.x] = 0.25 * (I[i] + I[i + 1u] + I[i + p.W] + I[i + p.W + 1u]);
}`;

const CH = 32;

export class UNetGPU {
  static async tryCreate(manifest, buf, T, gpu = (typeof navigator !== "undefined" ? navigator.gpu : null)) {
    try {
      if (!gpu) return null;
      const adapter = await gpu.requestAdapter();
      if (!adapter) return null;
      const device = await adapter.requestDevice();
      const u = new UNetGPU(device, manifest, buf, T);
      u.adapter = adapter.info ? `${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.description || adapter.info.device}`.trim() : "unknown";
      u.software = !!(adapter.isFallbackAdapter || (adapter.info && /software|llvmpipe|swiftshader/i.test(`${adapter.info.architecture} ${adapter.info.device} ${adapter.info.description}`)));
      return u;
    } catch (e) {
      return null;
    }
  }

  constructor(device, manifest, buf, T) {
    this.device = device; this.T = T;
    const G = this.G = manifest.config.grid, L = this.L = manifest.config.levels;
    const S = GPUBufferUsage.STORAGE;
    const mk = (n, usage = S) => device.createBuffer({ size: Math.max(16, n * 4), usage });
    this.wbuf = mk(buf.byteLength / 4, S | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(this.wbuf, 0, buf);
    this.in = mk(T * 6 * G * G, S | GPUBufferUsage.COPY_DST);
    this.outBuf = mk(T * G * G, S | GPUBufferUsage.COPY_SRC);
    this.read = device.createBuffer({ size: T * G * G * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.dummy = mk(4);
    const sz = (l) => T * CH * (G >> l) * (G >> l);
    const h0 = mk(sz(0)), mid = mk(sz(0));
    const skip = Array.from({ length: L + 1 }, (_, l) => mk(sz(l)));
    const pooled = Array.from({ length: L + 1 }, (_, l) => (l ? mk(sz(l)) : null));
    const dec = Array.from({ length: L }, (_, l) => mk(sz(l)));
    const mod = (code) => device.createShaderModule({ code });
    this.convPipe = device.createComputePipeline({ layout: "auto", compute: { module: mod(CONV), entryPoint: "main" } });
    this.poolPipe = device.createComputePipeline({ layout: "auto", compute: { module: mod(POOL), entryPoint: "main" } });
    this.steps = [];
    const off = (name) => manifest.tensors[name].offset;
    const conv = (name, a, cinA, b, cinB, o, cout, H, up, gelu) => {
      const ng = Math.ceil(cout / 8), data = new Uint32Array([cinA, cinB, cout, H, H, up, gelu, off(name + ".weight"), off(name + ".bias"), ng]);
      const ub = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(ub, 0, data);
      const bg = device.createBindGroup({ layout: this.convPipe.getBindGroupLayout(0), entries: [ub, this.wbuf, a, b || this.dummy, o].map((r, i) => ({ binding: i, resource: { buffer: r } })) });
      this.steps.push({ pipe: this.convPipe, bg, x: Math.ceil(H / 8), y: Math.ceil(H / 8), z: T * ng });
    };
    const pool = (i, o, H) => {
      const ub = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(ub, 0, new Uint32Array([CH, H, H, 0]));
      const bg = device.createBindGroup({ layout: this.poolPipe.getBindGroupLayout(0), entries: [ub, i, o].map((r, k) => ({ binding: k, resource: { buffer: r } })) });
      this.steps.push({ pipe: this.poolPipe, bg, x: Math.ceil(H / 16), y: Math.ceil(H / 16), z: T * CH });
    };
    conv("net.inp", this.in, 6, null, 0, h0, CH, G, 0, 1);
    conv("net.enc.0.0", h0, CH, null, 0, mid, CH, G, 0, 1);
    conv("net.enc.0.2", mid, CH, null, 0, skip[0], CH, G, 0, 1);
    for (let l = 1; l <= L; l++) {
      const H = G >> l;
      pool(skip[l - 1], pooled[l], H * 2);
      conv(`net.enc.${l}.0`, pooled[l], CH, null, 0, mid, CH, H, 0, 1);
      conv(`net.enc.${l}.2`, mid, CH, null, 0, skip[l], CH, H, 0, 1);
    }
    let cur = skip[L];
    for (let l = L - 1; l >= 0; l--) {
      const H = G >> l;
      conv(`net.dec.${l}.0`, cur, CH, skip[l], CH, mid, CH, H, 1, 1);
      conv(`net.dec.${l}.2`, mid, CH, null, 0, dec[l], CH, H, 0, 1);
      cur = dec[l];
    }
    conv("net.out", cur, CH, null, 0, this.outBuf, 1, G, 0, 0);
  }

  encode() {
    const enc = this.device.createCommandEncoder();
    for (const s of this.steps) {
      const pass = enc.beginComputePass();
      pass.setPipeline(s.pipe); pass.setBindGroup(0, s.bg); pass.dispatchWorkgroups(s.x, s.y, s.z); pass.end();
    }
    return enc;
  }

  // x: Float32Array [T][6][G][G]; returns Float32Array [T][G][G]. timing (ms): upload, compute (submit to done), readback
  async run(x, timing = null) {
    const { device, G, T } = this;
    let t0 = performance.now();
    device.queue.writeBuffer(this.in, 0, x);
    await device.queue.onSubmittedWorkDone();
    const t1 = performance.now();
    const enc = this.encode();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const t2 = performance.now();
    const cp = device.createCommandEncoder();
    cp.copyBufferToBuffer(this.outBuf, 0, this.read, 0, T * G * G * 4);
    device.queue.submit([cp.finish()]);
    await this.read.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.read.getMappedRange().slice(0));
    this.read.unmap();
    if (timing) { timing.upload = t1 - t0; timing.compute = t2 - t1; timing.readback = performance.now() - t2; }
    return out;
  }

  // median of n runs of the compute pass alone (inputs already resident), ms
  async timeCompute(n = 5) {
    const ts = [];
    for (let i = 0; i < n + 1; i++) {
      const t0 = performance.now();
      this.device.queue.submit([this.encode().finish()]);
      await this.device.queue.onSubmittedWorkDone();
      if (i) ts.push(performance.now() - t0);
    }
    ts.sort((a, b) => a - b);
    return ts[ts.length >> 1];
  }
}
