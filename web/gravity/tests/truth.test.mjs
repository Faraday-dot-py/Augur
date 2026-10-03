// The ghost truth must attract: two bodies 3 apart, scripts/gravity_sim.accel gives +0.10663718 on body 0.
import * as truth from "../js/truth.js";

const a = new Float64Array(4);
truth.accel(new Float64Array([0, 0, 3, 0]), 2, a);
if (Math.abs(a[0] - 0.10663718) > 1e-7 || Math.abs(a[2] + 0.10663718) > 1e-7) { console.error("FAIL: truth.accel sign", a); process.exit(1); }
console.log("truth attracts OK");
