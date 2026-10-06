export function collect(tr, bufs) {
  const add = (a) => { bufs.push(a.buffer); return a; };
  for (const k of ["pos", "vel", "input", "phiK", "inp", "phi", "gather", "pairAcc", "accel"]) add(tr[k]);
  for (const k of ["enc", "dec", "gradPhi", "aGrid"]) tr[k].forEach(add);
}
