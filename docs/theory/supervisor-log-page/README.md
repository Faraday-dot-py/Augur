# Supervisor log page standard

The published supervisor-log artifact (Jb4ufo2BV3fucZKknc8p3w, copy DBwgufJJZGvp4a5apxjtgz) is built from `supervisor-log-page-reference.html` (set as the standard 2026-10-26) and `docs/theory/supervisor-log.md`.

Keep when updating:
- Liquid-glass-on-CRT look: dull red/green/yellow/blue palette, static fog, ASCII-art titles, no box borders, straight square-ended rules, 250% root size, soft letter halo. No script, no animation.
- Performance (user reported scroll lag): at most 3 text-shadow layers on headings and labels, 2 on body and log text; no animated fog; no background-attachment: fixed; one scanline gradient, no RGB stripe layer, no blurred inset box-shadow; content-visibility: auto with contain-intrinsic-size on log entries and day groups.
- Live status table at the top; every Log entry appended, dated; 'log current through' updated on every republish.
- Descriptive names only; legacy labels in small type only where needed.

Rebuild by editing the reference (or refresh_build.py), republishing to both URLs, and reading back to verify.
