# Supervisor log page standard

`supervisor-log-page-reference.html` is the ONE canonical file for the published supervisor-log artifact (https://claude.ai/artifact/Ub74UinSTQu35kU4jAYbCG and https://claude.ai/artifact/7FQpySdYjbec7QLWURg8Ju). It is tracked in git and is updated and committed on every republish: publish, copy the exact published HTML over this file, commit it with any log changes. No extra copies or versioned filenames; git history is the version history. Source of content: `docs/theory/supervisor-log.md`. `refresh_build.py` is the older one-shot build script, kept for reference.

Keep when updating:
- Liquid-glass-on-CRT look: dull red/green/yellow/blue palette, static fog, ASCII-art titles, no box borders, straight square-ended rules, 250% root size, soft letter halo. No script, no animation.
- Performance (user reported scroll lag): at most 3 text-shadow layers on headings and labels, 2 on body and log text; no animated fog; no background-attachment: fixed; one scanline gradient, no RGB stripe layer, no blurred inset box-shadow; content-visibility: auto with contain-intrinsic-size on log entries and day groups.
- Live status table at the top (columns Agent, State, Priority, Doing; Priority = rating name and number from docs/debugging/priority-registry.md, "-" when finished); every Log entry appended, dated; 'log current through' updated on every republish.
- Descriptive names only; legacy labels in small type only where needed.

Republish to both URLs and read back to verify (compare sha256 of the readback with the file).
