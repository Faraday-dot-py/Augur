# Contact rollout grid review prompt (hypothesis-free)

Render: `python3 scripts/render_scatter_contact.py --family <fam> --tag <tag>` (writes videos/contact_<fam>_<tag>_grid.png), then dispatch a fresh agent with:

---
Read the image at `<path>`. It shows simulated balls moving inside a container (black outlines, possibly with solid obstacles) under a constant sideways pull. Blue is the reference simulation, red is a learned model's rollout of the same scene, shown at several time steps (labelled). Describe plainly what you see: whether red stays near blue and until about which step, whether red balls stay inside the container and avoid obstacles, whether any red balls leave the frame, whether the red pattern looks different in kind from blue (clumped, spread, frozen, scattered), and anything else notable. Do not assume a cause. Report in a short paragraph per image.
---
