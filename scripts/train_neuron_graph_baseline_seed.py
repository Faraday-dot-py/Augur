import argparse
import os

import torch

from scripts.train_neuron_graph_smoke import make_train_data, train_arm

ARM = "full-synaptic-messages"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="results/neuron_extra_training_seeds/baseline")
    p.add_argument("--stage1-iters", type=int, default=1500)
    p.add_argument("--stage2-iters", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--circuits-per-size", type=int, default=64)
    args = p.parse_args()
    device = torch.device(args.device)
    out = os.path.join(args.out_dir, f"seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{ARM}.pt")
    if os.path.exists(path):
        print(f"{ARM} seed {args.seed} already trained, skipping", flush=True)
        return
    data = make_train_data(args.seed, device, args.circuits_per_size, 2, 300)
    model = train_arm(ARM, data, args.seed, args, device, lambda m: print(m, flush=True))
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    main()
