"""scripts/train_tier0_harvest.py with the velocity and recurrent-field inputs of the tier 0 removed (needs scripts/apply_velocity_recurrent_input_removal.py
applied to scatter_field.py). Same recipe and flags; the saved checkpoint args carry blind=True so loaders can restore the switches."""
import sys

from scripts import train_tier0_ext as tte
from scripts import train_tier0_harvest as th

_build, _save = tte.build, tte.save


def build(args, dev):
    model = _build(args, dev)
    model.vblind = model.norec = True
    return model


def save(path, model, opt, it, a):
    a.blind = True
    return _save(path, model, opt, it, a)


tte.build, tte.save = build, save

if __name__ == "__main__":
    th.main()
