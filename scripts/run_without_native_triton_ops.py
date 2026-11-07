import runpy
import sys

import torch._native.registry as registry

registry.deregister_op_overrides(disable_dsl_names="triton")
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
