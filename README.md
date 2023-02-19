# DriveTorch

PyTorch Module wrapper for low-memory inference.

---

## Planned Functionality

- DriveTensor (class): Drop-in replacement for a Pytorch Tensor or Parameter
  object but data is stored with zarr and loaded when needed.
- wrap (function): Converts params in PyTorch modules to DriveTensors.
  Calls store if nothing found at path location.
- trace(class): Traces a model and records which params are loaded in which
  order. Returns wrapper that loads parameters ahead of time. Should also
  accept a n_threads or pre_load argument for specifying how many parameters
  to load ahead of time.
- store (function): Converts any unwrapped parameters to DriveTensors then
  saves the model.
- load (function): Load a wrapped model saved with "store"
- hf_automodel (function): Reduced memory version of HuggingFace's Automodel 
  that loads models with DriveTensors.

