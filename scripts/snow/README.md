# Snow Scripts

Directory layout:

- `data/`: dataset preparation launchers
- `fineweb/`: FineWeb experiments
  - `pretrain/<size>/`
  - `finetune/<size>/<dataset>/`
  - `eval/<size>/<dataset>/`
- `wiki/`: Wiki experiments (`pretrain/<size>/`)
- `wmdp/`: WMDP experiments (`finetune/<size>/`)

Naming convention inside leaf folders:

- use short names like `run.sh`, `run_multi.sh`, `run_kl0.sh`
- do not repeat size or dataset in file names (already encoded by folders)
