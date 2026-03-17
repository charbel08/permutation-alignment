# FineWeb Snow Scripts

This folder is grouped by workflow, then model size, then dataset.

- `pretrain/<size>/`: baseline and tiered pretraining launchers
- `finetune/<size>/<dataset>/`: private fine-tuning launchers
- `eval/<size>/<dataset>/`: inference/evaluation launchers

Examples:

- `pretrain/150m/run_multi_cumulative.sh`
- `finetune/150m/fineweb2/run_multi_cumulative.sh`
- `eval/150m/fineweb2/run_clean_prompts_3langs.sh`
