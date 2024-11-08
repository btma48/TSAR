#!/usr/bin/env bash
python -m torch.distributed.launch --nproc_per_node=4  --master_port=4324 basicsr/train_hformer.py  --opt ./OCT/Options/OCTA.yml --launcher pytorch

