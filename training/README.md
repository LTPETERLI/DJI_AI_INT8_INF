# Training Record

The packaged files are the exact YOLO26n training configuration and metric log
from `train46-yolo26n-sangu`.

Key settings:

- model: external `yolo26n.pt`
- data: external `loose_brand_yolo/dataset.yaml`
- input: 1280x1280
- epochs: 100
- batch: 2
- GPU: device 7 on the training host
- AMP: enabled

Best validation result occurs at epoch 74:

- mAP50: 0.93801
- mAP50-95: 0.66364
- Precision: 0.91737
- Recall: 0.92661

The full 100-epoch history is in `yolo26n_train46_results.csv`.

No training entry script was found in the original workspace. Reconstruct the
training invocation from `yolo26n_train46_args.yaml`; do not silently reuse
those paths on another machine.

