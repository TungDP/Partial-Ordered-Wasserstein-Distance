#! /bin/bash
for reg in $(seq 0.2 0.02 0.4); do
    # for kp in $(seq 0.3 0.02 0.4); do
    #     python -m src.experiments.step_localization.evaluate  --algorithm=POW --keep_percentile $kp --reg $reg --reg2 0 --use_unlabeled
    # done
    # # python -m src.experiments.step_localization.evaluate  --algorithm=POW-reg --keep_percentile 0.36 --reg 0.25 --reg2 $reg --use_unlabeled --metric cosine
    python -m src.experiments.step_localization.evaluate --algorithm=POW-reg --keep_percentile 0.38 --reg 0.39 --reg2 $reg --use_unlabeled --metric cosine
    # python -m src.experiments.step_localization.evaluate  --algorithm=POW --keep_percentile 0.4 --reg $reg --reg2 0 --use_unlabeled
done
