# POW

## For Developers

### Setup

#### Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements/dev.txt
pre-commit install
```


#### Download datasets

```bash
python -m scripts.download_data --data all
```

## Run POW experiments

* Step localization experiments

Example:

```bash
python -m src.experiments.step_localization.evaluate --algorithm=POW --keep_percentile 0.3 --reg 3 --use_unlabeled
```

Read src/evaluate.py for more details.

* Weizmann classification 1-nn experiments

Example :

```bash
python -m src.experiments.weizmann.knn_eval --test_size 0.5 --outlier_ratio 0.1 --metric pow  --m 0.9 --reg 1 --distance euclidean
```

Read src/experiments/weizmann/knn_eval.py for more details.


* UCR classification k-nn experiments

Example

```bash
python -m src.experiments.ucr.knn_eval --dataset=Chinatown --outlier_ratio 0.1 --metric pow  --m 0.9 --reg 1 --distance euclidean --seed 1
```

* Note : to run softdtw follow instructions in [this repo](https://github.com/mblondel/soft-dtw)



* Sample notebook for POW and POW with segment regularization in [notebook](notebooks/sample.ipynb)
