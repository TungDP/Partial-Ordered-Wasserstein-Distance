from src.experiments.step_localization.evaluate import main as evaluate_main
from argparse import Namespace
import pytest

def test_evaluate_main_POW():

    args = Namespace(
        algorithm="POW",
        dataset="COIN",
        drop_cost="logit",
        keep_percentile=0.3,
        name="",
        reg=3,
        use_unlabeled=True,
        metric="cosine",
    )

    result = evaluate_main(args)
    assert result['accuracy'] == pytest.approx(46.8, 0.1)
    assert result['iou'] == pytest.approx(14.4, 0.1)

def test_evaluate_main_DropDTW():

    args = Namespace(
        algorithm="DropDTW",
        dataset="COIN",
        metric="inner",
        drop_cost="logit",
        keep_percentile=0.3,
        name="",
        use_unlabeled=True,
    )

    result = evaluate_main(args)
    assert result['accuracy'] == pytest.approx(51.2, 0.1)
    assert result['iou'] == pytest.approx(23.6, 0.1)
