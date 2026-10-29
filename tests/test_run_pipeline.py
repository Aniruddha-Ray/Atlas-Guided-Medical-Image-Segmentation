import numpy as np
import pytest

import run
from scripts import validate_priors as vp


def test_quota_sums_to_size_and_keeps_every_source():
    for n in (6, 12, 24, 48, 100):
        for shares in (run.TRAIN_SHARE, run.VAL_SHARE):
            q = run.proportional_quota(n, shares)
            assert sum(q.values()) == n and q["img"] >= 1 and q["s"] >= 1 and q["amos"] >= 0
    assert run.proportional_quota(48, run.TRAIN_SHARE) == {"amos": 19, "img": 2, "s": 27}
    assert run.proportional_quota(24, run.VAL_SHARE) == {"amos": 10, "img": 1, "s": 13}


def test_quota_rejects_a_chunk_too_small_for_every_source():
    with pytest.raises(ValueError):
        run.proportional_quota(1, run.TRAIN_SHARE)


def test_estimate_arithmetic_and_monotonicity():
    e = run.estimate(n_train=48, n_val=24, to_register=60, epochs=200)
    assert e["registration"] == pytest.approx(60 * run.REG_S_PER_TARGET / 3600)
    per_epoch = 48 * run.TRAIN_S_PER_CASE + 24 * run.VAL_S_PER_CASE * 1.2
    assert e["training"] == pytest.approx(200 * per_epoch / 3600)
    assert e["total"] == pytest.approx(e["registration"] + e["validation"] + e["sample cache"] + e["training"])
    assert run.estimate(48, 24, 60, 400)["training"] == pytest.approx(2 * e["training"])
    assert run.estimate(48, 24, 0, 200)["registration"] == 0
    assert run.estimate(48, 24, 60, 200, physics=True)["training"] > e["training"]


def test_parse_args_defaults_and_stage_validation():
    a = run.parse_args(["--mode", "chunk", "--train-size", "48", "--val-size", "24", "--epochs", "200"])
    assert a.run_name == "chunk_48_24_e200" and a.epochs == 200 and a.stages.split(",") == run.STAGES
    assert run.parse_args(["--mode", "full", "--epochs", "5"]).run_name == "full_e5"
    with pytest.raises(SystemExit):
        run.parse_args(["--stages", "register,nonsense"])


def test_fmt_hours():
    assert run.fmt_hours(0.5) == "30 min" and run.fmt_hours(2.25) == "2.2 h"


def _good_priors(shape=(8, 8, 8)):
    prob = np.zeros((13,) + shape, np.float32)
    prob[0] = 1.0
    prob[1, :4] = 1.0
    prob[0, :4] = 0.0
    dist = np.ones((12,) + shape, np.float32)
    dist[0, :4] = 0.0  # organ 1: D = 0 on its region, which is P_1 >= 0.5 * max
    dist[0, 4] = 0.2
    conf = np.ones((1,) + shape, np.float32)
    return prob, dist, conf


def test_invariants_pass_on_valid_priors():
    assert vp.check_invariants(*_good_priors()) == []


def test_invariants_catch_each_kind_of_corruption():
    prob, dist, conf = _good_priors()
    bad_sum = prob.copy(); bad_sum[2] += 0.1
    assert any("sum to 1" in m for m in vp.check_invariants(bad_sum, dist, conf))
    nan = dist.copy(); nan[3, 0, 0, 0] = np.nan
    assert any("non-finite" in m for m in vp.check_invariants(prob, nan, conf))
    big_d = dist.copy(); big_d[5] = 2.0
    assert any("D outside" in m for m in vp.check_invariants(prob, big_d, conf))
    wrong_region = dist.copy(); wrong_region[0, 0] = 0.3  # D > 0 inside organ 1's own region
    assert any("organ 1: D is not 0" in m for m in vp.check_invariants(prob, wrong_region, conf))
    missing_but_near = dist.copy(); missing_but_near[6, 0, 0, 0] = 0.5  # nobody votes organ 7, so D must be 1 everywhere
    assert any("organ 7" in m for m in vp.check_invariants(prob, missing_but_near, conf))
    assert any("channel counts" in m for m in vp.check_invariants(prob[:12], dist, conf))


def test_dice_centroid_and_support_recall_known_answers():
    gt = np.zeros((10, 10, 10), int); gt[2:6, 2:6, 2:6] = 1
    assert vp.dice_per_organ(gt, gt)[0] == pytest.approx(1.0)
    shifted = np.zeros_like(gt); shifted[3:7, 2:6, 2:6] = 1
    assert vp.dice_per_organ(shifted, gt)[0] == pytest.approx(2 * 48 / (64 + 64))
    assert np.isnan(vp.dice_per_organ(shifted, gt)[1])  # organ 2 absent from both
    assert vp.centroid_error_mm(shifted, gt, (2.0, 3.0, 1.0)) == pytest.approx(2.0)  # 1 voxel along an axis of 2 mm
    prob = np.zeros((13, 10, 10, 10)); prob[1, 2:6, 2:6, 2:4] = 0.5  # half of the organ is covered
    assert vp.support_recall(prob, gt)[0] == pytest.approx(0.5)
