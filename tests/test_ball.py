import numpy as np
import pytest

from src.detection.ball import (ball_recall, interpolate_gaps, postprocess_trajectory,
                                reject_outliers, select_ball, slice_boxes, smooth_median)


def test_slice_boxes_cover_image_and_stay_inside():
    boxes = slice_boxes(1920, 1080, (640, 640), 0.2)
    assert all(0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080 for x1, y1, x2, y2 in boxes)
    cover = np.zeros((1080, 1920), bool)
    for x1, y1, x2, y2 in boxes:
        cover[y1:y2, x1:x2] = True
    assert cover.all()


def test_slice_boxes_small_image_single_slice():
    assert slice_boxes(300, 200, (640, 640), 0.2) == [(0, 0, 300, 200)]


def test_slice_invalid_overlap():
    with pytest.raises(ValueError):
        slice_boxes(100, 100, (50, 50), 1.0)


def test_select_ball_prefers_confident_but_plausible():
    c = np.array([[100, 100, 0.9], [200, 100, 0.4]])
    assert select_ball(c, None, 50)[2] == 0.9
    # a de 0.9 esta longe do ponto anterior -> escolhe a plausivel
    assert select_ball(c, np.array([195, 100]), 20)[2] == 0.4
    assert select_ball(c, np.array([500, 500]), 20) is None
    assert select_ball(np.empty((0, 3)), None, 50) is None


def test_interpolate_only_short_gaps():
    t = np.full((10, 2), np.nan)
    t[0], t[3], t[9] = (0, 0), (3, 6), (9, 0)
    out, mask = interpolate_gaps(t, max_gap=2)
    assert np.allclose(out[1], (1, 2)) and mask[1] and mask[2]
    assert np.isnan(out[5]).all() and not mask[5]  # buraco de 5 > 2


def test_reject_outliers_and_smooth():
    t = np.array([[0, 0], [1, 0], [500, 500], [3, 0], [4, 0]], float)
    out = reject_outliers(t, 50)
    assert np.isnan(out[2]).all()
    s = smooth_median(np.array([[0, 0], [100, 100], [2, 2]], float), 3)
    assert np.allclose(s[1], (2, 2))


def test_postprocess_and_recall():
    raw = np.array([[0, 0], [np.nan] * 2, [2, 0], [3, 0]], float)
    post, interp = postprocess_trajectory(raw, 50, 3, 1)
    assert interp[1] and np.allclose(post[1], (1, 0))
    gt = np.array([[0, 0], [1, 0], [2, 0], [np.nan] * 2], float)
    assert ball_recall(raw, gt) == pytest.approx(2 / 3)
    assert ball_recall(post, gt) == 1.0
