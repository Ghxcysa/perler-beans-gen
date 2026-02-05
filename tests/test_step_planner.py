import numpy as np

from perler_gen.step_planner import plan_steps


def _coverage_ok(masks: list[np.ndarray]) -> bool:
    stacked = np.stack(masks, axis=0).astype(int)
    summed = stacked.sum(axis=0)
    return bool((summed == 1).all())


def test_row_steps_cover_grid_once():
    w, h = 6, 5
    steps = plan_steps(w, h, mode="row", rows_per_step=2)
    masks = [s.mask for s in steps]
    assert _coverage_ok(masks)


def test_quadrant_steps_cover_grid_once():
    w, h = 7, 5
    steps = plan_steps(w, h, mode="quadrant", rows_per_step=1)
    masks = [s.mask for s in steps]
    assert _coverage_ok(masks)
