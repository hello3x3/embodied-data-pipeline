"""RoboDojo → dense trajectory (6.25 Hz) → action-chunk pipeline.

Contract (v3):
  - trajectory rate: one point every ``POINT_STRIDE=4`` action steps (6.25 Hz);
  - one plan = one primitive (reach/pick/move/place) + one object;
  - chunk = 25 action steps; near = 7 points (steps 0,4,..,24); far = 8 arc points;
  - execute ``min(EXEC_STEPS=12, remaining)`` steps, then replan (cursor in points).

Entry point: ``python -m data_pipeline.build_plan_queue`` (see README.md).
"""
