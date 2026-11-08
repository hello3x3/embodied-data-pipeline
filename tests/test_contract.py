import unittest

import numpy as np

from data_pipeline import config
from data_pipeline.task_timeline import (find_cycles, _ensure_min_transitions,
                                         _merge_short_stops)


class TestContract(unittest.TestCase):
    def test_contract(self):
        self.assertEqual(config.POINT_STRIDE, 4)
        self.assertEqual(config.NEAR_POINTS, 7)
        self.assertEqual(config.ACTION_HORIZON, 25)
        self.assertEqual(config.EXEC_STEPS, 12)
        self.assertEqual(config.FAR_POINTS, 8)

    def test_find_cycles_synthetic(self):
        g = np.ones(100)
        g[20:40] = 0.0
        g[60:80] = 0.0
        self.assertEqual(find_cycles(g), [(20, 40), (60, 80)])

    def test_min_transitions_keep_place_end(self):
        phases = [("reach", 0, 100), ("pick", 101, 106),
                  ("move", 107, 130), ("place", 131, 136)]
        out = _ensure_min_transitions(phases, 25)
        d = {n: (a, b) for n, a, b in out}
        self.assertGreaterEqual(d["pick"][1] - d["pick"][0] + 1, 25)
        self.assertGreaterEqual(d["place"][1] - d["place"][0] + 1, 25)
        # place still ends at fully_open (the original right boundary)
        self.assertEqual(d["place"][1], 136)

    def test_short_stop_is_merged(self):
        phases = [("reach", 0, 100), ("stop", 101, 105), ("move", 106, 200)]
        out = _merge_short_stops(phases)
        self.assertTrue(all(not (n == "stop" and b - a + 1 < 25)
                            for n, a, b in out))
        self.assertEqual(out[0][0], "reach")
        self.assertEqual(out[-1][0], "move")


if __name__ == "__main__":
    unittest.main()
