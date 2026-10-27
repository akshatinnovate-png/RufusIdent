"""Decision-logic tests (no models or photos needed).

    python tests.py
"""
import unittest

import numpy as np

from config import Config
from fusion import Fusion
from gallery import Gallery

rng = np.random.default_rng(42)


def unit(v):
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def school(n=300, photos=20, noise=0.03):
    centers = unit(rng.normal(size=(n, 512)))
    centers[1] = unit(centers[0] * 0.85 + centers[1] * 0.15)       # 1 is a look-alike of 0
    per = {str(1000 + i): list(unit(centers[i] + rng.normal(scale=noise, size=(photos, 512)))) for i in range(n)}
    g = Gallery.build(per, {})
    g.calibrate(3, 0.55)
    return g, centers


class Decisions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g, cls.c = school()

    def probe(self, i, noise=0.03):
        return unit(self.c[i] + rng.normal(scale=noise, size=512))

    def run_track(self, frames, live=0.95, cfg=None):
        f = Fusion(cfg or Config())
        r = None
        for emb in frames:
            r = f.update("t", emb, 1.0, live, self.g)
            if r["status"] != "pending":
                break
        return r

    def test_enrolled_student_is_matched_after_min_frames(self):
        r = self.run_track([self.probe(5) for _ in range(10)])
        self.assertEqual(r["status"], "match")
        self.assertEqual(r["student_id"], "1005")
        self.assertEqual(r["frames"], 3)

    def test_single_frame_never_marks(self):
        f = Fusion(Config())
        self.assertEqual(f.update("t", self.probe(5), 1.0, 0.95, self.g)["status"], "pending")

    def test_stranger_is_unknown(self):
        stranger = unit(rng.normal(size=512))
        r = self.run_track([unit(stranger + rng.normal(scale=0.03, size=512)) for _ in range(10)])
        self.assertEqual(r["status"], "unknown")

    def test_lookalikes_are_flagged(self):
        self.assertIn("1001", self.g.lookalikes.get("1000", []))

    def test_ambiguous_between_lookalikes_goes_to_review(self):
        mid = unit(self.c[0] + self.c[1])
        r = self.run_track([unit(mid + rng.normal(scale=0.01, size=512)) for _ in range(12)])
        self.assertEqual(r["status"], "review")

    def test_photo_attack_is_spoof(self):
        r = self.run_track([self.probe(7) for _ in range(10)], live=0.1)
        self.assertEqual(r["status"], "spoof")

    def test_threshold_is_calibrated_and_sane(self):
        self.assertEqual(self.g.calib["source"], "calibrated")
        self.assertGreater(self.g.t_accept, self.g.t_review)
        self.assertGreater(self.g.probability(0.7), 0.99)
        self.assertLess(self.g.probability(0.1), 0.01)

    def test_save_and_load_round_trip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.g.save(d)
            g2 = Gallery.load(d)
        q = self.probe(9)
        self.assertEqual(self.g.match(q)[0][0], g2.match(q)[0][0])
        self.assertAlmostEqual(self.g.match(q)[0][1], g2.match(q)[0][1], places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
