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
    per = {}
    for i in range(n):
        photos_i = unit(centers[i] + rng.normal(scale=noise, size=(photos, 512)))
        far = unit(photos_i + rng.normal(scale=noise * 2, size=(photos, 512)))       # stand-in for the far versions
        per[str(1000 + i)] = [np.stack([p, f]) for p, f in zip(photos_i, far)]
    g = Gallery.build(per, {})
    g.calibrate(3, 0.55)
    return g, centers


class Decisions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g, cls.c = school()

    def probe(self, i, noise=0.03):
        return unit(self.c[i] + rng.normal(scale=noise, size=512))

    def run_track(self, frames, live=0.95, cfg=None, width=120):
        f = Fusion(cfg or Config())
        r = None
        for emb in frames:
            r = f.update("t", emb, 1.0, width, live, self.g)
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
        self.assertEqual(f.update("t", self.probe(5), 1.0, 120, 0.95, self.g)["status"], "pending")

    def test_stranger_is_unknown(self):
        stranger = unit(rng.normal(size=512))
        r = self.run_track([unit(stranger + rng.normal(scale=0.03, size=512)) for _ in range(10)])
        self.assertEqual(r["status"], "unknown")

    def test_lookalikes_are_flagged(self):
        self.assertIn("1001", self.g.lookalikes.get("1000", []))

    def test_ambiguous_between_lookalikes_goes_to_review(self):
        mid = unit(self.c[0] + self.c[1])
        r = self.run_track([unit(mid + rng.normal(scale=0.01, size=512)) for _ in range(20)])
        self.assertEqual(r["status"], "review")

    def test_photo_attack_is_spoof(self):
        r = self.run_track([self.probe(7) for _ in range(10)], live=0.1)
        self.assertEqual(r["status"], "spoof")

    def test_threshold_is_calibrated_and_sane(self):
        self.assertEqual(self.g.calib["source"], "calibrated")
        for near in (True, False):
            th = self.g.thresholds(near)
            self.assertGreater(th["t_accept"], th["t_review"])
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


    def test_far_face_is_recognised_but_waits_for_liveness(self):
        r = self.run_track([self.probe(11) for _ in range(30)], live=None, width=36)
        self.assertEqual(r["status"], "pending")
        self.assertEqual(r["student_id"], "1011")
        self.assertEqual(r["range"], "far")
        self.assertIn("closer", r["reason"])

    def test_far_then_near_is_matched(self):
        f = Fusion(Config())
        for _ in range(6):
            r = f.update("t", self.probe(12), 0.3, 36, None, self.g)
        for _ in range(3):
            r = f.update("t", self.probe(12), 1.0, 90, 0.95, self.g)
            if r["status"] != "pending":
                break
        self.assertEqual((r["status"], r["student_id"]), ("match", "1012"))

    def test_identity_switch_restarts_the_track(self):
        f = Fusion(Config())
        f.update("t", self.probe(20), 1.0, 120, 0.95, self.g)
        f.update("t", self.probe(20), 1.0, 120, 0.95, self.g)
        r = f.update("t", self.probe(21), 1.0, 120, 0.95, self.g)      # tracker swapped people
        self.assertEqual(r["frames"], 1)
        for _ in range(3):
            r = f.update("t", self.probe(21), 1.0, 120, 0.95, self.g)
        self.assertEqual((r["status"], r["student_id"]), ("match", "1021"))

    def test_four_students_at_once(self):
        f = Fusion(Config())
        done = {}
        for _ in range(4):
            for i in (30, 31, 32, 33):
                r = f.update(f"t{i}", self.probe(i), 1.0, 100, 0.95, self.g)
                if r["status"] == "match":
                    done[f"t{i}"] = r["student_id"]
        self.assertEqual(done, {"t30": "1030", "t31": "1031", "t32": "1032", "t33": "1033"})

    def test_confident_match_teaches_the_gallery(self):
        f = Fusion(Config())
        learned = None
        for _ in range(3):
            r = f.update("t", self.probe(40, noise=0.01), 1.0, 120, 0.95, self.g)
            learned = r.get("_learn", learned)
        self.assertEqual(r["status"], "match")
        self.assertIsNotNone(learned)
        g = self.g
        g.add_adaptive("1040", learned, 5)
        self.assertEqual(int((g.adapt_owner == g.index["1040"]).sum()), 1)
        for _ in range(7):
            g.add_adaptive("1040", self.probe(40), 5)
        self.assertEqual(int((g.adapt_owner == g.index["1040"]).sum()), 5)     # only the newest few are kept
        self.assertEqual(g.match(self.probe(40))[0][0], "1040")


class Tracking(unittest.TestCase):
    def test_four_people_keep_their_ids(self):
        from tracker import Tracker
        t = Tracker()
        boxes = [np.array([x, 100, x + 60, 170], np.float32) for x in (100, 400, 700, 1000)]
        first = t.assign(boxes)
        self.assertEqual(len(set(first)), 4)
        moved = [b + np.array([15, 10, 18, 14], np.float32) for b in boxes]     # everyone walks a bit closer
        self.assertEqual(t.assign(moved[::-1]), first[::-1])

    def test_new_person_gets_new_id(self):
        from tracker import Tracker
        t = Tracker()
        a = t.assign([np.array([100, 100, 160, 170], np.float32)])
        b = t.assign([np.array([100, 100, 160, 170], np.float32), np.array([900, 100, 940, 150], np.float32)])
        self.assertEqual(b[0], a[0])
        self.assertNotEqual(b[1], a[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
