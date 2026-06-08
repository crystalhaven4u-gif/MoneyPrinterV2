import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))

import produce_video as pv


class RunLockTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # Point the lock dir at a temp location so tests don't touch .mp/locks.
        self._patch = patch.object(pv, "LOCK_DIR", self._tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_acquire_release_roundtrip(self):
        lock = pv._RunLock("run-a")
        self.assertTrue(lock.acquire())
        self.assertTrue(os.path.exists(lock.path))
        lock.release()
        self.assertFalse(os.path.exists(lock.path))

    def test_live_other_process_blocks(self):
        lock = pv._RunLock("run-b")
        # Simulate a different, still-running owner.
        os.makedirs(self._tmp.name, exist_ok=True)
        with open(lock.path, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid() + 1))
        with patch.object(pv, "_pid_alive", return_value=True):
            self.assertFalse(lock.acquire())
        self.assertTrue(os.path.exists(lock.path))  # not removed

    def test_stale_lock_is_reclaimed(self):
        lock = pv._RunLock("run-c")
        os.makedirs(self._tmp.name, exist_ok=True)
        with open(lock.path, "w", encoding="utf-8") as handle:
            handle.write("999999")  # a PID we treat as dead
        with patch.object(pv, "_pid_alive", return_value=False):
            self.assertTrue(lock.acquire())
        lock.release()

    def test_own_pid_lock_is_reclaimable(self):
        # A leftover lock owned by THIS pid should not block (own stale lock).
        lock = pv._RunLock("run-d")
        os.makedirs(self._tmp.name, exist_ok=True)
        with open(lock.path, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        self.assertTrue(lock.acquire())
        lock.release()


if __name__ == "__main__":
    unittest.main()
