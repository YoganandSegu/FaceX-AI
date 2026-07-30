import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("main", ROOT / "main.py")
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)


class ReIDTests(unittest.TestCase):
    def test_guest_id_is_stable_for_the_same_person_across_emotions(self):
        guest_map = {}
        next_guest_id = 1

        first_guest, next_guest_id = main.resolve_guest_id(7, guest_map, next_guest_id)
        second_guest, next_guest_id = main.resolve_guest_id(7, guest_map, next_guest_id)

        self.assertEqual(first_guest, "Guest 1")
        self.assertEqual(second_guest, "Guest 1")
        self.assertEqual(next_guest_id, 2)


if __name__ == "__main__":
    unittest.main()
