import importlib
import json
import os
import pathlib
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config


class TestConfigLoading(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(config.__file__).parent.parent  # config lives in core/
        self.live = self.root / "config.json"
        self.backup = self.root / "config.json.test-backup"

    def tearDown(self):
        if self.backup.exists():
            self.backup.rename(self.live)
        importlib.reload(config)

    def test_falls_back_to_defaults_json_when_config_json_missing(self):
        moved = False
        try:
            if self.live.exists():
                self.live.rename(self.backup)
                moved = True
            importlib.reload(config)
            defaults = json.loads((self.root / "config.defaults.json").read_text())
            self.assertEqual(config.BASE_URL, defaults["server"]["url"])
            self.assertEqual(config.CURRENT_MODEL, defaults["server"]["model"])
            self.assertEqual(config.CLIENT, defaults.get("client", "terminal"))
            self.assertEqual(config.COMPACTION_TAIL_TURNS, defaults["compaction"]["tail_turns"])
            self.assertEqual(config.COMPACTION_TAIL_TURNS, 10)
        finally:
            if moved:
                self.backup.rename(self.live)
                importlib.reload(config)

    def test_config_json_takes_precedence_over_defaults(self):
        if not self.live.exists():
            self.skipTest("no local config.json in this environment")
        importlib.reload(config)
        live = json.loads(self.live.read_text())
        self.assertEqual(config.BASE_URL, live["server"]["url"])
        self.assertEqual(config.CURRENT_MODEL, live["server"]["model"])


if __name__ == "__main__":
    unittest.main()
