import json
import pathlib

# Last-resort built-ins; config.defaults.json (repo) and config.json (local, untracked) override them.
_DEFAULTS = {
    "client": "terminal",
    "server": {"url": "", "api_key": "", "model": ""},
    "compaction": {"tail_turns": 10, "trigger_pct": 80.0, "target_pct": 30.0},
}


def _find_root() -> pathlib.Path:
    # cwd when it holds a config file (pip-installed runs), else the source tree.
    cwd = pathlib.Path.cwd()
    if (cwd / "config.defaults.json").exists() or (cwd / "config.json").exists():
        return cwd
    return pathlib.Path(__file__).parent.parent  # project root (this file lives in core/)


ROOT_DIR = _find_root()


def _load() -> dict:
    """Built-ins <- config.defaults.json (repo) <- config.json (local, wins)."""
    data = json.loads(json.dumps(_DEFAULTS))
    for name in ("config.defaults.json", "config.json"):
        path = ROOT_DIR / name
        if path.exists():
            with open(path) as fh:
                user = json.load(fh)
            for key, value in user.items():
                if isinstance(value, dict) and isinstance(data.get(key), dict):
                    data[key].update(value)
                else:
                    data[key] = value
    return data


_data = _load()

CLIENT = _data["client"]
BASE_URL = _data["server"]["url"]
API_KEY = _data["server"]["api_key"]
CURRENT_MODEL = _data["server"]["model"]
COMPACTION_TAIL_TURNS = _data["compaction"]["tail_turns"]
COMPACTION_TRIGGER_PCT = _data["compaction"]["trigger_pct"]
COMPACTION_TARGET_PCT = _data["compaction"]["target_pct"]


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
