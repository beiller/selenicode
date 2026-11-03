import json
import pathlib

# Last-resort built-ins; config.defaults.json (repo) and config.json (local, untracked) override them.
_DEFAULTS = {
    "client": "terminal",
    "server": {"url": "", "api_key": "", "model": ""},
    "compaction": {"tail_turns": 6, "trigger_pct": 80.0, "target_pct": 30.0},
}


def _config_path() -> pathlib.Path:
    """Prefer the local config.json; fall back to the repo's config.defaults.json."""
    path = pathlib.Path(__file__).parent / "config.json"
    if not path.exists():
        return pathlib.Path(__file__).parent / "config.defaults.json"
    return path


def _load() -> dict:
    data = json.loads(json.dumps(_DEFAULTS))
    path = _config_path()
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
