import json
import pathlib

# Single source of truth: built-in defaults, overridden by config.json (project root).
_DEFAULTS = {
    "client": "terminal",
    "server": {"url": "http://127.0.0.1:8080", "api_key": "", "model": "Qwen_Qwen3.5-27B-Q4_1"},
    "compaction": {"tail_turns": 6, "trigger_pct": 80.0, "target_pct": 30.0},
}


def _load() -> dict:
    data = json.loads(json.dumps(_DEFAULTS))
    path = pathlib.Path(__file__).parent / "config.json"
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
