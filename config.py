import os

# The single edge where environment is read: .env.defaults (base), then .env (user overrides).
try:
    from dotenv import load_dotenv

    load_dotenv(".env.defaults", override=False)
    if os.path.exists(".env"):
        load_dotenv(".env", override=True)
except ImportError:
    pass

if "LLAMA_BASE_URL" not in os.environ:
    os.environ["LLAMA_BASE_URL"] = "http://127.0.0.1:8080"

# Runtime source of truth for the model server URL; /server reassigns this.
BASE_URL = os.environ["LLAMA_BASE_URL"]
CURRENT_MODEL = os.environ.get("CURRENT_MODEL", "Qwen_Qwen3.5-27B-Q4_1")
API_KEY = os.environ.get("LLAMA_API_KEY", "")


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
