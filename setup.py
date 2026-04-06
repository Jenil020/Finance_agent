"""
setup.py — AI Investment Analyst dependency installer & environment verifier
Run: python setup.py
"""

import subprocess
import sys
import os
from pathlib import Path

# ── Dependencies ───────────────────────────────────────────────────────────────
# All free-tier compatible. No paid APIs except Google Gemini free quota.

DEPENDENCIES = [
    # ── Web Framework ──────────────────────────────────────
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.29.0",
    "python-multipart>=0.0.9",

    # ── LLM — Google Gemini (free API) ────────────────────
    # NOTE: google-generativeai is DEPRECATED. Use google-genai instead.
    "google-genai>=1.0.0",

    # ── LangGraph — Agent Orchestration ───────────────────
    "langgraph>=0.2.0",
    "langgraph-checkpoint-sqlite>=1.0.0",  # SQLite state persistence
    "langchain>=0.2.0",
    "langchain-google-genai>=1.0.5",
    "langchain-community>=0.2.0",

    # ── LlamaIndex — RAG Pipeline ─────────────────────────
    "llama-index>=0.10.50",
    "llama-index-vector-stores-qdrant>=0.2.8",
    "llama-index-embeddings-gemini>=0.1.7",  # still works but wraps old SDK
    "llama-index-retrievers-bm25>=0.1.3",

    # ── Vector DB — Qdrant local (free, no cloud) ─────────
    "qdrant-client>=1.9.0",

    # ── Memory — Redis ────────────────────────────────────
    "redis>=5.0.4",

    # ── Data Validation ───────────────────────────────────
    "pydantic>=2.7.0",
    "pydantic-settings>=2.2.0",

    # ── Web Search Tool (free, no API key) ────────────────
    "duckduckgo-search>=6.1.0",

    # ── Retry logic ───────────────────────────────────────
    "tenacity>=8.3.0",

    # ── Observability — LangSmith (free tier) ─────────────
    "langsmith>=0.1.75",

    # ── Document loaders ──────────────────────────────────
    "pypdf>=4.2.0",
    "python-docx>=1.1.0",
    "pandas>=2.2.0",

    # ── Async HTTP ────────────────────────────────────────
    "httpx>=0.27.0",
    "aiohttp>=3.9.5",

    # ── Environment ───────────────────────────────────────
    "python-dotenv>=1.0.1",

    # ── Dev / Testing ─────────────────────────────────────
    "pytest>=8.2.0",
    "pytest-asyncio>=0.23.7",
    "pytest-mock>=3.14.0",
    "ruff>=0.4.7",
]

# ── Env checks ────────────────────────────────────────────────────────────────
REQUIRED_ENV_VARS = {
    "GOOGLE_API_KEY": "Required for Gemini LLM + embeddings (free tier)",
}

OPTIONAL_ENV_VARS = {
    "REDIS_URL":          "Redis for memory/cache (defaults to localhost:6379)",
    "LANGCHAIN_API_KEY":  "LangSmith tracing — free tier available at smith.langchain.com",
}

FREE_TIER_LINKS = {
    "Google Gemini API": "https://aistudio.google.com/app/apikey",
    "LangSmith":         "https://smith.langchain.com (free: 5k traces/month)",
    "Upstash Redis":     "https://upstash.com (free: 10k req/day)",
}


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=False)


def pip_install(packages: list[str]) -> None:
    print("\n📦 Installing dependencies...\n" + "─" * 50)
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q",
         "--break-system-packages"])
    # Install in batches to show progress cleanly
    batch_size = 5
    for i in range(0, len(packages), batch_size):
        batch = packages[i : i + batch_size]
        print(f"  Installing: {', '.join(p.split('>=')[0] for p in batch)}")
        run([sys.executable, "-m", "pip", "install"] + batch + ["-q",
            "--break-system-packages"])
    print("\n✅ All dependencies installed!")


def check_env() -> tuple[list, list]:
    """Check .env file and environment variables. Returns (missing, warnings)."""
    missing = []
    warnings = []

    # Try loading .env
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
        print("\n  ✅ .env file found and loaded")
    else:
        warnings.append(".env file not found — copy .env.example and fill in your keys")
        print("\n  ⚠️  .env file not found — run: cp .env.example .env")

    # Check required vars
    for var, desc in REQUIRED_ENV_VARS.items():
        val = os.getenv(var, "")
        if not val or val.startswith("your_"):
            missing.append((var, desc))
            print(f"  ❌ {var} — NOT SET ({desc})")
        else:
            print(f"  ✅ {var} — set ({val[:8]}...)")

    # Check optional vars
    for var, desc in OPTIONAL_ENV_VARS.items():
        val = os.getenv(var, "")
        if not val or val.startswith("your_"):
            warnings.append(f"{var} not set — {desc}")
            print(f"  ⚠️  {var} — not set (optional: {desc})")
        else:
            print(f"  ✅ {var} — set")

    return missing, warnings


def create_data_dirs() -> None:
    """Create local data directories needed at runtime."""
    dirs = [
        Path("data/qdrant"),
        Path("data/sample_docs"),
        Path("data"),   # for checkpoints.db (SQLite)
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"\n  ✅ Data directories ready")


def verify_imports() -> list[str]:
    """Try importing key packages and report failures."""
    checks = [
        ("fastapi",                          "FastAPI"),
        ("langgraph",                        "LangGraph"),
        ("langgraph.checkpoint.sqlite",      "LangGraph SQLite checkpointer"),
        ("llama_index.core",                 "LlamaIndex core"),
        ("llama_index.vector_stores.qdrant", "LlamaIndex Qdrant store"),
        ("llama_index.retrievers.bm25",      "LlamaIndex BM25 retriever"),
        ("qdrant_client",                    "Qdrant client"),
        ("redis",                            "Redis client"),
        ("google.genai",                     "Google Genai (Gemini)"),
        ("duckduckgo_search",                "DuckDuckGo Search"),
        ("tenacity",                         "Tenacity (retry)"),
        ("pydantic",                         "Pydantic v2"),
        ("pydantic_settings",                "Pydantic Settings"),
    ]
    failed = []
    for module, label in checks:
        try:
            __import__(module)
            print(f"  ✅ {label}")
        except ImportError as e:
            print(f"  ❌ {label} — {e}")
            failed.append(label)
    return failed


def print_free_tier_info() -> None:
    print("\n🆓 Free resources needed:\n" + "─" * 50)
    for service, url in FREE_TIER_LINKS.items():
        print(f"  • {service}: {url}")


def print_next_steps(missing_env: list) -> None:
    print("\n🚀 Next steps:\n" + "─" * 50)
    step = 1

    if missing_env:
        print(f"  {step}. Set missing env vars in .env:")
        for var, _ in missing_env:
            print(f"       {var}=<your_value>")
        step += 1

    print(f"  {step}. Run the project structure generator (if not done):")
    print(f"       python template.py")
    step += 1

    print(f"  {step}. (Optional) Ingest sample documents:")
    print(f"       python scripts/ingest_sample.py")
    step += 1

    print(f"  {step}. Start the API server:")
    print(f"       uvicorn app.main:app --reload --port 8000")
    step += 1

    print(f"  {step}. Run tests:")
    print(f"       pytest tests/ -v")
    step += 1

    print(f"  {step}. Smoke test the agents:")
    print(f"       python scripts/test_chat.py")

    print("\n" + "─" * 50)


def main() -> None:
    print("=" * 50)
    print("  AI Investment Analyst — Setup Script")
    print("=" * 50)

    # 1. Install deps
    pip_install(DEPENDENCIES)

    # 2. Create runtime dirs
    print("\n📁 Creating runtime directories...")
    create_data_dirs()

    # 3. Verify imports
    print("\n🔍 Verifying package imports...\n" + "─" * 50)
    failed_imports = verify_imports()

    # 4. Check environment
    print("\n🔑 Checking environment variables...\n" + "─" * 50)
    missing_env, env_warnings = check_env()

    # 5. Print free tier info
    print_free_tier_info()

    # 6. Summary
    print("\n📋 Setup Summary\n" + "─" * 50)
    if not failed_imports and not missing_env:
        print("  🎉 Everything looks good — ready to run!")
    else:
        if failed_imports:
            print(f"  ❌ Failed imports: {failed_imports}")
        if missing_env:
            print(f"  ❌ Missing env vars: {[v for v, _ in missing_env]}")
        if env_warnings:
            for w in env_warnings:
                print(f"  ⚠️  {w}")

    print_next_steps(missing_env)


if __name__ == "__main__":
    main()