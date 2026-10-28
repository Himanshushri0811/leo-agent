# LEO AI Backend — PowerShell Developer Tasks Helper
param (
    [Parameter(Mandatory=$true, Position=0)]
    [ValidateSet("install", "dev", "test", "lint", "format", "typecheck", "migrate", "seed", "docker-up", "docker-down")]
    [string]$Task
)

switch ($Task) {
    "install"     { pip install -e ".[dev]" }
    "dev"         { uvicorn app.main:app --reload --port 8000 }
    "test"        { pytest }
    "lint"        { ruff check . }
    "format"      { ruff format . }
    "typecheck"   { mypy app }
    "migrate"     { alembic upgrade head }
    "seed"        { python scripts/seed_knowledge.py }
    "docker-up"   { docker compose up -d }
    "docker-down" { docker compose down }
}
