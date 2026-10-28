"""
Integration tests for Alembic migration definitions and revision graph.
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic import command


def test_alembic_revisions_chain():
    """Verify that migration revisions form an unbroken chain to head."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    alembic_cfg_path = repo_root / "alembic.ini"
    assert alembic_cfg_path.exists(), "alembic.ini must exist in project root"

    cfg = Config(str(alembic_cfg_path))
    cfg.set_main_option("script_location", str(repo_root / "migrations"))

    script = ScriptDirectory.from_config(cfg)
    heads = script.get_heads()
    assert heads == ["002_demo_call_tables"], f"Expected head '002_demo_call_tables', got {heads}"

    # Verify revision chain
    rev_002 = script.get_revision("002_demo_call_tables")
    assert rev_002 is not None
    assert rev_002.down_revision == "001_initial_schema"

    rev_001 = script.get_revision("001_initial_schema")
    assert rev_001 is not None
    assert rev_001.down_revision is None


def test_alembic_offline_sql_generation(capsys):
    """Verify that offline migration generates valid DDL for all tables."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    alembic_cfg_path = repo_root / "alembic.ini"

    cfg = Config(str(alembic_cfg_path))
    cfg.set_main_option("script_location", str(repo_root / "migrations"))

    # Generate offline SQL up to head
    command.upgrade(cfg, "head", sql=True)
    captured = capsys.readouterr()
    sql_output = captured.out

    assert "CREATE TABLE knowledge_sources" in sql_output
    assert "CREATE TABLE knowledge_chunks" in sql_output
    assert "CREATE TABLE conversations" in sql_output
    assert "CREATE TABLE conversation_messages" in sql_output
    assert "CREATE TABLE demo_calls" in sql_output
    assert "CREATE TABLE demo_call_transcripts" in sql_output
    assert "CREATE TABLE demo_agent_handoffs" in sql_output
