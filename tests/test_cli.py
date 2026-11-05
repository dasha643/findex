from typer.testing import CliRunner
from findex.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    """Перевірка виведення довідки CLI."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.stdout or "Options" in result.stdout


def test_cli_search_missing_index() -> None:
    """Перевірка обробки помилки при відсутності файлу індексу (без трейсбеку)."""
    result = runner.invoke(app, ["search", "nonexistent_query", "--index", "nonexistent.pkl"])
    assert result.exit_code != 0
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr