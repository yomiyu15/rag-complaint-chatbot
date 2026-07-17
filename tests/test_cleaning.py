"""Unit tests for the Task 1 text-cleaning logic."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from eda_preprocessing import clean_narrative, PRODUCT_MAP  # noqa: E402


def test_lowercases_and_strips_special_chars():
    out = clean_narrative("My CREDIT card was CHARGED $$$ twice!!!")
    assert out == out.lower()
    assert "$$$" not in out
    assert "credit card" in out


def test_removes_xxxx_redactions():
    out = clean_narrative("On XX/XX/XXXX my account XXXX was closed")
    assert "x" not in out.replace("account", "")  # no stray redaction runs
    assert "account was closed" in out


def test_removes_boilerplate_opener():
    out = clean_narrative("I am writing to file a complaint about my late fees")
    assert "writing to file a complaint" not in out
    assert "late fees" in out


def test_handles_non_string():
    assert clean_narrative(None) == ""
    assert clean_narrative(float("nan")) == ""


def test_product_map_covers_four_families():
    assert set(PRODUCT_MAP.values()) == {
        "Credit Card", "Personal Loan", "Savings Account", "Money Transfer",
    }
