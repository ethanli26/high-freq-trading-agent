"""Safety-guard tests (critical): the DU-account guard and DRY_RUN must block orders.

A fake broker records any placement, so we can prove that orders flow ONLY when the
account is a paper (DU) account AND DRY_RUN is False — and that neither guard can be
bypassed. No live IBKR connection is required.
"""

import pytest

import config
from decision.autonomy import (
    _place_proposal,
    assert_paper_account,
    is_paper_account,
    run_gate,
)


class FakeBroker:
    """Minimal broker stub: configurable account id; records placements."""

    def __init__(self, account_id):
        self._account_id = account_id
        self.placed = []

    def get_account_id(self):
        return self._account_id

    def place_market_order(self, symbol, shares, action):
        self.placed.append((symbol, shares, action))
        return ("trade", symbol)


def _proposal():
    return {"symbol": "AAA", "action": "BUY", "shares": 10, "entry_ref": 100.0,
            "stop": 96.0, "atr": 2.0, "risk_dollars": 1000.0, "est_value": 1000.0}


def test_is_paper_account_only_accepts_du():
    assert is_paper_account("DU1234567") is True
    assert is_paper_account("U1234567") is False   # live account
    assert is_paper_account("") is False
    assert is_paper_account(None) is False


def test_assert_paper_account_rejects_live():
    with pytest.raises(RuntimeError):
        assert_paper_account(FakeBroker("U999"))


def test_assert_paper_account_accepts_paper():
    assert assert_paper_account(FakeBroker("DU1")) == "DU1"


def test_dry_run_blocks_placement_even_on_paper(monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", True)
    broker = FakeBroker("DU1")  # a valid paper account ...
    assert _place_proposal(broker, _proposal()) is None  # ... but DRY_RUN refuses
    assert broker.placed == []


def test_live_account_blocks_placement_even_when_not_dry_run(monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", False)  # DRY_RUN off ...
    broker = FakeBroker("U999")                    # ... but a live account
    with pytest.raises(RuntimeError):
        _place_proposal(broker, _proposal())
    assert broker.placed == []


def test_placement_only_when_paper_and_not_dry_run(monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", False)
    broker = FakeBroker("DU1")
    _place_proposal(broker, _proposal())
    assert broker.placed == [("AAA", 10, "BUY")]  # both guards satisfied -> order flows


def test_run_gate_approve_rejects_live_before_any_order(monkeypatch):
    monkeypatch.setattr(config, "AUTONOMY_MODE", "approve")
    monkeypatch.setattr(config, "DRY_RUN", False)
    broker = FakeBroker("U999")
    with pytest.raises(RuntimeError):  # fails fast on the DU guard, before prompting
        run_gate([_proposal()], broker)
    assert broker.placed == []


def test_run_gate_signal_only_never_places(monkeypatch):
    monkeypatch.setattr(config, "AUTONOMY_MODE", "signal_only")
    monkeypatch.setattr(config, "DRY_RUN", False)
    broker = FakeBroker("DU1")
    run_gate([_proposal()], broker)  # alerts only
    assert broker.placed == []
