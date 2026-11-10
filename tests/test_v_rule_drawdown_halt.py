"""The account-wide drawdown halt really stops trading (rule V settings).

PLAIN ENGLISH: a "drawdown" is how far the account has fallen from its
highest value.  The "halt" (kill switch) is a hard limit in code: when the
fall is deeper than the limit, the bot sells everything and blocks new
orders.  It must work no matter what the strategy thinks.

These tests *manufacture* a crash with fake files and a fake broker, then
run the real halt code (nothing inside the halt is mocked):

  equity file (peak) + broker equity (now)
      -> check_portfolio_drawdown      (is the fall deeper than the limit?)
      -> _emergency_liquidate          (cancel orders, sell every position)
      -> halt lock file                (blocks new orders on later runs)

The limit under test is 40%, the value the paper workflow sets for rule V
(see Documentation/V_RULE_FORWARD_TEST.md).  No real broker, no network.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import alpaca_paper_trading as apt

# The workflow file that starts the daily paper run.
WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "daily_paper_trading.yml"


def _crashed_account(monkeypatch, tmp_path, *, equity_now: float):
    """Build a fake account whose best-ever value was $100,000.

    Returns (broker, orders): `orders` collects every order the halt sends,
    so a test can check what was sold.
    """
    # 1. A fake equity history.  The peak ($100,000) is what the fall is
    #    measured from.
    equity_file = tmp_path / "equity.csv"
    pd.DataFrame({
        "date": ["2026-11-02", "2026-11-03", "2026-11-04"],
        "equity": [98_000.0, 100_000.0, 97_000.0],
    }).to_csv(equity_file, index=False)

    # 2. Point the bot at temporary files, so the test never touches the
    #    real signals/ folder.
    monkeypatch.setattr(apt, "EQUITY_FILE", equity_file)
    monkeypatch.setattr(apt, "_HALT_SENTINEL_FILE", tmp_path / "alpaca_halt_active.txt")
    monkeypatch.setattr(apt, "_DRAWDOWN_PEAK_RESET_FILE", tmp_path / "peak_reset.json")
    # 3. The rule V limit: halt when more than 40% below the peak.
    monkeypatch.setattr(apt, "PORTFOLIO_DRAWDOWN_HALT_PCT", 0.40)
    # 4. Recording the run's outcome writes report files; not needed here.
    monkeypatch.setattr(apt, "_set_submit_outcome", lambda *args, **kwargs: {})

    orders = []

    def place_order(order):
        orders.append(order)          # remember the order instead of sending it
        return f"fake-order-{len(orders)}"

    # 5. A fake broker holding rule V's two funds.
    broker = SimpleNamespace(
        get_equity=lambda: equity_now,
        is_market_open=lambda: True,
        cancel_all_orders=lambda: True,
        get_positions=lambda: [
            SimpleNamespace(ticker="QQQ", quantity=100),
            SimpleNamespace(ticker="BIL", quantity=50),
        ],
        place_order=place_order,
    )
    return broker, orders


def test_fall_beyond_the_limit_sells_everything_and_locks(monkeypatch, tmp_path):
    # $59,000 is 41% below the $100,000 peak: deeper than the 40% limit.
    broker, orders = _crashed_account(monkeypatch, tmp_path, equity_now=59_000.0)

    exit_code = apt._no_order_day_drawdown_check(broker)

    # Exit code 2 = "blocked": the run stops here and sends no strategy orders.
    assert exit_code == 2
    # Every position is closed with a market sell.
    assert sorted((o.ticker, o.side, o.quantity, o.type) for o in orders) == [
        ("BIL", "sell", 50, "market"),
        ("QQQ", "sell", 100, "market"),
    ]
    # The lock file exists, so later runs keep new orders blocked.
    assert apt._HALT_SENTINEL_FILE.exists()


def test_halt_keeps_blocking_on_the_next_run(monkeypatch, tmp_path):
    # Still below the limit the next day: the halt must fire again, not
    # quietly let trading resume.
    broker, orders = _crashed_account(monkeypatch, tmp_path, equity_now=59_000.0)

    assert apt._no_order_day_drawdown_check(broker) == 2
    assert apt._no_order_day_drawdown_check(broker) == 2
    assert apt._HALT_SENTINEL_FILE.exists()


def test_fall_inside_the_tested_range_does_not_halt(monkeypatch, tmp_path):
    # $65,000 is 35% below the peak.  Rule V's worst fall in its test was
    # 38%, so this is a bad but expected loss: hold, do not sell.
    broker, orders = _crashed_account(monkeypatch, tmp_path, equity_now=65_000.0)

    assert apt._no_order_day_drawdown_check(broker) is None
    assert orders == []
    assert not apt._HALT_SENTINEL_FILE.exists()


def test_drawdown_number_is_measured_from_the_peak(monkeypatch, tmp_path):
    broker, _orders = _crashed_account(monkeypatch, tmp_path, equity_now=59_000.0)

    halted, drawdown = apt.check_portfolio_drawdown(broker)

    # (59,000 - 100,000) / 100,000 = -0.41
    assert halted is True
    assert round(drawdown, 4) == -0.41


def test_paper_workflow_sets_real_halt_limits_for_rule_v():
    """The limits live in the workflow file, so pin them here.

    If someone edits the workflow and removes or loosens a halt, this test
    fails and the change has to be made on purpose.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    # Drawdown halt: 40% for rule V, 12% for core-alpha.
    assert "ALPACA_DD_HALT_PCT=${{ env.PAPER_MODE == 'v_rule' && '0.40' || '0.12' }}" in text
    # One-day loss halt: -12% for rule V, -8% for core-alpha.
    assert "GUARD_PNL_HALT_PCT=${{ env.PAPER_MODE == 'v_rule' && '-12.0' || '-8.0' }}" in text
