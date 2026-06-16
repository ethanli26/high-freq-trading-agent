"""The autonomy gate: the single place where proposals become orders.

The gate reads ``AUTONOMY_MODE`` and behaves accordingly. Phase 2 implements
``signal_only`` (alert, place nothing) and ``approve`` (confirm each trade in the
terminal). Every path runs through two hard safety checks before any order:

  1. The connected account id must start with ``DU`` (an IBKR paper account).
  2. ``DRY_RUN`` must be False.

If either check fails, the gate refuses to place and logs an error.
"""

import logging

import config

log = logging.getLogger(__name__)

# IBKR paper account ids start with "DU"; live accounts start with "U".
PAPER_ACCOUNT_PREFIX = "DU"


def is_paper_account(account_id: str | None) -> bool:
    """True only for an IBKR paper account id (starts with ``DU``)."""
    return bool(account_id) and account_id.startswith(PAPER_ACCOUNT_PREFIX)


def assert_paper_account(broker) -> str:
    """Return the connected account id if it is a paper account, else raise.

    This is the safety guard: it refuses to let any caller proceed toward placing
    orders on a non-paper account.
    """
    account_id = broker.get_account_id()
    if not is_paper_account(account_id):
        log.error(
            "SAFETY GUARD: account %r is not a paper (DU) account; refusing to place orders.",
            account_id,
        )
        raise RuntimeError(f"Refusing to trade on non-paper account: {account_id!r}")
    log.info("Safety guard passed: paper account %s confirmed.", account_id)
    return account_id


def format_proposal(proposal: dict) -> str:
    """One readable multi-line block describing a proposal."""
    return (
        f"PROPOSAL  {proposal['action']} {proposal['shares']} {proposal['symbol']}\n"
        f"  entry ref : ${proposal['entry_ref']:,.2f}\n"
        f"  stop      : ${proposal['stop']:,.2f}  (ATR {proposal['atr']:,.2f})\n"
        f"  risk      : ${proposal['risk_dollars']:,.2f}\n"
        f"  est value : ${proposal['est_value']:,.2f}"
    )


def _place_proposal(broker, proposal: dict):
    """Place one proposal as a paper market order, after re-checking safety."""
    # Defense in depth: never place while DRY_RUN is set.
    if config.DRY_RUN:
        log.error("DRY_RUN is True; refusing to place order for %s.", proposal["symbol"])
        return None

    assert_paper_account(broker)  # re-confirm immediately before placing
    trade = broker.place_market_order(proposal["symbol"], proposal["shares"], proposal["action"])
    log.info(
        "PLACED %s %d %s (est value $%.2f).",
        proposal["action"],
        proposal["shares"],
        proposal["symbol"],
        proposal["est_value"],
    )
    return trade


def run_gate(proposals: list[dict], broker) -> None:
    """Route proposals through the autonomy gate according to ``AUTONOMY_MODE``."""
    mode = config.AUTONOMY_MODE
    log.info("Autonomy gate: mode=%s, %d proposal(s).", mode, len(proposals))

    if not proposals:
        log.info("No proposals to act on.")
        return

    if mode == "signal_only":
        for proposal in proposals:
            print("\n" + format_proposal(proposal))
            log.info("SIGNAL_ONLY: alerted on %s, placed nothing.", proposal["symbol"])
        return

    if mode == "approve":
        # Fail fast on the safety guard before prompting for anything.
        assert_paper_account(broker)
        for proposal in proposals:
            print("\n" + format_proposal(proposal))
            answer = input(f"Place this paper order for {proposal['symbol']}? [y/N]: ").strip().lower()
            if answer == "y":
                _place_proposal(broker, proposal)
            else:
                log.info("DECLINED by user: %s.", proposal["symbol"])
                print(f"Skipped {proposal['symbol']}.")
        return

    # semi_auto / full_auto are deliberately not implemented yet (Phase 5).
    log.warning("Autonomy mode %r is not implemented yet; placing nothing.", mode)
