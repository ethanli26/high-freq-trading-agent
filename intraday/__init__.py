"""Event-driven intraday engine (minute bars) — a real-time systems showcase.

Architecture: a feed yields ``BarEvent``s; the engine dispatches each through a
per-bar event queue to strategy -> risk gate -> simulated execution, producing
``SignalEvent`` -> ``OrderEvent`` -> ``FillEvent``. Per-symbol state is updated
incrementally; out-of-order/duplicate timestamps and gaps are handled explicitly.

This is an architecture demo on replayed/synthetic data — NOT a validated or
profitable strategy. Execution is simulated and places no real orders; a live IBKR
feed/execution path (see feed.LiveFeed) would route through the existing
paper-safety guards. Modules: events, feed, engine, strategy.
"""
