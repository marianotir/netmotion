"""The interbank debt network and its crisis contagion.

Thesis section 5.6.4 and the 2012 IJMPC article.  A community is a country, a
node is a bank, a link is a lending relationship, and the weight on that link is
the debt along it.  Crash one bank and watch how far the failure travels.

Two cascades are implemented, because the two sources model contagion
differently and the difference matters: the 2011 thesis has banks either
healthy or crashed, while the 2012 article adds a reaction threshold and lets a
damaged bank call in its loans and partially recover.
"""

from __future__ import annotations

from .bank_growth import confidence_strategy_growth, strategy_growth
from .contagion import (
    BankState,
    CascadeRun,
    four_state_cascade,
    systemic_impact,
    two_state_cascade,
)
from .interbank import InterbankNetwork, bank_capital, debt_matrix, too_big_to_fail

__all__ = [
    "InterbankNetwork",
    "debt_matrix",
    "bank_capital",
    "too_big_to_fail",
    "BankState",
    "CascadeRun",
    "two_state_cascade",
    "four_state_cascade",
    "systemic_impact",
    "strategy_growth",
    "confidence_strategy_growth",
]
