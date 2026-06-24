from __future__ import annotations

from src.options.black_scholes import bsm_gamma, bsm_vega


def test_gamma_positive_for_call_and_put() -> None:
    # Gamma does not depend on option type in BSM.
    gamma = float(bsm_gamma(100.0, 100.0, 0.5, 0.03, 0.01, 0.2))
    assert gamma > 0.0


def test_vega_positive_for_calls_and_puts() -> None:
    # Vega is identical for calls and puts in BSM.
    vega = float(bsm_vega(100.0, 105.0, 0.75, 0.03, 0.01, 0.25))
    assert vega > 0.0
