from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _to_array(x: float | np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)


def d1(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    sigma: float | np.ndarray,
) -> np.ndarray:
    """Black-Scholes-Merton d1 with continuous dividend yield."""
    S_a = _to_array(S)
    K_a = _to_array(K)
    T_a = _to_array(T)
    r_a = _to_array(r)
    q_a = _to_array(q)
    sigma_a = _to_array(sigma)

    out = np.full(np.broadcast(S_a, K_a, T_a, r_a, q_a, sigma_a).shape, np.nan, dtype=float)
    valid = (S_a > 0.0) & (K_a > 0.0) & (T_a > 0.0) & (sigma_a > 0.0)
    if np.any(valid):
        out[valid] = (
            np.log(S_a[valid] / K_a[valid])
            + (r_a[valid] - q_a[valid] + 0.5 * sigma_a[valid] ** 2) * T_a[valid]
        ) / (sigma_a[valid] * np.sqrt(T_a[valid]))
    return out


def d2(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    sigma: float | np.ndarray,
) -> np.ndarray:
    """Black-Scholes-Merton d2 with continuous dividend yield."""
    d1_v = d1(S, K, T, r, q, sigma)
    return d1_v - _to_array(sigma) * np.sqrt(_to_array(T))


def bsm_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: str | np.ndarray,
) -> np.ndarray:
    """Black-Scholes-Merton call/put price with continuous dividend yield."""
    S_a = _to_array(S)
    K_a = _to_array(K)
    T_a = _to_array(T)
    r_a = _to_array(r)
    q_a = _to_array(q)
    sigma_a = _to_array(sigma)
    d1_v = d1(S_a, K_a, T_a, r_a, q_a, sigma_a)
    d2_v = d2(S_a, K_a, T_a, r_a, q_a, sigma_a)

    disc_q = np.exp(-q_a * T_a)
    disc_r = np.exp(-r_a * T_a)
    call = S_a * disc_q * norm.cdf(d1_v) - K_a * disc_r * norm.cdf(d2_v)
    put = K_a * disc_r * norm.cdf(-d2_v) - S_a * disc_q * norm.cdf(-d1_v)

    opt = np.asarray(option_type)
    is_call = np.char.lower(opt.astype(str)) == "call"
    is_put = np.char.lower(opt.astype(str)) == "put"

    out = np.full(call.shape, np.nan, dtype=float)
    out[is_call] = call[is_call]
    out[is_put] = put[is_put]
    return out


def bsm_delta(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: str | np.ndarray,
) -> np.ndarray:
    d1_v = d1(S, K, T, r, q, sigma)
    disc_q = np.exp(-_to_array(q) * _to_array(T))
    call_delta = disc_q * norm.cdf(d1_v)
    put_delta = disc_q * (norm.cdf(d1_v) - 1.0)
    opt = np.asarray(option_type)
    is_call = np.char.lower(opt.astype(str)) == "call"
    is_put = np.char.lower(opt.astype(str)) == "put"
    out = np.full(call_delta.shape, np.nan, dtype=float)
    out[is_call] = call_delta[is_call]
    out[is_put] = put_delta[is_put]
    return out


def bsm_gamma(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    sigma: float | np.ndarray,
) -> np.ndarray:
    S_a = _to_array(S)
    T_a = _to_array(T)
    q_a = _to_array(q)
    sigma_a = _to_array(sigma)
    d1_v = d1(S, K, T, r, q, sigma)
    out = np.full(d1_v.shape, np.nan, dtype=float)
    valid = (S_a > 0.0) & (T_a > 0.0) & (sigma_a > 0.0)
    if np.any(valid):
        out[valid] = (
            np.exp(-q_a[valid] * T_a[valid])
            * norm.pdf(d1_v[valid])
            / (S_a[valid] * sigma_a[valid] * np.sqrt(T_a[valid]))
        )
    return out


def bsm_vega(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    sigma: float | np.ndarray,
) -> np.ndarray:
    S_a = _to_array(S)
    T_a = _to_array(T)
    q_a = _to_array(q)
    d1_v = d1(S, K, T, r, q, sigma)
    return S_a * np.exp(-q_a * T_a) * norm.pdf(d1_v) * np.sqrt(T_a)


def bsm_theta(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: str | np.ndarray,
) -> np.ndarray:
    S_a = _to_array(S)
    K_a = _to_array(K)
    T_a = _to_array(T)
    r_a = _to_array(r)
    q_a = _to_array(q)
    d1_v = d1(S, K, T, r, q, sigma)
    d2_v = d2(S, K, T, r, q, sigma)

    first_term = -(
        S_a * np.exp(-q_a * T_a) * norm.pdf(d1_v) * _to_array(sigma) / (2.0 * np.sqrt(T_a))
    )
    call_theta = first_term - r_a * K_a * np.exp(-r_a * T_a) * norm.cdf(d2_v) + q_a * S_a * np.exp(-q_a * T_a) * norm.cdf(d1_v)
    put_theta = first_term + r_a * K_a * np.exp(-r_a * T_a) * norm.cdf(-d2_v) - q_a * S_a * np.exp(-q_a * T_a) * norm.cdf(-d1_v)

    opt = np.asarray(option_type)
    is_call = np.char.lower(opt.astype(str)) == "call"
    is_put = np.char.lower(opt.astype(str)) == "put"
    out = np.full(call_theta.shape, np.nan, dtype=float)
    out[is_call] = call_theta[is_call]
    out[is_put] = put_theta[is_put]
    return out


def bsm_rho(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    sigma: float | np.ndarray,
    option_type: str | np.ndarray,
) -> np.ndarray:
    K_a = _to_array(K)
    T_a = _to_array(T)
    r_a = _to_array(r)
    d2_v = d2(S, K, T, r, q, sigma)

    call_rho = K_a * T_a * np.exp(-r_a * T_a) * norm.cdf(d2_v)
    put_rho = -K_a * T_a * np.exp(-r_a * T_a) * norm.cdf(-d2_v)

    opt = np.asarray(option_type)
    is_call = np.char.lower(opt.astype(str)) == "call"
    is_put = np.char.lower(opt.astype(str)) == "put"
    out = np.full(call_rho.shape, np.nan, dtype=float)
    out[is_call] = call_rho[is_call]
    out[is_put] = put_rho[is_put]
    return out
