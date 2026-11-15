"""
agent.py — Repte UPC Sostenible, MODE DIFÍCIL (Datathon 2026)

WHAT CHANGES IN HARD MODE
-------------------------
  * Battery x3 (6,000 kWh, +-1,500 kW) and solar x12: at sunny middays solar
    exceeds consumption (free surplus), and a blind 1,500 kW discharge can
    exceed what the campus consumes (the extra is exported for nothing).
  * obs only gives 3 hours of price and CO2 (this hour and the next 2).

So, unlike standard mode, here we DO need forecasts. The agent is:

    forecasts (LightGBM)  ->  24 h plan (linear program)  ->  first hour applied

  1. FORECAST with gradient-boosted trees trained on data/train:
       - CO2 and price for hours +3..+23 (0..2 are given exactly),
       - campus load and solar for hours 0..23,
       - 10 quantiles of THIS hour's net load (load - solar), because the
         real value is unknown when we decide and errors are costly.
  2. PLAN with an LP over 24 hours that minimises
         sum_t  w_t * grid_t,     w_t = co2_t + LAMBDA * price_t
     where grid_t = max(0, net_t + charge_t - discharge_t). For the current
     hour the LP averages the grid import over the 10 net-load scenarios, so
     it automatically hedges: it does not discharge more than the campus is
     likely to consume, and it only counts solar surplus that is likely real.
  3. APPLY only the first hour, clamp it to the physical limits, and replan
     next hour (MPC).

The feature code below is imported by train_models.py, so training and
inference use exactly the same features.
"""

import math
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import linprog

ETA = 0.9 ** 0.5            # one-way efficiency
LAMBDA = 0.264              # price weight vs CO2, from hard-mode train baselines (see train_models.py)
EPS = 1e-4                  # safety margin recommended by the organisers (kW)
NAN = float("nan")
HOUR = datetime.timedelta(hours=1)

MARKET_HORIZONS = list(range(3, 24))   # hours ahead we must forecast for CO2 / price
NET_HORIZONS = list(range(0, 24))      # hours ahead we forecast for load / solar
QUANTILES = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
MODEL_DIR = Path(__file__).parent

# Public holidays in Barcelona during the challenge year (Sep 2025 - Aug 2026).
# On these days the campus consumes like on a Sunday (about half a normal
# weekday), so the load forecast treats them as Sundays.
HOLIDAYS = {datetime.date(*d) for d in [
    (2025, 9, 11), (2025, 9, 24), (2025, 10, 12), (2025, 11, 1), (2025, 12, 6), (2025, 12, 8),
    (2025, 12, 25), (2025, 12, 26), (2026, 1, 1), (2026, 1, 6), (2026, 4, 3), (2026, 4, 6),
    (2026, 5, 1), (2026, 5, 25), (2026, 6, 24), (2026, 8, 15)]}


def _num(x):
    """float(x), or NaN if missing / not finite."""
    try:
        v = float(x)
        return v if math.isfinite(v) else NAN
    except Exception:
        return NAN


# ---------------------------------------------------------------------------
# Memory: everything observed since the last time jump
# ---------------------------------------------------------------------------
class History:
    """The test is 12 separate weeks run back to back. When the timestamp
    jumps (new week, a month later) the recent past is useless, so we forget
    it. Hours are indexed by n = steps since the last jump."""

    def __init__(self):
        self.prev_ts = None
        self.reset()

    def reset(self):
        self.n = -1
        self.co2, self.price = {}, {}                         # hour -> value, known up to n+2
        self.load, self.pv, self.irr, self.tamb = {}, {}, {}, {}   # hour -> value, known up to n-1

    def update(self, obs, ts):
        if self.prev_ts is not None and ts is not None:
            gap = (ts - self.prev_ts).total_seconds() / 3600.0
            if gap > 2.0 or gap <= 0.0:
                self.reset()
                # The "last hour" readings now come from the end of the previous
                # period (another day, another time of day): do not use them.
                obs = dict(obs, noves_dades_5min={})
        self.prev_ts = ts
        self.n += 1
        n = self.n
        for j, v in enumerate(obs.get("co2_properes_24h") or []):
            self.co2[n + j] = _num(v)
        for j, v in enumerate(obs.get("preus_properes_24h") or []):
            self.price[n + j] = _num(v)
        if n not in self.co2:
            self.co2[n] = _num(obs.get("co2_actual"))
        if n not in self.price:
            self.price[n] = _num(obs.get("preu_actual"))
        sensors = obs.get("noves_dades_5min") or {}
        self.load[n - 1] = _num(sensors.get("ultim_consum_kw", NAN))
        self.pv[n - 1] = _num(sensors.get("ultima_solar_kw", NAN))
        self.irr[n - 1] = _num(sensors.get("irradiancia_wm2", NAN))
        self.tamb[n - 1] = _num(sensors.get("temp_ambient_c", NAN))


# ---------------------------------------------------------------------------
# Features (missing values stay NaN: LightGBM handles them natively)
# ---------------------------------------------------------------------------
def _stats(series, lo, hi):
    """mean, min, max of the known values of `series` between hours lo..hi."""
    v = [series[i] for i in range(lo, hi + 1) if i in series and series[i] == series[i]]
    if not v:
        return NAN, NAN, NAN
    return sum(v) / len(v), min(v), max(v)


def _calendar(ts, k, holidays=False):
    """Hour, weekday, weekend flag and season (sin/cos of day of year) of hour ts+k.
    With holidays=True (used for the load forecast) a public holiday counts as a Sunday."""
    if ts is None:
        return [NAN, NAN, NAN, NAN, NAN]
    t = ts + k * HOUR
    angle = 2 * math.pi * t.timetuple().tm_yday / 365.25
    weekday = t.weekday()
    if holidays and t.date() in HOLIDAYS:
        weekday = 6
    return [t.hour, weekday, 1.0 if weekday >= 5 else 0.0, math.sin(angle), math.cos(angle)]


def market_features(h, ts, which):
    """One row per target hour n+3..n+23 for `which` in {'co2', 'price'}.
    Returns (X, anchor). The model predicts (target - anchor), where anchor is
    the last known value, so it learns the SHAPE of the day from the level."""
    y = h.co2 if which == "co2" else h.price          # series to forecast
    z = h.price if which == "co2" else h.co2          # the other market series
    n = h.n
    g = lambda d, i: d.get(i, NAN)
    known = [g(y, n), g(y, n + 1), g(y, n + 2)]
    known_other = [g(z, n), g(z, n + 1), g(z, n + 2)]
    finite = [v for v in known if v == v]
    anchor = finite[-1] if finite else NAN
    mean24, min24, max24 = _stats(y, n - 21, n + 2)
    other_mean24, _, _ = _stats(z, n - 21, n + 2)
    change_now = g(y, n) - g(y, n - 24)               # today vs yesterday, same hours
    change_end = g(y, n + 2) - g(y, n - 22)
    _, _, irr_max = _stats(h.irr, n - 24, n - 1)
    base = known + known_other + [mean24, min24, max24, other_mean24, change_now, change_end,
                                  g(h.irr, n - 1), irr_max, g(h.tamb, n - 1), float(min(n, 60))]
    rows = []
    for k in MARKET_HORIZONS:
        t = n + k
        rows.append([k] + _calendar(ts, k) + base
                    + [g(y, t - 24), g(y, t - 48), g(z, t - 24), g(y, t - 24) - g(y, n - 22)])
    return np.array(rows, dtype=float), anchor


def net_features(h, ts, which):
    """One row per target hour n..n+23 for `which` in {'load', 'pv'}.
    Returns (X, last observed value)."""
    y = h.load if which == "load" else h.pv
    n = h.n
    g = lambda d, i: d.get(i, NAN)
    last = g(y, n - 1)
    mean24, min24, max24 = _stats(y, n - 24, n - 1)
    irr_mean, _, irr_max = _stats(h.irr, n - 24, n - 1)
    base = [last, g(y, n - 2), g(y, n - 3), mean24, min24, max24, last - g(y, n - 25),
            g(h.irr, n - 1), g(h.irr, n - 2), irr_mean, irr_max, g(h.tamb, n - 1),
            g(h.pv, n - 1), g(h.load, n - 1), float(min(n, 60))]
    hour_now = ts.hour if ts is not None else NAN
    rows = []
    for k in NET_HORIZONS:
        t = n + k
        rows.append([k, hour_now] + _calendar(ts, k, which == "load") + base
                    + [g(y, t - 24), g(y, t - 48), g(h.irr, t - 24)])
    return np.array(rows, dtype=float), last


def now_features(h, ts):
    """One row describing the current hour, to predict the distribution of its
    net load. Returns (X, last observed net load)."""
    x_load, last_load = net_features(h, ts, "load")
    x_pv, last_pv = net_features(h, ts, "pv")
    return np.concatenate([x_load[0], x_pv[0][-3:], x_pv[0][7:13]])[None, :], last_load - last_pv


# ---------------------------------------------------------------------------
# The plan: a linear program
# ---------------------------------------------------------------------------
_MATRICES = {}


def solve_plan(weights, net, scenarios, soc, capacity, power):
    """Best battery power (kW) for the current hour.

    weights   : w_t for each hour of the horizon (H values)
    net       : forecast net load (load - solar) per hour, kW (hour 0 is ignored)
    scenarios : S equally likely values of THIS hour's net load, kW
    Variables : x = [charge (H), discharge (H), grid (H), grid_now (S)]
        grid_t     >= net_t + charge_t - discharge_t,  >= 0      (hours 1..H-1)
        grid_now_s >= scenario_s + charge_0 - discharge_0, >= 0  (current hour)
        0 <= SOC_t <= capacity, with SOC_t = soc + ETA*cumsum(charge) - cumsum(discharge)/ETA
    Objective : sum_{t>=1} w_t*grid_t + w_0 * mean_s(grid_now_s)
    """
    H, S = len(weights), len(scenarios)
    if (H, S) not in _MATRICES:
        cumulative = np.tril(np.ones((H, H)))
        zeros, zeros_s = np.zeros((H, H)), np.zeros((H, S))
        soc_delta = np.hstack([ETA * cumulative, -cumulative / ETA, zeros, zeros_s])
        grid_rows = np.hstack([np.eye(H), -np.eye(H), -np.eye(H), zeros_s])
        now_rows = np.zeros((S, 3 * H + S))
        now_rows[:, 0], now_rows[:, H] = 1.0, -1.0
        now_rows[:, 3 * H:] = -np.eye(S)
        _MATRICES[(H, S)] = np.vstack([soc_delta, -soc_delta, grid_rows, now_rows])
    a_ub = _MATRICES[(H, S)]
    b_ub = np.concatenate([np.full(H, capacity - soc), np.full(H, soc),
                           -np.asarray(net, dtype=float), -np.asarray(scenarios, dtype=float)])
    b_ub[2 * H] = 1e9                    # hour 0 is handled by the scenario rows instead
    cost = np.concatenate([np.full(2 * H, 1e-9),          # tiny cost: never charge and discharge at once
                           weights, np.full(S, weights[0] / S)])
    cost[2 * H] = 0.0
    bounds = [(0.0, power)] * (2 * H) + [(0.0, None)] * (H + S)
    result = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success:
        return 0.0
    return float(result.x[0] - result.x[H])


def safe_setpoint(power_wanted, soc, capacity, power):
    """Clamp to what the simulator accepts, so there are never infractions."""
    max_charge = max(0.0, min(power, (capacity - soc) / ETA - EPS))
    max_discharge = max(0.0, min(power, soc * ETA - EPS))
    p = float(max(-max_discharge, min(max_charge, power_wanted)))
    return p if math.isfinite(p) else 0.0


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
class BatteryAgent:
    def __init__(self):
        self.history = History()
        self.models, self.quantile_models = {}, []
        try:
            for name in ("co2", "price", "load", "pv"):
                self.models[name] = lgb.Booster(model_file=str(MODEL_DIR / f"model_{name}.txt"))
            self.quantile_models = [lgb.Booster(model_file=str(MODEL_DIR / f"model_now_{int(q * 100):02d}.txt"))
                                    for q in QUANTILES]
        except Exception:
            self.models, self.quantile_models = {}, []    # fall back to the no-forecast plan

    # -- forecasts -----------------------------------------------------------
    def _forecast(self, ts):
        """Returns (co2, price, load, pv, scenarios) for the planning horizon."""
        h, n = self.history, self.history.n
        known_co2 = [h.co2[n + j] for j in range(3) if (n + j) in h.co2]
        known_price = [h.price[n + j] for j in range(3) if (n + j) in h.price]
        last_net = h.load.get(n - 1, NAN) - h.pv.get(n - 1, NAN)
        if not math.isfinite(last_net):
            last_net = 1000.0

        # Fewer than 3 given hours means the simulation is ending: plan only to the end.
        ending = min(len(known_co2), len(known_price)) < 3
        if ending or not self.models:
            H = min(len(known_co2), len(known_price))
            return (np.array(known_co2[:H]), np.array(known_price[:H]),
                    np.full(H, max(last_net, 0.0)), np.zeros(H), self._scenarios(ts, last_net))

        x, anchor = market_features(h, ts, "co2")
        co2 = np.concatenate([known_co2, anchor + self.models["co2"].predict(x)])
        x, anchor = market_features(h, ts, "price")
        price = np.concatenate([known_price, anchor + self.models["price"].predict(x)])
        x, last = net_features(h, ts, "load")
        load = np.maximum(0.0, (last if math.isfinite(last) else 0.0) + self.models["load"].predict(x)) \
            if math.isfinite(last) else np.full(24, max(last_net, 0.0))
        x, _ = net_features(h, ts, "pv")
        pv = np.maximum(0.0, self.models["pv"].predict(x))
        return co2, price, load, pv, self._scenarios(ts, last_net)

    def _scenarios(self, ts, last_net):
        """10 equally likely values for this hour's net load."""
        if not self.quantile_models:
            return np.full(len(QUANTILES), last_net)
        x, anchor = now_features(self.history, ts)
        if not math.isfinite(anchor):
            return np.full(len(QUANTILES), last_net)
        return np.sort([anchor + m.predict(x)[0] for m in self.quantile_models])

    # -- called by the simulator once per hour -------------------------------
    def act(self, obs: dict) -> float:
        soc = _num(obs.get("soc_kwh"))
        capacity = _num(obs.get("cap_max_kwh", 6000.0))
        power = _num(obs.get("power_max_kw", 1500.0))
        if not (math.isfinite(soc) and math.isfinite(capacity) and math.isfinite(power)):
            return 0.0
        try:
            try:
                ts = pd.Timestamp(obs.get("timestamp")).to_pydatetime()
                if ts != ts:            # NaT
                    ts = None
            except Exception:
                ts = None
            self.history.update(obs, ts)
            co2, price, load, pv, scenarios = self._forecast(ts)
            weights = co2 + LAMBDA * price
            net = load - pv
            ok = np.isfinite(weights) & np.isfinite(net)
            if len(weights) == 0 or not ok.all() or not np.isfinite(scenarios).all():
                return 0.0
            wanted = solve_plan(weights, net, scenarios, soc, capacity, power)
        except Exception:
            wanted = 0.0                 # never crash the evaluation
        return safe_setpoint(wanted, soc, capacity, power)
