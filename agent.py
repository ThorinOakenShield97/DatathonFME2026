"""
agent.py — Repte UPC Sostenible (Datathon 2026)

THE PROBLEM IN ONE PARAGRAPH
----------------------------
The campus has a 2,000 kWh / 500 kW battery. Every hour the simulator calls
`act(obs)` and we return one number: the battery power in kW
(positive = charge, negative = discharge). The score rewards importing less
energy from the grid in hours when it is dirty (70%) and expensive (30%).

THE IDEA (MPC = Model Predictive Control)
-----------------------------------------
Every hour:
  1. Look at the next 24 hours of CO2 and price, which `obs` gives us exactly.
  2. Solve a linear program (LP) that finds the best charge/discharge plan
     for those 24 hours.
  3. Apply only the first hour of that plan.
  4. Next hour, the window slides by one hour: plan again from scratch.

WHY AN LP IS ENOUGH
-------------------
  * The campus always consumes more than 500 kW (99.5% of hours), so every
    kW the battery charges adds 1 kW of grid import, and every kW it
    discharges removes 1 kW. The score is therefore linear in battery power.
  * We merge both goals into one number per hour:
        w_t = co2_t + LAMBDA * price_t
    and minimise  sum_t  w_t * (charge_t - discharge_t).
  * The battery limits (power and state of charge) are linear constraints.
  A problem with a linear objective and linear constraints is exactly an LP,
  and an LP solver returns the guaranteed best plan in about 1.5 ms.
"""

import numpy as np
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# Battery physics (from starter_kit/battery_spec.json)
# ---------------------------------------------------------------------------
CAPACITY_KWH = 2000.0     # the battery holds between 0 and 2,000 kWh
POWER_KW = 500.0          # max charge or discharge power of the inverter
ETA = 0.9 ** 0.5          # one-way efficiency ~0.9487 (round trip = 90%)
#   Charging   500 kW for 1 h -> the battery gains 500 * ETA = 474 kWh
#   Discharging 500 kW for 1 h -> the battery loses 500 / ETA = 527 kWh

# ---------------------------------------------------------------------------
# How much 1 EUR of cost is worth compared with 1 kg of CO2.
# ---------------------------------------------------------------------------
# The score is 0.70 * (% CO2 saved) + 0.30 * (% cost saved). Percentages are
# relative to the no-battery baseline, so 1 kg CO2 is worth 0.70 / co2_base
# points and 1 EUR is worth 0.30 / cost_base points. Dividing both by the CO2
# value gives the price weight: LAMBDA = (0.30 / cost_base) / (0.70 / co2_base).
# With the training data this is 0.252 (evaluate.py prints it).
LAMBDA = 0.25

# Tiny safety margin (kW). The solver works in floating point and may return
# e.g. 500.0000000001 kW; the simulator would count that as an infraction.
MARGIN = 1e-6


class BatteryAgent:
    def __init__(self):
        # The constraint matrix depends only on the horizon length (24 hours
        # most of the time, fewer at the very end of the simulation), never on
        # the data. We build it once per length and reuse it every hour.
        self._matrices = {}

    # -----------------------------------------------------------------------
    # LP building blocks
    # -----------------------------------------------------------------------
    def _constraints(self, horizon: int):
        """Build the matrix that keeps the state of charge (SOC) in [0, 2000].

        DECISION VARIABLES (what the LP chooses), for horizon H:
            x = [c_0, c_1, ..., c_{H-1},  d_0, d_1, ..., d_{H-1}]
                 \\_______ charge ______/   \\______ discharge ______/
            all in kW, all >= 0.

        Charge and discharge are separate variables because losses work
        differently in each direction (multiply by ETA vs divide by ETA).
        With one signed variable the SOC would not be linear.

        SOC AFTER HOUR k:
            SOC_k = SOC_0 + ETA * (c_0 + ... + c_k) - (d_0 + ... + d_k) / ETA

        Example for H = 3 ("cumulative" is a lower-triangular matrix of ones,
        so row k adds up hours 0..k):

            cumulative = [[1, 0, 0],
                          [1, 1, 0],
                          [1, 1, 1]]

            soc_delta  =      c0   c1   c2     d0     d1     d2
                    hour 0: [ η,   0,   0,   -1/η,    0,     0  ]
                    hour 1: [ η,   η,   0,   -1/η,  -1/η,    0  ]
                    hour 2: [ η,   η,   η,   -1/η,  -1/η,  -1/η ]

            soc_delta @ x  =  SOC_k - SOC_0  (change in SOC by end of hour k)

        linprog only accepts constraints written as  A_ub @ x <= b_ub,  so:
            SOC_k <= 2000   ->   soc_delta @ x <= 2000 - SOC_0   (top rows)
            SOC_k >= 0      ->  -soc_delta @ x <= SOC_0          (bottom rows)
        The right-hand sides (b_ub) are built in _plan, because they depend on
        the current SOC.
        """
        if horizon not in self._matrices:
            cumulative = np.tril(np.ones((horizon, horizon)))
            soc_delta = np.hstack([ETA * cumulative, -cumulative / ETA])
            self._matrices[horizon] = np.vstack([soc_delta, -soc_delta])
        return self._matrices[horizon]

    def _plan(self, weights: np.ndarray, soc: float, discharge_cap_now: float) -> float:
        """Solve the LP and return the power (kW) for the current hour only.

        weights           : w_t for each hour of the horizon (CO2 + LAMBDA*price)
        soc               : energy in the battery right now (kWh)
        discharge_cap_now : max useful discharge this hour (kW), see act()
        """
        horizon = len(weights)

        # OBJECTIVE: minimise  sum_t w_t * c_t  -  sum_t w_t * d_t
        # Charging adds grid import (we pay w_t per kWh), discharging removes
        # it (we save w_t per kWh).  cost = [w_0..w_{H-1}, -w_0..-w_{H-1}]
        cost = np.concatenate([weights, -weights])

        # SOC LIMITS (see _constraints for the derivation)
        a_ub = self._constraints(horizon)
        b_ub = np.concatenate([
            np.full(horizon, CAPACITY_KWH - soc),    # room left to charge
            np.full(horizon, soc),                   # energy left to discharge
        ])

        # POWER LIMITS: one (min, max) pair per variable, in the order of x.
        bounds = [(0.0, POWER_KW)] * (2 * horizon)
        # Index `horizon` is d_0, this hour's discharge. Energy discharged
        # beyond what the campus consumes is exported for 0 EUR and 0 CO2,
        # i.e. wasted, so we cap it.
        bounds[horizon] = (0.0, discharge_cap_now)

        # SOLVE. "highs" is the fast, modern LP solver bundled with scipy.
        result = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not result.success:
            return 0.0                               # idle rather than crash

        # result.x = [c_0, ..., c_{H-1}, d_0, ..., d_{H-1}]
        # We keep only hour 0: c_0 - d_0 (positive = charge, negative =
        # discharge). The rest of the plan is discarded and recomputed next
        # hour with fresh data.
        return float(result.x[0] - result.x[horizon])

    # -----------------------------------------------------------------------
    # Called by the simulator once per hour
    # -----------------------------------------------------------------------
    def act(self, obs: dict) -> float:
        soc = float(obs["soc_kwh"])

        # 1. Known future: CO2 and price for this hour and the next ones
        #    (usually 24 values; fewer near the end of the simulation, which
        #    automatically makes the LP empty the battery before the end).
        co2 = np.asarray(obs["co2_properes_24h"], dtype=float)
        price = np.asarray(obs["preus_properes_24h"], dtype=float)
        horizon = min(len(co2), len(price))
        if horizon == 0:
            return 0.0

        # 2. One merged signal per hour: how "bad" 1 kWh of grid import is.
        weights = co2[:horizon] + LAMBDA * price[:horizon]

        #    Guard against missing values: a single NaN makes the whole LP
        #    fail, and it would stay in the 24 h window for a day, leaving the
        #    battery idle all that time. Fill gaps with the mean of the rest.
        finite = np.isfinite(weights)
        if not finite.any():
            return 0.0
        weights = np.where(finite, weights, weights[finite].mean())

        # 3. Max useful discharge this hour = current campus net load
        #    (consumption - solar). We only know last hour's value, which is
        #    a good estimate. In practice it is almost always above 500 kW,
        #    so this cap rarely matters.
        last = obs.get("noves_dades_5min") or {}
        net_load = float(last.get("ultim_consum_kw", POWER_KW)) - float(last.get("ultima_solar_kw", 0.0))
        if not np.isfinite(net_load):
            net_load = POWER_KW      # unknown reading: assume the usual case (> 500 kW)
        discharge_cap_now = min(POWER_KW, max(0.0, net_load))

        # 4. Optimal plan -> power for this hour. Any unexpected error means
        #    "do nothing" this hour instead of crashing the whole evaluation.
        try:
            power = self._plan(weights, soc, discharge_cap_now)
        except Exception:
            power = 0.0

        # 5. Safety clamp. Each infraction costs score points, so we make sure
        #    the setpoint is strictly inside what the simulator accepts:
        #      - never above 500 kW in either direction,
        #      - never charge more than the room left: (2000 - soc) / ETA kW,
        #      - never discharge more than what is stored: soc * ETA kW.
        max_charge = max(0.0, min(POWER_KW, (CAPACITY_KWH - soc) / ETA) - MARGIN)
        max_discharge = max(0.0, min(POWER_KW, soc * ETA) - MARGIN)
        return float(max(-max_discharge, min(max_charge, power)))
