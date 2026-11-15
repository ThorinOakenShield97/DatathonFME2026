"""
evaluate.py — local evaluation, and the evidence behind our design decisions.

Usage (from the Project/ folder):
    python evaluate.py <path to paquet_participants_datathon_upc>

It prints four things, each backing up one claim we make to the jury:
  1. Net-load statistics  -> "forecasting load/solar is not needed"
  2. The implied LAMBDA   -> "the price weight is derived, not guessed"
  3. Three agents compared on 12 training weeks (2,016 h, same size as test):
       - starter heuristic          (the organisers' baseline agent)
       - our MPC agent              (agent.py)
       - perfect-information optimum (knows the whole future, incl. load)
     -> "our agent reaches the optimum"
  4. Score for several LAMBDA values -> "the result is robust"
"""

import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

import agent as mpc_agent     # our agent (agent.py in this folder)

ETA = 0.9 ** 0.5
CAPACITY = 2000.0
POWER = 500.0


def load_module(path: Path, name: str):
    """Import a .py file by path (used for the organisers' simulator and the
    starter agent, which live in another folder)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_hourly_table(data_dir: Path) -> pd.DataFrame:
    """Turn the raw CSVs into one hourly table with the 4 columns the
    simulator needs. Same preprocessing as the official tutorial notebook.

    Output columns:
      load_kw            campus consumption (kW, hourly mean)
      pv_kw              solar production of the 3 inverters (kW, hourly mean)
      precio_eur_per_kwh grid price (EUR/kWh)
      co2_kg_per_kwh     grid CO2 intensity (kg/kWh)
    """
    # Meter: "exported power" is negative when the campus imports, so
    # consumption = -exported. Clip absurd values (sensor glitches).
    meter = pd.read_csv(data_dir / "contador.csv", parse_dates=["Fecha"],
                        usecols=["Fecha", "Contador - POTENCIA ACTIVA EXPORTADA (kW)"]).set_index("Fecha")
    load = (-meter.iloc[:, 0]).clip(0, 5000).resample("1h").mean().dropna()

    # Solar: sum the total active power of inverters 1, 2 and 3 (not per phase).
    inverters = pd.read_csv(data_dir / "inversor.csv", parse_dates=["Fecha"]).set_index("Fecha")
    power_cols = [
        [c for c in inverters.columns
         if f"INVERSOR {i}" in c.upper() and "POTENCIA ACTIVA" in c.upper() and "FASE" not in c.upper()][0]
        for i in (1, 2, 3)
    ]
    solar = sum(inverters[c].fillna(0) for c in power_cols).clip(lower=0).resample("1h").mean().dropna()

    # Price and CO2 are already hourly.
    price = pd.read_csv(data_dir / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
    co2 = pd.read_csv(data_dir / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")

    # Align everything on the hourly timestamp; drop hours with missing data.
    table = pd.DataFrame({
        "load_kw": load,
        "pv_kw": solar,
        "precio_eur_per_kwh": price["precio_eur_per_kwh"],
        "co2_kg_per_kwh": co2["co2_kg_per_kwh"],
    }).dropna()
    table.index.name = "Fecha"
    return table


def implied_lambda(table: pd.DataFrame) -> float:
    """Price weight implied by the score formula (see LAMBDA in agent.py).

    Without a battery the campus imports max(0, load - solar) every hour.
    Summing that times CO2 (or price) gives the baseline totals the score
    divides by.
    """
    base_grid = (table.load_kw - table.pv_kw).clip(lower=0)
    co2_base = (base_grid * table.co2_kg_per_kwh).sum()       # kg
    cost_base = (base_grid * table.precio_eur_per_kwh).sum()  # EUR
    return (0.30 / cost_base) / (0.70 / co2_base)


def simulate(env_module, agent, table: pd.DataFrame) -> dict:
    """Run an agent hour by hour through the official simulator and return
    its score dictionary."""
    env = env_module.CampusBatteryEnv(table)
    obs, done = env.reset(), False
    while not done:
        obs, done = env.step(agent.act(obs))
    return env.evaluate_results()


class ReplayAgent:
    """An 'agent' that just replays a plan computed in advance. Used to run
    the perfect-information optimum through the same simulator, so it is
    scored exactly like the real agents."""

    def __init__(self, plan):
        self.plan, self.step = plan, 0

    def act(self, obs):
        power = self.plan[self.step]
        self.step += 1
        # Same safety clamp as agent.py, to avoid rounding infractions.
        soc = obs["soc_kwh"]
        max_charge = max(0.0, min(POWER, (CAPACITY - soc) / ETA) - 1e-6)
        max_discharge = max(0.0, min(POWER, soc * ETA) - 1e-6)
        return float(max(-max_discharge, min(max_charge, power)))


def perfect_information_plan(table: pd.DataFrame, lam: float) -> np.ndarray:
    """The best possible plan if we knew EVERYTHING in advance: all prices,
    all CO2 and the real load of every hour. One single LP over the whole
    period. No real agent can beat this, so it is the upper bound.

    Same LP as agent.py, but written differently: here the SOC of each hour
    is an explicit variable linked by equality constraints, which is more
    efficient for 2,016 hours (sparse matrix).

    Variables: x = [charge (n), discharge (n), soc (n)]
    Equality for each hour k:
        soc_k - soc_{k-1} - ETA*charge_k + discharge_k/ETA = 0
        (for k = 0, soc_{-1} is the initial 1,000 kWh, moved to b_eq)
    """
    n = len(table)
    weights = (table.co2_kg_per_kwh + lam * table.precio_eur_per_kwh).values
    net_load = (table.load_kw - table.pv_kw).clip(lower=0).values
    cost = np.concatenate([weights, -weights, np.zeros(n)])   # SOC has no cost

    a_eq = lil_matrix((n, 3 * n))
    b_eq = np.zeros(n)
    for k in range(n):
        a_eq[k, k] = -ETA            # charge_k
        a_eq[k, n + k] = 1 / ETA     # discharge_k
        a_eq[k, 2 * n + k] = 1       # soc_k
        if k > 0:
            a_eq[k, 2 * n + k - 1] = -1   # soc_{k-1}
        else:
            b_eq[k] = CAPACITY / 2        # starting SOC = 1,000 kWh

    bounds = ([(0, POWER)] * n                              # charge
              + [(0, min(POWER, x)) for x in net_load]      # discharge <= real net load
              + [(0, CAPACITY)] * n)                        # SOC in [0, 2000]
    result = linprog(cost, A_eq=a_eq.tocsr(), b_eq=b_eq, bounds=bounds, method="highs")
    return result.x[:n] - result.x[n:2 * n]                 # signed power per hour


def show(name: str, r: dict):
    print(f"  {name:34s} score={r['score']:5.2f}  co2={r['co2_estalviat_pct']:5.2f}%  "
          f"cost={r['cost_estalviat_pct']:5.2f}%  infractions={r['infraccions']}")


def main():
    package = Path(sys.argv[1])
    env_module = load_module(package / "starter_kit" / "environment.py", "environment")
    starter = load_module(package / "starter_kit" / "agent.py", "starter_agent")
    table = build_hourly_table(package / "data" / "train")

    # --- 1. Why we do not forecast load or solar ---------------------------
    net = table.load_kw - table.pv_kw
    print("1. Net load (consumption - solar)")
    print(f"  hours: {len(table)}   mean load: {table.load_kw.mean():.0f} kW   max solar: {table.pv_kw.max():.0f} kW")
    print(f"  share of hours with net load > 500 kW: {(net > POWER).mean() * 100:.1f}%")

    # --- 2. Where LAMBDA comes from ----------------------------------------
    lam = implied_lambda(table)
    print(f"\n2. LAMBDA implied by the score: {lam:.3f}   (agent.py uses {mpc_agent.LAMBDA})")

    # --- 3. Compare agents on a test-sized sample --------------------------
    # Cut the training year into consecutive 168-hour weeks and keep every
    # third one: 12 weeks spread over the year = 2,016 hours, like the test.
    weeks = [table.iloc[i:i + 168] for i in range(0, len(table) - 167, 168)]
    validation = pd.concat(weeks[::3])
    print(f"\n3. Agents on {len(validation)} validation hours")
    show("starter heuristic", simulate(env_module, starter.BatteryAgent(), validation))
    start = time.time()
    show("MPC with LP (ours)", simulate(env_module, mpc_agent.BatteryAgent(), validation))
    print(f"  (MPC run time: {time.time() - start:.1f} s, server limit 60 s)")
    plan = perfect_information_plan(validation, lam)
    show("perfect-information optimum", simulate(env_module, ReplayAgent(plan), validation))

    # --- 4. How sensitive is the score to LAMBDA? --------------------------
    print("\n4. Sensitivity to LAMBDA")
    original = mpc_agent.LAMBDA
    for value in (0.0, 0.1, 0.25, 0.4, 0.6, 1.0):
        mpc_agent.LAMBDA = value      # the agent reads the module constant
        show(f"MPC, LAMBDA={value}", simulate(env_module, mpc_agent.BatteryAgent(), validation))
    mpc_agent.LAMBDA = original


if __name__ == "__main__":
    main()
