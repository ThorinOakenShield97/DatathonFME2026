"""
train_models.py — trains the forecast models used by agent.py in this folder.
Both final submissions (final_A_holidays and final_B_rundrain) use the same models.

Usage (from this folder; the repository root contains data/train and starter_kit):
    python train_models.py ../..             # validate, then train on everything
    python train_models.py ../.. --no-final  # validation only

The models committed next to agent.py are the ones we submitted, and this script
never overwrites them: it writes into ./retrained/ (use --out DIR to choose another
folder). To use retrained models, copy them next to agent.py yourself.

How the training set is built: the training data is replayed hour by hour
through the SAME History/feature code the agent uses (imported from agent.py),
with obs dictionaries shaped like the hard-mode simulator's. So what the models
see in training is exactly what they will see in the test.

Validation: 12 one-week blocks (every third week) are held out, concatenated
like the hidden test, and the full agent is scored on them in the simulator.
"""

import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

import agent as A

HERE = Path(__file__).parent
SPEC = {"capacity_kwh": 6000.0, "power_kw": 1500.0, "efficiency": 0.90, "soc_init": 0.50,
        "soc_min": 0.0, "soc_max": 1.0, "dt_hours": 1.0}
ROUNDS = {"co2": 430, "price": 600, "load": 600, "pv": 100}      # chosen on the validation weeks
PARAMS = dict(objective="l2", learning_rate=0.05, num_leaves=31, min_data_in_leaf=60, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)
Q_PARAMS = dict(objective="quantile", learning_rate=0.05, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
Q_ROUNDS = 250
COLUMNS = {"co2": "co2_kg_per_kwh", "price": "precio_eur_per_kwh", "load": "load_kw", "pv": "pv_kw"}


def hourly_table(package: Path) -> pd.DataFrame:
    """Hourly table in HARD MODE (solar x12), including the weather station."""
    d = package / "data" / "train"
    meter = pd.read_csv(d / "contador.csv", parse_dates=["Fecha"],
                        usecols=["Fecha", "Contador - POTENCIA ACTIVA EXPORTADA (kW)"]).set_index("Fecha")
    inv = pd.read_csv(d / "inversor.csv", parse_dates=["Fecha"]).set_index("Fecha")
    cols = [[c for c in inv.columns if f"INVERSOR {i}" in c.upper() and "POTENCIA ACTIVA" in c.upper()
             and "FASE" not in c.upper()][0] for i in (1, 2, 3)]
    meteo = pd.read_csv(d / "meteo.csv", parse_dates=["Fecha"]).set_index("Fecha")
    price = pd.read_csv(d / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
    co2 = pd.read_csv(d / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")
    table = pd.DataFrame({
        "load_kw": (-meter.iloc[:, 0]).clip(0, 5000).resample("1h").mean(),
        "pv_kw": 12.0 * sum(inv[c].fillna(0) for c in cols).clip(lower=0).resample("1h").mean(),
        "precio_eur_per_kwh": price["precio_eur_per_kwh"],
        "co2_kg_per_kwh": co2["co2_kg_per_kwh"],
        "irr": meteo["Atersa - IRRADIANCIA (W/m2)"].resample("1h").mean(),
        "tamb": meteo["Atersa - TEMPERATURA AMBIENTE (SENSOR 1) (ºC)"].resample("1h").mean(),
        "tpan": meteo["Atersa - TEMPERATURA PANEL (SENSOR 1) (ºC)"].resample("1h").mean(),
        "wind": meteo["Atersa - VELOCIDAD VIENTO (SENSOR 1) (m/s)"].resample("1h").mean(),
    }).dropna(subset=["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"])
    table.index.name = "Fecha"
    return table


def obs_stream(d: pd.DataFrame):
    """Yield (obs, timestamp, row) as the hard-mode simulator would."""
    co2, pr = d.co2_kg_per_kwh.values, d.precio_eur_per_kwh.values
    ld, pv, irr, ta = d.load_kw.values, d.pv_kw.values, d.irr.values, d.tamb.values
    for t in range(len(d)):
        e, j = min(t + 3, len(d)), (t - 1 if t > 0 else t)
        yield ({"co2_properes_24h": co2[t:e].tolist(), "preus_properes_24h": pr[t:e].tolist(),
                "co2_actual": co2[t], "preu_actual": pr[t],
                "noves_dades_5min": {"ultim_consum_kw": ld[j], "ultima_solar_kw": pv[j],
                                     "irradiancia_wm2": irr[j], "temp_ambient_c": ta[j]}},
               d.index[t].to_pydatetime(), t)


def build(d: pd.DataFrame, reset_every=None, phase=0) -> dict:
    """Replay d and collect (features, target) for every model.
    reset_every: also forget the history every N hours, so the models learn to
    work with a short past, as at the start of each test week."""
    truth = {k: d[c].to_dict() for k, c in COLUMNS.items()}
    net = (d.load_kw - d.pv_kw).values
    out = {k: ([], []) for k in ("co2", "price", "load", "pv", "now")}
    h = A.History()
    for i, (obs, ts, t) in enumerate(obs_stream(d)):
        if reset_every and (i + phase) % reset_every == 0:
            h.reset()
        h.update(obs, ts)
        for name, fn, ks in (("co2", A.market_features, A.MARKET_HORIZONS), ("price", A.market_features, A.MARKET_HORIZONS),
                             ("load", A.net_features, A.NET_HORIZONS), ("pv", A.net_features, A.NET_HORIZONS)):
            x, anchor = fn(h, ts, name)
            y = np.array([truth[name].get(pd.Timestamp(ts) + pd.Timedelta(hours=k), np.nan) for k in ks])
            if name != "pv":
                y = y - anchor                   # models predict the change from the last known value
            ok = np.isfinite(y)
            out[name][0].append(x[ok]); out[name][1].append(y[ok])
        x, anchor = A.now_features(h, ts)
        if np.isfinite(anchor):
            out["now"][0].append(x); out["now"][1].append(np.array([net[t] - anchor]))
    return {k: (np.vstack(a), np.concatenate(b)) for k, (a, b) in out.items()}


def build_augmented(d: pd.DataFrame) -> dict:
    parts = [build(d), build(d, 72, 0), build(d, 72, 36)]
    return {k: (np.vstack([p[k][0] for p in parts]), np.concatenate([p[k][1] for p in parts])) for k in parts[0]}


def train(data: dict, out_dir: Path):
    for name in ("co2", "price", "load", "pv"):
        x, y = data[name]
        lgb.train(PARAMS, lgb.Dataset(x, y), ROUNDS[name]).save_model(str(out_dir / f"model_{name}.txt"))
    x, y = data["now"]
    for q in A.QUANTILES:
        lgb.train(dict(Q_PARAMS, alpha=q), lgb.Dataset(x, y), Q_ROUNDS).save_model(
            str(out_dir / f"model_now_{int(q * 100):02d}.txt"))


def implied_lambda(d: pd.DataFrame) -> float:
    base = (d.load_kw - d.pv_kw).clip(lower=0)
    return (0.30 / (base * d.precio_eur_per_kwh).sum()) / (0.70 / (base * d.co2_kg_per_kwh).sum())


def simulate(env_module, agent, d: pd.DataFrame):
    """Run an agent in the official simulator configured as hard mode.
    The local simulator has no 3-hour option, so we wrap its observations."""
    class HardEnv(env_module.CampusBatteryEnv):
        def _get_obs(self):
            o = super()._get_obs()
            if o is None:
                return o
            e = min(self.t + 3, self.n)
            o["preus_properes_24h"] = self.df["precio_eur_per_kwh"].iloc[self.t:e].tolist()
            o["co2_properes_24h"] = self.df["co2_kg_per_kwh"].iloc[self.t:e].tolist()
            r = self.df.iloc[self.t - 1 if self.t > 0 else self.t]
            o["noves_dades_5min"].update(irradiancia_wm2=float(r["irr"]), temp_ambient_c=float(r["tamb"]),
                                         temp_panel_c=float(r["tpan"]), vent_ms=float(r["wind"]))
            o["cap_max_kwh"], o["power_max_kw"] = self.cap, self.power
            return o
    env = HardEnv(d, spec=SPEC)
    obs, done = env.reset(), False
    while not done:
        obs, done = env.step(agent.act(obs))
    return env.evaluate_results()


class NoForecastAgent(A.BatteryAgent):
    """Same planner without models: only the 3 given hours and last hour's net load."""
    def __init__(self):
        super().__init__()
        self.models, self.quantile_models = {}, []


def main():
    package = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location("environment", package / "starter_kit" / "environment.py")
    env_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(env_module)

    table = hourly_table(package)
    weeks = [table.iloc[i:i + 168] for i in range(0, len(table) - 167, 168)]
    validation = pd.concat(weeks[::3])
    training = table.drop(validation.index)
    net = table.load_kw - table.pv_kw
    print(f"hard-mode table: {len(table)} h | solar surplus in {(net < 0).mean() * 100:.1f}% of hours "
          f"| net load < 1500 kW in {(net < 1500).mean() * 100:.1f}%")
    print(f"LAMBDA implied by the score: {implied_lambda(table):.3f} (agent.py uses {A.LAMBDA})")

    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else HERE / "retrained"
    if out.resolve() == HERE.resolve():
        sys.exit("refusing to overwrite the submitted models next to agent.py; choose another --out")
    out.mkdir(parents=True, exist_ok=True)
    A.MODEL_DIR = out                        # the agent loads its models from here during validation

    # 1. Honest validation: train WITHOUT the validation weeks, score the agent on them.
    start = time.time()
    train(build_augmented(training), out)
    print(f"models trained on {len(training)} h in {time.time() - start:.0f} s")
    r = simulate(env_module, NoForecastAgent(), validation)
    print(f"  no forecasts (3 h only)   score={r['score']:.2f}  co2={r['co2_estalviat_pct']:.2f}%  "
          f"cost={r['cost_estalviat_pct']:.2f}%  infractions={r['infraccions']}")
    start = time.time()
    r = simulate(env_module, A.BatteryAgent(), validation)
    print(f"  forecasts + LP (ours)     score={r['score']:.2f}  co2={r['co2_estalviat_pct']:.2f}%  "
          f"cost={r['cost_estalviat_pct']:.2f}%  infractions={r['infraccions']}  ({time.time() - start:.0f} s for {len(validation)} h)")

    # 2. Final models for submission: train on ALL the data.
    if "--no-final" not in sys.argv:
        train(build_augmented(table), out)
        print(f"final models (trained on all data) written to {out}")


if __name__ == "__main__":
    main()
