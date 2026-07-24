import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import *


SEED = 994
OUT = "results_acceler"
CACHE = "cache"
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

SAMPLE_NUM = 2 ** 14
N_COARSE = 2 ** 8
COARSE_MASS = 0.50
COARSE_MIN_WIDTH = 0.10
COARSE_MARGIN = 0.02
QMSA_LAMBDA = 1.25
RETRAIN_SURROGATE = False

N_REF_POWER = 14
QAE_STATE_QUBITS = 6
QMSA_POWERS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
QMSA_REPEAT = 5
QAE_RESOURCE_POWERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MC_REPEAT = 20

OBS_KEYS = ["PH", "Mg", "Ca", "Na", "K", "HCO3"]
TARGET_PARAMS = [
    ("cation_ecof", 1),
    ("cation_ecof", 2),
    ("cation_ecof", 3),
    ("CEC", 0),
    ("bouncon", 1),
    ("bouncon", 5),
]
PARA_BOUNDS = {
    "cation_ecof": {"low": [0.07, 0.1, 0.1, 1e-6], "up": [0.5, 0.9, 0.9, 8e-6]},
    "CEC": {"low": [1.0], "up": [6.0]},
    "bouncon": {
        "low": [10 ** -9.0, 8e-4, 8e-7, 7e-5, 2e-5, 1e-3],
        "up": [10 ** -6.5, 4e-3, 4e-6, 4e-4, 9e-5, 7e-3],
    },
}


def prepare_problem():
    np.random.seed(SEED)
    x_data = load_x(SAMPLE_NUM, TARGET_PARAMS)
    obs_df = pd.read_excel("concentration.xlsx")
    e0, sigma_df = compute_e0(SAMPLE_NUM, obs_df, OBS_KEYS, return_sigma_report=True)
    e0_norm, e0_meta = normalize_e0(e0)
    model = train_surrogate(
        x_data, e0_norm, seed=SEED,
        model_path=os.path.join(CACHE, "surrogate_weighted.joblib"),
        retrain=RETRAIN_SURROGATE,
    )

    d = len(TARGET_PARAMS)
    x_coarse = lhs_samples(N_COARSE, d, bounds=[(0.0, 1.0)] * d, seed=SEED + 1)
    jc, wc, pc, _, _ = build_likelihood_table(model, x_coarse, e0_meta, TARGET_PARAMS, PARA_BOUNDS)
    ref_bounds = refined_bounds(x_coarse, wc, COARSE_MASS, COARSE_MIN_WIDTH, COARSE_MARGIN)
    ess = 1.0 / np.sum(pc ** 2)

    meta = pd.DataFrame([{
        "SAMPLE_NUM": SAMPLE_NUM,
        "N_COARSE": N_COARSE,
        "N_REF": 2 ** N_REF_POWER,
        "coarse_J_min": float(jc.min()),
        "coarse_J_mean": float(jc.mean()),
        "coarse_Z_EW": float(wc.mean()),
        "coarse_ESS": float(ess),
        "coarse_ESS_ratio": float(ess / N_COARSE),
    }])
    return model, e0_meta, ref_bounds, sigma_df, meta


def sobol_samples(bounds, power):
    sampler = qmc.Sobol(d=len(bounds), scramble=False)
    u = sampler.random_base2(m=int(power))
    x = np.zeros_like(u)
    for j, (lo, hi) in enumerate(bounds):
        x[:, j] = lo + u[:, j] * (hi - lo)
    return np.clip(x, 0.0, 1.0)


def _qmsa_raw_row(p, n, r, exact_idx, exact_j, qmsa):
    calls = int(qmsa["oracle_calls"])
    return {
        "method": "QMSA",
        "resource_power": int(p),
        "N": int(n),
        "sqrt_N": float(np.sqrt(n)),
        "repeat": int(r),
        "exact_min_index": exact_idx,
        "exact_min_J": exact_j,
        "QMSA_index": int(qmsa["best_index"]),
        "QMSA_J": float(qmsa["best_value"]),
        "abs_J_error": float(abs(qmsa["best_value"] - exact_j)),
        "success_by_value": bool(qmsa["success_by_value"]),
        "oracle_calls": calls,
        "classical_queries": int(n),
        "query_reduction_factor": float(n / max(calls, 1)),
    }


def run_qmsa_comparison(model, e0_meta, ref_bounds):
    rows = []
    for p in QMSA_POWERS:
        n = 2 ** int(p)
        x = lhs_samples(n, len(ref_bounds), bounds=ref_bounds, seed=SEED + 1000 + p)
        j, _, _, _, _ = build_likelihood_table(model, x, e0_meta, TARGET_PARAMS, PARA_BOUNDS)
        exact_idx = int(np.argmin(j))
        exact_j = float(j[exact_idx])
        for r in range(QMSA_REPEAT):
            qmsa = run_qmsa(
                j,
                seed=SEED + 2000 + 97 * p + r,
                max_oracle_calls=int(4 * np.sqrt(n)),
                lambda_growth=QMSA_LAMBDA,
                verbose=False,
            )
            rows.append(_qmsa_raw_row(p, n, r, exact_idx, exact_j, qmsa))
    raw = pd.DataFrame(rows)
    agg = raw.groupby(["method", "resource_power", "N"], as_index=False).agg(
        mean_oracle_calls=("oracle_calls", "mean"),
        std_oracle_calls=("oracle_calls", "std"),
        mean_abs_J_error=("abs_J_error", "mean"),
        success_rate=("success_by_value", "mean"),
        mean_query_reduction=("query_reduction_factor", "mean"),
    )
    return raw, agg


def _evidence_row(method, p, n_or_q, n_ref, z_true, z_mean, z_std, mae, rmse,
                  state_q=np.nan, eval_q=np.nan, hist=np.nan):
    return {
        "method": method,
        "resource_power": int(p),
        "N_or_Q": int(n_or_q),
        "N_ref": int(n_ref),
        "Z_true": z_true,
        "Z_est_mean": float(z_mean),
        "Z_est_std": float(z_std),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "QAE_state_qubits": state_q,
        "QAE_eval_qubits": eval_q,
        "histogram_target_E_f": hist,
    }


def run_qae_comparison(model, e0_meta, ref_bounds):
    x_ref = sobol_samples(ref_bounds, N_REF_POWER)
    _, w_ref, _, _, _ = build_likelihood_table(model, x_ref, e0_meta, TARGET_PARAMS, PARA_BOUNDS)
    z_true = float(np.mean(w_ref))
    n_ref = len(w_ref)
    rng = np.random.default_rng(SEED + 3000)
    rows = []

    for p in QAE_RESOURCE_POWERS:
        n_or_q = 2 ** int(p)
        mc_values = np.asarray(
            [float(np.mean(w_ref[rng.integers(0, n_ref, size=n_or_q)])) for _ in range(MC_REPEAT)],
            dtype=float,
        )
        mc_rmse = float(np.sqrt(np.mean((mc_values - z_true) ** 2)))
        mc_mae = float(np.mean(np.abs(mc_values - z_true)))
        rows.append(_evidence_row(
            "MC", p, n_or_q, n_ref, z_true,
            np.mean(mc_values), np.std(mc_values), mc_mae, mc_rmse,
        ))

        z_qae, _, z_hist = qae_circuit(
            w_ref, n_bin_qubits=QAE_STATE_QUBITS, eval_qubits=p, seed=SEED + 4000 + p,
        )
        qae_error = float(abs(z_qae - z_true))
        rows.append(_evidence_row(
            "QAE", p, n_or_q, n_ref, z_true,
            z_qae, 0.0, qae_error, qae_error,
            state_q=int(QAE_STATE_QUBITS), eval_q=int(p), hist=float(z_hist),
        ))
        print(f"QAE/MC p={p:2d}: MC_RMSE={mc_rmse:.4e}, QAE_error={qae_error:.4e}, Z_true={z_true:.6e}")

    return pd.DataFrame(rows)


def fit_slope(df, method):
    d = df[(df["method"] == method) & (df["RMSE"] > 0)].copy()
    if len(d) < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(d["N_or_Q"].astype(float)), np.log(d["RMSE"].astype(float)), 1)
    return float(slope)



model, e0_meta, ref_bounds, sigma_df, meta = prepare_problem()
bounds_df = bounds_dataframe(ref_bounds, TARGET_PARAMS, PARA_BOUNDS)

print("\n--- QMSA comparison ---")
qmsa_raw, qmsa_agg = run_qmsa_comparison(model, e0_meta, ref_bounds)

print("\n--- QAE comparison ---")
qae_df = run_qae_comparison(model, e0_meta, ref_bounds)
slope_mc = fit_slope(qae_df, "MC")
slope_qae = fit_slope(qae_df, "QAE")
slope_df = pd.DataFrame([{"MC_slope": slope_mc, "QAE_slope": slope_qae}])

pack_sections({
    "meta": meta,
    "sigma": sigma_df,
    "refined_bounds": bounds_df,
    "qmsa_raw": qmsa_raw,
    "qmsa_summary": qmsa_agg,
    "qae_scaling": qae_df,
    "slope_summary": slope_df,
}).to_csv(os.path.join(OUT, "acceler_comparison.csv"), index=False, encoding="utf-8-sig")
