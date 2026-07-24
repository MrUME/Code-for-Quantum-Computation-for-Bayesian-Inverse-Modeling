# @Time: 2026.6.30
# @Author: Zihao Wang
# @Email: zihaow24@mails.jlu.edu.cn
# @Memo: Code for study titled "Quantum Computation for Bayesian Inverse Modeling in Fractured and Porous Media"

from utils import *

SEED = 994
np.random.seed(SEED)
OUT = "results"

SAMPLE_NUM = 2 ** 14
N_COARSE = 2 ** 8
N_QMSA = 2 ** 10
COARSE_MASS = 0.50
COARSE_MIN_WIDTH = 0.10
COARSE_MARGIN = 0.02
QMSA_LAMBDA = 1.25
QAE_STATE_QUBITS = 6
QAE_EVAL_QUBITS = 6
QAE_ESTIMATE_CROSS_MOMENTS = True
RETRAIN_SURROGATE = False

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
    "cation_ecof": {"low": [0.07, 0.1, 0.1, 1E-6], "up": [0.5, 0.9, 0.9, 8E-6]},
    "CEC": {"low": [1.0], "up": [6.0]},
    "bouncon": {
        "low": [10 ** -9.0, 8e-4, 8e-7, 7e-5, 2e-5, 1e-3],
        "up": [10 ** -6.5, 4e-3, 4e-6, 4e-4, 9e-5, 7e-3],
    },
}

print("--- 1. Loading data and training surrogate ---")
X_data = load_x(SAMPLE_NUM, TARGET_PARAMS)
obs_df = pd.read_excel("concentration.xlsx")
E0, sigma_df = compute_e0(SAMPLE_NUM, obs_df, OBS_KEYS, return_sigma_report=True)
E0_norm, E0_meta = normalize_e0(E0)
model = train_surrogate(
    X_data,
    E0_norm,
    seed=SEED,
    model_path=os.path.join("results", "surrogate_weight.joblib"),
    retrain=RETRAIN_SURROGATE,
)

print("\n--- 2. Coarse bound refinement ---")
d = len(TARGET_PARAMS)
X_coarse = lhs_samples(N_COARSE, d, bounds=[(0.0, 1.0)] * d, seed=SEED + 1)
Jc, Wc, Pc, _, _ = build_likelihood_table(model, X_coarse, E0_meta, TARGET_PARAMS, PARA_BOUNDS)
ESS_c = 1.0 / np.sum(Pc ** 2)
ref_bounds = refined_bounds(X_coarse, Wc, COARSE_MASS, COARSE_MIN_WIDTH, COARSE_MARGIN)

bounds_df = bounds_dataframe(ref_bounds, TARGET_PARAMS, PARA_BOUNDS)
coarse_meta = pd.DataFrame([{
    **E0_meta,
    "SAMPLE_NUM": SAMPLE_NUM,
    "N_COARSE": N_COARSE,
    "J_min": float(Jc.min()),
    "J_max": float(Jc.max()),
    "J_mean": float(Jc.mean()),
    "Z_EW": float(Wc.mean()),
    "ESS": float(ESS_c),
    "ESS_ratio": float(ESS_c / N_COARSE),
    "coarse_mass": COARSE_MASS,
}])
pack_sections({"meta": coarse_meta, "sigma": sigma_df, "refined_bounds": bounds_df}).to_csv(
    os.path.join(OUT, "coarse.csv"), index=False, encoding="utf-8-sig"
)

print("\n--- 3. QMSA MAP search ---")
X_q = lhs_samples(N_QMSA, d, bounds=ref_bounds, seed=SEED + 2)
J_q, W_q, P_q, _, table_q = build_likelihood_table(model, X_q, E0_meta, TARGET_PARAMS, PARA_BOUNDS)

qmsa = run_qmsa(J_q, seed=SEED + 3, max_oracle_calls=int(4 * np.sqrt(N_QMSA)), lambda_growth=QMSA_LAMBDA)

map_df = qmsa_map_dataframe(qmsa, X_q, table_q, TARGET_PARAMS, PARA_BOUNDS)
scalar_df = qmsa_scalar_dataframe(qmsa, table_q)
likelihood_meta = pd.DataFrame([{
    "N_QMSA": N_QMSA,
    "J_min": float(J_q.min()),
    "J_max": float(J_q.max()),
    "J_mean": float(J_q.mean()),
}])
pack_sections({
    "likelihood_meta": likelihood_meta,
    "map": map_df,
    "scalar": scalar_df,
    "history": qmsa["history"],
}).to_csv(os.path.join(OUT, "qmsa.csv"), index=False, encoding="utf-8-sig")

print("\n--- 4. QAE posterior uncertainty quantification ---")
qae = run_qae(
    X_q,
    W_q,
    TARGET_PARAMS,
    PARA_BOUNDS,
    map_index=int(qmsa["best_index"]),
    state_qubits=QAE_STATE_QUBITS,
    eval_qubits=QAE_EVAL_QUBITS,
    seed=SEED + 4,
    estimate_cross_moments=QAE_ESTIMATE_CROSS_MOMENTS,
)
# Keep only three CSV files, but include the refined table in qae.csv
# so the plotting script can reproduce the previous KDE diagnostics.
pack_sections({
    "meta": qae["meta"],
    "summary": qae["summary"],
    "terms": qae["terms"],
    "kde_samples": table_q,
    "correlation": matrix_to_long(qae["correlation"], "correlation"),
    "covariance_norm": matrix_to_long(qae["covariance_norm"], "covariance_norm"),
}).to_csv(os.path.join(OUT, "qae.csv"), index=False, encoding="utf-8-sig")

print("\n--- 5. Results validation ---")
xq_path = os.path.join(OUT, "Xq_for_UQ.npz")
np.savez_compressed(
    xq_path,
    X_q_norm=X_q,
    X_q_real=denorm_matrix(X_q, TARGET_PARAMS, PARA_BOUNDS),
    J_q=J_q,
    W_q=W_q,
    P_q=P_q,
    map_index=np.array(int(qmsa["best_index"]), dtype=np.int64),
)
