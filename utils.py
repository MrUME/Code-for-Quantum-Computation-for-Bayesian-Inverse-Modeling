import os
import math
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import Diagonal, StatePreparation
from qiskit_algorithms import AmplitudeEstimation, EstimationProblem
from qiskit.primitives import StatevectorSampler
from scipy import interpolate


def load_x(n, target_params):
    cols = []
    for item, idx in target_params:
        cols.append(np.load(os.path.join("Parameter", f"{item}_norm.npy"))[:n, idx].astype(float))
    return np.clip(np.array(cols).T, 0.0, 1.0)


def compute_e0(n, obs_df, obs_keys, floor=1e-8, return_sigma_report=False):
    e0 = np.zeros(n, dtype=float)
    rows = []

    for key in obs_keys:
        sim = np.load(os.path.join("Concentrate", f"{key}_model.npy")).astype(float)
        obs = np.asarray(obs_df[key].values, dtype=float)
        obs = obs[np.isfinite(obs)]

        sig = max(0.10 * float(np.mean(np.abs(obs))), floor)
        rows.append({
            "key": key,
            "obs_mean": float(np.mean(obs)),
            "obs_min": float(np.min(obs)),
            "obs_max": float(np.max(obs)),
            "obs_std": float(np.std(obs)),
            "obs_mean_abs": float(np.mean(np.abs(obs))),
            "sigma": float(sig),
            "sigma_source": "10pct_mean_abs_obs",
        })

        if sim.ndim == 2:
            nt, ns = sim.shape
            m = min(ns, n)
            target = obs.reshape(-1, 1) if len(obs) == nt else np.full((nt, 1), np.mean(obs))
            e0[:m] += 0.5 * np.mean(((sim[:, :m] - target) / sig) ** 2, axis=0)
        elif sim.ndim == 1:
            m = min(len(sim), n)
            e0[:m] += 0.5 * ((sim[:m] - np.mean(obs)) / sig) ** 2

    finite = e0[np.isfinite(e0)]
    fallback = np.max(finite) if finite.size else 1.0
    e0 = np.nan_to_num(e0, nan=fallback, posinf=fallback, neginf=fallback)
    return (e0, pd.DataFrame(rows)) if return_sigma_report else e0


def normalize_e0(e0):
    e0_min = float(np.min(e0))
    e0_max = float(np.max(e0))
    scale = max(e0_max - e0_min, 1e-300)
    return (np.asarray(e0, dtype=float) - e0_min) / scale, {
        "E0_min": e0_min,
        "E0_max": e0_max,
        "E0_scale": scale,
    }


def denormalize_e0(e0_norm, meta):
    return np.asarray(e0_norm, dtype=float) * float(meta["E0_scale"]) + float(meta["E0_min"])


def denorm_x(x, low, up):
    return low + np.asarray(x, dtype=float) * (up - low)


def get_param_bounds(target_params, para_bounds):
    names, bounds = [], []
    for item, idx in target_params:
        names.append(f"{item}[{idx}]")
        bounds.append((float(para_bounds[item]["low"][idx]), float(para_bounds[item]["up"][idx])))
    return names, bounds


def denorm_matrix(x_norm, target_params, para_bounds):
    _, bounds = get_param_bounds(target_params, para_bounds)
    x_real = np.zeros_like(x_norm, dtype=float)
    for j, (lo, hi) in enumerate(bounds):
        x_real[:, j] = denorm_x(x_norm[:, j], lo, hi)
    return x_real


def train_surrogate(x, y_norm, seed, model_path, retrain=False):
    cache_dir = os.path.dirname(model_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(model_path) and not retrain:
        model = joblib.load(model_path)
        pred = model.predict(x)
        print(f"Loaded surrogate: {model_path}")
        print(f"Surrogate R2 all={r2_score(y_norm, pred):.4e}, RMSE all={math.sqrt(mean_squared_error(y_norm, pred)):.4e}")
        return model

    xtr, xte, ytr, yte = train_test_split(x, y_norm, test_size=0.2, random_state=seed)
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=800,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
        random_state=seed,
    )
    model.fit(xtr, ytr)
    print(f"Surrogate R2 test={r2_score(yte, model.predict(xte)):.4e}, RMSE test={math.sqrt(mean_squared_error(yte, model.predict(xte))):.4e}")
    joblib.dump(model, model_path)
    return model


def lhs_samples(n, d, bounds=None, seed=0):
    u = qmc.LatinHypercube(d=d, seed=seed).random(n)
    if bounds is None:
        return np.clip(u, 0.0, 1.0)
    x = np.zeros_like(u)
    for j, (lo, hi) in enumerate(bounds):
        x[:, j] = lo + u[:, j] * (hi - lo)
    return np.clip(x, 0.0, 1.0)


def build_likelihood_table(model, x, e0_meta, target_params=None, para_bounds=None):
    pred_norm = np.asarray(model.predict(x), dtype=float)
    finite = pred_norm[np.isfinite(pred_norm)]
    fallback = float(np.max(finite)) if finite.size else 1.0
    pred_norm = np.nan_to_num(pred_norm, nan=fallback, posinf=fallback, neginf=fallback)
    pred_e0 = denormalize_e0(pred_norm, e0_meta)

    j = pred_e0 - float(np.min(pred_e0))
    j = np.maximum(np.nan_to_num(j, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    w = np.exp(np.maximum(-j, -745.0))
    w = np.clip(np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    p = w / (np.sum(w) + 1e-300)

    df = pd.DataFrame({
        "sample_index": np.arange(len(x), dtype=int),
        "E0_norm_pred_raw": pred_norm,
        "E0_pred_raw": pred_e0,
        "J_shift": j,
        "weight": w,
        "posterior_prob": p,
    })

    if target_params is not None and para_bounds is not None:
        names, _ = get_param_bounds(target_params, para_bounds)
        x_real = denorm_matrix(x, target_params, para_bounds)
        for col, name in enumerate(names):
            safe = name.replace("[", "_").replace("]", "")
            df[f"{safe}_norm"] = x[:, col]
            df[f"{safe}_real"] = x_real[:, col]
    return j, w, p, pred_norm, df


def weighted_quantile(x, w, qs):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if np.sum(w) <= 0:
        w = np.ones_like(x)
    idx = np.argsort(x)
    xs, ws = x[idx], w[idx]
    cdf = np.cumsum(ws) / np.sum(ws)
    return np.interp(qs, cdf, xs)


def refined_bounds(x, w, mass=0.5, min_width=0.1, margin=0.02):
    ql = (1.0 - mass) / 2.0
    qh = 1.0 - ql
    wp = w / (np.sum(w) + 1e-300)
    bounds = []
    for j in range(x.shape[1]):
        lo, hi = weighted_quantile(x[:, j], wp, [ql, qh])
        lo, hi = float(lo - margin), float(hi + margin)
        if hi - lo < min_width:
            c = 0.5 * (lo + hi)
            lo, hi = c - 0.5 * min_width, c + 0.5 * min_width
        lo, hi = max(0.0, lo), min(1.0, hi)
        bounds.append((lo, hi) if hi > lo else (0.0, 1.0))

    print("Refined bounds:")
    print(bounds)
    return bounds


def bounds_dataframe(bounds, target_params, para_bounds):
    names, real_bounds = get_param_bounds(target_params, para_bounds)
    rows = []
    for j, name in enumerate(names):
        lo, hi = bounds[j]
        rlo, rhi = real_bounds[j]
        rows.append({
            "param": name,
            "low_norm": lo,
            "up_norm": hi,
            "low_real": denorm_x(lo, rlo, rhi),
            "up_real": denorm_x(hi, rlo, rhi),
            "width_norm": hi - lo,
        })
    return pd.DataFrame(rows)


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.z(0) if n_qubits == 1 else qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover(objective, threshold, n_iters):
    obj = np.asarray(objective, dtype=float)
    n_qubits = int(round(math.log2(len(obj))))
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    signs = np.ones(len(obj), dtype=complex)
    signs[obj < threshold] = -1.0 + 0j
    oracle = Diagonal(signs.tolist())
    diffuser = build_diffuser(n_qubits)
    for _ in range(int(n_iters)):
        qc.append(oracle, range(n_qubits))
        qc.compose(diffuser, inplace=True)
    return qc


def run_qmsa(objective, seed=0, max_oracle_calls=None, lambda_growth=1.25, verbose=True):
    obj = np.asarray(objective, dtype=float)
    n = len(obj)
    n_qubits = int(round(math.log2(n)))

    sqrt_n = math.sqrt(n)
    max_oracle_calls = int(4 * sqrt_n) if max_oracle_calls is None else int(max_oracle_calls)
    rng = np.random.default_rng(seed)
    table_min_index = int(np.argmin(obj))
    table_min_value = float(obj[table_min_index])
    tol = max(1e-12, 1e-10 * max(1.0, abs(table_min_value)))
    best_index = int(rng.integers(0, n))
    best_value = float(obj[best_index])
    m, calls, step = 1.0, 0, 0
    history = []

    while calls < max_oracle_calls and best_value > table_min_value + tol:
        r = int(rng.integers(0, max(1, int(math.ceil(m)))))
        r = min(r, max_oracle_calls - calls)
        if r == 0:
            candidate = int(rng.integers(0, n))
        else:
            qc = build_grover(obj, best_value, r)
            prob = np.real(Statevector.from_instruction(qc).probabilities())
            prob = np.maximum(prob, 0.0)
            candidate = int(rng.choice(n, p=prob / np.sum(prob)))

        calls += r
        step += 1
        candidate_value = float(obj[candidate])
        improved = candidate_value < best_value
        history.append({
            "step": step,
            "grover_iters": r,
            "oracle_calls_cumulative": calls,
            "candidate_index": candidate,
            "candidate_J": candidate_value,
            "best_index_before_update": best_index,
            "best_J_before_update": best_value,
            "improved": improved,
        })
        if improved:
            best_index, best_value = candidate, candidate_value
            if verbose:
                print(f"  step={step:03d}, iters={r:3d}, calls={calls:4d}, index={best_index}, J={best_value:.4e}")
        m = min(lambda_growth * m, sqrt_n)

    return {
        "best_index": int(best_index),
        "best_value": float(best_value),
        "table_min_index": table_min_index,
        "table_min_value": table_min_value,
        "success_by_value": bool(best_value <= table_min_value + tol),
        "near_min_count": int(np.sum(obj <= table_min_value + tol)),
        "tol": tol,
        "N": int(n),
        "state_qubits": int(n_qubits),
        "sqrt_N": float(sqrt_n),
        "oracle_calls": int(calls),
        "classical_query_count_np_min": int(n),
        "query_reduction_factor": float(n / max(calls, 1)),
        "history": pd.DataFrame(history),
    }


def qmsa_map_dataframe(qmsa, x, table, target_params, para_bounds):
    names, _ = get_param_bounds(target_params, para_bounds)
    x_real = denorm_matrix(x, target_params, para_bounds)
    qidx, midx = int(qmsa["best_index"]), int(qmsa["table_min_index"])
    rows = []
    for j, name in enumerate(names):
        rows.append({
            "param": name,
            "QMSA_index": qidx,
            "QMSA_norm": x[qidx, j],
            "QMSA_real": x_real[qidx, j],
            "np_min_index": midx,
            "np_min_norm": x[midx, j],
            "np_min_real": x_real[midx, j],
        })
    return pd.DataFrame(rows)


def qmsa_scalar_dataframe(qmsa, table):
    qidx, midx = int(qmsa["best_index"]), int(qmsa["table_min_index"])
    return pd.DataFrame([{
        "N_QMSA": qmsa["N"],
        "sqrt_N": qmsa["sqrt_N"],
        "QMSA_index": qidx,
        "QMSA_J": qmsa["best_value"],
        "QMSA_E0_pred_raw": float(table.loc[qidx, "E0_pred_raw"]),
        "np_min_index": midx,
        "np_min_J": qmsa["table_min_value"],
        "np_min_E0_pred_raw": float(table.loc[midx, "E0_pred_raw"]),
        "abs_J_error": abs(qmsa["best_value"] - qmsa["table_min_value"]),
        "success_by_value": qmsa["success_by_value"],
        "oracle_calls_QMSA": qmsa["oracle_calls"],
        "classical_query_count_np_min": qmsa["classical_query_count_np_min"],
        "query_reduction_factor": qmsa["query_reduction_factor"],
    }])


def values_histogram(f_values, n_bin_qubits=6):
    f = np.clip(np.asarray(f_values, dtype=float), 0.0, 1.0)
    n_bins = 2 ** int(n_bin_qubits)
    idx = np.floor(f * (n_bins - 1)).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    sums = np.bincount(idx, weights=f, minlength=n_bins).astype(float)
    probs = counts / (np.sum(counts) + 1e-300)
    values = np.zeros(n_bins, dtype=float)
    values[counts > 0] = sums[counts > 0] / counts[counts > 0]
    return values, probs


def amplitude(values, probs):
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    probs = np.maximum(np.asarray(probs, dtype=float), 0.0)
    probs = probs / (np.sum(probs) + 1e-300)
    n_bins = len(values)
    n_bin_qubits = int(round(math.log2(n_bins)))
    state = np.zeros(2 * n_bins, dtype=complex)
    state[0::2] = np.sqrt(probs * (1.0 - values))
    state[1::2] = np.sqrt(probs * values)
    state = state / (np.linalg.norm(state) + 1e-300)
    qc = QuantumCircuit(n_bin_qubits + 1, name="A_hist")
    try:
        qc.append(StatePreparation(state, normalize=False), range(n_bin_qubits + 1))
    except TypeError:
        qc.append(StatePreparation(state), range(n_bin_qubits + 1))
    return qc, 0


def qae_circuit(f_values, n_bin_qubits=6, eval_qubits=6, seed=0):
    f = np.clip(np.asarray(f_values, dtype=float), 0.0, 1.0)
    values, probs = values_histogram(f, n_bin_qubits=n_bin_qubits)
    state_preparation, objective_qubit = amplitude(values, probs)
    problem = EstimationProblem(state_preparation=state_preparation, objective_qubits=[objective_qubit])
    result = AmplitudeEstimation(
        num_eval_qubits=int(eval_qubits),
        sampler=StatevectorSampler(seed=seed),
    ).estimate(problem)
    estimate = getattr(result, "mle_processed", None)
    if estimate is None:
        estimate = getattr(result, "estimation_processed", getattr(result, "estimation", np.nan))
    estimate = float(np.clip(float(estimate), 0.0, 1.0))
    raw_target = float(np.mean(f))
    hist_target = float(np.sum(values * probs))
    return estimate, raw_target, hist_target


def run_qae(x, w, target_params, para_bounds, map_index, state_qubits=6, eval_qubits=6, seed=0, estimate_cross_moments=True):
    x = np.asarray(x, dtype=float)
    w = np.clip(np.asarray(w, dtype=float), 0.0, 1.0)
    n, d = x.shape
    names, real_bounds = get_param_bounds(target_params, para_bounds)
    term_rows = []

    def estimate(label, f_values, local_seed):
        print(f"QAE predicting: {label}")
        est, raw, hist = qae_circuit(f_values, state_qubits, eval_qubits, local_seed)
        term_rows.append({
            "quantity": label,
            "qae_E_f": est,
            "raw_mean_E_f_for_validation": raw,
            "histogram_target_E_f": hist,
            "abs_error_qae_vs_raw": abs(est - raw),
            "abs_error_qae_vs_histogram": abs(est - hist),
        })
        return est

    z = max(estimate("Z_EW", w, seed + 10), 1e-300)
    m1 = np.zeros(d)
    m2 = np.zeros(d)
    for j, name in enumerate(names):
        m1[j] = estimate(f"M1_E_xW__{name}", x[:, j] * w, seed + 100 + j)
        m2[j] = estimate(f"M2_E_x2W__{name}", (x[:, j] ** 2) * w, seed + 200 + j)

    if estimate_cross_moments:
        for j in range(d):
            for k in range(j, d):
                estimate(f"Mjk_E_xxW__{names[j]}__{names[k]}", x[:, j] * x[:, k] * w, seed + 300 + 31 * j + k)

    mean = np.clip(m1 / z, 0.0, 1.0)
    second = np.clip(m2 / z, 0.0, 1.0)
    var = np.maximum(second - mean ** 2, 0.0)
    std = np.sqrt(var)

    w_post = w / (np.sum(w) + 1e-300)
    mean_corr = np.sum(x * w_post[:, None], axis=0)
    xc = x - mean_corr[None, :]
    cov = (xc * w_post[:, None]).T @ xc
    std_corr = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = cov / (std_corr[:, None] * std_corr[None, :] + 1e-300)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    x_real = denorm_matrix(x, target_params, para_bounds)
    exact_mean_norm = np.sum(x * w_post[:, None], axis=0)
    exact_var_norm = np.sum(((x - exact_mean_norm[None, :]) ** 2) * w_post[:, None], axis=0)
    exact_std_norm = np.sqrt(np.maximum(exact_var_norm, 0.0))

    rows = []
    for j, name in enumerate(names):
        lo, hi = real_bounds[j]
        ci65 = weighted_quantile(x[:, j], w_post, [0.175, 0.825])
        ci95 = weighted_quantile(x[:, j], w_post, [0.025, 0.975])
        rows.append({
            "param": name,
            "MAP_norm": x[int(map_index), j],
            "MAP_real": x_real[int(map_index), j],
            "mean_norm_qae": mean[j],
            "mean_real_qae": denorm_x(mean[j], lo, hi),
            "var_norm_qae": var[j],
            "std_norm_qae": std[j],
            "std_real_qae": std[j] * (hi - lo),
            "mean_norm_exact": exact_mean_norm[j],
            "mean_real_exact": denorm_x(exact_mean_norm[j], lo, hi),
            "var_norm_exact": exact_var_norm[j],
            "std_norm_exact": exact_std_norm[j],
            "std_real_exact": exact_std_norm[j] * (hi - lo),
            "CI65_low_norm": ci65[0],
            "CI65_high_norm": ci65[1],
            "CI65_low_real": denorm_x(ci65[0], lo, hi),
            "CI65_high_real": denorm_x(ci65[1], lo, hi),
            "CI95_low_norm": ci95[0],
            "CI95_high_norm": ci95[1],
            "CI95_low_real": denorm_x(ci95[0], lo, hi),
            "CI95_high_real": denorm_x(ci95[1], lo, hi),
        })

    meta = pd.DataFrame([{
        "N": int(n),
        "state_qubits": int(state_qubits),
        "eval_qubits": int(eval_qubits),
        "estimate_cross_moments": bool(estimate_cross_moments),
        "Z_EW_qae": float(z),
        "Z_EW_raw_exact_for_validation": float(np.mean(w)),
        "ESS_exact_from_weights": float(1.0 / np.sum(w_post ** 2)),
        "ESS_ratio_exact_from_weights": float((1.0 / np.sum(w_post ** 2)) / n),
        "map_index": int(map_index),
    }])
    return {
        "summary": pd.DataFrame(rows),
        "correlation": pd.DataFrame(corr, index=names, columns=names),
        "covariance_norm": pd.DataFrame(cov, index=names, columns=names),
        "terms": pd.DataFrame(term_rows),
        "meta": meta,
    }


def pack_sections(sections):
    frames = []
    for name, df in sections.items():
        tmp = df.copy()
        tmp.insert(0, "section", name)
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True, sort=False)


def matrix_to_long(mat, value_name="value"):
    df = mat.copy()
    df.index.name = "row_param"
    return df.reset_index().melt(id_vars="row_param", var_name="col_param", value_name=value_name)


def site_concentration(column):
    site_concentration=pd.read_excel('.\\concentration.xlsx')
    column_values = site_concentration.columns.values
    site_value=np.ones((site_concentration.shape[0],2))
    column_0=np.where(column_values==column[0])[0][0]
    column_1=np.where(column_values==column[1])[0][0]
    site_value[:, 0] = site_concentration[column_values[column_0]]
    site_value[:, 1] = site_concentration[column_values[column_1]]
    value_nan_index = np.where(np.isnan(site_value))
    value_nan = site_value[value_nan_index[0], value_nan_index[1]].reshape((-1, 2))
    site_value = site_value[0:site_value.shape[0]-value_nan.shape[0]]
    return site_value


def aqui_con_read(t,elem_num,skiprow=10):
    for i in range (t+1):
        aqui_con=np.loadtxt('.\\aqui_con.dat', skiprows=skiprow, max_rows=elem_num, dtype=float)
        aqui_con=np.expand_dims(aqui_con,axis=1)
        if i==0:
            aqui_con_result=aqui_con
        else:
            aqui_con_result=np.concatenate((aqui_con_result,aqui_con),axis=1)
        skiprow+=elem_num+1
    return aqui_con_result


def get_aquicon_index(name):
    result = open('.\\aqui_con.dat', 'r')
    result_lines = result.readlines()
    line = result_lines[8]
    line = line.split(',')
    line_index=line[0:len(line)-1]
    for i in range (len(line_index)):
        a=line_index[i]
        if name in a:
            name_index=i
    return name_index


def simulation_data(name_aqui,name_site,aqui_con_t=5,aqui_con_enum=17):
    aqui_con = aqui_con_read(t=aqui_con_t, elem_num=aqui_con_enum)
    index = get_aquicon_index(name_aqui)
    Dis_index = get_aquicon_index('VARIABLES =X')
    model_data = aqui_con[:, -1, [Dis_index, index]]
    site_data = site_concentration(name_site)
    site_D=site_data[:,0]
    # 插值
    f=interpolate.interp1d(model_data[:,0],model_data[:,1],kind='quadratic')
    site_model=f(site_D)
    return site_model


def run_exe():
    cmd_root='echo.|SOWCOM_V2_EOS9.exe'
    os.system(cmd_root)
    if not os.path.exists('.\\run_exe'):
        os.mkdir('.\\run_exe')
    return


def chemical_keywords_index(chemical_lines,Keywords):
    keywords_index = np.zeros(len(Keywords), dtype=int)
    for i in range(len(Keywords)):
        key=Keywords[i]
        for line in chemical_lines:
            index_key=chemical_lines.index(line)
            line1=chemical_lines[index_key]
            if key in line1:
                keywords_line=line1
                index=chemical_lines.index(keywords_line)
                keywords_index[i]=index
    return keywords_index


def flow_open():
    flow = open('.\\Multiphase Flow.inp', 'r')
    flow_lines = flow.readlines()
    flow.close()
    return flow_lines


def chemical_open():
    chemical=open('.\\Geochemical.inp','r')
    chemical_lines=chemical.readlines()
    chemical.close()
    return chemical_lines


def flow_write(flow_lines):
    newfile=open('.\\Multiphase Flow.inp','w')
    for newline in flow_lines:
        newfile.write(newline)
    newfile.close()
    return


def chemical_write(chemical_lines):
    newfile=open('.\\Geochemical.inp','w')
    for newline in chemical_lines:
        newfile.write(newline)
    newfile.close()
    return


def per_para(flow_lines,PER1,PER2,PER3):
    ROCKS_index=flow_lines.index("ROCKS----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8\n")
    ROCKS1=flow_lines[ROCKS_index+1]
    A=ROCKS1[0:30]
    B=ROCKS1[60:82]

    flow_lines[ROCKS_index+1]=A+"%-10.4e"%(PER1)+"%-10.4e"%(PER2)+"%-10.4e"%(PER3)+B

    return flow_lines


def GENER_para(flow_lines,GENER_value):
    GENER_num = GENER_value.shape[0]
    GENER_index0 = flow_lines.index(
        'GENER----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8\n')
    for i in range(GENER_num):
        GENER_index = GENER_index0 + i + 1
        GENER_line = flow_lines[GENER_index]
        GENER_line1 = GENER_line[0:40]
        GENER_line2 = GENER_value[i]
        GENER_line = GENER_line1 + "{:10.3E}".format(GENER_line2) + '\n'
        flow_lines[GENER_index] = GENER_line
    return flow_lines


def Exchange_cation_coeff(chemical_lines,parameter,add_index=3):
    Keywords=['EXCHANGEABLE CATIONS','INITIAL AND BOUDARY WATER TYPES']
    index=chemical_keywords_index(chemical_lines,Keywords)
    for i in range(len(parameter)):
        index_para=index[0]+add_index+i
        para_line=chemical_lines[index_para]
        para_line=para_line.split()
        para_line[3]=parameter[i]
        chemical_lines[index_para]="%-20s"%(para_line[0])+"%-13s"%(para_line[1])+"%-13s"%(para_line[2])+\
                                   "%-10.4e"%(para_line[3])+'\n'
    return chemical_lines


def Initial_boudary(chemical_lines,parameter,add_ini_index=5,reduce_bou_index=2):
    Keywords=['INITIAL AND BOUDARY WATER TYPES','INITIAL MINERAL ZONES']
    index=chemical_keywords_index(chemical_lines,Keywords)
    for i in range (max(parameter.shape)):
        # 初始水
        index_inipara=index[0]+add_ini_index+i
        inipara_line=chemical_lines[index_inipara]
        inipara_line=inipara_line.split()
        inipara_line[2]=parameter[0,i]
        inipara_line[3]=parameter[0,i]
        chemical_lines[index_inipara]="%-11s"%(inipara_line[0])+"%-9s"%(inipara_line[1])+"%-15.3e"%(inipara_line[2])+\
                                      "%-12.3e"%(inipara_line[3])+"%-4s"%(inipara_line[4])+"%-5s"%(inipara_line[5])+\
                                      "%-8s"%(inipara_line[6])+'\n'
        # 边界水
        index_boupara=index[1]-reduce_bou_index-parameter.shape[1]+i
        boupara_line=chemical_lines[index_boupara]
        boupara_line=boupara_line.split()
        boupara_line[2]=parameter[1,i]
        boupara_line[3]=parameter[1,i]
        chemical_lines[index_boupara]="%-11s"%(boupara_line[0])+"%-9s"%(boupara_line[1])+"%-15.3e"%(boupara_line[2])+\
                                      "%-12.3e"%(boupara_line[3])+"%-4s"%(boupara_line[4])+"%-5s"%(boupara_line[5])+\
                                      "%-8s"%(boupara_line[6])+'\n'
    return chemical_lines


def para_CEC(parameter,chemical_lines,add_index=3):
    Keywords=['INITIAL ZONES OF CATION EXCHANGE','end']
    index = chemical_keywords_index(chemical_lines, Keywords)
    index_CEC=index[0]+add_index
    CEC_line=chemical_lines[index_CEC]
    CEC_line=CEC_line.split()
    CEC_line[1]=float(np.asarray(parameter, dtype=float).reshape(-1)[0])
    chemical_lines[index_CEC]="%-20s"%(CEC_line[0])+"%-6.4f"%(CEC_line[1])+'\n'
    return chemical_lines


def parameter_sample(sample_num,num_para,para_low,para_up):
    para_norm=np.ones((sample_num,num_para))
    for i in range (num_para):
        para_norm[:,i]=np.random.uniform(0,1,sample_num)
    para_low_tile=np.tile(para_low,(sample_num,1))
    para_up_tile=np.tile(para_up,(sample_num,1))
    para=para_norm*(para_up_tile-para_low_tile)+para_low_tile
    return para_norm,para


def para_inverse_fix(fix_para,fix_para_item,inverse_para):
    total_dim=len(fix_para)+len(inverse_para)
    total_item=np.arange(total_dim)
    inverse_item=np.setdiff1d(total_item,fix_para_item)
    para=np.ones(total_dim)
    for i in range(total_dim):
        if i in fix_para_item:
            fix_idx = np.where(fix_para_item == i)[0][0]
            para[i] = float(fix_para[fix_idx])
        else:
            inverse_idx = np.where(inverse_item == i)[0][0]
            para[i] = float(inverse_para[inverse_idx])
    return para
