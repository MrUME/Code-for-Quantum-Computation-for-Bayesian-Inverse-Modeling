from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "results_quantum/quantum_results.txt"
OUT_DIR = ROOT / "results_quantum"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_QUBITS = 1
EVAL_QUBITS = 2

B5_MIN = 1.0e-3
B5_MAX = 7.0e-3
TRUE_B5 = 2.8628e-3
IDEAL_EVIDENCE = 0.127703
IDEAL_MOMENT_B5 = 0.040956

CASE_LABELS = ["s0d0", "s1d1", "s5d1", "s1d5", "s5d5"]
SINGLE_GATE_ERROR_PERCENT = np.array([0.0, 1.0, 5.0, 1.0, 5.0], dtype=float)
DOUBLE_GATE_ERROR_PERCENT = np.array([0.0, 1.0, 1.0, 5.0, 5.0], dtype=float)
BITSTRINGS = ["00", "01", "10", "11"]


def reorder_reverse_bits(probabilities):
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    expected = 2 ** EVAL_QUBITS

    reordered = np.zeros(expected, dtype=float)
    for bitstring, p in zip(BITSTRINGS, probabilities):
        reordered[int(bitstring[::-1], 2)] += float(p)
    total = float(np.sum(reordered))

    return reordered / total


def dirichlet_kernel_probability(delta, m):
    denom = np.sin(np.pi * delta)
    if abs(denom) < 1.0e-12:
        return 1.0
    return (np.sin(m * np.pi * delta) / (m * denom)) ** 2


def qae_probability(amplitude):
    m = 2 ** EVAL_QUBITS
    amplitude = np.clip(float(amplitude), 1.0e-12, 1.0 - 1.0e-12)
    phi = np.arcsin(np.sqrt(amplitude)) / np.pi
    probs = np.zeros(m, dtype=float)
    for y in range(m):
        probs[y] = (
            0.5 * dirichlet_kernel_probability(phi - y / m, m)
            + 0.5 * dirichlet_kernel_probability(1.0 - phi - y / m, m)
        )
    return probs / np.sum(probs)


def mle_amplitude(probabilities):
    observed = reorder_reverse_bits(probabilities)

    def nll(amplitude):
        fitted = np.clip(qae_probability(amplitude), 1.0e-300, 1.0)
        return -float(np.sum(observed * np.log(fitted)))

    result = minimize_scalar(
        nll,
        bounds=(1.0e-12, 1.0 - 1.0e-12),
        method="bounded",
        options={"xatol": 1.0e-14},
    )
    return float(result.x)


def load_quantum_results(path=INPUT_FILE):
    raw = np.loadtxt(path, dtype=float)
    n = len(CASE_LABELS)
    return raw[:n, :], raw[n:, :]


def denorm_b5(mean_norm):
    return B5_MIN + mean_norm * (B5_MAX - B5_MIN)


def convert_cases(evidence_probs, moment_probs):
    rows = []
    for i, case in enumerate(CASE_LABELS):
        e_hat = mle_amplitude(evidence_probs[i])
        m_hat = mle_amplitude(moment_probs[i])
        mean_norm = m_hat / e_hat
        mean_real = denorm_b5(mean_norm)

        e_abs = abs(e_hat - IDEAL_EVIDENCE)
        m_abs = abs(m_hat - IDEAL_MOMENT_B5)
        b5_abs = abs(mean_real - TRUE_B5)

        rows.append({
            "case": case,
            "single_gate_error_percent": float(SINGLE_GATE_ERROR_PERCENT[i]),
            "double_gate_error_percent": float(DOUBLE_GATE_ERROR_PERCENT[i]),
            "evidence_estimate": e_hat,
            "evidence_relative_error_percent": e_abs / abs(IDEAL_EVIDENCE) * 100.0,
            "ExW_bouncon5_estimate": m_hat,
            "ExW_bouncon5_relative_error_percent": m_abs / abs(IDEAL_MOMENT_B5) * 100.0,
            "bouncon5_mean_normalized": mean_norm,
            "bouncon5_mean_real": mean_real,
            "bouncon5_absolute_error": b5_abs,
            "bouncon5_relative_error_percent": b5_abs / abs(TRUE_B5) * 100.0,
        })
    return rows


def write_summary(rows, path):
    ideal_mean_norm = IDEAL_MOMENT_B5 / IDEAL_EVIDENCE
    ideal_mean_real = denorm_b5(ideal_mean_norm)
    ideal_b5_abs = abs(ideal_mean_real - TRUE_B5)
    ideal_b5_rel = ideal_b5_abs / abs(TRUE_B5) * 100.0

    lines = [
        "QAE GATE-ERROR",
        "=" * 100,
        "",
        f"STATE_QUBITS = {STATE_QUBITS}",
        f"EVAL_QUBITS = {EVAL_QUBITS}",
        f"B5 range = [{B5_MIN:.12e}, {B5_MAX:.12e}]",
        "",
        "Ideal values",
        "-" * 100,
        f"Ideal Evidence E[W] = {IDEAL_EVIDENCE:.12e}",
        f"Ideal E[xW] for bouncon[5] = {IDEAL_MOMENT_B5:.12e}",
        f"Ideal normalized posterior mean = {ideal_mean_norm:.12e}",
        f"Ideal real bouncon[5] = {ideal_mean_real:.12e}",
        f"True bouncon[5] = {TRUE_B5:.12e}",
        f"Ideal bouncon[5] absolute error = {ideal_b5_abs:.12e}",
        f"Ideal bouncon[5] relative error = {ideal_b5_rel:.6f}%",
        "",
        "Five gate-error cases",
        "-" * 100,
    ]

    for row in rows:
        lines.extend([
            (
                f"{row['case']} "
                f"(single={row['single_gate_error_percent']:.0f}%, "
                f"double={row['double_gate_error_percent']:.0f}%)"
            ),
            f"  Evidence = {row['evidence_estimate']:.12e}",
            f"  Evidence relative error = {row['evidence_relative_error_percent']:.6f}%",
            f"  E[xW] = {row['ExW_bouncon5_estimate']:.12e}",
            f"  E[xW] relative error = {row['ExW_bouncon5_relative_error_percent']:.6f}%",
            f"  bouncon[5] normalized mean = {row['bouncon5_mean_normalized']:.12e}",
            f"  bouncon[5] real mean = {row['bouncon5_mean_real']:.12e}",
            f"  bouncon[5] absolute error = {row['bouncon5_absolute_error']:.12e}",
            f"  bouncon[5] relative error = {row['bouncon5_relative_error_percent']:.6f}%",
            "",
        ])

    path.write_text("\n".join(lines), encoding="utf-8")


# main
evidence_probs, moment_probs = load_quantum_results()
rows = convert_cases(evidence_probs, moment_probs)
summary_path = OUT_DIR / "qae_gate_error.txt"
write_summary(rows, summary_path)

