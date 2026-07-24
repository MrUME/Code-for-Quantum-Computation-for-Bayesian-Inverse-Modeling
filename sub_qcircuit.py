import os
import math
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, qasm2
from qiskit_algorithms import AmplitudeEstimation, EstimationProblem
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import StatePreparation


SEED = 994
np.random.seed(SEED)

RESULT_DIR = "results"
OUT = "results_quantum"
os.makedirs(OUT, exist_ok=True)

QMSA_SMALL_QUBITS = 6
QMSA_SMALL_N = 2 ** QMSA_SMALL_QUBITS
QMSA_DEMO_MODE = "threshold_top_k"
QMSA_MARKED_TOP_K = 1
QMSA_GROVER_ITERS = 1

QAE_STATE_QUBITS = 1
QAE_EVAL_QUBITS = 2

TARGET_NORM_COL = "bouncon_5_norm"


def save_qasm_txt(path, circuit):
    qasm_text = qasm2.dumps(circuit)
    with open(path, "w", encoding="utf-8") as f:
        f.write(qasm_text)



def section_csv(path, section_name):
    df = pd.read_csv(path)

    sec = df[df["section"] == section_name].copy()

    sec = sec.drop(columns=["section"]).dropna(axis=1, how="all")
    return sec.reset_index(drop=True)


def load_refined_table(result_dir=RESULT_DIR):
    table = section_csv(os.path.join(result_dir, "qae.csv"), "kde_samples")
    for col in table.columns:
        table[col] = pd.to_numeric(table[col], errors="ignore")

    return table


# QMSA circuit

def select_small_qmsa_table(table, n_state):
    order = np.argsort(table["J_shift"].values.astype(float))
    small = table.iloc[order[:n_state]].reset_index(drop=True).copy()
    return small


def append_single_basis_phase_flip(qc, marked_index, n_qubits):
    for q in range(n_qubits):
        if ((marked_index >> q) & 1) == 0:
            qc.x(q)

    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    for q in range(n_qubits):
        if ((marked_index >> q) & 1) == 0:
            qc.x(q)


def append_marked_state_oracle(qc, marked_indices, n_qubits):
    for idx in marked_indices:
        append_single_basis_phase_flip(qc, int(idx), n_qubits)


def append_diffuser(qc, n_qubits):
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


def choose_marked_indices(objective, mode="threshold_top_k", top_k=4):
    objective = np.asarray(objective, dtype=float)
    n = len(objective)
    order = np.argsort(objective)

    if mode == "single_best":
        return np.array([int(order[0])], dtype=int)
    if mode == "threshold_top_k":
        k = int(max(1, min(top_k, n - 1)))
        return order[:k].astype(int)


def build_grover_measured_circuit(n_qubits, marked_indices, n_iters=1):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(int(n_iters)):
        append_marked_state_oracle(qc, marked_indices, n_qubits)
        append_diffuser(qc, n_qubits)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def build_qmsa_measured_circuit(small_table, mode="threshold_top_k", top_k=4, n_iters=1):
    objective = small_table["J_shift"].values.astype(float)
    n = len(objective)
    n_qubits = int(round(math.log2(n)))
    marked = choose_marked_indices(objective, mode=mode, top_k=top_k)
    return build_grover_measured_circuit(n_qubits, marked, n_iters=n_iters)


# QAE circuit

def compress_values_to_histogram(raw_values, n_bin_qubits):
    raw_values = np.clip(
        np.nan_to_num(np.asarray(raw_values, dtype=float), nan=0.0, posinf=1.0, neginf=0.0),
        0.0, 1.0,
    )
    n_bins = 2 ** int(n_bin_qubits)
    order = np.argsort(raw_values)
    groups = np.array_split(order, n_bins)

    bin_values = np.zeros(n_bins, dtype=float)
    probs = np.zeros(n_bins, dtype=float)
    for k, g in enumerate(groups):
        if len(g) > 0:
            bin_values[k] = float(np.mean(raw_values[g]))
            probs[k] = float(len(g) / len(raw_values))
    probs = probs / (np.sum(probs) + 1e-300)
    return bin_values, probs


def amplitude_circuit(values, probs):
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    probs = np.maximum(np.asarray(probs, dtype=float), 0.0)
    probs = probs / (np.sum(probs) + 1e-300)

    n_bins = len(values)
    n_state = int(round(math.log2(n_bins)))

    objective_qubit = n_state
    qc = QuantumCircuit(n_state + 1, name="A")
    amps = np.sqrt(probs).astype(float)

    qc.append(StatePreparation(amps.tolist()), range(n_state))

    controls = list(range(n_state))
    for k, val in enumerate(values):
        theta = 2.0 * math.asin(math.sqrt(float(val)))
        if abs(theta) < 1e-15:
            continue
        for q in range(n_state):
            if ((k >> q) & 1) == 0:
                qc.x(q)
        if n_state == 0:
            qc.ry(theta, objective_qubit)
        elif n_state == 1:
            qc.cry(theta, controls[0], objective_qubit)
        else:
            qc.mcry(theta, controls, objective_qubit, None, mode="noancilla")

        for q in range(n_state):
            if ((k >> q) & 1) == 0:
                qc.x(q)
    return qc, objective_qubit


def build_qae_complete_circuit(raw_values, state_qubits=2, eval_qubits=2):
    raw_values = np.clip(
        np.nan_to_num(np.asarray(raw_values, dtype=float), nan=0.0, posinf=1.0, neginf=0.0),
        0.0, 1.0,
    )
    values, probs = compress_values_to_histogram(raw_values, n_bin_qubits=state_qubits)
    a_circuit, objective_qubit = amplitude_circuit(values, probs)
    problem = EstimationProblem(
        state_preparation=a_circuit,
        objective_qubits=[objective_qubit],
    )
    ae = AmplitudeEstimation(
        num_eval_qubits=int(eval_qubits),
        sampler=StatevectorSampler(),
    )

    return ae.construct_circuit(estimation_problem=problem, measurement=True)




table = load_refined_table(RESULT_DIR)
small = select_small_qmsa_table(table, QMSA_SMALL_N)

# QMSA circuit
qmsa = build_qmsa_measured_circuit(
    small, mode=QMSA_DEMO_MODE, top_k=QMSA_MARKED_TOP_K, n_iters=QMSA_GROVER_ITERS,
)
path_qmsa = os.path.join(OUT, "qmsa_circuit_qasm.txt")
save_qasm_txt(path_qmsa, qmsa)

# QAE for E[W] and E[x W]
w = np.clip(np.nan_to_num(small["weight"].values.astype(float), nan=0.0), 0.0, 1.0)
x = np.clip(np.nan_to_num(small[TARGET_NORM_COL].values.astype(float), nan=0.0), 0.0, 1.0)

path_ew = os.path.join(OUT, "qae_evidence_circuit_qasm.txt")
path_m1 = os.path.join(OUT, "qae_E_xW_bouncon_5_circuit_qasm.txt")
save_qasm_txt(path_ew, build_qae_complete_circuit(w, QAE_STATE_QUBITS, QAE_EVAL_QUBITS))
save_qasm_txt(path_m1, build_qae_complete_circuit(x * w, QAE_STATE_QUBITS, QAE_EVAL_QUBITS))
