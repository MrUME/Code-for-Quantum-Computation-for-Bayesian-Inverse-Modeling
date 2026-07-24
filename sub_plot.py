from __future__ import annotations
import argparse
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager, gridspec
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
QMSA_FILE = RESULTS_DIR / "qmsa.csv"
QAE_FILE = RESULTS_DIR / "qae.csv"
COARSE_FILE = RESULTS_DIR / "coarse.csv"
UQ_DIR = RESULTS_DIR / "uq"
BASE_FONT_SIZE = 10.5
TIMES_FONTS = [
    Path(r"C:\Windows\Fonts\times.ttf"),
    Path(r"C:\Windows\Fonts\timesbd.ttf"),
    Path(r"C:\Windows\Fonts\timesi.ttf"),
    Path(r"C:\Windows\Fonts\timesbi.ttf"),
]

PARAMS = [
    "cation_ecof_1", "cation_ecof_2", "cation_ecof_3",
    "CEC_0", "bouncon_1", "bouncon_5",
]
PARAM_LABELS = {
    "cation_ecof_1": "cation-selectivity coefficient 1",
    "cation_ecof_2": "cation-selectivity coefficient 2",
    "cation_ecof_3": "cation-selectivity coefficient 3",
    "CEC_0": "cation-exchange capacity",
    "bouncon_1": "boundary-concentration parameter 1",
    "bouncon_5": "boundary-concentration parameter 2",
}
SOURCE_LABELS = {
    "cation_ecof_1": "cation_ecof[1]",
    "cation_ecof_2": "cation_ecof[2]",
    "cation_ecof_3": "cation_ecof[3]",
    "CEC_0": "CEC[0]",
    "bouncon_1": "bouncon[1]",
    "bouncon_5": "bouncon[5]",
}
PRIOR_RANGES = {
    "cation_ecof_1": (0.1, 0.9),
    "cation_ecof_2": (0.1, 0.9),
    "cation_ecof_3": (1e-6, 8e-6),
    "CEC_0": (1.0, 6.0),
    "bouncon_1": (8e-4, 4e-3),
    "bouncon_5": (1e-3, 7e-3),
}
VARIABLES = ["PH", "Mg", "Ca", "Na", "K", "HCO3"]

PALETTE = {
    "white": "#FFFFFF", "blue": "#386793", "blue_light": "#ABDCE0",
    "orange": "#F18C48", "green": "#518EAB", "red": "#E76156",
    "teal": "#72BED8", "grey": "#204770", "dark": "#000000",
}
CORR_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "approved_diverging", ["#204770", "#72BED8", "#FFFFFF", "#FFCE71", "#E76156"],
)


def set_style():
    for f in TIMES_FONTS:
        if f.exists():
            font_manager.fontManager.addfont(str(f))
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "font.size": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE, "axes.titlesize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE, "ytick.labelsize": BASE_FONT_SIZE,
        "legend.fontsize": BASE_FONT_SIZE, "axes.linewidth": 0.65,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "legend.frameon": False, "pdf.fonttype": 42, "axes.unicode_minus": False,
        "axes.facecolor": "white", "figure.facecolor": "white",
    })


def read_sections(path):
    df = pd.read_csv(path)
    if "section" not in df.columns:
        return {"data": df}
    out = {}
    for sec, g in df.groupby("section", sort=False):
        out[str(sec)] = g.drop(columns=["section"]).dropna(axis=1, how="all").reset_index(drop=True)
    return out


def num_col(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def get_weights(table):
    raw = np.maximum(np.nan_to_num(num_col(table, "weight").to_numpy(float), nan=0.0), 0.0)
    return raw, raw / (raw.sum() + 1e-300)


def weighted_kde(x, weights, grid):
    x = np.asarray(x, float)
    w = np.asarray(weights, float)
    m = np.isfinite(x) & np.isfinite(w)
    x, w = x[m], w[m]
    if x.size < 3 or np.nanmax(x) <= np.nanmin(x):
        return np.zeros_like(grid)
    if gaussian_kde is not None:
        try:
            return gaussian_kde(x, weights=w)(grid)

        except Exception:
            pass
    hist, edges = np.histogram(x, bins=35, weights=w, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.interp(grid, centers, hist, left=0.0, right=0.0)


def map_index(qmsa, qae, table):
    for src in (qmsa.get("map"), qmsa.get("scalar"), qae.get("meta")):
        if src is None or src.empty:
            continue
        for col in ("QMSA_index", "map_index", "np_min_index"):
            if col in src.columns:
                v = pd.to_numeric(src[col], errors="coerce").dropna()
                if not v.empty:
                    return int(v.iloc[0])
    if "sample_index" in table.columns and "J_shift" in table.columns:
        pos = int(num_col(table, "J_shift").idxmin())
        return int(num_col(table, "sample_index").iloc[pos])
    return int(num_col(table, "J_shift").idxmin())


def row_for_param(summary, name):
    if summary is None or summary.empty or "param" not in summary.columns:
        return None
    aliases = {PARAM_LABELS[name], SOURCE_LABELS[name], name}
    m = summary[summary["param"].astype(str).isin(aliases)]
    return None if m.empty else m.iloc[0]


def load_refined_bounds():
    if not COARSE_FILE.exists():
        return {}
    bounds = read_sections(COARSE_FILE).get("refined_bounds")
    if bounds is None or bounds.empty:
        return {}
    out = {}
    for _, row in bounds.iterrows():
        param = str(row.get("param", ""))
        key = next((k for k in PARAMS if param in {PARAM_LABELS[k], SOURCE_LABELS[k], k}), None)
        if key is None:
            continue
        lo = pd.to_numeric(pd.Series([row.get("low_real", np.nan)]), errors="coerce").iloc[0]
        hi = pd.to_numeric(pd.Series([row.get("up_real", np.nan)]), errors="coerce").iloc[0]
        if np.isfinite(lo) and np.isfinite(hi):
            out[key] = (float(lo), float(hi))
    return out


def value_from(row, cols, default=np.nan):
    if row is None:
        return default
    for c in cols:
        if c in row.index:
            v = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
            if np.isfinite(v):
                return float(v)
    return default


def param_range(table, name):
    if name in PRIOR_RANGES:
        return PRIOR_RANGES[name]
    x = num_col(table, f"{name}_real").to_numpy(float)
    x = x[np.isfinite(x)]
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    pad = 0.04 * (hi - lo) if hi > lo else max(abs(lo), 1.0) * 0.04
    return lo - pad, hi + pad


def map_row(table, idx):
    if "sample_index" in table.columns:
        m = table[pd.to_numeric(table["sample_index"], errors="coerce") == idx]
        if not m.empty:
            return m.iloc[0]
    if 0 <= idx < len(table):
        return table.iloc[idx]
    return table.iloc[int(num_col(table, "J_shift").idxmin())]


def draw_likelihood(ax, table):
    raw_w, _ = get_weights(table)
    obj = num_col(table, "J_shift").to_numpy(float)
    order = np.argsort(obj)
    x = np.linspace(0, 1, len(obj))
    ax.plot(x, obj[order], color=PALETTE["blue"], lw=1.35)
    ax.grid(True, lw=0.35, alpha=0.22)
    ax.spines["left"].set_visible(True)
    axw = ax.twinx()
    axw.plot(x, raw_w[order], color=PALETTE["orange"], lw=1.05, ls="--")
    axw.spines["top"].set_visible(False)
    axw.spines["right"].set_visible(True)


def draw_search(ax, history):
    if history is None or history.empty or "candidate_J" not in history.columns:
        ax.set_axis_off()
        return
    x = num_col(history, "step").to_numpy(float) if "step" in history.columns else np.arange(1, len(history) + 1)
    best, cur = [], np.inf
    for _, row in history.iterrows():
        cand = pd.to_numeric(pd.Series([row.get("candidate_J", np.inf)]), errors="coerce").iloc[0]
        prev = pd.to_numeric(pd.Series([row.get("best_J_before_update", np.inf)]), errors="coerce").iloc[0]
        cur = min(cur, cand, prev)
        best.append(cur)
    ax.plot(x, best, color=PALETTE["green"], marker="o", ms=3.0, lw=1.25)
    ax.grid(True, lw=0.35, alpha=0.22)


def draw_kde(ax, table, summary, name, map_values, weights, refined):
    col = f"{name}_real"
    x = num_col(table, col).to_numpy(float)
    lo, hi = param_range(table, name)
    grid = np.linspace(lo, hi, 320)
    if name in refined:
        a, b = refined[name]
        ax.axvspan(a, b, color=PALETTE["blue_light"], alpha=0.26, zorder=0)
        ax.axvline(a, color=PALETTE["teal"], lw=1.0, zorder=1)
        ax.axvline(b, color=PALETTE["teal"], lw=1.0, zorder=1)
    ax.plot(grid, weighted_kde(x, weights, grid), color=PALETTE["blue"], lw=1.35, zorder=3)
    mv = pd.to_numeric(pd.Series([map_values.get(col, np.nan)]), errors="coerce").iloc[0]
    if np.isfinite(mv):
        ax.axvline(mv, color=PALETTE["red"], lw=1.0, ls="--", zorder=4)
    mean = value_from(row_for_param(summary, name), ["mean_real_qae", "mean_real_exact"])
    if np.isfinite(mean):
        ax.axvline(mean, color=PALETTE["dark"], lw=0.9, ls=":")
    ax.set_xlim(lo, hi)
    ax.grid(True, lw=0.3, alpha=0.18)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))


def correlation_matrix(corr_df):
    labels = [PARAM_LABELS[p] for p in PARAMS]
    if corr_df is not None and not corr_df.empty:
        need = {"row_param", "col_param", "correlation"}
        if need.issubset(corr_df.columns):
            alias = {a: PARAM_LABELS[k] for k in PARAMS for a in (k, SOURCE_LABELS[k], PARAM_LABELS[k])}
            w = corr_df.copy()
            w["row_param"] = w["row_param"].astype(str).map(lambda v: alias.get(v, v))
            w["col_param"] = w["col_param"].astype(str).map(lambda v: alias.get(v, v))
            mat = w.pivot(index="row_param", columns="col_param", values="correlation")
            mat = mat.reindex(index=labels, columns=labels)
            values = mat.apply(pd.to_numeric, errors="coerce").to_numpy(float)
            if np.isfinite(values).any():
                return values
    return np.eye(len(PARAMS), dtype=float)


def draw_joint(fig, spec, table, summary, correlation, map_values, weights):
    cols = [f"{p}_real" for p in PARAMS]
    raw_w, _ = get_weights(table)
    xm = np.column_stack([num_col(table, c).to_numpy(float) for c in cols])
    corr = np.clip(correlation_matrix(correlation), -1, 1)
    n = len(PARAMS)

    outer = spec.subgridspec(1, 3, width_ratios=[1.0, 0.024, 0.024], wspace=0.36)
    sub = outer[0, 0].subgridspec(n, n, wspace=0.10, hspace=0.10)
    cmap_w, cmap_c = plt.get_cmap("jet"), CORR_CMAP
    norm_w = mpl.colors.Normalize(vmin=float(np.nanmin(raw_w)), vmax=float(np.nanmax(raw_w)))
    sizes = 4.0 + 28.0 * (raw_w - np.nanmin(raw_w)) / (np.nanmax(raw_w) - np.nanmin(raw_w) + 1e-300)

    for i in range(n):
        for j in range(n):
            ax = fig.add_subplot(sub[i, j])
            if i == j:
                name = PARAMS[j]
                lo, hi = param_range(table, name)
                g = np.linspace(lo, hi, 260)
                ax.plot(g, weighted_kde(xm[:, j], weights, g), lw=0.95, color=PALETTE["blue"])
                mv = pd.to_numeric(pd.Series([map_values.get(f"{name}_real", np.nan)]), errors="coerce").iloc[0]
                if np.isfinite(mv):
                    ax.axvline(mv, lw=0.75, ls="--", color=PALETTE["red"])
                mean = value_from(row_for_param(summary, name), ["mean_real_qae", "mean_real_exact"])
                if np.isfinite(mean):
                    ax.axvline(mean, lw=0.75, ls=":", color=PALETTE["dark"])
                ax.set_xlim(lo, hi)
            elif i > j:
                ax.scatter(xm[:, j], xm[:, i], s=sizes, c=raw_w, cmap=cmap_w, norm=norm_w, alpha=0.58, linewidths=0)
                mx = pd.to_numeric(pd.Series([map_values.get(cols[j], np.nan)]), errors="coerce").iloc[0]
                my = pd.to_numeric(pd.Series([map_values.get(cols[i], np.nan)]), errors="coerce").iloc[0]
                if np.isfinite(mx) and np.isfinite(my):
                    ax.scatter(mx, my, marker="o", s=34, facecolors="white",
                               edgecolors=PALETTE["red"], linewidths=1.0)
            else:
                ax.imshow([[corr[i, j]]], vmin=-1, vmax=1, cmap=cmap_c, aspect="auto")
                ax.set_xticks([])
                ax.set_yticks([])
            if i < n - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=BASE_FONT_SIZE, pad=1.0, length=1.8)

    for col, cmap, vmin, vmax in (
        (1, cmap_w, float(np.nanmin(raw_w)), float(np.nanmax(raw_w))),
        (2, cmap_c, -1.0, 1.0),
    ):
        cax = fig.add_subplot(outer[0, col])
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, cax=cax)


def draw_uq(ax, variable):
    data = pd.read_csv(UQ_DIR / f"{variable}_uncertainty_data.csv")
    x = num_col(data, "distance").to_numpy(float)
    obs = num_col(data, "observation").to_numpy(float)
    map_sim = num_col(data, "QMSA_MAP_simulation").to_numpy(float)
    lo = num_col(data, "CI95_low").to_numpy(float)
    hi = num_col(data, "CI95_high").to_numpy(float)
    ax.fill_between(x, lo, hi, color=PALETTE["blue_light"], alpha=0.45, zorder=1)
    ax.plot(x, map_sim, color=PALETTE["blue"], marker="o", ms=2.3, lw=1.25, zorder=3)
    ax.scatter(x, obs, marker="o", s=11, facecolors="none", edgecolors=PALETTE["red"],
               linewidths=1.0, zorder=2)
    ax.grid(True, lw=0.3, alpha=0.20)
    ax.margins(x=0.02)


def strip_text(fig):
    for t in fig.findobj(match=matplotlib.text.Text):
        t.set_visible(False)
    for lg in fig.findobj(match=matplotlib.legend.Legend):
        lg.set_visible(False)


def build_figure(out_dir: Path, prefix: str):
    set_style()
    qmsa = read_sections(QMSA_FILE)
    qae = read_sections(QAE_FILE)
    table = qae.get("kde_samples")
    summary = qae.get("summary")
    if summary is None:
        summary = pd.DataFrame()
    correlation = qae.get("correlation")
    history = qmsa.get("history")

    mi = map_index(qmsa, qae, table)
    mv = map_row(table, mi)
    _, weights = get_weights(table)
    refined = load_refined_bounds()

    fig = plt.figure(figsize=(8.27, 14.8), constrained_layout=False)
    outer = fig.add_gridspec(
        6, 12, height_ratios=[1.10, 0.98, 0.98, 1.0, 1.0, 5.40], hspace=0.95, wspace=1.62,
    )

    draw_likelihood(fig.add_subplot(outer[0, :6]), table)
    draw_search(fig.add_subplot(outer[0, 7:]), history)

    for i, param in enumerate(PARAMS):
        r, c = 1 + i // 3, i % 3
        ax = fig.add_subplot(outer[r, c * 4: c * 4 + 4])
        draw_kde(ax, table, summary, param, mv, weights, refined)

    for i, var in enumerate(VARIABLES):
        r, c = 3 + i // 3, i % 3
        draw_uq(fig.add_subplot(outer[r, c * 4: c * 4 + 4]), var)

    draw_joint(fig, outer[5, 1:11], table, summary, correlation, mv, weights)

    fig.subplots_adjust(top=0.970, left=0.075, right=0.985, bottom=0.070)
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    strip_text(fig)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{prefix}.pdf", bbox_inches=bbox)
    plt.close(fig)


p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, default=RESULTS_DIR)
p.add_argument("--prefix", default="fig3")
args = p.parse_args()
build_figure(args.out, args.prefix)
