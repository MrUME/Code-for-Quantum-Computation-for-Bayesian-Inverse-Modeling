from __future__ import annotations
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager, gridspec
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parent
QMF_FILE = ROOT / "qmf_result.xlsx"
QAE_FILE = ROOT / "qae_result.xlsx"
OBS_FILE = ROOT / "concentration.xlsx"
FRACTURE_CANDIDATES = [ROOT / "fractures.txt", ROOT / "fracture.txt"]
DATAY_DIR = ROOT / ("datay - " + "\u526f\u672c")
RESULT_PATTERN = "result-{}.txt"

N_UNCERTAINTY = 100
MAP_INDEX = 101
KEEP_BEST_FRACTION = 0.95
BASE_FONT_SIZE = 10.5
TIMES_FONTS = [
    Path(r"C:\Windows\Fonts\times.ttf"),
    Path(r"C:\Windows\Fonts\timesbd.ttf"),
    Path(r"C:\Windows\Fonts\timesi.ttf"),
    Path(r"C:\Windows\Fonts\timesbi.ttf"),
]

PARAMS = ["center_x", "center_y", "length"]
PRIOR_RANGES = {
    "center_x": (-8.0, 10.0),
    "center_y": (-46.0, -15.0),
    "length": (6.0, 10.0),
}
OBS_SHEETS = ["\u89c2\u6d4b\u503c", "Observation", "observations", "Sheet1"]
OBS_X, OBS_Y = "M3x", "M3_delta"

PALETTE = {
    "white": "#FFFFFF", "black": "#000000", "blue": "#386793",
    "light_blue": "#ABDCE0", "orange": "#F18C48", "green": "#518EAB",
    "red": "#E76156", "dark": "#000000",
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
        "legend.fontsize": BASE_FONT_SIZE, "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 2.8, "ytick.major.size": 2.8,
        "legend.frameon": False, "pdf.fonttype": 42, "axes.unicode_minus": False,
        "axes.facecolor": PALETTE["white"], "figure.facecolor": PALETTE["white"],
        "savefig.facecolor": PALETTE["white"],
        "text.color": PALETTE["black"], "axes.edgecolor": PALETTE["black"],
        "axes.labelcolor": PALETTE["black"], "xtick.color": PALETTE["black"],
        "ytick.color": PALETTE["black"],
    })


def load_results():
    table = pd.read_excel(QMF_FILE, sheet_name="refined_table")
    scalar = pd.read_excel(QMF_FILE, sheet_name="scalar")
    qmap = pd.read_excel(QMF_FILE, sheet_name="map")
    try:
        history = pd.read_excel(QMF_FILE, sheet_name="history")
    except Exception:
        history = pd.DataFrame()
    summary = pd.read_excel(QAE_FILE, sheet_name="summary")
    try:
        corr = pd.read_excel(QAE_FILE, sheet_name="correlation", index_col=0)
    except Exception:
        corr = None
    return table, scalar, qmap, history, summary, corr


def get_weights(table):
    raw = np.maximum(np.nan_to_num(table["weight"].to_numpy(float), nan=0.0), 0.0)
    return raw, raw / (raw.sum() + 1e-300)


def kde(x, weights, grid):
    try:
        return gaussian_kde(x, weights=weights)(grid)
    except Exception:
        hist, edges = np.histogram(x, bins=30, weights=weights, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return np.interp(grid, centers, hist, left=hist[0], right=hist[-1])


def map_index(table, scalar):
    if scalar is not None and "QMF_index" in scalar.columns:
        return int(scalar.loc[0, "QMF_index"])
    return int(table["J_shift"].idxmin())


def load_result_file(path):
    return np.asarray(np.loadtxt(path, dtype=float), dtype=float).reshape(-1)


def resolve_datay():
    if DATAY_DIR.exists():
        return DATAY_DIR
    for p in ROOT.iterdir():
        if p.is_dir() and p.name.lower().startswith("datay"):
            return p
    return DATAY_DIR


def load_ensemble():
    datay = resolve_datay()
    return np.vstack([
        load_result_file(datay / RESULT_PATTERN.format(i))
        for i in range(1, N_UNCERTAINTY + 1)
    ])


def read_obs():
    xls = pd.ExcelFile(OBS_FILE)
    sheet = next((s for s in OBS_SHEETS if s in xls.sheet_names), xls.sheet_names[0])
    df = pd.read_excel(OBS_FILE, sheet_name=sheet)
    data = df[[OBS_X, OBS_Y]].apply(pd.to_numeric, errors="coerce").dropna()
    return data[OBS_X].to_numpy(float), data[OBS_Y].to_numpy(float)


def build_validation():
    x_obs, y_obs = read_obs()
    ensemble = load_ensemble()
    map_sim = load_result_file(resolve_datay() / RESULT_PATTERN.format(MAP_INDEX))
    order = np.argsort(x_obs)
    x_obs, y_obs = x_obs[order], y_obs[order]
    ensemble, map_sim = ensemble[:, order], map_sim[order]
    valid = np.isfinite(y_obs)
    rmse = np.sqrt(np.mean((ensemble[:, valid] - y_obs[None, valid]) ** 2, axis=1))
    rank = np.argsort(rmse)
    n_keep = max(1, min(N_UNCERTAINTY, int(np.ceil(KEEP_BEST_FRACTION * N_UNCERTAINTY))))
    selected = ensemble[rank[:n_keep], :]
    return {
        "x": x_obs,
        "observation": y_obs,
        "envelope_low": np.min(selected, axis=0),
        "envelope_high": np.max(selected, axis=0),
        "map_simulation": map_sim,
    }


def read_fractures(path):
    if path is None or not path.exists():
        return []
    segs, pts = [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            nums = []
            for tok in line.replace(",", " ").replace(";", " ").split():
                try:
                    nums.append(float(tok))
                except Exception:
                    pass
            if len(nums) >= 4:
                segs.append(nums[:4])
            elif len(nums) >= 2:
                pts.append(nums[:2])
    if not segs and len(pts) >= 2:
        for i in range(0, len(pts) - 1, 2):
            segs.append([*pts[i], *pts[i + 1]])
    return segs


def draw_diagnostics(ax, table):
    raw_w, _ = get_weights(table)
    obj = table["J_shift"].to_numpy(float)
    order = np.argsort(obj)
    x = np.linspace(0, 1, len(obj))
    ax.plot(x, obj[order], color=PALETTE["blue"], lw=1.5)
    ax.spines["left"].set_visible(True)
    ax.grid(True, lw=0.4, alpha=0.22)
    axw = ax.twinx()
    axw.plot(x, raw_w[order], color=PALETTE["orange"], lw=1.2, ls="--")
    axw.spines["top"].set_visible(False)
    axw.spines["right"].set_visible(True)


def draw_search_history(ax, history):
    if history is None or history.empty or "candidate_J" not in history.columns:
        ax.set_axis_off()
        return
    best, cur = [], np.inf
    for v in history["candidate_J"].to_numpy(float):
        cur = min(cur, v)
        best.append(cur)
    x = history["step"].to_numpy() if "step" in history.columns else np.arange(1, len(best) + 1)
    ax.plot(x, best, marker="o", ms=3.2, lw=1.4, color=PALETTE["blue"])
    ax.grid(True, lw=0.4, alpha=0.22)


def draw_kde(ax, table, summary, scalar, param):
    _, w = get_weights(table)
    mi = map_index(table, scalar)
    x = table[f"{param}_real"].to_numpy(float)
    xmin, xmax = PRIOR_RANGES[param]
    grid = np.linspace(xmin, xmax, 400)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    ax.axvspan(lo, hi, color=PALETTE["light_blue"], alpha=0.26, zorder=0)
    ax.plot(grid, kde(x, w, grid), lw=1.6, color=PALETTE["blue"], zorder=3)
    ax.axvline(x[mi], lw=1.1, ls="--", color=PALETTE["red"], zorder=4)
    ax.axvline(lo, lw=1.0, color=PALETTE["green"])
    ax.axvline(hi, lw=1.0, color=PALETTE["green"])
    row = summary[summary["param"].astype(str) == param]
    if not row.empty:
        mean = float(row.iloc[0].get("mean_real_analytic", np.nan))
        if np.isfinite(mean):
            ax.axvline(mean, lw=1.0, ls=":", color=PALETTE["dark"])
    ax.set_xlim(xmin, xmax)
    ax.grid(True, lw=0.35, alpha=0.18)


def weighted_corr(table, weights):
    x = np.column_stack([table[f"{p}_real"].to_numpy(float) for p in PARAMS])
    mean = np.sum(x * weights[:, None], axis=0)
    c = x - mean
    cov = (c * weights[:, None]).T @ c
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = np.clip(cov / (std[:, None] * std[None, :] + 1e-300), -1, 1)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=PARAMS, columns=PARAMS)


def draw_joint(fig, spec, table, scalar, summary, corr):
    raw_w, weights = get_weights(table)
    if corr is None:
        corr = weighted_corr(table, weights)
    mi = map_index(table, scalar)
    xm = np.column_stack([table[f"{p}_real"].to_numpy(float) for p in PARAMS])
    sizes = 5.0 + 26.0 * (raw_w - raw_w.min()) / (raw_w.max() - raw_w.min() + 1e-300)

    outer = spec.subgridspec(1, 3, width_ratios=[1.0, 0.026, 0.026], wspace=0.34)
    sub = outer[0, 0].subgridspec(3, 3, wspace=0.16, hspace=0.16)
    cmap_w, cmap_r = plt.get_cmap("jet"), CORR_CMAP
    norm_w = mpl.colors.Normalize(vmin=raw_w.min(), vmax=raw_w.max())

    for i in range(3):
        for j in range(3):
            ax = fig.add_subplot(sub[i, j])
            if i == j:
                p = PARAMS[j]
                lo, hi = PRIOR_RANGES[p]
                g = np.linspace(lo, hi, 240)
                ax.plot(g, kde(xm[:, j], weights, g), lw=1.0, color=PALETTE["blue"])
                ax.axvline(xm[mi, j], lw=0.9, ls="--", color=PALETTE["red"])
                row = summary[summary["param"].astype(str) == p]
                if not row.empty:
                    ax.axvline(float(row.iloc[0]["mean_real_analytic"]), lw=0.9, ls=":", color=PALETTE["dark"])
                ax.set_xlim(lo, hi)
            elif i > j:
                ax.scatter(xm[:, j], xm[:, i], c=raw_w, s=sizes, cmap=cmap_w, norm=norm_w, alpha=0.64, linewidths=0)
                ax.scatter(xm[mi, j], xm[mi, i], marker="o", s=42, facecolors=PALETTE["white"],
                           edgecolors=PALETTE["red"], linewidths=1.2)
            else:
                try:
                    rho = float(corr.loc[PARAMS[i], PARAMS[j]])
                except Exception:
                    rho = np.nan
                ax.imshow([[rho]], cmap=cmap_r, vmin=-1, vmax=1, aspect="auto")
                ax.set_xticks([])
                ax.set_yticks([])
            if i < 2:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=BASE_FONT_SIZE, pad=1.5)

    for col, cmap, vmin, vmax in (
        (1, cmap_w, raw_w.min(), raw_w.max()),
        (2, cmap_r, -1.0, 1.0),
    ):
        cax = fig.add_subplot(outer[0, col])
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, cax=cax)


def draw_temperature(ax, val):
    x, obs = val["x"], val["observation"]
    lo, hi, map_sim = val["envelope_low"], val["envelope_high"], val["map_simulation"]
    ax.fill_between(x, lo, hi, color=PALETTE["light_blue"], alpha=0.45, zorder=1)
    ax.plot(x, map_sim, lw=1.8, marker="o", ms=2.8, color=PALETTE["blue"], zorder=3)
    ax.scatter(x, obs, marker="o", s=12, facecolors="none", edgecolors=PALETTE["red"],
               linewidths=1.1, zorder=2)
    ax.grid(True, lw=0.4, alpha=0.24)
    ax.margins(x=0.01)


def draw_fracture(ax, qmap):
    path = next((p for p in FRACTURE_CANDIDATES if p.exists()), None)
    segs = read_fractures(path)
    vals = {str(r["param"]): float(r["QMF_real"]) for _, r in qmap.iterrows()}
    xc, yc, length = vals["center_x"], vals["center_y"], vals["length"]
    theta = 0.4678
    dx, dy = 0.5 * length * np.cos(theta), 0.5 * length * np.sin(theta)
    p1 = np.array([xc - dx, yc - dy])
    p2 = np.array([xc + dx, yc + dy])
    xs, ys = [p1[0], p2[0], 0.0, 1.8], [p1[1], p2[1]]
    for x1, y1, x2, y2 in segs:
        ax.plot([x1, x2], [y1, y2], lw=0.65, color=PALETTE["light_blue"], alpha=0.75, zorder=1)
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    ax.plot([-4.85, 3.76], [-30.9, -26.7], lw=2.0, color=PALETTE["light_blue"], alpha=0.85, zorder=2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], lw=2.0, color=PALETTE["red"], zorder=3)
    ax.scatter([xc], [yc], marker="o", s=24, color=PALETTE["dark"], zorder=4)
    ax.axvline(0.0, ls="--", lw=0.9, color=PALETTE["green"])
    ax.axvline(1.8, ls="--", lw=0.9, color=PALETTE["green"])
    xpad = max(1.0, 0.05 * (max(xs) - min(xs)))
    ypad = max(1.0, 0.05 * (max(ys) - min(ys)))
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)
    ax.set_aspect("equal", adjustable="box")


def strip_text(fig):
    for t in fig.findobj(match=matplotlib.text.Text):
        t.set_visible(False)
    for lg in fig.findobj(match=matplotlib.legend.Legend):
        lg.set_visible(False)


def build_figure(out_dir: Path, prefix: str):
    set_style()
    table, scalar, qmap, history, summary, corr = load_results()
    val = build_validation()

    fig = plt.figure(figsize=(11.69, 9.2), constrained_layout=False)
    gs = fig.add_gridspec(3, 12, height_ratios=[0.98, 1.35, 3.45], hspace=1.02, wspace=1.45)

    draw_diagnostics(fig.add_subplot(gs[0, :7]), table)
    draw_search_history(fig.add_subplot(gs[0, 8:]), history)
    for i, param in enumerate(PARAMS):
        draw_kde(fig.add_subplot(gs[1, 2 * i: 2 * i + 2]), table, summary, scalar, param)
    draw_temperature(fig.add_subplot(gs[1, 7:]), val)
    draw_joint(fig, gs[2, :6], table, scalar, summary, corr)
    draw_fracture(fig.add_subplot(gs[2, 7:]), qmap)

    fig.subplots_adjust(top=0.965, left=0.060, right=0.985, bottom=0.085)
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    strip_text(fig)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{prefix}.pdf", bbox_inches=bbox)
    plt.close(fig)


p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, default=ROOT / "results")
p.add_argument("--prefix", default="fig2")
args = p.parse_args()
build_figure(args.out, args.prefix)
print(f"Saved: {args.out / (args.prefix + '.pdf')}")
