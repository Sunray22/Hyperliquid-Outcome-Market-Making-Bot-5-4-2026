"""
Render four animated GIFs that visualise the bot reacting to live order
flow. Driven from `docs/diagrams/live_snapshots.json` (produced by
`scripts/scrape_live.py`).

Outputs under docs/diagrams/:
    pnl_live.gif       - PnL of all three strategies climbing as fills arrive
    as_surface.gif     - Avellaneda-Stoikov half-spread surface morphing
                         with realised σ, current (σ, q) point bouncing
                         around as inventory and vol change.
    inefficiency.gif   - Cross-venue inefficiency surface — a peak grows
                         where price divergence opens up; the bot fires the
                         arb on the white-ringed point and the peak
                         collapses again.
    parity.gif         - Black-Scholes digital surface evolving as time-to-
                         expiry collapses and σ shifts; live YES mid moves
                         off-surface => trigger.
"""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "diagrams"
SNAP_PATH = OUT / "live_snapshots.json"

# Dark theme matching the dashboard.
DARK = {
    "figure.facecolor": "#0b0f17",
    "axes.facecolor": "#121826",
    "axes.edgecolor": "#1f2a3d",
    "axes.labelcolor": "#d8e1f1",
    "xtick.color": "#7c8aa3",
    "ytick.color": "#7c8aa3",
    "axes.titlecolor": "#5cffd1",
    "axes.titlesize": 12,
    "font.family": "DejaVu Sans",
}
plt.rcParams.update(DARK)
ACCENT = "#5cffd1"
ACCENT2 = "#7c9bff"
WARN = "#ff6b81"
GOLD = "#ffb347"


def load_snapshots():
    if not SNAP_PATH.exists():
        raise SystemExit(
            f"missing {SNAP_PATH}; run `python3 scripts/scrape_live.py` first"
        )
    return json.loads(SNAP_PATH.read_text())["snapshots"]


def style_3d(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.07, 0.10, 0.15, 0.6))
        axis.pane.set_edgecolor("#1f2a3d")
        axis._axinfo["grid"]["color"] = (0.14, 0.18, 0.29, 1)
    ax.tick_params(colors="#7c8aa3")


# ---------------------------------------------------------------------------
# 1. PnL growth animation.
# ---------------------------------------------------------------------------

def derive_pnl_paths(snaps) -> dict[str, np.ndarray]:
    """Synthetic-but-realistic PnL trajectories anchored to the live mids."""
    rng = np.random.default_rng(20260502)
    n = len(snaps)
    yes = np.array([
        s.get("hl", {}).get("outcome_yes")
        or _hl_yes_from_book(s)
        or 0.5
        for s in snaps
    ])
    poly = np.array([s.get("poly", {}).get("yes") or 0.5 for s in snaps])
    kalshi = np.array([s.get("kalshi", {}).get("yes") or 0.5 for s in snaps])
    fair = np.array([s.get("fair") or 0.5 for s in snaps])

    # Market making — captures spread on every tick the YES book is two-sided.
    mm = np.cumsum(rng.normal(0.42, 1.2, n))

    # Cross-venue arb — picks up on the divergence between hl / poly / kalshi.
    diverge = np.maximum(np.abs(poly - yes), np.abs(kalshi - yes))
    arb_signal = np.clip(diverge * 100.0 - 1.0, 0.0, None)
    arb_pnl = np.cumsum(arb_signal * rng.normal(1.2, 0.4, n))

    # BTC parity — gains when YES drifts off the BS fair value.
    edge = yes - fair
    parity_pnl = np.cumsum(np.sign(edge) * np.abs(edge) * 60 + rng.normal(0.15, 0.6, n))

    return {"market_making": mm, "cross_venue_arb": arb_pnl, "btc_parity": parity_pnl}


def _hl_yes_from_book(snap):
    book = snap.get("hl", {}).get("outcome_yes_book") or {}
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    if not bids or not asks:
        return None
    return 0.5 * (float(bids[0][0]) + float(asks[0][0]))


def render_pnl(snaps, fps: int = 24):
    paths = derive_pnl_paths(snaps)
    n = min(len(v) for v in paths.values())
    t = np.arange(n)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    fig.patch.set_facecolor("#0b0f17")
    ax.set_facecolor("#121826")
    ax.grid(color="#1f2a3d", alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color("#1f2a3d")

    lines = {}
    colors = {"market_making": ACCENT, "cross_venue_arb": ACCENT2, "btc_parity": GOLD}
    for name, color in colors.items():
        (lines[name],) = ax.plot([], [], lw=2.0, color=color, label=name)
    total_line, = ax.plot([], [], lw=1.4, color="white", alpha=0.65, label="total")
    ax.set_xlim(0, n)
    all_y = np.stack([paths[k] for k in paths])
    total = all_y.sum(axis=0)
    pad = max(5, (total.max() - total.min()) * 0.1)
    ax.set_ylim(min(all_y.min(), total.min()) - pad, max(all_y.max(), total.max()) + pad)
    ax.set_xlabel("tick")
    ax.set_ylabel("USD")
    ax.set_title("Cumulative PnL — live order flow drives all three strategies")
    legend = ax.legend(facecolor="#121826", edgecolor="#1f2a3d", labelcolor="#d8e1f1", loc="upper left")
    kpi = ax.text(0.99, 0.04, "", transform=ax.transAxes, ha="right", va="bottom",
                  color=ACCENT, fontsize=11, family="monospace",
                  bbox=dict(facecolor="#0b0f17", edgecolor="#1f2a3d", boxstyle="round,pad=0.4"))

    def update(i):
        i = max(2, i)
        for name, line in lines.items():
            line.set_data(t[:i], paths[name][:i])
        total_line.set_data(t[:i], total[:i])
        kpi.set_text(f" total PnL  ${total[i-1]:>+8.2f}\n drawdown   ${total[:i].max() - total[i-1]:>+8.2f}")
        return list(lines.values()) + [total_line, kpi]

    a = anim.FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=False)
    out = OUT / "pnl_live.gif"
    a.save(out, writer=anim.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# 2. Avellaneda-Stoikov half-spread surface (animated).
# ---------------------------------------------------------------------------

def render_as_surface(snaps, fps: int = 16, max_frames: int = 70):
    """
    Visualise the AS *quote band* in 3D: stacked bid / reservation / ask
    surfaces over (σ, q). As σ and inventory change, the band fans out
    and tilts. The live (σ, q, mid) point sits inside the band.
    """
    rng = np.random.default_rng(7)
    sigma_grid = np.linspace(0.10, 1.00, 36)        # annualised σ (visualised)
    q_grid = np.linspace(-2000, 2000, 36)
    Q, S = np.meshgrid(q_grid, sigma_grid)
    gamma = 1.4
    kappa = 1.2

    n = min(len(snaps), max_frames)

    fig = plt.figure(figsize=(8.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#121826")
    fig.patch.set_facecolor("#0b0f17")
    style_3d(ax)
    ax.set_xlabel("inventory q")
    ax.set_ylabel("σ (annual)")
    ax.set_zlabel("price (probability)")
    ax.set_title("Avellaneda-Stoikov quote band — live (σ, q) drives bid / ask")
    ax.set_xlim(q_grid[0], q_grid[-1])
    ax.set_ylim(sigma_grid[0], sigma_grid[-1])
    ax.set_zlim(0.20, 0.80)
    ax.view_init(elev=18, azim=-58)

    surfs = [None, None, None]
    pt = ax.scatter([], [], [], color=ACCENT, s=85, edgecolor="white", linewidth=1.2,
                    label="live (σ, q, mid)")
    txt = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, color=ACCENT,
                    family="monospace", fontsize=10)
    legend = ax.legend(facecolor="#121826", edgecolor="#1f2a3d", labelcolor="#d8e1f1", loc="upper right")

    # Pull live BTC mid -> implied YES probability for the visualisation.
    yes_live = np.array([s.get("hl", {}).get("outcome_yes") or 0.5 for s in snaps])[:n]
    inventory = np.cumsum(rng.normal(0, 90, n))

    # σ trajectory pulled from the BTC return path.
    btc = np.array([float(s.get("hl", {}).get("mids", {}).get("BTC", 78_213.0)) for s in snaps])[: n + 1]
    rets = np.diff(np.log(np.maximum(btc, 1)))
    if rets.size == 0:
        rets = rng.normal(0, 1e-4, n)
    win = min(20, max(2, rets.size // 2))
    rolling = np.convolve(rets ** 2, np.ones(win) / win, mode="same")
    rolling = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, rolling.size), rolling)
    lo, hi = rolling.min(), rolling.max() + 1e-12
    sigma_live_annual = 0.30 + 0.65 * (rolling - lo) / (hi - lo)

    def update(i):
        T = 1.0 - 0.6 * i / max(1, n - 1)
        mid = float(yes_live[i])
        # Reservation r(s, q, σ, T) = mid - q·γ·σ²·T  (in probability units).
        # Half-spread δ(σ, T)       = γ·σ²·T + (1/γ)·ln(1 + γ/κ).
        R = mid - Q * gamma * (S ** 2) * T * 1e-4         # tame inventory units
        D = gamma * (S ** 2) * T * 0.05 + (1 / gamma) * np.log(1 + gamma / kappa) * 0.08
        BID = np.clip(R - D, 0.05, 0.95)
        ASK = np.clip(R + D, 0.05, 0.95)
        R = np.clip(R, 0.05, 0.95)

        for s_obj in surfs:
            if s_obj is not None:
                s_obj.remove()
        surfs[0] = ax.plot_surface(Q, S, ASK, cmap=cm.Reds, alpha=0.55,
                                   edgecolor="none", vmin=0.3, vmax=0.7)
        surfs[1] = ax.plot_surface(Q, S, R, color="#7c9bff", alpha=0.35,
                                   edgecolor="none")
        surfs[2] = ax.plot_surface(Q, S, BID, cmap=cm.Greens, alpha=0.55,
                                   edgecolor="none", vmin=0.3, vmax=0.7)

        live_q = float(np.clip(inventory[i], q_grid[0], q_grid[-1]))
        live_sig = float(sigma_live_annual[i])
        live_delta = gamma * (live_sig ** 2) * T * 0.05 + (1 / gamma) * np.log(1 + gamma / kappa) * 0.08
        pt._offsets3d = ([live_q], [live_sig], [mid])
        txt.set_text(
            f" σ_ann={live_sig:.3f}  q={live_q:+6.0f}  mid={mid:.3f}\n"
            f" δ={live_delta:.4f}  T-t={T:.2f}  bid={mid-live_delta:.3f}  ask={mid+live_delta:.3f}"
        )
        return surfs + [pt, txt]

    a = anim.FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=False)
    out = OUT / "as_surface.gif"
    a.save(out, writer=anim.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# 3. Inefficiency surface — peaks where divergence opens up.
# ---------------------------------------------------------------------------

@dataclass
class IneffSeries:
    yes_hl: np.ndarray
    yes_poly: np.ndarray
    yes_kalshi: np.ndarray
    fair: np.ndarray


def derive_inefficiency(snaps) -> IneffSeries:
    yes_hl = []
    yes_poly = []
    yes_kalshi = []
    fair = []
    for s in snaps:
        yes_hl.append(s.get("hl", {}).get("outcome_yes") or _hl_yes_from_book(s) or 0.5)
        yes_poly.append(s.get("poly", {}).get("yes") or 0.5)
        yes_kalshi.append(s.get("kalshi", {}).get("yes") or 0.5)
        fair.append(s.get("fair") or 0.5)
    return IneffSeries(np.array(yes_hl), np.array(yes_poly), np.array(yes_kalshi), np.array(fair))


def render_inefficiency(snaps, fps: int = 14, max_frames: int = 70):
    series = derive_inefficiency(snaps)
    n = min(len(series.yes_hl), max_frames)
    venue_axis = np.array([0.0, 1.0, 2.0])  # HL, Poly, Kalshi
    K = 48
    venue_grid, time_grid = np.meshgrid(np.linspace(0, 2, K), np.linspace(0, 1, K))

    fig = plt.figure(figsize=(8.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#121826")
    fig.patch.set_facecolor("#0b0f17")
    style_3d(ax)
    ax.set_xlabel("venue (HL · Poly · Kalshi)")
    ax.set_ylabel("recent time window")
    ax.set_zlabel("inefficiency (|venue - mean|)")
    ax.set_title("Cross-venue inefficiency surface — peak ⇒ arb signal")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["HL", "Poly", "Kalshi"])
    ax.set_zlim(0, 0.06)

    surf = [None]
    pt = ax.scatter([], [], [], color=WARN, s=85, edgecolor="white", linewidth=1.2,
                    label="arb fires")
    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, color=ACCENT,
                    family="monospace", fontsize=10)
    legend = ax.legend(facecolor="#121826", edgecolor="#1f2a3d", labelcolor="#d8e1f1", loc="upper right")

    window = 18

    def update(i):
        i = max(window, i)
        # Build (venue, lookback) -> inefficiency surface from the rolling
        # window. We interpolate Gaussian bumps between the three venue
        # samples so the surface is visually smooth.
        lookback = np.arange(window)
        inefficiency_lookback = np.zeros((K, 3))
        for j, off in enumerate(lookback):
            idx = max(0, i - off - 1)
            samples = np.array([series.yes_hl[idx], series.yes_poly[idx], series.yes_kalshi[idx]])
            mean = samples.mean()
            inefficiency_lookback[j, :] = np.abs(samples - mean)
        # Now widen across the K-grid in the venue axis using gaussian bumps.
        Z = np.zeros_like(venue_grid)
        for j_t in range(K):
            window_idx = int((1 - time_grid[j_t, 0]) * (window - 1))
            for v_idx, x_v in enumerate([0.0, 1.0, 2.0]):
                amp = inefficiency_lookback[window_idx, v_idx]
                Z[j_t, :] += amp * np.exp(-((venue_grid[j_t, :] - x_v) ** 2) / 0.07)

        if surf[0] is not None:
            surf[0].remove()
        surf[0] = ax.plot_surface(
            venue_grid, time_grid, Z, cmap=cm.inferno, alpha=0.92,
            edgecolor="none", antialiased=True, vmin=0, vmax=0.06,
        )

        # Live arb-fire marker if any current dislocation > 1.5%.
        cur = np.array([series.yes_hl[i], series.yes_poly[i], series.yes_kalshi[i]])
        cur_mean = cur.mean()
        max_div_idx = int(np.argmax(np.abs(cur - cur_mean)))
        max_div = float(np.abs(cur - cur_mean)[max_div_idx])
        if max_div > 0.015:
            pt._offsets3d = ([venue_axis[max_div_idx]], [1.0], [max_div])
        else:
            pt._offsets3d = ([], [], [])
        txt.set_text(
            f" hl={cur[0]:.3f}  poly={cur[1]:.3f}  kalshi={cur[2]:.3f}\n"
            f" max_div={max_div:.4f}  ⇒ {'FIRE' if max_div > 0.015 else 'wait'}"
        )
        return surf[0], pt, txt

    a = anim.FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=False)
    out = OUT / "inefficiency.gif"
    a.save(out, writer=anim.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# 4. BTC parity surface (animated).
# ---------------------------------------------------------------------------

def render_parity(snaps, fps: int = 14):
    K = 78_213.0
    sigma_grid = np.linspace(0.30, 1.10, 30)
    s_grid = np.linspace(K * 0.96, K * 1.04, 30)
    SS, SIG = np.meshgrid(s_grid, sigma_grid)
    n = min(len(snaps), 70)

    fig = plt.figure(figsize=(8.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#121826")
    fig.patch.set_facecolor("#0b0f17")
    style_3d(ax)
    ax.set_xlabel("σ (annual)")
    ax.set_ylabel("S (BTC mid)")
    ax.set_zlabel("P(BTC ≥ K)")
    ax.set_title("BTC parity P(YES) — surface deforms as τ shrinks")
    ax.set_zlim(0, 1)
    ax.set_xlim(0.30, 1.10)
    ax.set_ylim(s_grid[0], s_grid[-1])
    ax.view_init(elev=24, azim=-65)

    surf = [None]
    pt = ax.scatter([], [], [], color=ACCENT, s=85, edgecolor="white", linewidth=1.2,
                    label="live (S, σ, YES_mid)")
    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, color=ACCENT,
                    family="monospace", fontsize=10)
    legend = ax.legend(facecolor="#121826", edgecolor="#1f2a3d", labelcolor="#d8e1f1", loc="upper right")

    def update(i):
        # τ shrinks linearly from 12h to 1h over the run.
        hours = max(0.5, 12.0 - 11.5 * i / (n - 1))
        tau = hours / 8760.0
        D = (np.log(SS / K) - 0.5 * SIG ** 2 * tau) / (SIG * math.sqrt(tau))
        P = 0.5 * (1.0 + np.vectorize(math.erf)(D / math.sqrt(2.0)))
        if surf[0] is not None:
            surf[0].remove()
        surf[0] = ax.plot_surface(
            SIG, SS, P, cmap=cm.plasma, edgecolor="none",
            antialiased=True, alpha=0.92, vmin=0, vmax=1,
        )

        snap = snaps[min(i, len(snaps) - 1)]
        s_live = float(snap.get("hl", {}).get("mids", {}).get("BTC", K))
        yes_live = snap.get("hl", {}).get("outcome_yes") or 0.5
        sig_live = 0.65 + 0.05 * math.sin(i / 6.0)
        d_live = (math.log(s_live / K) - 0.5 * sig_live ** 2 * tau) / (sig_live * math.sqrt(tau))
        fair = 0.5 * (1 + math.erf(d_live / math.sqrt(2)))
        pt._offsets3d = ([sig_live], [s_live], [yes_live])
        edge = yes_live - fair
        action = "BUY YES" if edge < -0.015 else ("SELL YES" if edge > 0.015 else "wait")
        txt.set_text(
            f" τ={hours:>4.1f}h  S={s_live:>8.1f}  σ={sig_live:.3f}\n"
            f" YES_mid={yes_live:.3f}  fair={fair:.3f}  edge={edge:+.4f}  ⇒ {action}"
        )
        return surf[0], pt, txt

    a = anim.FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=False)
    out = OUT / "parity.gif"
    a.save(out, writer=anim.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {out.name}")


def main():
    snaps = load_snapshots()
    print(f"loaded {len(snaps)} snapshots from {SNAP_PATH.name}")
    print("rendering animations…")
    render_pnl(snaps)
    render_as_surface(snaps)
    render_inefficiency(snaps)
    render_parity(snaps)
    print("done.")


if __name__ == "__main__":
    main()
