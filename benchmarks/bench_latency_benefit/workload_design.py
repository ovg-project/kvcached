#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""
Google/Meta-style synthetic workload generator for systems benchmarking.

Design goals
------------
Generate *arrivals* (not just a smooth rate curve) with realistic properties:
  1) Diurnal / slow trend (shared) + per-instance trend
  2) Correlated short-term fluctuations (shared + per-instance)
  3) Flash crowds / burst clusters (self-exciting-ish, but simple & controllable)
  4) Heavy-tailed request durations (service times)
  5) Optional hard capacity cap applied by thinning arrivals (rarely active)

This is closer to what many systems papers do: model an intensity λ(t) and
sample arrivals from an inhomogeneous Poisson process (discretized).

Outputs
-------
  - workload_rates.png: per-instance arrival rate (req/s) + total (smoothed for viz)
  - workload_inflight.png: expected in-flight (from realized arrivals + durations)
  - workload_trace.npz: arrays for reuse in benchmarks

Key knobs
---------
  - R_peak: system capacity / intended peak. We target mean load at mean_frac * R_peak.
  - mean_frac: average utilization (0.5–0.7 typical)
  - burstiness: how often/strong flash crowds are
  - corr: correlation across instances (0–1)
  - durations: lognormal heavy tail (mu, sigma)

Usage
-----
  python3 workload_google_style.py --N 6 --R-peak 30 --T 300 --dt 0.1 \
    --mean-frac 0.6 --corr 0.35 --burstiness 0.6 --seed 42

Notes
-----
- This samples discrete arrivals. If you need an *exact* per-second rate curve,
  you can compute it from arrivals with a windowed average.
- The "cap" is implemented as thinning: if total intensity exceeds R_peak, we
  scale down all instance intensities proportionally for that timestep.
  If mean_frac is sane, this cap is rarely active.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --------------------------- helper processes --------------------------------


def ou_process(rng: np.random.Generator, n: int, dt: float, theta: float, sigma: float) -> np.ndarray:
    """Zero-mean OU via Euler-Maruyama."""
    x = np.zeros(n, dtype=np.float64)
    for k in range(1, n):
        x[k] = x[k - 1] + theta * (-x[k - 1]) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
    return x


def smooth_ma(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(x, kernel, mode="same")


def make_diurnal(t: np.ndarray, T: float, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Shared slow-ish trend; for short experiments, it's just a slow wave."""
    period = rng.uniform(0.7, 1.3) * T
    phase = rng.uniform(0, 2 * np.pi)
    return strength * np.sin(2 * np.pi * t / period + phase)


def make_flash_crowd_component(
    rng: np.random.Generator,
    n: int,
    dt: float,
    base_event_rate: float,
    mean_amp: float,
    amp_jitter: float,
    tau_s: float,
    cluster_strength: float,
) -> np.ndarray:
    """
    Simple clustered flash crowd model:
      - background events: Poisson(base_event_rate)
      - each event triggers an exponential "crowd" that increases intensity
      - cluster_strength > 0 makes follow-up events more likely (approx Hawkes-like)
    This is a lightweight approximation that yields burst clusters.

    Implementation:
      - We keep a latent "excitation" e[k] that decays exponentially.
      - At each step, event rate is base_event_rate * (1 + cluster_strength * e[k]).
      - When events happen, we add hump(s) to an output signal and bump excitation.
    """
    crowd = np.zeros(n, dtype=np.float64)
    e = 0.0
    decay = np.exp(-dt / max(1e-6, tau_s))

    for k in range(n):
        e *= decay
        lam = base_event_rate * (1.0 + cluster_strength * e)  # events/s
        m = rng.poisson(lam * dt)
        if m > 0:
            # bump excitation for clustering
            e += float(m)
            # each event adds an exponential bump to crowd
            for _ in range(m):
                amp = mean_amp * max(0.0, 1.0 + amp_jitter * rng.standard_normal())
                # add a short exponential hump starting at k
                L = max(1, int((6.0 * tau_s) / dt))
                end = min(n, k + L)
                u = np.arange(0, end - k) * dt
                crowd[k:end] += amp * np.exp(-u / max(1e-6, tau_s))

    return crowd


def sample_lognormal_durations(rng: np.random.Generator, m: int, mu: float, sigma: float, clip_s: float) -> np.ndarray:
    """Sample heavy-tailed service times in seconds, optionally clipped."""
    d = rng.lognormal(mean=mu, sigma=sigma, size=m)
    if clip_s is not None and clip_s > 0:
        d = np.minimum(d, clip_s)
    return d


# --------------------------- inflight from arrivals ---------------------------


def inflight_from_arrivals(
    arrivals: np.ndarray,  # shape (N, n), integer count per dt
    dt: float,
    rng: np.random.Generator,
    dur_mu: float,
    dur_sigma: float,
    dur_clip_s: float,
) -> np.ndarray:
    """
    Convert arrivals-per-bin into an in-flight time series by sampling a duration
    for each arrival and adding +1 over its active interval.

    Returns:
      inflight: shape (N, n), float (actually integer-ish)
    """
    N, n = arrivals.shape
    inflight = np.zeros((N, n), dtype=np.float64)

    for i in range(N):
        # We'll maintain a difference array for efficient interval additions.
        diff = np.zeros(n + 1, dtype=np.float64)
        for k in range(n):
            m = int(arrivals[i, k])
            if m <= 0:
                continue
            durs = sample_lognormal_durations(rng, m, dur_mu, dur_sigma, dur_clip_s)
            # For each request, mark active interval [k, k+L)
            Ls = np.maximum(1, (durs / dt).astype(int))
            for L in Ls:
                end = min(n, k + int(L))
                diff[k] += 1.0
                diff[end] -= 1.0
        inflight[i] = np.cumsum(diff[:-1])

    return inflight


# --------------------------- main generator -----------------------------------


@dataclass
class WorkloadConfig:
    R_peak: float
    T: float
    dt: float
    N: int
    seed: int

    mean_frac: float
    corr: float
    burstiness: float

    trend_strength: float
    noise_strength: float

    # flash crowd params (shared + per-instance scaled)
    crowd_event_rate: float
    crowd_tau_s: float
    crowd_cluster_strength: float

    # duration model
    dur_mu: float
    dur_sigma: float
    dur_clip_s: float

    # visualization
    smooth_s: float

    # output
    out_dir: Path
    cap_enabled: bool


def generate_google_style_workload(cfg: WorkloadConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      intensities: (N, n) instantaneous λ_i(t) in req/s
      arrivals:    (N, n) arrivals per dt bin (ints)
      inflight:    (N, n) in-flight requests (from arrivals+durations)
      t:           (n,) time points
    """
    rng = np.random.default_rng(cfg.seed)

    t = np.arange(0, cfg.T, cfg.dt)
    n = len(t)

    # Target mean total rate
    target_mean_total = cfg.mean_frac * cfg.R_peak
    target_mean_per = target_mean_total / cfg.N

    # Shared components (create correlation)
    shared_trend = make_diurnal(t, cfg.T, rng, strength=cfg.trend_strength * target_mean_per)
    shared_ou = ou_process(
        rng, n=n, dt=cfg.dt,
        theta=rng.uniform(0.015, 0.05),
        sigma=cfg.noise_strength * 0.30 * target_mean_per,
    )
    shared_crowd = make_flash_crowd_component(
        rng=rng,
        n=n,
        dt=cfg.dt,
        base_event_rate=cfg.crowd_event_rate * (0.6 + 0.8 * cfg.burstiness),
        mean_amp=cfg.burstiness * 0.9 * target_mean_per,
        amp_jitter=0.40,
        tau_s=cfg.crowd_tau_s * (0.8 + 0.6 * cfg.burstiness),
        cluster_strength=cfg.crowd_cluster_strength * (0.6 + 0.8 * cfg.burstiness),
    )

    # Convert shared components into a nonnegative shared "multiplier-ish" signal.
    # We will build intensity as:
    #   λ_i(t) = base_i + corr * shared + (1-corr) * unique_i
    # but keep everything in additive space then clip.

    shared = shared_trend + shared_ou + shared_crowd

    intensities = np.zeros((cfg.N, n), dtype=np.float64)

    for i in range(cfg.N):
        # Instance-specific baseline with heterogeneity
        base = target_mean_per * rng.uniform(0.8, 1.2)

        # Per-instance trend/noise/crowds
        per_trend = make_diurnal(t, cfg.T, rng, strength=cfg.trend_strength * target_mean_per * rng.uniform(0.3, 0.8))
        per_ou = ou_process(
            rng, n=n, dt=cfg.dt,
            theta=rng.uniform(0.02, 0.08),
            sigma=cfg.noise_strength * rng.uniform(0.25, 0.55) * target_mean_per,
        )
        per_crowd = make_flash_crowd_component(
            rng=rng,
            n=n,
            dt=cfg.dt,
            base_event_rate=cfg.crowd_event_rate * rng.uniform(0.6, 1.4) * (0.5 + 0.9 * cfg.burstiness),
            mean_amp=cfg.burstiness * rng.uniform(0.5, 1.3) * target_mean_per,
            amp_jitter=0.45,
            tau_s=cfg.crowd_tau_s * rng.uniform(0.7, 1.5),
            cluster_strength=cfg.crowd_cluster_strength * rng.uniform(0.7, 1.5),
        )

        unique = per_trend + per_ou + per_crowd

        mix = cfg.corr * shared + np.sqrt(max(0.0, 1 - cfg.corr**2)) * unique
        lam = base + mix
        intensities[i] = np.maximum(0.0, lam)

    # Optional cap (thinning)
    if cfg.cap_enabled:
        total = intensities.sum(axis=0)
        factor = np.minimum(1.0, cfg.R_peak / (total + 1e-12))
        intensities = intensities * factor  # broadcast

    # Sample arrivals: arrivals[k] ~ Poisson(λ(t_k) * dt)
    arrivals = rng.poisson(intensities * cfg.dt).astype(np.int32)

    # Compute in-flight by sampling durations
    inflight = inflight_from_arrivals(
        arrivals=arrivals, dt=cfg.dt, rng=rng,
        dur_mu=cfg.dur_mu, dur_sigma=cfg.dur_sigma, dur_clip_s=cfg.dur_clip_s
    )

    return intensities, arrivals, inflight, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R-peak", type=float, default=20.0, help="Capacity / peak total rate cap (req/s)")
    ap.add_argument("--T", type=float, default=300.0, help="Duration (s)")
    ap.add_argument("--dt", type=float, default=1, help="Time step for intensity sampling (s)")
    ap.add_argument("--N", type=int, default=6, help="Number of instances")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--mean-frac", type=float, default=0.5, help="Mean utilization fraction of R_peak (0.5-0.7 typical)")
    ap.add_argument("--corr", type=float, default=0, help="Cross-instance correlation strength (0-1)")
    ap.add_argument("--burstiness", type=float, default=0.3, help="Flash crowd intensity (0-1ish)")

    ap.add_argument("--trend-strength", type=float, default=0.55, help="Strength of slow trend (relative)")
    ap.add_argument("--noise-strength", type=float, default=0.7, help="Strength of OU noise (relative)")

    ap.add_argument("--crowd-event-rate", type=float, default=0.03, help="Base flash event rate (events/s)")
    ap.add_argument("--crowd-tau-s", type=float, default=6.0, help="Decay timescale of crowd humps (s)")
    ap.add_argument("--crowd-cluster-strength", type=float, default=1.2, help="Clustering strength (>0)")

    ap.add_argument("--dur-mu", type=float, default=np.log(3.0), help="Lognormal mu for duration (ln seconds)")
    ap.add_argument("--dur-sigma", type=float, default=0.9, help="Lognormal sigma for duration")
    ap.add_argument("--dur-clip-s", type=float, default=120.0, help="Clip durations at this many seconds (0 disables)")

    ap.add_argument("--smooth-s", type=float, default=2.0, help="Smoothing window for plotting (s)")
    ap.add_argument("--cap-enabled", action="store_true", help="Enable hard cap via thinning (usually rarely active)")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = WorkloadConfig(
        R_peak=args.R_peak,
        T=args.T,
        dt=args.dt,
        N=args.N,
        seed=args.seed,
        mean_frac=args.mean_frac,
        corr=args.corr,
        burstiness=args.burstiness,
        trend_strength=args.trend_strength,
        noise_strength=args.noise_strength,
        crowd_event_rate=args.crowd_event_rate,
        crowd_tau_s=args.crowd_tau_s,
        crowd_cluster_strength=args.crowd_cluster_strength,
        dur_mu=args.dur_mu,
        dur_sigma=args.dur_sigma,
        dur_clip_s=(None if args.dur_clip_s <= 0 else args.dur_clip_s),
        smooth_s=args.smooth_s,
        out_dir=args.out_dir,
        cap_enabled=args.cap_enabled,
    )

    intensities, arrivals, inflight, t = generate_google_style_workload(cfg)

    # Rates for plotting (smoothed realized arrivals converted to req/s)
    n = len(t)
    smooth_w = max(1, int(cfg.smooth_s / cfg.dt))
    rates_realized = arrivals.astype(np.float64) / cfg.dt
    rates_plot = np.vstack([smooth_ma(rates_realized[i], smooth_w) for i in range(cfg.N)])
    total_plot = rates_plot.sum(axis=0)

    inflight_plot = np.vstack([smooth_ma(inflight[i], smooth_w) for i in range(cfg.N)])
    total_inflight_plot = inflight_plot.sum(axis=0)

    # ----------------------------- plot: rates --------------------------------
    cmap = plt.get_cmap("tab10")
    fig1, ax1 = plt.subplots(figsize=(13, 5))
    for i in range(cfg.N):
        ax1.plot(t, rates_plot[i], linewidth=1.3, alpha=0.85, color=cmap(i % 10), label=f"inst{i+1}")
    ax1.plot(t, total_plot, color="black", linewidth=3.0, linestyle="--", label="Total", zorder=5)
    if cfg.cap_enabled:
        ax1.axhline(cfg.R_peak, color="red", linewidth=1.2, linestyle=":", alpha=0.7, label=f"Cap R_peak={cfg.R_peak}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Arrival rate (req/s)")
    ax1.set_title(
        f"Google/Meta-style synthetic workload (arrivals + flash crowds)\n"
        f"N={cfg.N}, mean≈{cfg.mean_frac:.2f}*R_peak={cfg.mean_frac*cfg.R_peak:.1f} req/s, "
        f"corr={cfg.corr:.2f}, burstiness={cfg.burstiness:.2f}"
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, cfg.T)
    ax1.set_ylim(bottom=0)
    ax1.legend(ncol=min(4, cfg.N + 2), fontsize=9)
    plt.tight_layout()
    out1 = cfg.out_dir / "workload_rates.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ---------------------------- plot: inflight ------------------------------
    fig2, ax2 = plt.subplots(figsize=(13, 5))
    for i in range(cfg.N):
        ax2.plot(t, inflight_plot[i], linewidth=1.3, alpha=0.85, color=cmap(i % 10), label=f"inst{i+1}")
    ax2.plot(t, total_inflight_plot, color="black", linewidth=3.0, linestyle="--", label="Total", zorder=5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("In-flight requests")
    ax2.set_title(
        f"In-flight derived from realized arrivals + lognormal durations\n"
        f"durations: mu={cfg.dur_mu:.2f}, sigma={cfg.dur_sigma:.2f}"
    )
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, cfg.T)
    ax2.set_ylim(bottom=0)
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.legend(ncol=min(4, cfg.N + 2), fontsize=9)
    plt.tight_layout()
    out2 = cfg.out_dir / "workload_inflight.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ------------------------------ save trace --------------------------------
    out_trace = cfg.out_dir / "workload_trace.npz"
    np.savez_compressed(
        out_trace,
        t=t,
        intensities=intensities,
        arrivals=arrivals,
        inflight=inflight,
        dt=cfg.dt,
        R_peak=cfg.R_peak,
        mean_frac=cfg.mean_frac,
        corr=cfg.corr,
        burstiness=cfg.burstiness,
        dur_mu=cfg.dur_mu,
        dur_sigma=cfg.dur_sigma,
    )

    # ------------------------------ stats -------------------------------------
    total_rate = rates_realized.sum(axis=0)
    print(f"Saved: {out1}")
    print(f"Saved: {out2}")
    print(f"Saved: {out_trace}\n")

    print(f"Mean total arrival rate: {total_rate.mean():.2f} req/s (target {cfg.mean_frac*cfg.R_peak:.2f})")
    print(f"Std total arrival rate:  {total_rate.std():.2f} req/s")
    print(f"Max total arrival rate:  {total_rate.max():.2f} req/s")
    if cfg.cap_enabled:
        print(f"Hard cap (thinning) set at R_peak={cfg.R_peak} req/s")
    print(f"Max total inflight:      {inflight.sum(axis=0).max():.1f} reqs")
    print(f"Per-instance mean rates: {(rates_realized.mean(axis=1)).round(2)}")


if __name__ == "__main__":
    main()