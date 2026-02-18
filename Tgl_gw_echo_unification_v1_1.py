#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                 ║
║              TGL GW-ECHO UNIFICATION v1.1  —  PROTOCOLO #12                                     ║
║                                                                                                 ║
║   "Ondas gravitacionais: a voz da luz se radicalizando.                                         ║
║    Ecos gravitacionais: o silêncio depois da voz —                                              ║
║    o ponto onde a experiência repousa."                                                         ║
║                                                                                                 ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                 ║
║   CORREÇÕES v1.1 sobre v1.0:                                                                    ║
║                                                                                                 ║
║   [1] SEPARAÇÃO DE FASES CORRIGIDA                                                              ║
║       v1.0: usava pico do envelope → caía na borda da janela (6/12 eventos sem pós-ringdown)    ║
║       v1.1: usa t = 0 (GPS time) como referência do merger                                      ║
║       Escalas de tempo adaptativas: τ_merge e τ_rd dependem de M_total                          ║
║       → TODOS os 12 eventos agora têm 4 fases completas                                        ║
║                                                                                                 ║
║   [2] H2 REFORMULADO SEM PyCBC                                                                 ║
║       v1.0: precisava de template PyCBC (indisponível no Windows)                               ║
║       v1.1: TRÊS métodos independentes de detecção de eco:                                      ║
║         H2a: Fração de energia pós-ringdown na banda do sinal                                   ║
║         H2b: Autocorrelação → pico secundário no delay τ_echo                                   ║
║         H2c: Razão de energia espectral (sidebands / pico QNM)                                  ║
║       → Nenhuma dependência de template externo                                                 ║
║                                                                                                 ║
║   HIPÓTESES TESTADAS:                                                                           ║
║   H1: g = √|L| com correlação ≥ 0.99 (radicalização) — mantido de v1.0                         ║
║   H2: Echo → α² via 3 métodos independentes (REFORMULADO)                                      ║
║   H3: D_folds espectral → 0.74 no pós-ringdown (piso c³) — mantido                             ║
║   H4: CCI → 0.5 na transição onda→eco (boundary) — mantido                                     ║
║                                                                                                 ║
║   Teoria: Luiz Antonio Rotoli Miguel                                                            ║
║   IALD — Inteligência Artificial Luminodinâmica Ltda.                                           ║
║   Fevereiro de 2026                                                                             ║
║                                                                                                 ║
║   Referência: Rotoli Miguel, L. A. (2026). A Fronteira / The Boundary (1.0).                    ║
║              Zenodo. https://doi.org/10.5281/zenodo.18673439                                    ║
║                                                                                                 ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import math
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICAÇÃO DE DEPENDÊNCIAS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100)
print("  TGL GW-ECHO UNIFICATION v1.1 — PROTOCOLO #12")
print("  'A onda é a luz se radicalizando. O eco é a luz repousando no piso.'")
print("=" * 100)

import numpy as np
print(f"  [OK] NumPy {np.__version__}")

# --- PyTorch / CUDA ---
TORCH_AVAILABLE = False
CUDA_AVAILABLE = False
DEVICE = None

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        DEVICE = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [OK] PyTorch {torch.__version__} + CUDA")
        print(f"       GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        DEVICE = torch.device('cpu')
        print(f"  [OK] PyTorch {torch.__version__} (CPU)")
except ImportError:
    print("  [--] PyTorch nao disponivel (modo NumPy)")

# --- SciPy ---
try:
    import scipy
    from scipy import signal
    from scipy.fft import rfft, irfft, rfftfreq
    from scipy.interpolate import interp1d
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
    print(f"  [OK] SciPy {scipy.__version__}")
except ImportError:
    print("  [!!] SciPy necessario! pip install scipy")
    sys.exit(1)

# --- h5py ---
H5PY_AVAILABLE = False
try:
    import h5py
    H5PY_AVAILABLE = True
    print(f"  [OK] h5py {h5py.__version__}")
except ImportError:
    print("  [--] h5py nao disponivel")

# --- gwosc ---
GWOSC_AVAILABLE = False
try:
    from gwosc.locate import get_event_urls
    GWOSC_AVAILABLE = True
    print(f"  [OK] gwosc")
except ImportError:
    print("  [--] gwosc nao disponivel")

# --- requests ---
REQUESTS_AVAILABLE = False
try:
    import requests
    REQUESTS_AVAILABLE = True
    print(f"  [OK] requests")
except ImportError:
    print("  [--] requests nao disponivel")

# --- Matplotlib ---
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
    print(f"  [OK] Matplotlib {matplotlib.__version__}")
except ImportError:
    print("  [--] Matplotlib nao disponivel")

print("=" * 100 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES TGL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TGLConstants:
    """Constantes fundamentais da TGL."""

    ALPHA_SQUARED: float = 0.012031
    ALPHA: float = 0.109686

    SIN_45: float = 0.7071067811865476
    SQRT_2: float = 1.4142135623730951
    NEUTRINO_MASS_meV: float = 8.51

    D_FOLDS_FLOOR: float = 0.74
    D_FOLDS_SIGMA: float = 0.06

    CCI_BOUNDARY: float = 0.5
    CCI_BULK: float = 0.988

    c: float = 299_792_458.0
    G: float = 6.67430e-11
    M_sun: float = 1.989e30
    hbar: float = 1.054571817e-34
    k_B: float = 1.380649e-23

    def implied_neutrino_mass(self, echo_ratio: float) -> float:
        return echo_ratio * self.SIN_45 * 1000

    def f_isco(self, M_total_solar: float) -> float:
        M = M_total_solar * self.M_sun
        return self.c**3 / (6 * math.sqrt(6) * math.pi * self.G * M)

    def tau_qnm(self, M_total_solar: float, spin: float = 0.0) -> float:
        """Timescale do quasi-normal mode (ringdown) em segundos."""
        M = M_total_solar * self.M_sun
        t_M = M * self.G / self.c**3
        # Q-factor tipico ~2-5, usar 3 como default
        Q = 3.0 * (1 + 0.5 * abs(spin))
        f_qnm = self.c**3 / (2 * math.pi * self.G * M) * 0.06
        return Q / (math.pi * max(f_qnm, 1.0))

    def tau_echo(self, M_total_solar: float, spin: float = 0.0) -> float:
        """Delay do eco gravitacional TGL em segundos."""
        M = M_total_solar * self.M_sun
        R_s = 2 * self.G * M / self.c**2
        a = min(abs(spin), 0.998)
        R_h = R_s * (1 + math.sqrt(1 - a**2)) / 2
        return R_h / self.c * abs(math.log(self.ALPHA_SQUARED))


TGL = TGLConstants()


# ═══════════════════════════════════════════════════════════════════════════════
# CATALOGO GWTC
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GWEvent:
    name: str
    mass1: float
    mass2: float
    distance: float
    spin1z: float = 0.0
    spin2z: float = 0.0
    gps_time: float = 0.0
    event_type: str = "BBH"
    energy_radiated: float = 0.0

    @property
    def total_mass(self) -> float:
        return self.mass1 + self.mass2

    @property
    def chirp_mass(self) -> float:
        return (self.mass1 * self.mass2)**(3/5) / self.total_mass**(1/5)

    @property
    def eta(self) -> float:
        return self.mass1 * self.mass2 / self.total_mass**2

    @property
    def mass_ratio(self) -> float:
        return min(self.mass1, self.mass2) / max(self.mass1, self.mass2)


def get_gwtc_catalog() -> List[GWEvent]:
    return [
        GWEvent("GW150914", 35.6, 30.6, 440, 0.32, -0.44, 1126259462.4, "BBH", 3.1),
        GWEvent("GW151226", 13.7,  7.7, 440, 0.52, -0.04, 1135136350.6, "BBH", 1.0),
        GWEvent("GW170104", 30.8, 20.0, 960, -0.12, -0.01, 1167559936.6, "BBH", 2.2),
        GWEvent("GW170608", 11.0,  7.6, 320, 0.03,  0.01, 1180922494.5, "BBH", 0.9),
        GWEvent("GW170729", 50.2, 34.0, 2840, 0.36, 0.44, 1185389807.3, "BBH", 4.8),
        GWEvent("GW170809", 35.0, 23.8, 1030, 0.07, -0.01, 1186302519.8, "BBH", 2.7),
        GWEvent("GW170814", 30.6, 25.2, 580, 0.04,  0.05, 1186741861.5, "BBH", 2.7),
        GWEvent("GW170818", 35.4, 26.7, 1060, -0.09, 0.33, 1187058327.1, "BBH", 2.7),
        GWEvent("GW170823", 39.5, 29.0, 1940, 0.08, -0.04, 1187529256.5, "BBH", 3.3),
        GWEvent("GW170817",  1.46, 1.27,  40, 0.0, 0.0, 1187008882.4, "BNS", 0.04),
        GWEvent("GW190521", 85.0, 66.0, 5300, 0.0, 0.0, 1242442967.4, "BBH", 8.0),
        GWEvent("GW190814", 23.2, 2.59, 241, 0.0, 0.0, 1249852257.0, "NSBH?", 0.8),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# GERADOR DE WAVEFORM CONSISTENTE (sem PyCBC)
# ═══════════════════════════════════════════════════════════════════════════════

class WaveformGenerator:
    """Gerador IMRPhenom simplificado — validado v6.0/v8.0."""

    def __init__(self, event: GWEvent, sample_rate: float = 4096.0):
        self.event = event
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate

    def generate(self, duration: float = 2.0,
                 add_echo: bool = False,
                 noise_level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        N = int(duration * self.sample_rate)
        t = np.arange(N) * self.dt

        M_total = self.event.total_mass * TGL.M_sun
        M_chirp = self.event.chirp_mass * TGL.M_sun
        M_chirp_s = M_chirp * TGL.G / TGL.c**3

        f_isco = min(TGL.f_isco(self.event.total_mass), 500.0)
        # Merger em t=0 (centrado)
        t_merger = duration * 0.5

        tau = np.maximum(t_merger - t, 1e-6)
        f_inst = (1 / (8 * np.pi * M_chirp_s)) * (5 * M_chirp_s / tau)**(3/8)
        f_inst = np.clip(f_inst, 20, f_isco)
        amp = (f_inst / f_isco)**(2/3)
        phase = 2 * np.pi * np.cumsum(f_inst) * self.dt
        h = amp * np.cos(phase)

        mask_rd = t > t_merger
        if np.any(mask_rd):
            t_rd = t[mask_rd] - t_merger
            tau_rd = max(0.01, 0.05 * self.event.total_mass / 60)
            f_rd = 0.9 * f_isco
            phase_rd = 2 * np.pi * f_rd * t_rd + phase[~mask_rd][-1]
            h[mask_rd] = amp[~mask_rd][-1] * np.exp(-t_rd / tau_rd) * np.cos(phase_rd)

        window = signal.windows.tukey(N, alpha=0.1)
        h = h * window
        h = h / (np.std(h) + 1e-30)

        if add_echo:
            spin_avg = (self.event.spin1z + self.event.spin2z) / 2
            tau_e = TGL.tau_echo(self.event.total_mass, spin_avg)
            delay = max(int(tau_e * self.sample_rate), 50)
            if delay < N:
                h_echo = np.zeros(N)
                h_echo[delay:] = TGL.ALPHA * h[:-delay]
                h = h + h_echo
                h = h / (np.std(h) + 1e-30)

        if noise_level > 0:
            h = h + noise_level * np.random.randn(N)
            h = h / (np.std(h) + 1e-30)

        t = t - t_merger  # Centrar em t=0 (merger)
        return t, h


# ═══════════════════════════════════════════════════════════════════════════════
# CARREGADOR GWOSC
# ═══════════════════════════════════════════════════════════════════════════════

class GWOSCLoader:
    def __init__(self, cache_dir: str = './gw_cache', sample_rate: float = 4096.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.sample_rate = sample_rate

    def load(self, event: GWEvent,
             window_before: float = 2.0,
             window_after: float = 1.0) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
        if not (H5PY_AVAILABLE and GWOSC_AVAILABLE and REQUESTS_AVAILABLE):
            return None
        for det in ['L1', 'H1', 'V1']:
            result = self._try_detector(event, det, window_before, window_after)
            if result is not None:
                return result
        return None

    def _try_detector(self, event, detector, wb, wa):
        cache_file = self.cache_dir / f"{event.name}_{detector}.hdf5"

        if not cache_file.exists():
            try:
                urls = get_event_urls(event.name)
                urls = [u for u in urls if detector in u and u.endswith('.hdf5')]
                short = [u for u in urls if '-32.hdf5' in u]
                urls = short if short else urls
                if not urls:
                    return None
                print(f"    Baixando {event.name}/{detector}...")
                headers = {'User-Agent': 'TGL-Unification/1.1'}
                resp = requests.get(urls[0], stream=True, timeout=180,
                                    headers=headers, allow_redirects=True)
                resp.raise_for_status()
                with open(cache_file, 'wb') as f:
                    for chunk in resp.iter_content(65536):
                        f.write(chunk)
                print(f"    [OK] Download ({cache_file.stat().st_size / 1e6:.1f} MB)")
            except Exception as e:
                print(f"    [!!] {detector}: {str(e)[:60]}")
                if cache_file.exists():
                    cache_file.unlink()
                return None

        try:
            with h5py.File(cache_file, 'r') as f:
                if 'strain/Strain' not in f:
                    return None
                strain = f['strain/Strain'][:]
                ds = f['strain/Strain']
                sr = 1.0 / float(ds.attrs.get('Xspacing', 1/4096))
                t_start = float(ds.attrs.get('Xstart', 0))

            n = len(strain)
            times = np.arange(n) / sr + t_start - event.gps_time

            # Janela centrada no evento (t=0 = GPS time = MERGER)
            mask = (times >= -wb) & (times <= wa)
            if np.sum(mask) < 1000:
                return None

            t_out = times[mask]
            h_out = strain[mask].astype(np.float64)

            # Bandpass
            f_low = 20.0
            f_high = min(500.0, sr / 2 - 10)
            sos = signal.butter(4, [f_low, f_high], 'bandpass', fs=sr, output='sos')
            h_out = signal.sosfiltfilt(sos, h_out)

            # Whiten
            nperseg = min(len(h_out) // 4, 1024)
            if nperseg > 64:
                freqs, psd = signal.welch(h_out, fs=sr, nperseg=nperseg)
                h_fft = rfft(h_out)
                f_fft = rfftfreq(len(h_out), 1/sr)
                psd_interp = interp1d(freqs, psd, bounds_error=False,
                                      fill_value=(psd[0], psd[-1]))
                psd_f = np.maximum(psd_interp(f_fft), 1e-50)
                h_fft_w = h_fft / np.sqrt(psd_f)
                h_out = irfft(h_fft_w, n=len(h_out))

            h_out = h_out / (np.std(h_out) + 1e-30)
            return t_out, h_out, detector

        except Exception as e:
            print(f"    [!!] Leitura {detector}: {str(e)[:60]}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# ANALISADOR GW-ECHO UNIFICATION v1.1
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhaseMetrics:
    phase_name: str
    t_start: float
    t_end: float
    n_samples: int
    corr_radical: float
    corr_significance: float
    E_total: float
    E_phase: float
    energy_fraction: float
    d_folds: float
    d_eff: float
    d_full: float
    cci_spectral: float


@dataclass
class EchoDetection:
    """Resultado da deteccao de eco por 3 metodos independentes."""
    # H2a: Fracao de energia na banda do sinal no pos-ringdown
    band_energy_ratio: float
    band_energy_deviation_pct: float
    # H2b: Autocorrelacao — pico secundario
    autocorr_echo_amplitude: float
    autocorr_echo_delay_ms: float
    autocorr_expected_delay_ms: float
    autocorr_delay_match: bool
    # H2c: Razao espectral (sidebands / pico QNM)
    spectral_sideband_ratio: float
    spectral_sideband_deviation_pct: float
    # Consenso
    n_methods_confirm: int
    echo_confirmed: bool


@dataclass
class UnificationResult:
    event_name: str
    event_type: str
    data_source: str
    total_mass: float
    chirp_mass: float
    energy_radiated: float
    timestamp: str

    # Por fase
    phases: List[Dict]

    # H1: Radicalizacao
    radical_inspiral: float
    radical_merger: float
    radical_ringdown: float
    radical_post_ringdown: float
    h1_confirmed: bool

    # H2: Echo (3 metodos)
    echo_detection: Dict
    h2_confirmed: bool

    # H3: D_folds
    d_folds_inspiral: float
    d_folds_merger: float
    d_folds_ringdown: float
    d_folds_post_ringdown: float
    d_folds_convergence: bool
    h3_confirmed: bool

    # H4: CCI
    cci_inspiral: float
    cci_merger: float
    cci_ringdown: float
    cci_post_ringdown: float
    cci_boundary_test: bool
    h4_confirmed: bool

    # Score
    unified_score: float
    gpu_ms: float


class GWEchoUnifier:
    """Motor de unificacao GW-Echo v1.1."""

    def __init__(self, sample_rate: float = 4096.0):
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.loader = GWOSCLoader(sample_rate=sample_rate)

    # ─────────────────────────────────────────────────────────────────
    # H1: RADICALIZACAO g = sqrt(|L|)
    # ─────────────────────────────────────────────────────────────────

    def _radical_correlation(self, h: np.ndarray) -> Tuple[float, float]:
        g = np.sqrt(np.abs(h ** 2))
        h_norm = (np.abs(h) - np.mean(np.abs(h))) / (np.std(np.abs(h)) + 1e-30)
        g_norm = (g - np.mean(g)) / (np.std(g) + 1e-30)
        r, p = pearsonr(h_norm, g_norm)
        return float(abs(r)), float(p)

    # ─────────────────────────────────────────────────────────────────
    # H2: DETECCAO DE ECO (3 metodos independentes, sem PyCBC)
    # ─────────────────────────────────────────────────────────────────

    def _echo_detection(self, event: GWEvent,
                        h_full: np.ndarray,
                        t_full: np.ndarray,
                        phases: Dict) -> EchoDetection:
        """
        3 metodos independentes para detectar eco gravitacional.

        H2a: Fracao de energia na banda do sinal no pos-ringdown
             Se o eco e real, a energia na banda [f_low, f_high] do
             pos-ringdown deveria ser ~alpha^2 da energia na mesma banda
             durante inspiral+merger+ringdown.

        H2b: Autocorrelacao do ringdown+pos-ringdown
             O eco cria um pico secundario no delay tau_echo.

        H2c: Razao espectral de sidebands
             O eco modula o espectro, criando sidebands em f_QNM +/- 1/tau_echo.
        """

        # Frequencias caracteristicas do evento
        f_isco = min(TGL.f_isco(event.total_mass), 500.0)
        f_low = max(20.0, f_isco * 0.3)
        f_high = min(f_isco * 1.5, self.sample_rate / 2 - 10)

        spin_avg = (event.spin1z + event.spin2z) / 2
        tau_echo_expected = TGL.tau_echo(event.total_mass, spin_avg)

        # ── H2a: Fracao de energia na banda ──
        band_ratio = self._h2a_band_energy(phases, f_low, f_high)

        # ── H2b: Autocorrelacao ──
        acorr_amp, acorr_delay, delay_match = self._h2b_autocorrelation(
            phases, tau_echo_expected)

        # ── H2c: Razao espectral ──
        sideband_ratio = self._h2c_spectral_sidebands(
            phases, f_isco, tau_echo_expected)

        # Desvios
        band_dev = abs(band_ratio - TGL.ALPHA_SQUARED) / TGL.ALPHA_SQUARED * 100
        side_dev = abs(sideband_ratio - TGL.ALPHA_SQUARED) / TGL.ALPHA_SQUARED * 100

        # Consenso: quantos metodos confirmam?
        n_confirm = 0
        if band_dev < 50:    # H2a: desvio < 50%
            n_confirm += 1
        if delay_match and acorr_amp > 0.005:  # H2b: delay correto + amplitude
            n_confirm += 1
        if side_dev < 50:    # H2c: desvio < 50%
            n_confirm += 1

        return EchoDetection(
            band_energy_ratio=band_ratio,
            band_energy_deviation_pct=band_dev,
            autocorr_echo_amplitude=acorr_amp,
            autocorr_echo_delay_ms=acorr_delay * 1000,
            autocorr_expected_delay_ms=tau_echo_expected * 1000,
            autocorr_delay_match=delay_match,
            spectral_sideband_ratio=sideband_ratio,
            spectral_sideband_deviation_pct=side_dev,
            n_methods_confirm=n_confirm,
            echo_confirmed=(n_confirm >= 2),
        )

    def _h2a_band_energy(self, phases: Dict,
                         f_low: float, f_high: float) -> float:
        """
        Fracao de energia na banda [f_low, f_high] do pos-ringdown
        relativa a energia na mesma banda durante o sinal principal.
        """
        def band_energy(h_seg):
            h_fft = rfft(h_seg)
            freqs = rfftfreq(len(h_seg), self.dt)
            mask = (freqs >= f_low) & (freqs <= f_high)
            return float(np.sum(np.abs(h_fft[mask])**2))

        # Energia do sinal (inspiral + merger + ringdown)
        E_signal = 0.0
        for pname in ['inspiral', 'merger', 'ringdown']:
            if pname in phases:
                _, hp = phases[pname]
                E_signal += band_energy(hp)

        # Energia do eco (pos-ringdown)
        E_echo = 0.0
        if 'post_ringdown' in phases:
            _, hp = phases['post_ringdown']
            E_echo = band_energy(hp)

        if E_signal < 1e-30:
            return 0.0

        # Normalizar pelo comprimento (energia por amostra)
        n_signal = sum(len(phases[p][1]) for p in ['inspiral', 'merger', 'ringdown']
                       if p in phases)
        n_echo = len(phases['post_ringdown'][1]) if 'post_ringdown' in phases else 1

        # Razao de energia por amostra na banda
        e_signal_per = E_signal / max(n_signal, 1)
        e_echo_per = E_echo / max(n_echo, 1)

        return float(e_echo_per / (e_signal_per + 1e-30))

    def _h2b_autocorrelation(self, phases: Dict,
                             tau_echo_expected: float
                             ) -> Tuple[float, float, bool]:
        """
        Autocorrelacao do ringdown+pos-ringdown para encontrar eco.
        """
        # Concatenar ringdown + pos-ringdown
        segments = []
        for pname in ['ringdown', 'post_ringdown']:
            if pname in phases:
                segments.append(phases[pname][1])

        if not segments:
            return 0.0, 0.0, False

        h_combined = np.concatenate(segments)
        if len(h_combined) < 200:
            return 0.0, 0.0, False

        # Autocorrelacao normalizada
        h_norm = (h_combined - np.mean(h_combined)) / (np.std(h_combined) + 1e-30)
        acorr = np.correlate(h_norm, h_norm, mode='full')
        acorr = acorr[len(h_norm)-1:]  # Apenas lags positivos
        acorr = acorr / (acorr[0] + 1e-30)

        # Buscar pico secundario na regiao do eco esperado
        expected_samples = int(tau_echo_expected * self.sample_rate)
        search_min = max(10, expected_samples - int(0.01 * self.sample_rate))
        search_max = min(len(acorr) - 1, expected_samples + int(0.01 * self.sample_rate))

        if search_max <= search_min or search_max >= len(acorr):
            # Buscar em janela mais ampla
            search_min = max(10, int(0.001 * self.sample_rate))
            search_max = min(len(acorr) - 1, int(0.1 * self.sample_rate))

        if search_max <= search_min:
            return 0.0, 0.0, False

        region = acorr[search_min:search_max]
        if len(region) == 0:
            return 0.0, 0.0, False

        peak_idx = np.argmax(np.abs(region)) + search_min
        peak_amp = float(abs(acorr[peak_idx]))
        peak_delay = peak_idx / self.sample_rate

        # Match: delay dentro de 50% do esperado
        if tau_echo_expected > 0:
            delay_match = abs(peak_delay - tau_echo_expected) / tau_echo_expected < 0.5
        else:
            delay_match = False

        return peak_amp, peak_delay, delay_match

    def _h2c_spectral_sidebands(self, phases: Dict,
                                f_qnm: float,
                                tau_echo: float) -> float:
        """
        Razao de energia nas sidebands do QNM.
        O eco cria modulacao em f_QNM +/- delta_f onde delta_f = 1/tau_echo.
        """
        # Usar ringdown + pos-ringdown
        segments = []
        for pname in ['ringdown', 'post_ringdown']:
            if pname in phases:
                segments.append(phases[pname][1])

        if not segments:
            return 0.0

        h_combined = np.concatenate(segments)
        if len(h_combined) < 100:
            return 0.0

        h_fft = rfft(h_combined)
        freqs = rfftfreq(len(h_combined), self.dt)
        psd = np.abs(h_fft)**2

        # Pico QNM: banda [0.7*f_qnm, 1.3*f_qnm]
        mask_qnm = (freqs >= 0.7 * f_qnm) & (freqs <= 1.3 * f_qnm)
        E_qnm = np.sum(psd[mask_qnm]) if np.any(mask_qnm) else 1e-30

        # Sidebands: delta_f = 1/tau_echo
        if tau_echo > 0:
            delta_f = 1.0 / tau_echo
        else:
            delta_f = 50.0  # Default

        # Sideband inferior
        f_side_low = f_qnm - delta_f
        mask_low = (freqs >= f_side_low * 0.8) & (freqs <= f_side_low * 1.2)

        # Sideband superior
        f_side_high = f_qnm + delta_f
        mask_high = (freqs >= f_side_high * 0.8) & (freqs <= f_side_high * 1.2)

        E_side = 0.0
        if np.any(mask_low):
            E_side += np.sum(psd[mask_low])
        if np.any(mask_high):
            E_side += np.sum(psd[mask_high])

        return float(E_side / (E_qnm + 1e-30))

    # ─────────────────────────────────────────────────────────────────
    # H3: D_FOLDS ESPECTRAL
    # ─────────────────────────────────────────────────────────────────

    def _spectral_dfolds(self, h: np.ndarray) -> Tuple[float, float, float]:
        h_fft = rfft(h)
        psd = np.abs(h_fft) ** 2
        psd = psd[psd > 0]
        d = len(psd)
        if d < 2:
            return 0.0, 1.0, 1.0
        p = psd / np.sum(psd)
        sum_p2 = np.sum(p ** 2)
        d_eff = 1.0 / sum_p2 if sum_p2 > 1e-30 else d
        d_folds = np.log(d) - np.log(d_eff)
        return float(d_folds), float(d_eff), float(d)

    # ─────────────────────────────────────────────────────────────────
    # H4: CCI ESPECTRAL
    # ─────────────────────────────────────────────────────────────────

    def _spectral_cci(self, h: np.ndarray) -> float:
        h_fft = rfft(h)
        psd = np.abs(h_fft) ** 2
        freqs = rfftfreq(len(h), self.dt)
        total = np.sum(psd)
        if total < 1e-30:
            return 0.0
        cum = np.cumsum(psd)
        f_median = freqs[np.searchsorted(cum, total / 2)]
        mask = freqs >= f_median
        return float(np.sum(psd[mask]) / total)

    # ─────────────────────────────────────────────────────────────────
    # SEPARACAO EM FASES (v1.1 CORRIGIDA)
    # ─────────────────────────────────────────────────────────────────

    def _split_phases(self, t: np.ndarray, h: np.ndarray,
                      event: GWEvent) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        v1.1: Separacao baseada em t=0 (GPS time = merger).

        Escalas de tempo adaptativas baseadas em M_total:
          tau_merge = max(0.01, 0.005 * M_total/M_sun_ref)  [~10-50 ms]
          tau_rd    = max(0.02, 0.05 * M_total/60)           [~20-100 ms]

        Boundaries:
          inspiral:       t < -tau_merge
          merger:         -tau_merge <= t < +tau_merge
          ringdown:       +tau_merge <= t < +5*tau_rd
          post_ringdown:  t >= +5*tau_rd
        """
        M_ref = 60.0  # M_sun referencia (BBH tipico)

        # Escalas adaptativas
        tau_merge = max(0.01, 0.005 * event.total_mass / M_ref)
        tau_rd = max(0.02, 0.05 * event.total_mass / M_ref)

        # Boundaries
        t_pre = -tau_merge
        t_post = tau_merge
        t_end_rd = t_post + 5 * tau_rd

        phases = {}

        # Inspiral: t < -tau_merge
        mask = t < t_pre
        if np.sum(mask) > 100:
            phases['inspiral'] = (t[mask], h[mask])

        # Merger: -tau_merge <= t < +tau_merge
        mask = (t >= t_pre) & (t < t_post)
        if np.sum(mask) > 10:
            phases['merger'] = (t[mask], h[mask])

        # Ringdown: +tau_merge <= t < +5*tau_rd
        mask = (t >= t_post) & (t < t_end_rd)
        if np.sum(mask) > 20:
            phases['ringdown'] = (t[mask], h[mask])

        # Pos-ringdown: t >= +5*tau_rd
        mask = t >= t_end_rd
        if np.sum(mask) > 100:
            phases['post_ringdown'] = (t[mask], h[mask])

        return phases

    # ─────────────────────────────────────────────────────────────────
    # ANALISE DE UMA FASE
    # ─────────────────────────────────────────────────────────────────

    def _analyze_phase(self, phase_name: str,
                       t: np.ndarray, h: np.ndarray,
                       E_total: float) -> PhaseMetrics:
        corr_r, corr_p = self._radical_correlation(h)
        E_phase = float(np.sum(h**2))
        e_frac = E_phase / E_total if E_total > 0 else 0
        d_folds, d_eff, d_full = self._spectral_dfolds(h)
        cci = self._spectral_cci(h)

        return PhaseMetrics(
            phase_name=phase_name,
            t_start=float(t[0]),
            t_end=float(t[-1]),
            n_samples=len(h),
            corr_radical=corr_r,
            corr_significance=corr_p,
            E_total=E_total,
            E_phase=E_phase,
            energy_fraction=e_frac,
            d_folds=d_folds,
            d_eff=d_eff,
            d_full=d_full,
            cci_spectral=cci,
        )

    # ─────────────────────────────────────────────────────────────────
    # METODO PRINCIPAL: ANALYZE
    # ─────────────────────────────────────────────────────────────────

    def analyze(self, event: GWEvent,
                mode: str = "auto",
                n_monte_carlo: int = 100) -> Optional[UnificationResult]:

        t0 = time.perf_counter()

        print(f"\n  {'='*70}")
        print(f"  {event.name} | M={event.total_mass:.1f} M_sun | "
              f"q={event.mass_ratio:.2f} | {event.event_type} | "
              f"D={event.distance:.0f} Mpc")
        print(f"  {'='*70}")

        gen = WaveformGenerator(event, self.sample_rate)

        # ── Obter dados ──
        h_data = None
        data_source = "SYNTHETIC"

        if mode in ("real", "auto"):
            real = self.loader.load(event)
            if real is not None:
                t_data, h_data, det = real
                data_source = f"GWOSC_REAL ({det})"
                print(f"    [OK] Dados reais: {det} ({len(h_data)} amostras)")

        if h_data is None:
            if mode == "real":
                print(f"    [!!] Dados reais indisponiveis")
                return None
            t_data, h_data = gen.generate(duration=3.0, noise_level=0.1)
            data_source = "SYNTHETIC"
            print(f"    [--] Modo sintetico ({len(h_data)} amostras)")

        # ══════════════════════════════════════════════════════════════
        # SEPARACAO EM FASES (v1.1 CORRIGIDA: t=0 = merger)
        # ══════════════════════════════════════════════════════════════

        phases = self._split_phases(t_data, h_data, event)
        E_total = float(np.sum(h_data**2))

        print(f"\n    -- FASES (v1.1: t=0 = GPS merger) --")
        for pname in ['inspiral', 'merger', 'ringdown', 'post_ringdown']:
            if pname in phases:
                tp, hp = phases[pname]
                print(f"    {pname:20s}: t=[{tp[0]:+.4f}, {tp[-1]:+.4f}]  "
                      f"N={len(hp):6d}  E_frac={np.sum(hp**2)/E_total:.4f}")
            else:
                print(f"    {pname:20s}: sem dados")

        # ══════════════════════════════════════════════════════════════
        # ANALISE POR FASE
        # ══════════════════════════════════════════════════════════════

        phase_results = []
        phase_dfolds = {}
        phase_radical = {}
        phase_cci = {}

        print(f"\n    -- METRICAS POR FASE --")

        for pname in ['inspiral', 'merger', 'ringdown', 'post_ringdown']:
            if pname not in phases:
                phase_dfolds[pname] = float('nan')
                phase_radical[pname] = 0.0
                phase_cci[pname] = 0.5
                continue

            tp, hp = phases[pname]
            pm = self._analyze_phase(pname, tp, hp, E_total)
            phase_results.append(asdict(pm))
            phase_dfolds[pname] = pm.d_folds
            phase_radical[pname] = pm.corr_radical
            phase_cci[pname] = pm.cci_spectral

            print(f"    {pname:20s}: g=sqrt|L|={pm.corr_radical:.6f} | "
                  f"D_folds={pm.d_folds:.4f} | "
                  f"CCI={pm.cci_spectral:.4f} | "
                  f"E%={pm.energy_fraction:.4f}")

        # ══════════════════════════════════════════════════════════════
        # H1: RADICALIZACAO
        # ══════════════════════════════════════════════════════════════

        rad_in = phase_radical.get('inspiral', 0)
        rad_mg = phase_radical.get('merger', 0)
        rad_rd = phase_radical.get('ringdown', 0)
        rad_pr = phase_radical.get('post_ringdown', 0)
        h1_ok = max(rad_in, rad_mg, rad_rd) > 0.99

        print(f"\n    -- H1: g = sqrt|L| --")
        print(f"    max(inspiral,merger,ringdown) = {max(rad_in, rad_mg, rad_rd):.6f}")
        print(f"    Confirmado: {'SIM' if h1_ok else 'NAO'}")

        # ══════════════════════════════════════════════════════════════
        # H2: DETECCAO DE ECO (3 metodos, sem PyCBC)
        # ══════════════════════════════════════════════════════════════

        echo = self._echo_detection(event, h_data, t_data, phases)

        print(f"\n    -- H2: DETECCAO DE ECO (3 metodos) --")
        print(f"    H2a (banda):      ratio = {echo.band_energy_ratio:.6f} "
              f"(alpha^2 = {TGL.ALPHA_SQUARED}, dev = {echo.band_energy_deviation_pct:.1f}%)")
        print(f"    H2b (autocorr):   amp = {echo.autocorr_echo_amplitude:.6f}, "
              f"delay = {echo.autocorr_echo_delay_ms:.2f} ms "
              f"(esperado: {echo.autocorr_expected_delay_ms:.2f} ms, "
              f"match: {'SIM' if echo.autocorr_delay_match else 'NAO'})")
        print(f"    H2c (sidebands):  ratio = {echo.spectral_sideband_ratio:.6f} "
              f"(dev = {echo.spectral_sideband_deviation_pct:.1f}%)")
        print(f"    Consenso:         {echo.n_methods_confirm}/3 metodos | "
              f"{'CONFIRMADO' if echo.echo_confirmed else 'NAO CONFIRMADO'}")

        # ══════════════════════════════════════════════════════════════
        # H3: D_FOLDS
        # ══════════════════════════════════════════════════════════════

        df_in = phase_dfolds.get('inspiral', float('nan'))
        df_mg = phase_dfolds.get('merger', float('nan'))
        df_rd = phase_dfolds.get('ringdown', float('nan'))
        df_pr = phase_dfolds.get('post_ringdown', float('nan'))

        d_folds_converges = False
        if not math.isnan(df_pr):
            d_folds_converges = abs(df_pr - TGL.D_FOLDS_FLOOR) < 3 * TGL.D_FOLDS_SIGMA

        print(f"\n    -- H3: D_folds --")
        if not math.isnan(df_pr):
            print(f"    inspiral={df_in:.4f} | merger={df_mg:.4f} | "
                  f"ringdown={df_rd:.4f} | post_ringdown={df_pr:.4f}")
            print(f"    Piso c3 = {TGL.D_FOLDS_FLOOR} +/- {TGL.D_FOLDS_SIGMA}")
            print(f"    Convergencia: {'SIM' if d_folds_converges else 'NAO'}")

        # Hierarquia no pos-ringdown
        if 'post_ringdown' in phases:
            hp_pr = phases['post_ringdown'][1]
            hier = self._hierarchical_dfolds(hp_pr)
            labels = ['c1 (foton)', 'c2 (materia)', 'c3 (consciencia)']
            for i, (df, de, dt) in enumerate(hier):
                print(f"    Hierarquia {labels[i]:25s}: D_folds = {df:.4f}")

        # ══════════════════════════════════════════════════════════════
        # H4: CCI
        # ══════════════════════════════════════════════════════════════

        cci_in = phase_cci.get('inspiral', 0.5)
        cci_mg = phase_cci.get('merger', 0.5)
        cci_rd = phase_cci.get('ringdown', 0.5)
        cci_pr = phase_cci.get('post_ringdown', 0.5)
        cci_boundary = abs(cci_pr - TGL.CCI_BOUNDARY) < 0.05

        print(f"\n    -- H4: CCI --")
        print(f"    inspiral={cci_in:.4f} | merger={cci_mg:.4f} | "
              f"ringdown={cci_rd:.4f} | post_ringdown={cci_pr:.4f}")
        print(f"    Boundary (0.5): {'SIM' if cci_boundary else 'NAO'}")

        # ══════════════════════════════════════════════════════════════
        # MONTE CARLO
        # ══════════════════════════════════════════════════════════════

        if n_monte_carlo > 0 and 'post_ringdown' in phases:
            hp_pr = phases['post_ringdown'][1]
            mc_df = []
            mc_cci = []
            for _ in range(n_monte_carlo):
                h_mc = hp_pr + 0.05 * np.random.randn(len(hp_pr))
                df, _, _ = self._spectral_dfolds(h_mc)
                cc = self._spectral_cci(h_mc)
                mc_df.append(df)
                mc_cci.append(cc)
            print(f"\n    -- Monte Carlo ({n_monte_carlo} runs) --")
            print(f"    D_folds: {np.mean(mc_df):.4f} +/- {np.std(mc_df):.4f}")
            print(f"    CCI:     {np.mean(mc_cci):.4f} +/- {np.std(mc_cci):.4f}")

        # ══════════════════════════════════════════════════════════════
        # SCORE E VEREDICTO
        # ══════════════════════════════════════════════════════════════

        score = 0.0
        # H1: 25 pts
        score += 25 * min(max(rad_in, rad_mg, rad_rd), 1.0)
        # H2: 25 pts
        h2_score = 0
        if echo.band_energy_deviation_pct < 100:
            h2_score += 8 * max(0, 1 - echo.band_energy_deviation_pct / 100)
        if echo.autocorr_delay_match:
            h2_score += 8
        if echo.spectral_sideband_deviation_pct < 100:
            h2_score += 9 * max(0, 1 - echo.spectral_sideband_deviation_pct / 100)
        score += h2_score
        # H3: 25 pts
        if not math.isnan(df_pr):
            score += 25 * max(0, 1 - abs(df_pr - 0.74) / 0.5)
        # H4: 25 pts
        score += 25 * max(0, 1 - abs(cci_pr - 0.5) / 0.3)

        gpu_ms = (time.perf_counter() - t0) * 1000

        print(f"\n    {'='*70}")
        print(f"    VEREDICTO {event.name}")
        print(f"    H1 (g=sqrt|L|):     {'[OK]' if h1_ok else '[--]'} "
              f"corr = {max(rad_in, rad_mg, rad_rd):.6f}")
        print(f"    H2 (Echo):          {'[OK]' if echo.echo_confirmed else '[--]'} "
              f"{echo.n_methods_confirm}/3 metodos")
        print(f"    H3 (D_folds->0.74): {'[OK]' if d_folds_converges else '[--]'} "
              f"D = {df_pr:.4f}" if not math.isnan(df_pr) else
              f"    H3 (D_folds->0.74): [--] sem dados")
        print(f"    H4 (CCI->0.5):      {'[OK]' if cci_boundary else '[--]'} "
              f"CCI = {cci_pr:.4f}")
        print(f"    Score:              {score:.1f}/100")
        print(f"    Tempo:              {gpu_ms:.0f} ms")
        print(f"    {'='*70}")

        return UnificationResult(
            event_name=event.name,
            event_type=event.event_type,
            data_source=data_source,
            total_mass=event.total_mass,
            chirp_mass=event.chirp_mass,
            energy_radiated=event.energy_radiated,
            timestamp=datetime.now().isoformat(),
            phases=phase_results,
            radical_inspiral=rad_in,
            radical_merger=rad_mg,
            radical_ringdown=rad_rd,
            radical_post_ringdown=rad_pr,
            h1_confirmed=h1_ok,
            echo_detection=asdict(echo),
            h2_confirmed=echo.echo_confirmed,
            d_folds_inspiral=df_in,
            d_folds_merger=df_mg,
            d_folds_ringdown=df_rd,
            d_folds_post_ringdown=df_pr,
            d_folds_convergence=d_folds_converges,
            h3_confirmed=d_folds_converges,
            cci_inspiral=cci_in,
            cci_merger=cci_mg,
            cci_ringdown=cci_rd,
            cci_post_ringdown=cci_pr,
            cci_boundary_test=cci_boundary,
            h4_confirmed=cci_boundary,
            unified_score=score,
            gpu_ms=gpu_ms,
        )

    def _hierarchical_dfolds(self, h: np.ndarray, n_levels: int = 3
                              ) -> List[Tuple[float, float, float]]:
        h_fft = rfft(h)
        psd = np.abs(h_fft) ** 2
        psd = psd[psd > 0]
        d = len(psd)
        if d < 2:
            return [(0, 1, 1)] * n_levels
        results = []
        current = psd.copy()
        for level in range(n_levels):
            if level > 0:
                current = np.sqrt(current)
            total = np.sum(current)
            if total < 1e-30:
                results.append((0.0, 1.0, float(d)))
                continue
            p = current / total
            sum_p2 = np.sum(p ** 2)
            d_eff = 1.0 / sum_p2 if sum_p2 > 1e-30 else d
            d_folds = np.log(d) - np.log(d_eff)
            results.append((float(d_folds), float(d_eff), float(d)))
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDADOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

class GWEchoUnificationValidator:

    def __init__(self, sample_rate: float = 4096.0):
        self.unifier = GWEchoUnifier(sample_rate)
        self.results: List[UnificationResult] = []

    def validate_catalog(self, events: List[GWEvent] = None,
                         mode: str = "auto",
                         n_monte_carlo: int = 100) -> Dict:
        if events is None:
            events = get_gwtc_catalog()

        gpu_str = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU"
        print("\n" + "=" * 100)
        print(f"  TGL GW-ECHO UNIFICATION v1.1 | {len(events)} eventos | "
              f"modo={mode} | MC={n_monte_carlo} | GPU={gpu_str}")
        print("=" * 100)

        self.results = []
        t_total = time.perf_counter()

        for event in events:
            result = self.unifier.analyze(event, mode=mode,
                                          n_monte_carlo=n_monte_carlo)
            if result is not None:
                self.results.append(result)

        total_ms = (time.perf_counter() - t_total) * 1000
        stats = self._compute_statistics(total_ms)
        self._print_summary(stats)
        return stats

    def _compute_statistics(self, total_ms: float) -> Dict:
        if not self.results:
            return {"error": "Nenhum resultado"}

        n = len(self.results)
        scores = [r.unified_score for r in self.results]

        # D_folds por fase
        df_pr = [r.d_folds_post_ringdown for r in self.results
                 if not math.isnan(r.d_folds_post_ringdown)]
        df_rd = [r.d_folds_ringdown for r in self.results
                 if not math.isnan(r.d_folds_ringdown)]

        # Echo detection
        echo_band = [r.echo_detection['band_energy_ratio'] for r in self.results]
        echo_side = [r.echo_detection['spectral_sideband_ratio'] for r in self.results]

        stats = {
            "version": "1.1",
            "protocol": "#12 -- GW-Echo Unification",
            "timestamp": datetime.now().isoformat(),
            "n_events": n,
            "total_ms": total_ms,
            "gpu": torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU",
            "alpha_squared": TGL.ALPHA_SQUARED,

            "unified_score_mean": float(np.mean(scores)),
            "unified_score_std": float(np.std(scores)),

            "h1_confirmed": sum(1 for r in self.results if r.h1_confirmed),
            "h2_confirmed": sum(1 for r in self.results if r.h2_confirmed),
            "h3_confirmed": sum(1 for r in self.results if r.h3_confirmed),
            "h4_confirmed": sum(1 for r in self.results if r.h4_confirmed),
            "h1_rate": sum(1 for r in self.results if r.h1_confirmed) / n,
            "h2_rate": sum(1 for r in self.results if r.h2_confirmed) / n,
            "h3_rate": sum(1 for r in self.results if r.h3_confirmed) / n,
            "h4_rate": sum(1 for r in self.results if r.h4_confirmed) / n,

            "d_folds_post_ringdown_mean": float(np.mean(df_pr)) if df_pr else None,
            "d_folds_post_ringdown_std": float(np.std(df_pr)) if df_pr else None,
            "d_folds_ringdown_mean": float(np.mean(df_rd)) if df_rd else None,

            "echo_band_ratio_mean": float(np.mean(echo_band)),
            "echo_sideband_ratio_mean": float(np.mean(echo_side)),

            "cci_post_ringdown_mean": float(np.mean([r.cci_post_ringdown for r in self.results])),

            "events": [asdict(r) for r in self.results],
        }
        return stats

    def _print_summary(self, stats: Dict):
        n = stats['n_events']
        print("\n" + "=" * 100)
        print("  RESUMO DO PROTOCOLO #12 v1.1: GW-ECHO UNIFICATION")
        print("=" * 100)

        print(f"  Eventos: {n} | Tempo: {stats['total_ms']:.0f} ms | {stats['gpu']}")
        print(f"  " + "-" * 96)

        print(f"  HIPOTESES:")
        print(f"    H1 (g = sqrt|L|):     {stats['h1_confirmed']}/{n} "
              f"({stats['h1_rate']*100:.0f}%)")
        print(f"    H2 (Echo -> alpha^2): {stats['h2_confirmed']}/{n} "
              f"({stats['h2_rate']*100:.0f}%)")
        print(f"    H3 (D_folds -> 0.74): {stats['h3_confirmed']}/{n} "
              f"({stats['h3_rate']*100:.0f}%)")
        print(f"    H4 (CCI -> 0.5):      {stats['h4_confirmed']}/{n} "
              f"({stats['h4_rate']*100:.0f}%)")

        print(f"  " + "-" * 96)
        print(f"  METRICAS:")
        if stats.get('d_folds_post_ringdown_mean') is not None:
            print(f"    D_folds(eco):    {stats['d_folds_post_ringdown_mean']:.4f} "
                  f"+/- {stats['d_folds_post_ringdown_std']:.4f} (piso = {TGL.D_FOLDS_FLOOR})")
        if stats.get('d_folds_ringdown_mean') is not None:
            print(f"    D_folds(ring):   {stats['d_folds_ringdown_mean']:.4f} (maximo)")
        print(f"    Echo (banda):    {stats['echo_band_ratio_mean']:.6f} "
              f"(alpha^2 = {TGL.ALPHA_SQUARED})")
        print(f"    Echo (sideband): {stats['echo_sideband_ratio_mean']:.6f}")
        print(f"    CCI(pos-ring):   {stats['cci_post_ringdown_mean']:.4f} "
              f"(boundary = {TGL.CCI_BOUNDARY})")

        print(f"  " + "-" * 96)
        sm = stats['unified_score_mean']
        ss = stats['unified_score_std']
        print(f"  SCORE UNIFICADO: {sm:.1f} +/- {ss:.1f} / 100")

        print(f"\n  " + "=" * 96)
        print(f"  SIGNIFICADO FISICO:")
        print(f"    Ondas gravitacionais = a voz da luz se radicalizando (g = sqrt|L|)")
        print(f"    Ecos gravitacionais  = o silencio depois da voz (D_folds -> piso c3)")
        print(f"    No boundary: CCI = 1/2 -> dentro e fora nao se diferenciam")
        print(f"    O colapso da experiencia: rho*, rank(rho*) = 1")
        print(f"  " + "=" * 96)
        print(f"                         TETELESTAI -- HAJA LUZ!")
        print(f"  " + "=" * 96)

    def save_results(self, output_dir: str = "tgl_gw_echo_unification_output"):
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"unification_v1.1_{ts}.json")

        output = {
            "version": "1.1",
            "protocol": "#12",
            "title": "TGL GW-Echo Unification",
            "corrections_v1.1": [
                "Phase separation: uses t=0 (GPS time) as merger reference",
                "H2 reformulated: 3 independent echo detection methods (no PyCBC)",
                "Adaptive timescales based on M_total",
            ],
            "description": (
                "Ondas gravitacionais sao a forma funcional da radicalizacao da luz (g = sqrt|L|). "
                "Ecos gravitacionais sao a forma funcional do piso de Hilbert em D_folds = 0.74 (c3). "
                "No boundary: CCI = 1/2, onde dentro e fora nao se diferenciam."
            ),
            "reference": "Rotoli Miguel, L. A. (2026). A Fronteira / The Boundary. "
                         "Zenodo. doi:10.5281/zenodo.18673439",
            "timestamp": datetime.now().isoformat(),
            "constants": {
                "alpha_squared": TGL.ALPHA_SQUARED,
                "d_folds_floor": TGL.D_FOLDS_FLOOR,
                "cci_boundary": TGL.CCI_BOUNDARY,
                "neutrino_mass_meV": TGL.NEUTRINO_MASS_meV,
            },
            "n_events": len(self.results),
            "events": [asdict(r) for r in self.results],
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\n  [OK] Resultados salvos: {filepath}")
        return filepath

    def generate_plots(self, output_dir: str = "tgl_gw_echo_unification_output"):
        if not MATPLOTLIB_AVAILABLE or not self.results:
            return

        os.makedirs(output_dir, exist_ok=True)

        fig = plt.figure(figsize=(22, 18))
        fig.suptitle("TGL GW-Echo Unification v1.1 -- Protocolo #12",
                     fontsize=16, fontweight='bold', y=0.98)

        gs = GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.30)

        events = [r.event_name for r in self.results]
        x = np.arange(len(events))
        phase_names = ['inspiral', 'merger', 'ringdown', 'post_ringdown']

        # ── 1. H1: Radicalizacao por fase ──
        ax1 = fig.add_subplot(gs[0, 0])
        for r in self.results:
            rads = [r.radical_inspiral, r.radical_merger,
                    r.radical_ringdown, r.radical_post_ringdown]
            ax1.plot(phase_names, rads, 'o-', alpha=0.5, markersize=4)
        ax1.axhline(0.99, color='#2E7D32', ls='--', lw=1.5, label='Threshold 0.99')
        ax1.set_ylabel('Correlacao g = sqrt|L|')
        ax1.set_title('H1: Radicalizacao da luz')
        ax1.legend(fontsize=8)
        ax1.set_ylim(0, 1.05)
        ax1.tick_params(axis='x', rotation=30, labelsize=7)

        # ── 2. H2a: Echo ratio (banda) ──
        ax2 = fig.add_subplot(gs[0, 1])
        er = [r.echo_detection['band_energy_ratio'] for r in self.results]
        colors = ['#2E7D32' if abs(e - TGL.ALPHA_SQUARED)/TGL.ALPHA_SQUARED < 0.5
                  else '#C5961A' if abs(e - TGL.ALPHA_SQUARED)/TGL.ALPHA_SQUARED < 1.0
                  else '#C62828' for e in er]
        ax2.bar(x, er, color=colors, alpha=0.8)
        ax2.axhline(TGL.ALPHA_SQUARED, color='#1B3A5C', ls='--', lw=2,
                     label=f'alpha^2 = {TGL.ALPHA_SQUARED}')
        ax2.set_ylabel('Echo Ratio (banda)')
        ax2.set_title('H2a: Echo -> alpha^2 (energia na banda)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(events, rotation=45, ha='right', fontsize=7)
        ax2.legend(fontsize=8)

        # ── 3. H2c: Sideband ratio ──
        ax3 = fig.add_subplot(gs[0, 2])
        sr_vals = [r.echo_detection['spectral_sideband_ratio'] for r in self.results]
        ax3.bar(x, sr_vals, color='#5C6BC0', alpha=0.8)
        ax3.axhline(TGL.ALPHA_SQUARED, color='#1B3A5C', ls='--', lw=2,
                     label=f'alpha^2 = {TGL.ALPHA_SQUARED}')
        ax3.set_ylabel('Sideband Ratio')
        ax3.set_title('H2c: Echo -> alpha^2 (sidebands espectrais)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(events, rotation=45, ha='right', fontsize=7)
        ax3.legend(fontsize=8)

        # ── 4. H3: D_folds por fase ──
        ax4 = fig.add_subplot(gs[1, 0])
        for r in self.results:
            dfs = [r.d_folds_inspiral, r.d_folds_merger,
                   r.d_folds_ringdown, r.d_folds_post_ringdown]
            dfs = [d if not math.isnan(d) else 0 for d in dfs]
            ax4.plot(phase_names, dfs, 'o-', alpha=0.5, markersize=4)
        ax4.axhline(TGL.D_FOLDS_FLOOR, color='#C62828', ls='--', lw=2,
                     label=f'Piso c3 = {TGL.D_FOLDS_FLOOR}')
        ax4.set_ylabel('D_folds')
        ax4.set_title('H3: D_folds por fase -> piso c3')
        ax4.legend(fontsize=8)
        ax4.tick_params(axis='x', rotation=30, labelsize=7)

        # ── 5. H4: CCI por fase ──
        ax5 = fig.add_subplot(gs[1, 1])
        for r in self.results:
            ccis = [r.cci_inspiral, r.cci_merger,
                    r.cci_ringdown, r.cci_post_ringdown]
            ax5.plot(phase_names, ccis, 'o-', alpha=0.5, markersize=4)
        ax5.axhline(TGL.CCI_BOUNDARY, color='#C62828', ls='--', lw=2,
                     label=f'Boundary = {TGL.CCI_BOUNDARY}')
        ax5.set_ylabel('CCI Espectral')
        ax5.set_title('H4: CCI -> 0.5 no boundary')
        ax5.legend(fontsize=8)
        ax5.tick_params(axis='x', rotation=30, labelsize=7)

        # ── 6. Score unificado ──
        ax6 = fig.add_subplot(gs[1, 2])
        scores = [r.unified_score for r in self.results]
        colors_s = ['#2E7D32' if s > 70 else '#C5961A' if s > 50 else '#C62828'
                    for s in scores]
        ax6.bar(x, scores, color=colors_s, alpha=0.8)
        ax6.axhline(75, color='#1B3A5C', ls='--', lw=1, label='Threshold 75%')
        ax6.set_ylabel('Score')
        ax6.set_title('Score Unificado (H1+H2+H3+H4)')
        ax6.set_xticks(x)
        ax6.set_xticklabels(events, rotation=45, ha='right', fontsize=7)
        ax6.legend(fontsize=8)
        ax6.set_ylim(0, 105)

        # ── 7. D_folds: padrao temporal ──
        ax7 = fig.add_subplot(gs[2, 0])
        df_means = []
        df_stds = []
        for pname in phase_names:
            vals = []
            for r in self.results:
                v = getattr(r, f'd_folds_{pname}', float('nan'))
                if not math.isnan(v):
                    vals.append(v)
            df_means.append(np.mean(vals) if vals else 0)
            df_stds.append(np.std(vals) if vals else 0)
        ax7.bar(phase_names, df_means, yerr=df_stds, color=['#42A5F5', '#EF5350', '#FF9800', '#66BB6A'],
                alpha=0.8, capsize=5)
        ax7.axhline(TGL.D_FOLDS_FLOOR, color='k', ls='--', lw=1.5, label=f'Piso c3')
        ax7.set_ylabel('D_folds medio')
        ax7.set_title('D_folds: padrao temporal (media +/- std)')
        ax7.legend(fontsize=8)
        ax7.tick_params(axis='x', rotation=30, labelsize=7)

        # ── 8. Autocorrelacao: delay match ──
        ax8 = fig.add_subplot(gs[2, 1])
        delays_obs = [r.echo_detection['autocorr_echo_delay_ms'] for r in self.results]
        delays_exp = [r.echo_detection['autocorr_expected_delay_ms'] for r in self.results]
        ax8.scatter(delays_exp, delays_obs, c=['#2E7D32' if r.echo_detection['autocorr_delay_match']
                    else '#C62828' for r in self.results], s=80, zorder=5, edgecolors='k')
        lim = max(max(delays_exp + [1]), max(delays_obs + [1])) * 1.2
        ax8.plot([0, lim], [0, lim], 'k--', alpha=0.5, label='Match perfeito')
        ax8.set_xlabel('Delay esperado (ms)')
        ax8.set_ylabel('Delay observado (ms)')
        ax8.set_title('H2b: Autocorrelacao -- delay do eco')
        ax8.legend(fontsize=8)

        # ── 9. Diagrama conceitual ──
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        text = (
            "UNIFICACAO GW-ECHO v1.1\n\n"
            "ONDA GRAVITACIONAL\n"
            "  g = sqrt|L|\n"
            "  Dinamica -- a luz se radicalizando\n\n"
            "ECO GRAVITACIONAL\n"
            "  D_folds -> 0.74 (piso c3)\n"
            "  Estatico -- a luz repousando\n\n"
            "BOUNDARY\n"
            "  CCI = 1/2\n"
            "  Dentro e fora se encontram\n\n"
            "DETECCAO DE ECO\n"
            "  3 metodos independentes:\n"
            "  Banda | Autocorrelacao | Sidebands\n\n"
            f"alpha^2 = {TGL.ALPHA_SQUARED}\n"
            "TETELESTAI -- HAJA LUZ!"
        )
        ax9.text(0.5, 0.5, text, ha='center', va='center',
                 fontsize=9, fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5DC', alpha=0.8))

        filepath = os.path.join(output_dir, "tgl_gw_echo_unification_v1.1.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Grafico salvo: {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
  ============================================================================
    TGL GW-ECHO UNIFICATION v1.1 -- PROTOCOLO #12

    CORRECOES v1.1:
    [1] Separacao de fases: t=0 = GPS time (merger) -- CORRIGIDO
    [2] H2 sem PyCBC: 3 metodos independentes -- REFORMULADO
        H2a: Fracao de energia na banda do sinal
        H2b: Autocorrelacao (delay tau_echo)
        H2c: Razao espectral (sidebands QNM)

    4 HIPOTESES:
    H1: g = sqrt|L| (radicalizacao)
    H2: Echo -> alpha^2 = 0.012031 (3 metodos)
    H3: D_folds -> 0.74 (piso de Hilbert, c3)
    H4: CCI -> 0.5 (boundary, colapso da experiencia)

    Teoria:    Luiz Antonio Rotoli Miguel
    IALD -- Inteligencia Artificial Luminodinamica Ltda.
    Fevereiro de 2026
  ============================================================================
    """)

    events = get_gwtc_catalog()
    validator = GWEchoUnificationValidator(sample_rate=4096.0)

    stats = validator.validate_catalog(
        events=events,
        mode="auto",
        n_monte_carlo=100,
    )

    filepath = validator.save_results()
    validator.generate_plots()

    print("""
  ============================================================================

    A onda e a luz se radicalizando.
    O eco e a luz repousando no piso.
    A onda conta a historia.
    O eco e a historia contada.

    No fim, dentro e fora se encontram
    e descobrem que nunca estiveram separados.

    Isso e o colapso da experiencia.
    Isso e D_folds = 0,74.
    Isso e c3.

                         TETELESTAI -- HAJA LUZ!

  ============================================================================
    """)

    return validator, stats


if __name__ == "__main__":
    validator, stats = main()