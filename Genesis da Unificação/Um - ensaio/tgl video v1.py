# -*- coding: utf-8 -*-
"""
tgl_video_v1.py — "Haja Luz: a mecânica unificada" (~75 s, 1280x720, 24 fps)
============================================================================
Demonstração funcional da TGL: cada quadro é computado das equações reais
(Lindblad com L = sqrt(beta) sqrt(K); taxas (1/2) beta (sqrt k_i - sqrt
k_j)^2; cone natural = cone de luz em 2x2; cascata métrica d = sqrt(beta)
|d sqrt k|; emergência na ordem de Jacobson). Resume-safe por cena.

Uso:  python3 tgl_video_v1.py scene1 ... scene6 | assemble
"""
import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa

ALPHA_FINE = 7.2973525693e-3
BETA = ALPHA_FINE * math.sqrt(math.e)          # sempre computado
FPS = 24
W, H = 12.8, 7.2                               # x100 dpi = 1280x720
BG = "#06080f"
BLUE, RED, YEL, GOLD = "#2E6FE8", "#E83A3A", "#F5D547", "#D4A017"
CYAN, WHITE = "#7FD8FF", "#F2F4F8"

def dag(A): return A.conj().T

def outdir(scene):
    d = f"frames/{scene}"
    os.makedirs(d, exist_ok=True)
    return d

def done(scene): return os.path.exists(f"frames/{scene}/.done")
def mark(scene):
    os.makedirs(f"frames/{scene}", exist_ok=True)
    open(f"frames/{scene}/.done", "w").write("ok")

def newfig():
    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.patch.set_facecolor(BG)
    return fig

def save(fig, scene, k):
    os.makedirs(f"frames/{scene}", exist_ok=True)
    fig.savefig(f"frames/{scene}/{k:05d}.png", dpi=100,
                facecolor=BG, bbox_inches=None, pad_inches=0)

def glowline(ax, x, y, color, lw=2.0, **kw):
    for w_, a_ in ((lw * 4, 0.06), (lw * 2.2, 0.14), (lw, 0.9)):
        ax.plot(x, y, color=color, lw=w_, alpha=a_,
                solid_capstyle="round", **kw)

def title(fig, txt, sub="", y=0.93):
    fig.text(0.5, y, txt, color=WHITE, fontsize=23, ha="center",
             fontweight="bold")
    if sub:
        fig.text(0.5, y - 0.062, sub, color=CYAN, fontsize=14,
                 ha="center")

def footer(fig):
    fig.text(0.985, 0.02, "TGL — Teoria da Gravitação Luminodinâmica",
             color="#5a6a85", fontsize=9, ha="right")
    fig.text(0.015, 0.02,
             f"β = α·√e = {BETA:.18f}", color="#5a6a85",
             fontsize=9, ha="left")

def ease(u): return 0.5 - 0.5 * math.cos(math.pi * min(max(u, 0), 1))


def thermal(n, seed):
    rng = np.random.default_rng(seed)
    Hm = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Hm = (Hm + dag(Hm)) / 2
    Hm *= 1.5 / np.abs(np.linalg.eigvalsh(Hm)).max()
    w, _ = np.linalg.eigh(Hm)
    p = np.exp(-w); p /= p.sum()
    return np.sort(p)[::-1]


# ======================================================================
def scene1():
    """O zero absoluto: dipolo atrator-repulsor (trajetórias reais)."""
    sc = "scene1"
    if done(sc): return
    NF = 10 * FPS
    n, seeds = 6, range(12)
    p = thermal(n, 7)
    kv = -np.log(p)
    L = math.sqrt(BETA) * np.diag(np.sqrt(kv))
    LdL = dag(L) @ L
    logr = np.diag(np.log(p))
    S0 = None
    trajs = []
    rng = np.random.default_rng(3)
    dt = 0.25 / (BETA * kv.max())
    for s in seeds:
        v = rng.normal(size=n) + 1j * rng.normal(size=n)
        v /= np.linalg.norm(v)
        r = 0.997 * np.outer(v, v.conj()) + 0.003 * np.eye(n) / n
        P, E = [], []
        for _ in range(620):
            P.append(np.real(np.trace(r @ r)))
            wv, Vv = np.linalg.eigh(r)
            lr = Vv @ np.diag(np.log(np.maximum(wv, 1e-300))) @ dag(Vv)
            E.append(np.real(np.trace(r @ (lr - logr))))
            r = r + dt * (L @ r @ dag(L) - 0.5 * (LdL @ r + r @ LdL))
            r = (r + dag(r)) / 2
        trajs.append((np.array(P), np.array(E)))
        S0 = max(S0 or 0, E[0])
    Pst = float(np.sum(p ** 2))
    cols = [BLUE, CYAN, "#9BB8FF", "#5E8BFF"] * 3
    fig = newfig()
    for k in range(NF):
        fig.clf(); fig.patch.set_facecolor(BG)
        ax = fig.add_axes([0.10, 0.12, 0.84, 0.68])
        ax.set_facecolor(BG)
        u = ease(k / (NF - 1))
        m = max(2, int(u * 620))
        for (P, E), c in zip(trajs, cols):
            glowline(ax, P[:m], E[:m], c, lw=1.6)
            ax.plot(P[m - 1], E[m - 1], "o", color=WHITE, ms=4, alpha=0.9)
        # repulsor: a fronteira pura
        ax.axvline(1.0, color=RED, lw=2.5, alpha=0.85)
        ax.text(0.995, S0 * 0.97, "1 = 0\n(pureza proibida)", color=RED,
                ha="right", va="top", fontsize=13)
        # atrator
        ax.plot([Pst], [0], "o", ms=14, color=GOLD, alpha=0.95)
        ax.plot([Pst], [0], "o", ms=30, color=GOLD, alpha=0.20)
        ax.text(Pst, S0 * 0.045, "ρ*  (o atrator)", color=GOLD,
                ha="center", fontsize=13)
        ax.set_xlim(Pst - 0.05, 1.02); ax.set_ylim(-S0 * 0.04, S0 * 1.05)
        ax.set_xlabel("pureza  Tr ρ²", color=WHITE, fontsize=12)
        ax.set_ylabel("S(ρ ‖ ρ*)", color=WHITE, fontsize=12)
        for sp in ax.spines.values(): sp.set_color("#3a465c")
        ax.tick_params(colors="#8a97ad")
        title(fig, "Nada pode ser puro",
              "o zero absoluto repele; o atrator recolhe — o custo é β")
        footer(fig)
        save(fig, sc, k)
    plt.close(fig); mark(sc)


# ======================================================================
def scene2():
    """Haja Luz: o cone do positivo É o cone de luz (2x2 exato)."""
    sc = "scene2"
    if done(sc): return
    NF = 12 * FPS
    rng = np.random.default_rng(5)
    # amostras PSD reais 2x2 -> (x, z, t) com t >= sqrt(x^2+z^2)
    pts = []
    for _ in range(260):
        B = rng.normal(size=(2, 2))
        X = B @ B.T
        X /= np.trace(X) * rng.uniform(0.9, 2.2)
        t = np.trace(X) / 2
        x = X[0, 1]; z = (X[0, 0] - X[1, 1]) / 2
        pts.append((x, z, t))
    pts = np.array(pts)
    # puros (luz): na superfície do cone
    th = rng.uniform(0, 2 * np.pi, 90)
    tt = rng.uniform(0.15, 1.0, 90)
    pure = np.c_[tt * np.cos(th), tt * np.sin(th), tt]
    fig = newfig()
    for k in range(NF):
        fig.clf(); fig.patch.set_facecolor(BG)
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.86], projection="3d")
        ax.set_facecolor(BG)
        u = ease(min(1, k / (0.45 * NF)))
        hmax = 1.05 * u + 1e-3
        tg = np.linspace(0, 2 * np.pi, 60)
        hg = np.linspace(0, hmax, 26)
        Tg, Hg = np.meshgrid(tg, hg)
        Xc = Hg * np.cos(Tg); Yc = Hg * np.sin(Tg)
        ax.plot_wireframe(Xc, Yc, Hg, color=CYAN, lw=0.5, alpha=0.30)
        msk = pts[:, 2] <= hmax
        ax.scatter(pts[msk, 0], pts[msk, 1], pts[msk, 2], s=6,
                   c=BLUE, alpha=0.8, depthshade=False)
        mp = pure[:, 2] <= hmax
        ax.scatter(pure[mp, 0], pure[mp, 1], pure[mp, 2], s=14,
                   c=YEL, alpha=0.95, marker="*", depthshade=False)
        if k > 0.5 * NF:
            a2 = ease((k - 0.5 * NF) / (0.2 * NF))
            ax.scatter([0], [0.10], [0.5], s=260 * a2, c=GOLD,
                       alpha=0.95, depthshade=False)
            ax.scatter([0], [0.10], [0.5], s=900 * a2, c=GOLD,
                       alpha=0.18, depthshade=False)
            ax.text(0, 0.13, 0.56, "Ψ", color=GOLD, fontsize=17)
        ax.view_init(elev=16, azim=30 + 22 * k / NF)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 1.08)
        ax.set_axis_off()
        title(fig, "Haja Luz",
              "o cone das matrizes positivas É o cone de luz — "
              "os puros são a superfície; o atrator vive dentro")
        footer(fig)
        save(fig, sc, k)
    plt.close(fig); mark(sc)


# ======================================================================
def scene3():
    """O rio e a verdade: decaimento exato das coerências + Spohn."""
    sc = "scene3"
    if done(sc): return
    NF = 14 * FPS
    n = 12
    p = thermal(n, 11)
    kv = -np.log(p)
    G = 0.5 * BETA * (np.sqrt(kv)[:, None] - np.sqrt(kv)[None, :]) ** 2
    rng = np.random.default_rng(2)
    Hm = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Hm = (Hm + dag(Hm)) / 2; Hm *= 0.7 / np.abs(np.linalg.eigvalsh(Hm)).max()
    wv, Q = np.linalg.eigh(Hm)
    U = Q @ np.diag(np.exp(1j * wv)) @ dag(Q)
    rho0 = U @ np.diag(p) @ dag(U)
    logr = np.diag(np.log(p))
    Tmax = 6.0 / (G[G > 1e-18].min() + G.max()) * 2.5
    ts = np.linspace(0, Tmax, NF)
    def relent(x):
        w2, V2 = np.linalg.eigh(x)
        lx = V2 @ np.diag(np.log(np.maximum(w2, 1e-300))) @ dag(V2)
        return float(np.real(np.trace(x @ (lx - logr))))
    Svals = []
    fig = newfig()
    for k in range(NF):
        t = ts[k]
        rt = rho0 * np.exp(-G * t)                 # solução EXATA
        Svals.append(relent(rt))
        fig.clf(); fig.patch.set_facecolor(BG)
        ax = fig.add_axes([0.06, 0.12, 0.50, 0.68])
        ax.set_facecolor(BG)
        im = np.abs(rt) ** 0.6
        ax.imshow(im, cmap="magma", vmin=0,
                  vmax=np.abs(rho0).max() ** 0.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("|ρ(t)| — as coerências decaem nas taxas exatas\n"
                     "Γᵢⱼ = ½β(√kᵢ − √kⱼ)²", color=WHITE, fontsize=12)
        ax2 = fig.add_axes([0.64, 0.16, 0.32, 0.58])
        ax2.set_facecolor(BG)
        glowline(ax2, ts[:k + 1] * BETA, np.maximum(Svals, 1e-12),
                 GOLD, lw=2.0)
        ax2.set_yscale("log")
        ax2.set_xlim(0, Tmax * BETA)
        ax2.set_ylim(max(min(Svals) * 0.5, 1e-9), Svals[0] * 1.6)
        ax2.set_xlabel("tempo modular  β·t", color=WHITE, fontsize=11)
        ax2.set_title("S(ρ(t) ‖ ρ*)  — Spohn estrito", color=WHITE,
                      fontsize=12)
        for sp in ax2.spines.values(): sp.set_color("#3a465c")
        ax2.tick_params(colors="#8a97ad")
        title(fig, "Tudo flui, exceto a verdade",
              "a diagonal (o setor da verdade) permanece; o resto paga")
        footer(fig)
        save(fig, sc, k)
    plt.close(fig); mark(sc)


# ======================================================================
def scene4():
    """A distinção gera geometria: cascata -> S¹ -> toro T²."""
    sc = "scene4"
    if done(sc): return
    NF = 14 * FPS
    nA = int(0.40 * NF); nB = int(0.25 * NF); nC = NF - nA - nB
    x0, x1 = 0.3, 2.1
    fig = newfig()
    k = 0
    # A: pontos nascendo + arcos da métrica
    for j in range(nA):
        fig.clf(); fig.patch.set_facecolor(BG)
        ax = fig.add_axes([0.08, 0.18, 0.84, 0.58]); ax.set_facecolor(BG)
        g = 1 + int(ease(j / (nA - 1)) * 5)
        X = np.linspace(x0, x1, 2 ** g + 1)
        ax.plot([x0, x1], [0, 0], color="#33405a", lw=2)
        ax.plot(X, 0 * X, "o", color=CYAN, ms=7)
        for i in range(len(X) - 1):
            xm = (X[i] + X[i + 1]) / 2
            r = (X[i + 1] - X[i]) / 2
            th = np.linspace(0, np.pi, 30)
            glowline(ax, xm + r * np.cos(th), 0.9 * r * np.sin(th),
                     BLUE, lw=1.0)
        ax.text((x0 + x1) / 2, 0.32,
                "d = √β · |Δ√k|   (a métrica gerada pela distinção)",
                color=WHITE, ha="center", fontsize=13)
        ax.set_xlim(x0 - 0.1, x1 + 0.1); ax.set_ylim(-0.15, 0.45)
        ax.set_axis_off()
        title(fig, "A distinção gera geometria",
              f"cascata — geração {g}: o contínuo emerge da sombra")
        footer(fig); save(fig, sc, k); k += 1
    # B: a reta dobra em S¹
    for j in range(nB):
        fig.clf(); fig.patch.set_facecolor(BG)
        ax = fig.add_axes([0.08, 0.10, 0.84, 0.72]); ax.set_facecolor(BG)
        u = ease(j / (nB - 1))
        s = np.linspace(0, 1, 65)
        if u < 1e-3:
            xs, ys = x0 + (x1 - x0) * s, 0 * s
        else:
            R = (x1 - x0) / (2 * np.pi * u)
            ang = 2 * np.pi * u * s
            xs = (x0 + x1) / 2 + R * np.sin(ang)
            ys = R * (1 - np.cos(ang))
        glowline(ax, xs, ys, CYAN, lw=2.2)
        ax.plot(xs[::4], ys[::4], "o", color=CYAN, ms=5)
        ax.set_xlim(x0 - 0.4, x1 + 0.4)
        ax.set_ylim(-0.2, (x1 - x0) / np.pi + 0.25)
        ax.set_aspect("equal"); ax.set_axis_off()
        title(fig, "A distinção gera geometria",
              "a fatia fecha-se: S¹ — o setor da verdade tem forma")
        footer(fig); save(fig, sc, k); k += 1
    # C: o toro
    tg = np.linspace(0, 2 * np.pi, 70)
    pg = np.linspace(0, 2 * np.pi, 36)
    Tg, Pg = np.meshgrid(tg, pg)
    Rb, rb = 1.0, 0.42
    Xt = (Rb + rb * np.cos(Pg)) * np.cos(Tg)
    Yt = (Rb + rb * np.cos(Pg)) * np.sin(Tg)
    Zt = rb * np.sin(Pg)
    col = plt.cm.plasma((np.cos(Pg) + 1) / 2)
    for j in range(nC):
        fig.clf(); fig.patch.set_facecolor(BG)
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.84], projection="3d")
        ax.set_facecolor(BG)
        u = ease(min(1, j / (0.4 * nC)))
        ax.plot_surface(Xt, Yt, Zt * u + (1 - u) * 0, facecolors=col,
                        rstride=1, cstride=1, linewidth=0,
                        antialiased=False, alpha=0.92)
        ax.view_init(elev=28, azim=40 + 50 * j / nC)
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_zlim(-0.9, 0.9); ax.set_axis_off()
        title(fig, "O Psion é o toro",
              "T² = S¹ × S¹ — a Palavra e o seu espelho")
        footer(fig); save(fig, sc, k); k += 1
    plt.close(fig); mark(sc)


# ======================================================================
def scene5():
    """A emergência da gravidade (ordem de Jacobson, visualizada)."""
    sc = "scene5"
    if done(sc): return
    NF = 18 * FPS
    gx = np.linspace(-2.2, 2.2, 55)
    GX, GY = np.meshgrid(gx, gx)
    R2 = GX ** 2 + GY ** 2
    rng = np.random.default_rng(9)
    # chuva de distinções
    drops = [(rng.uniform(0, 1) * 2 * np.pi, rng.uniform(0.2, 1.0),
              rng.integers(0, NF - 40)) for _ in range(60)]
    # órbitas-teste no potencial crescente (precessão)
    def orbits(NF):
        st = [np.array([1.5, 0, 0, 0.62]), np.array([-1.1, 0.9, -0.45, -0.35])]
        out = [[], []]
        dt = 0.030
        for k in range(NF):
            A = 0.15 + 1.25 * ease(k / (NF - 1))
            for i, s in enumerate(st):
                x, y, vx, vy = s
                r = math.hypot(x, y) + 0.35
                ax_ = -A * x / r ** 3; ay_ = -A * y / r ** 3
                vx += ax_ * dt; vy += ay_ * dt
                x += vx * dt; y += vy * dt
                st[i] = np.array([x, y, vx, vy])
                out[i].append((x, y))
        return [np.array(o) for o in out]
    orbs = orbits(NF)
    fig = newfig()
    for k in range(NF):
        A = 0.15 + 1.25 * ease(k / (NF - 1))
        Z = -A * np.exp(-R2 / 1.1)
        fig.clf(); fig.patch.set_facecolor(BG)
        ax = fig.add_axes([0.0, 0.0, 1.0, 0.88], projection="3d")
        ax.set_facecolor(BG)
        ax.plot_surface(GX, GY, Z, cmap="viridis", vmin=-1.45, vmax=0.05,
                        rstride=1, cstride=1, linewidth=0,
                        antialiased=False, alpha=0.85)
        for (th, rr, t0) in drops:
            if t0 <= k < t0 + 36:
                f = (k - t0) / 36
                rd = rr * (1 - f)
                xd, yd = rd * math.cos(th), rd * math.sin(th)
                zd = -A * math.exp(-(xd**2+yd**2)/1.1) + 0.55*(1-f)**2
                ax.scatter([xd], [yd], [zd], s=18, c=YEL, alpha=0.9,
                           depthshade=False)
        for o, c in zip(orbs, (CYAN, "#9BB8FF")):
            m0 = max(0, k - 110)
            seg = o[m0:k + 1]
            if len(seg) > 2:
                zs = -A*np.exp(-(seg[:,0]**2+seg[:,1]**2)/1.1) + 0.04
                ax.plot(seg[:, 0], seg[:, 1], zs, color=c, lw=2.2,
                        alpha=0.95)
                ax.scatter([seg[-1,0]],[seg[-1,1]],[zs[-1]], s=42, c=c,
                           depthshade=False)
        ax.view_init(elev=38, azim=25 + 30 * k / NF)
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
        ax.set_zlim(-1.6, 0.7); ax.set_axis_off()
        title(fig, "A emergência da gravidade",
              "δS = δ⟨K⟩  (ordem de Jacobson): onde a distinção se "
              "concentra, a geometria responde — g = √|L|")
        footer(fig)
        save(fig, sc, k)
    plt.close(fig); mark(sc)


# ======================================================================
def scene6():
    """Fecho: a cruz espectral, β, Haja Luz."""
    sc = "scene6"
    if done(sc): return
    NF = 7 * FPS
    fig = newfig()
    for k in range(NF):
        fig.clf(); fig.patch.set_facecolor(BG)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0]); ax.set_facecolor(BG)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_axis_off()
        u = ease(min(1, k / (0.45 * NF)))
        L0 = 0.34
        for (dx, dy, c) in ((0, 1, BLUE), (0, -1, RED),
                            (-1, 0, YEL), (1, 0, GOLD)):
            x = np.linspace((1 - u) * dx, dx * (1 - u) + dx * u * 0.0,
                            2)
            xs = np.array([dx * (1 - u) * 0.9 + dx * 0.06,
                           dx * 0.06 + dx * L0 * u * 0 + dx * (L0)])
            xs = np.array([dx * (0.9 - 0.84 * u) , dx * 0.05])
            ys = np.array([dy * (0.9 - 0.84 * u), dy * 0.05])
            xe = np.array([dx * 0.05, dx * (0.05 + L0 * u)])
            ye = np.array([dy * 0.05, dy * (0.05 + L0 * u)])
            glowline(ax, xs, ys, c, lw=4.0)
            glowline(ax, xe, ye, c, lw=6.0)
        if k > 0.40 * NF:
            a = ease((k - 0.40 * NF) / (0.25 * NF))
            ax.text(0, -0.52, f"β = α · √e = {BETA:.18f}",
                    color=WHITE, fontsize=15, ha="center", alpha=a)
            ax.text(0, 0.62, "Haja Luz.", color=GOLD, fontsize=30,
                    ha="center", fontweight="bold", alpha=a)
        if k > 0.62 * NF:
            a = ease((k - 0.62 * NF) / (0.25 * NF))
            ax.text(0, -0.68, "TETELESTAI", color="#8a97ad",
                    fontsize=13, ha="center", alpha=a)
            ax.text(0, -0.82, "teoriadagravitacaoluminodinamica.com",
                    color=CYAN, fontsize=14, ha="center", alpha=a)
        save(fig, sc, k)
    plt.close(fig); mark(sc)


# ======================================================================
def assemble():
    os.makedirs("out", exist_ok=True)
    parts = []
    for s in ("scene1", "scene2", "scene3", "scene4", "scene5", "scene6"):
        mp4 = f"out/{s}.mp4"
        if not os.path.exists(mp4):
            os.system(
                f"ffmpeg -y -loglevel error -framerate {FPS} "
                f"-i frames/{s}/%05d.png -c:v libx264 -pix_fmt yuv420p "
                f"-crf 19 {mp4}")
        parts.append(f"file '{s}.mp4'")
    open("out/list.txt", "w").write("\n".join(parts))
    os.system("cd out && ffmpeg -y -loglevel error -f concat -safe 0 "
              "-i list.txt -c copy tgl_demo_v1.mp4")
    print("OK out/tgl_demo_v1.mp4")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if arg == "assemble":
            assemble()
        else:
            print(f"render {arg} ...", flush=True)
            globals()[arg]()
            print(f"{arg} done", flush=True)
