# -*- coding: utf-8 -*-
"""
BANCADA T04 — OS FATOS DO SPARC, ANTES DE QUALQUER MODELO
                                                    (21/08/2026)

Este artefato NAO testa teoria nenhuma. Ele estabelece o DENOMINADOR: o que os
175 arquivos reais dizem, e qual e o desempenho da hipotese NULA — barions
sozinhos, sem materia escura alguma. Todo modelo de materia escura tem de
bater esse numero; quem nao bater esta morto antes de comecar.

Faz-se ANTES do protocolo do teste, de proposito: os fatos do dado nao podem
depender do modelo que se quer testar.

 F1 inventario: quantas galaxias, quantos pontos, faixas
 F2 a NULA (barions sozinhos, Upsilon_disk = 0,5 e Upsilon_bul = 0,7 — a
    pratica padrao de Lelli+2016): chi2_nu por galaxia
 F3 a NULA com Upsilon LIVRE por galaxia (1 parametro): quanto melhora
 F4 a RELACAO DE TULLY-FISHER BARIONICA medida no proprio dado: expoente e
    normalizacao — a relacao que QUALQUER teoria de materia escura tem de
    reproduzir
 F5 a aceleracao caracteristica g_dagger medida (onde g_obs desvia de g_bar)
    — e a comparacao com as combinacoes candidatas de beta
 C1 (CONTROLE) embaralhar Vobs entre galaxias TEM que piorar tudo

REGUA: beta jamais literal; nada aqui e' veredito sobre a TGL; estes sao
fatos do dado. CONFIRMED proibido.
"""
import numpy as np, math, json, io, os, glob

ALPHA = 7.2973525693e-3
BETA = ALPHA * math.sqrt(math.e)
SQB = math.sqrt(BETA)
c_kms = 299792.458
Mpc = 3.0856775814913673e22
G = 4.300917270e-6          # kpc (km/s)^2 / Msun

SPARC = r"C:/IALD/observaveis_tgl/tgl_cache/sparc"
SAIDA = "T04_sparc_fatos.json"
if os.path.exists(SAIDA): os.remove(SAIDA)

def ler(p):
    D = None
    R, V, E, Vg, Vd, Vb = [], [], [], [], [], []
    for ln in io.open(p, encoding="utf-8", errors="replace"):
        s = ln.strip()
        if s.startswith("#"):
            if "Distance" in s:
                try: D = float(s.split("=")[1].split()[0])
                except Exception: pass
            continue
        if not s: continue
        t = s.split()
        if len(t) < 6: continue
        try:
            R.append(float(t[0])); V.append(float(t[1])); E.append(float(t[2]))
            Vg.append(float(t[3])); Vd.append(float(t[4])); Vb.append(float(t[5]))
        except ValueError:
            continue
    return D, tuple(np.asarray(x, float) for x in (R, V, E, Vg, Vd, Vb))

arqs = sorted(glob.glob(os.path.join(SPARC, "*_rotmod.dat")))
print("=" * 84); print(" BANCADA T04 — OS FATOS DO SPARC (antes de qualquer modelo)"); print("=" * 84)
print(" beta = %.15f (derivado) | sqrt(beta) = %.6f" % (BETA, SQB))
print(" arquivos: %d" % len(arqs))

gals = []
for p in arqs:
    nome = os.path.basename(p).replace("_rotmod.dat", "")
    D, (R, V, E, Vg, Vd, Vb) = ler(p)
    if len(R) < 3: continue
    ok = (E > 0) & (V > 0) & (R > 0)
    if ok.sum() < 3: continue
    gals.append(dict(nome=nome, D=D, R=R[ok], V=V[ok], E=E[ok],
                     Vg=Vg[ok], Vd=Vd[ok], Vb=Vb[ok]))

npts = sum(len(g["R"]) for g in gals)
print("\n F1 — INVENTARIO")
print("   galaxias utilizaveis : %d" % len(gals))
print("   pontos totais        : %d" % npts)
print("   pontos por galaxia   : min %d | mediana %d | max %d"
      % (min(len(g["R"]) for g in gals),
         int(np.median([len(g["R"]) for g in gals])),
         max(len(g["R"]) for g in gals)))
vmax = np.array([g["V"].max() for g in gals])
print("   V_max [km/s]         : min %.1f | mediana %.1f | max %.1f"
      % (vmax.min(), np.median(vmax), vmax.max()))
rmax = np.array([g["R"].max() for g in gals])
print("   R_max [kpc]          : min %.2f | mediana %.2f | max %.2f"
      % (rmax.min(), np.median(rmax), rmax.max()))

def vbar2(g, yd, yb):
    return g["Vg"] ** 2 + yd * g["Vd"] ** 2 + yb * g["Vb"] ** 2

def chi2nu(g, vmod2, k):
    vm = np.sqrt(np.maximum(vmod2, 0.0))
    dof = max(len(g["R"]) - k, 1)
    return float(np.sum(((g["V"] - vm) / g["E"]) ** 2) / dof)

print("\n F2 — A HIPOTESE NULA: barions sozinhos (Upsilon fixo, pratica Lelli+2016)")
c2_fix = np.array([chi2nu(g, vbar2(g, 0.5, 0.7), 0) for g in gals])
print("   chi2_nu barions-so, Upsilon FIXO : mediana %.2f | media %.2f | frac<1,2: %.1f%%"
      % (np.median(c2_fix), c2_fix.mean(), 100.0 * np.mean(c2_fix < 1.2)))

print("\n F3 — A NULA com Upsilon LIVRE por galaxia (1 parametro)")
c2_liv, ys = [], []
grid = np.logspace(np.log10(0.05), np.log10(5.0), 60)
for g in gals:
    best, ybest = 1e30, None
    for y in grid:
        v = chi2nu(g, vbar2(g, y, 1.4 * y), 1)
        if v < best: best, ybest = v, y
    c2_liv.append(best); ys.append(ybest)
c2_liv = np.array(c2_liv); ys = np.array(ys)
print("   chi2_nu barions-so, Upsilon LIVRE: mediana %.2f | frac<1,2: %.1f%%"
      % (np.median(c2_liv), 100.0 * np.mean(c2_liv < 1.2)))
print("   Upsilon_disk ajustado            : mediana %.3f | 16-84%%: %.3f-%.3f"
      % (np.median(ys), np.percentile(ys, 16), np.percentile(ys, 84)))
print("   => a NULA e' o que qualquer modelo de MATERIA ESCURA precisa BATER.")

print("\n F4 — TULLY-FISHER BARIONICA medida no proprio dado")
Vf, Mb = [], []
for g in gals:
    n = max(1, len(g["R"]) // 4)
    vflat = float(np.median(g["V"][-n:]))
    Rout = float(g["R"][-1])
    vb2 = vbar2(g, 0.5, 0.7)[-1]
    Mbar = vb2 * Rout / G
    if vflat > 20 and Mbar > 0:
        Vf.append(vflat); Mb.append(Mbar)
Vf = np.array(Vf); Mb = np.array(Mb)
A = np.vstack([np.log10(Vf), np.ones_like(Vf)]).T
sl, ic = np.linalg.lstsq(A, np.log10(Mb), rcond=None)[0]
print("   N galaxias com V_flat > 20 km/s : %d" % len(Vf))
print("   log10(M_bar) = %.3f*log10(V_flat) + %.3f" % (sl, ic))
print("   => expoente medido %.3f (a BTFR canonica da ~4; MOND preve 4 EXATO)" % sl)
resid = np.log10(Mb) - (sl * np.log10(Vf) + ic)
print("   dispersao residual: %.3f dex" % np.std(resid))

print("\n F5 — A ACELERACAO CARACTERISTICA (relacao aceleracao radial)")
gobs, gbar = [], []
for g in gals:
    kpc_m = 3.0856775814913673e19
    go = (g["V"] * 1e3) ** 2 / (g["R"] * kpc_m)
    gb = (np.sqrt(np.maximum(vbar2(g, 0.5, 0.7), 0)) * 1e3) ** 2 / (g["R"] * kpc_m)
    m = (gb > 0) & (go > 0)
    gobs.append(go[m]); gbar.append(gb[m])
gobs = np.concatenate(gobs); gbar = np.concatenate(gbar)
# g_dagger da RAR: g_obs = g_bar / (1 - exp(-sqrt(g_bar/g_dag)))
def rar_chi(gd):
    pred = gbar / (1.0 - np.exp(-np.sqrt(gbar / gd)))
    return float(np.mean((np.log10(pred) - np.log10(gobs)) ** 2))
gg = np.logspace(-11.5, -9.0, 400)
gdag = gg[int(np.argmin([rar_chi(x) for x in gg]))]
print("   pontos usados: %d" % len(gobs))
print("   g_dagger AJUSTADO ao dado  : %.4e m/s^2" % gdag)
print("   [KNOWN] McGaugh+2016       : 1.20e-10 m/s^2")
H0 = 70.0 * 1e3 / Mpc
cands = {"alpha*c*H0": ALPHA * 299792458.0 * H0,
         "sqrt(beta)*c*H0": SQB * 299792458.0 * H0,
         "beta*c*H0": BETA * 299792458.0 * H0,
         "c*H0/(2pi)": 299792458.0 * H0 / (2 * math.pi),
         "c*H0": 299792458.0 * H0}
print("   candidatas (H0=70):")
for k, v in sorted(cands.items(), key=lambda t: abs(math.log10(t[1] / gdag))):
    print("      %-18s = %.4e   razao g_dag/cand = %7.3f" % (k, v, gdag / v))

print("\n C1 — CONTROLE: embaralhar Vobs entre galaxias tem que PIORAR")
rng = np.random.default_rng(20260821)
idx = rng.permutation(len(gals))
c2_emb = []
for i, g in enumerate(gals):
    h = gals[idx[i]]
    n = min(len(g["R"]), len(h["R"]))
    if n < 3: continue
    vm = np.sqrt(np.maximum(vbar2(g, 0.5, 0.7)[:n], 0))
    c2_emb.append(float(np.sum(((h["V"][:n] - vm) / h["E"][:n]) ** 2) / n))
c2_emb = np.array(c2_emb)
print("   chi2_nu embaralhado: mediana %.2f  (contra %.2f real)"
      % (np.median(c2_emb), np.median(c2_fix)))
disc = np.median(c2_emb) > np.median(c2_fix)
print("   controle %s" % ("DISCRIMINA (embaralhado e' pior)" if disc else "*FALHOU*"))

out = {
    "o_que_e": "fatos do conjunto SPARC, medidos ANTES de qualquer modelo; nao e veredito sobre a TGL",
    "n_galaxias": len(gals), "n_pontos": int(npts),
    "nula_barions_upsilon_fixo": {"chi2nu_mediano": float(np.median(c2_fix)),
                                  "frac_abaixo_1p2": float(np.mean(c2_fix < 1.2))},
    "nula_barions_upsilon_livre": {"chi2nu_mediano": float(np.median(c2_liv)),
                                   "frac_abaixo_1p2": float(np.mean(c2_liv < 1.2)),
                                   "upsilon_mediano": float(np.median(ys))},
    "btfr": {"expoente": float(sl), "intercepto": float(ic),
             "dispersao_dex": float(np.std(resid)), "n": int(len(Vf))},
    "rar": {"g_dagger_ajustado": float(gdag), "g_dagger_KNOWN_McGaugh2016": 1.20e-10,
            "candidatas_H0_70": {k: float(v) for k, v in cands.items()}},
    "controle_embaralhado": {"chi2nu_mediano": float(np.median(c2_emb)), "discrimina": bool(disc)},
    "beta_derivado": BETA, "alpha_input": ALPHA, "data": "2026-08-21",
    "ressalva": ("Upsilon fixo = 0,5/0,7 e a pratica padrao de Lelli+2016 [KNOWN]. A NULA e o "
                 "PISO que qualquer modelo de materia escura precisa bater. Nada aqui aprova ou "
                 "reprova a TGL: sao fatos do dado. CONFIRMED proibido."),
}
json.dump(out, io.open(SAIDA, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n gravado: %s" % SAIDA)
