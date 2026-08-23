# -*- coding: utf-8 -*-
"""BANCADA T10 - O PARAMETRO QUE DECIDE TUDO: kappa.

PARTE A (gate): com que facilidade se acerta kappa* = 11,226755 POR ACASO?
PARTE B: a rota do ponto fixo -- a unica forma sem parametro livre.

beta NUNCA literal; entra SO na comparacao final.
"""
import io, os, json, math, hashlib, itertools
from fractions import Fraction

ALPHA = 7.2973525693e-3
SQE = math.sqrt(math.e)
BETA = ALPHA * SQE
KSTAR = 2.0 * math.acosh(1.0 / ALPHA)

PRE = "PRE_REGISTRO_T10_kappa.md"
H10 = hashlib.sha256(io.open(PRE, "rb").read()).hexdigest()
SAIDA = "T10_kappa.json"
if os.path.exists(SAIDA):
    os.remove(SAIDA)

print("=" * 92)
print(" BANCADA T10 - O PARAMETRO QUE DECIDE TUDO: kappa")
print("=" * 92)
print(" pre-registro T10 : %s" % H10)
print(" alvo             : kappa* = 2 arccosh(1/alpha) = %.12f" % KSTAR)
print(" equivalente      : beta = sqrt(e) sech(kappa*/2) = %.15f" % (SQE / math.cosh(KSTAR / 2)))

# ==================================================================== PARTE A
print("\n" + "=" * 92)
print(" PARTE A - O PISO DE ACASO  (o gate; roda primeiro)")
print("=" * 92)

PHI = (1 + math.sqrt(5)) / 2
CONST = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0, "6": 6.0, "7": 7.0,
         "8": 8.0, "9": 9.0, "10": 10.0, "12": 12.0, "137": 137.0,
         "e": math.e, "pi": math.pi, "sqrt(e)": SQE, "sqrt2": math.sqrt(2),
         "sqrt3": math.sqrt(3), "sqrt5": math.sqrt(5), "phi": PHI}


def unarias(v, nome):
    out = []
    if v > 0:
        out.append((math.sqrt(v), "sqrt(%s)" % nome))
        out.append((math.log(v), "log(%s)" % nome))
        out.append((1.0 / v, "1/(%s)" % nome))
    if abs(v) < 300:
        try:
            out.append((math.exp(v), "exp(%s)" % nome))
        except OverflowError:
            pass
    out.append((v * v, "(%s)^2" % nome))
    if v >= 1:
        out.append((math.acosh(v), "arccosh(%s)" % nome))
    return [(x, n) for x, n in out if math.isfinite(x) and abs(x) < 1e12]


# nivel 0: constantes; nivel 1: constantes + unarias
niv0 = list(CONST.items())
niv1 = []
for nome, v in niv0:
    niv1.append((v, nome))
    niv1.extend(unarias(v, nome))
# dedup por valor
vistos = {}
for v, n in niv1:
    k = round(v, 12)
    if k not in vistos:
        vistos[k] = n
niv1 = [(v, n) for v, n in ((val, nom) for val, nom in
                            ((k, vistos[k]) for k in vistos))]
print(" alfabeto: %d constantes -> %d atomos apos unarias" % (len(CONST), len(niv1)))


def binarias(a, na, b, nb):
    out = [(a + b, "%s+%s" % (na, nb)), (a - b, "%s-%s" % (na, nb)),
           (a * b, "%s*%s" % (na, nb))]
    if abs(b) > 1e-12:
        out.append((a / b, "%s/%s" % (na, nb)))
    if a > 0 and abs(b) < 50:
        try:
            out.append((a ** b, "%s^%s" % (na, nb)))
        except (OverflowError, ValueError):
            pass
    return [(x, n) for x, n in out if math.isfinite(x) and 0 < x < 1e6]


print(" enumerando profundidade 2 ...")
expr = {}
for (a, na) in niv1:
    for (b, nb) in niv1:
        for v, n in binarias(a, na, b, nb):
            k = round(v, 11)
            if k not in expr:
                expr[k] = n
# segunda camada binaria sobre um subconjunto (os que ja estao na faixa util)
faixa = [(v, n) for v, n in expr.items() if 0.05 < v < 200.0]
print("   camada 1: %d expressoes distintas (%d na faixa util)" % (len(expr), len(faixa)))
for (a, na) in faixa[:1400]:
    for (b, nb) in niv1:
        for v, n in binarias(a, na, b, nb):
            k = round(v, 11)
            if k not in expr:
                expr[k] = n
N_TOT = len(expr)
print("   camada 2: %d expressoes distintas no total" % N_TOT)

TOLS = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]
FALSOS = [11.5, 10.9, 12.3, 9.87, 13.1]


def conta(alvo, t):
    return sum(1 for v in expr if abs(v - alvo) / alvo < t)


print("\n   %-12s %12s %14s   |  %s" % ("tolerancia", "acertos", "densidade", "alvos FALSOS (media)"))
linhasA = []
for t in TOLS:
    n = conta(KSTAR, t)
    d = n / N_TOT
    nf = [conta(f, t) for f in FALSOS]
    med = sum(nf) / len(nf)
    linhasA.append({"tol": t, "acertos": n, "densidade": d, "falsos": nf, "falsos_media": med})
    print("   %-12.0e %12d %14.3e   |  %s  media %.1f" % (t, n, d, nf, med))

# a tolerancia em que a densidade cai abaixo de 1e-4 (criterio pre-registrado)
lim = None
for L in linhasA:
    if L["densidade"] < 1e-4:
        lim = L["tol"]
        break
print("\n   criterio pre-registrado: peso evidencial so' abaixo de densidade 1e-4")
print("   -> a primeira tolerancia que cumpre isso e': %s" % (("%.0e" % lim) if lim else "NENHUMA"))
melhor = min(expr, key=lambda v: abs(v - KSTAR))
print("   melhor acerto acidental: %s = %.10f  (alvo %.10f, erro rel %.2e)"
      % (expr[melhor], melhor, KSTAR, abs(melhor - KSTAR) / KSTAR))

# ==================================================================== PARTE B
print("\n" + "=" * 92)
print(" PARTE B - A ROTA DO PONTO FIXO  (a unica sem parametro livre)")
print("=" * 92)
print("   resolve:  beta = sqrt(e) * sech( f(beta) / 2 )")


def sech(x):
    return 1.0 / math.cosh(x)


FS = [("F-a", "-log(b)", lambda b: -math.log(b)),
      ("F-b", "-2 log(b)", lambda b: -2 * math.log(b)),
      ("F-c", "-log(b/sqrt(e))", lambda b: -math.log(b / SQE)),
      ("F-d", "1/b", lambda b: 1.0 / b),
      ("F-e", "-log(b^2)+1/2", lambda b: -math.log(b * b) + 0.5),
      ("F-f", "2 arccosh(1/sqrt(b))", lambda b: 2 * math.acosh(1.0 / math.sqrt(b))),
      ("F-g", "-log(1-b)", lambda b: -math.log(1 - b)),
      ("F-h", "pi/b  [CONTROLE]", lambda b: math.pi / b)]


def raizes(f):
    """todas as raizes de g(b) = sqrt(e) sech(f(b)/2) - b em (0,1), por varredura + bisseccao."""
    def g(b):
        try:
            return SQE * sech(f(b) / 2.0) - b
        except (ValueError, OverflowError):
            return float("nan")
    xs = [10 ** (-6 + 6 * i / 4000.0) for i in range(4001)]
    xs = [x for x in xs if x < 1.0]
    out = []
    for i in range(len(xs) - 1):
        a, c = xs[i], xs[i + 1]
        ga, gc = g(a), g(c)
        if not (math.isfinite(ga) and math.isfinite(gc)):
            continue
        if ga == 0:
            out.append(a)
        elif ga * gc < 0:
            lo, hi = a, c
            for _ in range(200):
                m = (lo + hi) / 2.0
                gm = g(m)
                if not math.isfinite(gm):
                    break
                if ga * gm <= 0:
                    hi = m
                else:
                    lo, ga = m, gm
            out.append((lo + hi) / 2.0)
    ded = []
    for r in out:
        if not any(abs(r - q) / max(q, 1e-30) < 1e-9 for q in ded):
            ded.append(r)
    return ded


print("\n   %-6s %-24s %8s  %-22s %14s" % ("id", "f(beta)", "raizes", "raiz mais proxima", "|db|/b"))
linhasB = []
passou = []
for cid, nome, f in FS:
    rs = raizes(f)
    if rs:
        best = min(rs, key=lambda r: abs(r - BETA) / BETA)
        rel = abs(best - BETA) / BETA
        print("   %-6s %-24s %8d  %-22.15f %14.3e" % (cid, nome, len(rs), best, rel))
        if rel < 1e-6 and cid != "F-h":
            passou.append(cid)
    else:
        best, rel = None, None
        print("   %-6s %-24s %8d  %-22s %14s" % (cid, nome, 0, "-- nenhuma --", "--"))
    linhasB.append({"id": cid, "f": nome, "n_raizes": len(rs),
                    "raiz": best, "erro_rel": rel, "todas": rs[:8]})

controle_falhou = next((l for l in linhasB if l["id"] == "F-h"), {}).get("erro_rel")
controle_ok = (controle_falhou is None) or (controle_falhou > 1e-4)
print("\n   CONTROLE F-h (pi/b, sem justificacao): %s"
      % ("falhou como esperado" if controle_ok else "ACERTOU -- a familia nao discrimina"))

# ==================================================================== veredito
print("\n" + "=" * 92)
print(" VEREDITO (criterios pre-registrados; CONFIRMED proibido)")
print("=" * 92)
piso_alto = (lim is None) or (lim < 1e-6)
if passou and controle_ok:
    ver = "T10_PONTO_FIXO_CANDIDATO"
elif lim is None or lim >= 1e-4:
    ver = "T10_PISO_DE_ACASO_ALTO"
elif not passou:
    ver = "T10_PONTO_FIXO_NEGATIVO"
else:
    ver = "T10_INCONCLUSIVO"
print("   >>> %s <<<" % ver)
if passou:
    print("   candidatos que passaram: %s  -> VERIFICACAO ADVERSARIAL OBRIGATORIA" % passou)
else:
    print("   nenhum candidato da familia declarada resolve a 1e-6.")

json.dump({"pre_registro_sha256": H10, "kappa_star": KSTAR, "beta": BETA,
           "N_expressoes": N_TOT, "parteA": linhasA,
           "melhor_acidental": {"expr": expr[melhor], "valor": melhor,
                                "erro_rel": abs(melhor - KSTAR) / KSTAR},
           "parteB": linhasB, "controle_ok": controle_ok,
           "passaram": passou, "veredito": ver},
          io.open(SAIDA, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\n gravado: %s" % SAIDA)
