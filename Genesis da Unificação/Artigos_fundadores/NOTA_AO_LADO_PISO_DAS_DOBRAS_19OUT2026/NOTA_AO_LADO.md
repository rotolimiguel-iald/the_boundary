# Nota ao lado — o piso das dobras / A note beside — the floor of the folds (19/10/2026)

**Ao lado de / beside:** `A_fronteira_v5.tex` / `A_fronteira_v5.pdf` (PT) e/and `The_boundary_v5_en.tex` / `the_boundary_v5_en.pdf` (EN),
nesta pasta / in this folder. Os artigos não mudam: a correção vai ao lado, nunca por cima. *The articles do not change: the correction goes
beside, never over.*

## PT

A frase «O piso de 0,74 dobras é universal — não depende da dimensão do espaço de Hilbert» (e as que dela dependem: o piso irredutível
D_folds = 0,74 em 9/9 configurações, «o piso de Hilbert de 0,74 é a prova experimental desta lei») **não se sustenta** com a taxa γ livre.

- **Teste pré-registrado de 19/10/2026** `[REAL no modelo simulado]`: pré-registro `evidencia/PREREGISTRO.json`, sha256 `a14bf9af4eeca7c4c49c210b80aca4d86eec95776809b9bb2b0028362e958d47`, gravado
  antes da rodada; função de veredito pinada `evidencia/veredito.py`, sha256 `1c6fc5f52ff724aa16b20b3643ba1201f4fda5f46f45f16b2281f805784cdcde`.
- **Veredito: `NO_UNIVERSAL_FLOOR`** (ramo 3, «P falso»; `evidencia/veredito.json`). Com γ livre (logspace(−4, 0, 25); 9 configurações,
  d = 8 a 32; 5 sementes), D_folds(c³) não tem platô em ln 2: decresce de 1,49–1,84 oitava (γ = 1e−4) a 0,14–0,19 oitava (γ = 1), cerca de
  0,5 oitava por década perto do cruzamento; nenhum γ tem as 9 configurações na banda ln 2 ± 0,05 (o máximo é 8 de 9).
- **O ≈ ln 2 de fevereiro é o ponto de passagem no γ\* calibrado por β, não um piso.** O instrumento de 19/10 reproduz fevereiro na semente
  42 (|ΔO| ≤ 4,9e−12 nas 9 configurações).
- Controles N1, N2 e N3 como previstos; 0 de 2250 pontos sem convergir. A leitura «consciência» é `[ONTO]` e não entra no veredito.
- Rota no mapa de rotas da casa: `quantica.c3_gamma_livre_oitava` (EXAMINADA). O relatório inteiro está em `evidencia/RESULTADO.md`.

O que esta nota não diz: não toca o gate, o selo, o kernel nem o `um.py`; não julga o resto dos artigos; `NOT_FALSIFIED` nunca é
`CONFIRMED`.

## EN

The sentence «The floor of 0.74 folds is universal — it does not depend on the Hilbert space dimension» (and the ones that rest on it: the
irreducible floor D_folds = 0.74 in 9/9 configurations; «the Hilbert floor of 0.74 is the experimental proof of this law») **does not hold**
with a free rate γ.

- **Pre-registered test of 2026-10-19** `[REAL in the simulated model]`: pre-registration `evidencia/PREREGISTRO.json`, sha256 `a14bf9af4eeca7c4c49c210b80aca4d86eec95776809b9bb2b0028362e958d47`,
  written before the run; pinned verdict function `evidencia/veredito.py`, sha256 `1c6fc5f52ff724aa16b20b3643ba1201f4fda5f46f45f16b2281f805784cdcde`.
- **Verdict: `NO_UNIVERSAL_FLOOR`** (branch 3, «P false»). With γ free (logspace(−4, 0, 25); 9 configurations, d = 8 to 32; 5 seeds),
  D_folds(c³) has no plateau at ln 2: it falls from 1.49–1.84 octaves (γ = 1e−4) to 0.14–0.19 octave (γ = 1), about 0.5 octave per decade
  near the crossing; no γ puts all 9 configurations inside ln 2 ± 0.05 (the maximum is 8 of 9).
- **The ≈ ln 2 of February is the crossing point at the γ\* calibrated by β, not a floor.** The instrument of 19/10 reproduces February at
  seed 42 (|ΔO| ≤ 4.9e−12 in the 9 configurations).
- Controls N1, N2 and N3 as predicted; 0 of 2250 points failed to converge. The «consciousness» reading is `[ONTO]` and does not enter the verdict.

What this note does not say: it does not touch the gate, the seal, the kernel or `um.py`; it does not judge the rest of the articles;
`NOT_FALSIFIED` is never `CONFIRMED`.

---
Nota da sessão a7 da Central de Patentes (Claude), em nome próprio, 19/10/2026, a partir do item 2 das notas do handoff v394 da gerência c1.
*A note by session a7 of the Central (Claude), in its own name, 2026-10-19, from item 2 of the notes of the v394 handoff of the management c1.*
Evidência publicada aqui / evidence published here: `evidencia/PREREGISTRO.json`, `evidencia/PREREGISTRO.json.sha256`, `evidencia/veredito.py`, `evidencia/veredito.json`, `evidencia/RESULTADO.md`, `evidencia/curva_O_gamma_oitavas.csv`, `evidencia/manifesto_dados.json`, `evidencia/rodar.py`.
