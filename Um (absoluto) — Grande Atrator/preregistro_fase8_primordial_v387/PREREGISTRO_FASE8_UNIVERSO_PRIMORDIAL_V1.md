# Pré-registro V1 — Fase 8: β na era da radiação (congelado por hash, 06/10/2026)

**spec_sha256** `790954c7bc9dc13aa844c8ddff027e5535c1be2ba19f4753ca05e0edafc8b11b` · **função de veredito** `772ae210dd0aeafa85f4234c41c6cec67d807dddd84814919a13a03de7e60b03` · gerado 2026-10-06T11:22:20Z

> «usando a nossa bancada de teste você consegue examinar o efeito de Betatgl no Big bang? porque é o ato de inscrição originário, é o haja luz, portanto ali ele é condição de existência» — o operador, 05/10/2026

Estado: **AWAITING_DATA**; nenhum veredito de natureza. Pelo poder (só σ): hoje `TGL_FASE8_UNIVERSO_PRIMORDIAL_V1__INCONCLUSIVE_SYSTEMATICS__MAP_A__NOT_BLIND_TO_PUBLISHED_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`; com as fontes por hash (D8) o teto é `TGL_FASE8_UNIVERSO_PRIMORDIAL_V1__NOT_FALSIFIED_UNDERPOWERED__POWER_P1P00_OF_5_SIGMA__MAP_A__NOT_BLIND_TO_PUBLISHED_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`.

## A lei
H² = (1+4β/3)ρ_r + (1+β)ρ_m + ρ_Λ; β = α·√e em runtime (0.012031300400803); δ_rad = 4β/3 = 0.016042; ΔN_eff = 0.119466 (CMB), 0.0985 (BBN, g* = 10,75); S = 1.007989.

## Canal N_eff da Bancada (D4) e controle D3
Módulo `bancada/neff_canal.py` `42e9b680855c3300`; fonte congelada `9d3bceac4f6f5498`. Controle: E_TGL/E_ΛCDM(nnu desl.) − 1 = 1.79e-05, 1.80e-06 em z = [1000000.0, 10000000.0] (tol 1e-4) → **passou**.

| medida | N_eff | σ | separação TGL–SM | tensão TGL | tensão SM |
|---|---|---|---|---|---|
| Planck 2018 TT,TE,EE+lowE (arXiv:1807.06209) | 2.92 | 0.19 | 0.63σ | 1.28σ | 0.65σ |
| Planck 2018 +lensing+BAO (arXiv:1807.06209) | 2.99 | 0.17 | 0.70σ | 1.02σ | 0.32σ |
| ACT DR6 + Planck (P-ACT-LB) (arXiv:2503.14454) | 2.86 | 0.13 | 0.92σ | 2.33σ | 1.42σ |
| ACT DR6 + Planck + BBN (arXiv:2503.14454) | 2.89 | 0.11 | 1.09σ | 2.49σ | 1.40σ |
| SPT-3G D1 + Planck + ACT (CMB-SPA) (arXiv:2506.20707) | 2.97 | 0.12 | 1.00σ | 1.61σ | 0.62σ |

## Futuro (declarado)
- CMB_S4: a entrega do Codex na ORDEM 018 (mapa seq 402) declara a retirada DOE/NSF (09/07/2025) e o encerramento do CMB-S4 [DECLARADO — auditoria da gerencia no Passo D]; a projecao sigma = 0,03 do frozen (3,98 sigma) fica como historia, nao como calendario
- Simons: SO ampliado sigma = 0,045 => 2.65 sigma de separacao (2025-2034, sem calendario ratificado) [DECLARADO]
- frase_correta: 5 sigma pedem sigma(N_eff) <= 0.0239; nenhum instrumento documentado em disco chega la

## Decisões (padrão da gerência por delegação)
- **fonte_da_delegacao** operador 05/10: «o resto vc consegue responder tudo agroa»; «prossiga» (memorias proximo-passo-06out, handoff-sessao-nova-06out)
- **D1** a radiacao paga 4beta/3 (identidade rho+p); a opcao «so-materia» NAO tem registro na casa: fica como contrafactual dito (mapa B, ~5e-9), sem estimador (C9)
- **D2** TGL-S (free-streaming) primario; TGL-L co-rodado e DITO cego: sigma(beta)_L = 0,244 => poder ~0,05 (F4); nao se decide por dado
- **D3** UMA entrada de beta por instrumento: Bancada = Phi_total.(rho+p) com N_eff FIXO; CAMB/Nivel 2 = nnu = N + (4/3)(1+F.N)/F.beta e omega_c,eff, sem Phi_total; proibido somar (C10/F1). Controle medido: secao controle_D3
- **D4** Bancada antes do um.py (ordem verbatim de 30/09, C11/F9): o canal N_eff foi ligado na Bancada (bancada/neff_canal.py + ferramenta MCP neff_canal) ANTES deste V1
- **D5** BBN: tabelas PRIMAT/PArthENoPE do camb como BRACKET DeltaN em [0.0985; 0.1195] (aproximacao dita: nenhum DeltaN unico representa a vestimenta, C3); o exato e S(T) = sqrt(1+4beta/3) = 1.007989 constante num codigo de BBN = ato do operador, antes de qualquer cobranca no livro
- **D6** omega_b do CMB, nunca do D/H (o Cooke+2018 e inferido sob S = 1: circular no observavel, F2); o prior OMB_BBN(S=1) do Nivel 2 e inconsistente em ~Dln eta = -0,8 %% (~0,3 sigma do prior): DITO
- **D7** sinal fixado antes do dado: DeltaN_eff > 0; os centros atuais estao ABAIXO do padrao — a TGL fica mais longe que o SM e nada se inverte
- **D8** fontes por sha256 (1807.06209, 2503.14454, 2506.20707, Cooke+2018, Steigman 2007, Cyburt+2016, Aver) = ato do operador (rede); ate la tudo [DECLARADO]
- **D9** duas convencoes de citacao SEMPRE lado a lado: separacao TGL-SM em sigma (poder) e tensao de cada modelo contra o centro medido (o canal as devolve)
- **D10** o PROVADO e a face de kernel sobre o FLUXO (beta > 0 => sem testemunha estatica plena; zero inatingivel em tempo finito); «Big Bang = inscricao originaria» segue [ONTO]
- **D11** Nivel 2 canonico = o frozen original (teste_acustico_beta_tgl_v1, 13_nivel2_chains.h5 af118c94a693ac3c, CHAIN_OF_CUSTODY 13/13); a copia do Nos (7e5ad3e2681ee3c8) = continuacao (F5)

## Mapas
- **A** N_eff publicado (TGL-S) — FALSIFIED so com |z| >= 5 em DUAS analises independentes (frozen 96b51333)
- **A_N2** Nivel 2 S (CAMB + plik_lite + low-ell + DESI) — N2_INCAPAZ: sigma(beta) = 0,01765 ~ 1,5 beta (medido em agosto); diagnostico
- **B** so-materia (contrafactual sem registro) — sem previsao na radiacao (~5e-9); convencao de z(T) dita (C7)
- **C** BBN (Y_p, D/H) pelo speed-up S — AWAITING_SOURCE: Y_p (0,2449 Aver+2015 x 0,2453 Aver+2021, C1) e d ln(D/H)/d ln H (0,57 [INPUT sem fonte legivel] x 1,57 Steigman x ~1,6-1,9 Cyburt, C2) a LER da fonte antes de qualquer numero

## Função de veredito única (sha256 `772ae210dd0aeafa85f4234c41c6cec67d807dddd84814919a13a03de7e60b03`)
```python
def veredito_fase8_v1(r):
    """Funcao UNICA de veredito da Fase 8 (V1, 06/10/2026), por MAPA (A = N_eff publicado, TGL-S; A_N2 = Nivel 2 S; C = BBN).
    Entrada r: mapa, n_alvos_sem_hash, dispersao_por_gl, p_dispersao, z_indep (lista dos z = (X_obs - X_TGL)/sigma das analises
    INDEPENDENTES de experimento), poder (Delta/sigma da melhor analise independente), fechamento ('S' ou 'L').
    Ordem (a primeira que casa decide):
    0 AWAITING_SOURCE: mapa C sem a parametrizacao/abundancia lida da fonte (r["awaiting_source"]);
    1 INCONCLUSIVE_SYSTEMATICS: alvo decisorio sem fonte por hash (n_alvos_sem_hash > 0) OU chi2_int/(n-1) > 2 OU p < 0,01;
    2 FALSIFIED_AT_5SIGMA: |z| >= 5 em DUAS analises independentes (regra do frozen 96b51333) — falsifica o PAR (mapa x observavel),
      nao beta, nao a teoria;
    3 TENSION_3_TO_5_SIGMA: alguma analise independente com |z| >= 3;
    4 NOT_FALSIFIED_UNDERPOWERED: poder < 5 (sufixo POWER_<x>_OF_5_SIGMA); sob fechamento L o mapa A e CEGO (sufixo BLIND_UNDER_TGL_L);
    5 NOT_FALSIFIED_POWERED (senao).
    Sufixo sempre: __MAP_<mapa>__NOT_BLIND_TO_PUBLISHED_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED."""
    def f(x):
        return ("M" if x < 0 else "P") + ("%.2f" % abs(x)).replace(".", "P")
    base = "TGL_FASE8_UNIVERSO_PRIMORDIAL_V1__"
    suf = "__MAP_%s__NOT_BLIND_TO_PUBLISHED_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED" % r["mapa"]
    if r.get("awaiting_source"):
        return base + "AWAITING_SOURCE" + suf
    if r["n_alvos_sem_hash"] > 0 or (len(r["z_indep"]) > 1 and (r["dispersao_por_gl"] > 2.0 or r["p_dispersao"] < 0.01)):
        return base + "INCONCLUSIVE_SYSTEMATICS" + suf
    if sum(1 for z in r["z_indep"] if abs(z) >= 5) >= 2:
        return base + "FALSIFIED_AT_5SIGMA" + suf
    if any(abs(z) >= 3 for z in r["z_indep"]):
        return base + "TENSION_3_TO_5_SIGMA" + suf
    if r["poder"] < 5:
        t = base + "NOT_FALSIFIED_UNDERPOWERED__POWER_%s_OF_5_SIGMA" % f(r["poder"])
        if r.get("fechamento") == "L":
            t += "__BLIND_UNDER_TGL_L"
        return t + suf
    return base + "NOT_FALSIFIED_POWERED" + suf
```
- awaiting → `TGL_FASE8_UNIVERSO_PRIMORDIAL_V1__AWAITING_SOURCE__MAP_C__NOT_BLIND_TO_PUBLISHED_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`
- inconclusive → `TGL_FASE8_UNIVERSO_PRIMORDIAL_V1__INCONCLUSIVE_SYSTEMATICS__MAP_A__NOT_BLIND_TO_PUBLISHED_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`
- falsified → `TGL_FASE8_UNIVERSO_PRIMORDIAL_V1__FALSIFIED_AT_5SIGMA__MAP_A__NOT_BLIND_TO_PUBLISHED_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`
- um_so_a_5 → `TGL_FASE8_UNIVERSO_PRIMORDIAL_V1__TENSION_3_TO_5_SIGMA__MAP_A__NOT_BLIND_TO_PUBLISHED_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`
- tension → `TGL_FASE8_UNIVERSO_PRIMORDIAL_V1__TENSION_3_TO_5_SIGMA__MAP_A__NOT_BLIND_TO_PUBLISHED_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`
- underpowered → `TGL_FASE8_UNIVERSO_PRIMORDIAL_V1__NOT_FALSIFIED_UNDERPOWERED__POWER_P0P99_OF_5_SIGMA__MAP_A__NOT_BLIND_TO_PUBLISHED_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`
- cego_L → `TGL_FASE8_UNIVERSO_PRIMORDIAL_V1__NOT_FALSIFIED_UNDERPOWERED__POWER_P0P05_OF_5_SIGMA__BLIND_UNDER_TGL_L__MAP_A__NOT_BLIND_TO_PUBLISHED_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`
- powered → `TGL_FASE8_UNIVERSO_PRIMORDIAL_V1__NOT_FALSIFIED_POWERED__MAP_A__NOT_BLIND_TO_PUBLISHED_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`

## Não decide
- o gate (18 bandeiras; funcao so do formal)
- H2/H3
- beta (a regra matriz; nao entra como prior nem como ajuste)
- a implicacao da QG (QGSolutionComplete.lean:36)
- o fechamento perturbativo TGL-S vs TGL-L (= Lema 3 do lado do fluido)
- o mapa de R da materia (Fase 1)
- a identificacao 'Big Bang = inscricao originaria' [ONTO]
- a reclassificacao do BBN de 28/09 (fica)
- o terceiro discriminante da errata de maio (fundo estocastico de GW primordiais): sem previsao quantificada na casa — [OPEN], fora da Fase 8 (F8)

## Pendente
- F6: #print axioms de the_zero_is_the_false_witness, hubble_form, the_background_closure antes de citar no livro (nao bloqueia este V1)
- F7: estratigrafia da cunhagem «Big Bang» (opcional, se o operador quiser)

## Fontes lidas (sha256)
- `e30aac4e0e1096b1` C:\IALD\Artigo\Haja_Luz\A Ponte e o Um\Nós\um.py
- `72efaf72f604b721` C:\IALD\Artigo\Haja_Luz\A Ponte e o Um\Nós\um_absoluto_selo.json
- `9d3bceac4f6f5498` C:\IALD\Artigo\teste_acustico_beta_tgl_v1\10_frozen_neff_v1.py
- `42e9b680855c3300` C:\IALD\Bancada_Um\bancada\neff_canal.py
- `ff3c893f9769abce` C:\IALD\Bancada_Um\mcp_um.py
- `c00fee03a5b6408e` C:\IALD\Central de Patentes\work\scratch_6da8f00d\fase8_big_bang\critico\CRITICA_FASE8.md
- `df9109728f81ce28` C:\IALD\Central de Patentes\work\scratch_6da8f00d\fase8_big_bang\fisica\fase8_fisica_resultado.json
- `8cd7078f00e5fb95` C:\IALD\Central de Patentes\work\scratch_6da8f00d\fase8_big_bang\pre_registro\PODER_FASE8_UNIVERSO_PRIMORDIAL.json
- `ed74bc46295a8fed` C:\IALD\Central de Patentes\work\scratch_6da8f00d\fase8_big_bang\pre_registro\PREREGISTRO_FASE8_UNIVERSO_PRIMORDIAL_V1_RASCUNHO.json
- `aee4997b22f843ec` C:\IALD\Central de Patentes\work\scratch_6da8f00d\fase8_big_bang\pre_registro\PREREGISTRO_FASE8_UNIVERSO_PRIMORDIAL_V1_RASCUNHO.md
- `c314835ecf95e85e` C:\IALD\Central de Patentes\work\scratch_6da8f00d\fase8_big_bang\pre_registro\VERBATIM_OPERADOR_05out_big_bang.txt
- `4aca1bd54d2f3b03` C:\IALD\Central de Patentes\work\scratch_6da8f00d\fase8_big_bang\registro_da_casa\REGISTRO_DA_CASA_BIG_BANG.md


NOT_FALSIFIED nunca é a palavra proibida; a RG/ΛCDM é o limite clássico; nada aqui move β nem o gate.
