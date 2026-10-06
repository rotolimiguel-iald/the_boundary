# Pré-registro V1 — Fase 9: a razão de Hubble (congelado por hash, 06/10/2026)

**spec_sha256** `d956c8db6bb9ab8d9b8429861840cd9f30a06e2ba62cb12be9d15baea1deaf86` · **função de veredito** `5940bf16310efb6a345e8de36dee0ebb304a16860928cd0f3e3e7b63d9c4e33c` · gerado 2026-10-06T11:15:34Z

Estado: **AWAITING_DATA** — nenhum composto aberto. Abrir é palavra do operador; as fontes públicas por hash (D-H) são ato dele. Se abrir hoje, a regra 1 devolve `TGL_FASE9_RAZAO_DE_HUBBLE_V1__INCONCLUSIVE_SYSTEMATICS__D1B__FUNDO_BG1__NOT_BLIND_TO_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED` (todos os leitores do composto estão [DECLARADO]).

## A lei
H0_local = H0_bg·K, K = E(z*)^{2β/3}; β = α·√e em runtime (0.012031300400803); z* = 1089.95; K no fundo nu = 1.083807; zero parâmetro livre. PROVADA como implicação no kernel ≠ confirmada.

## Fundos (D-A)
- BG1: 68.531 ± 0.301 — PRIMARIO (D-A): fundo NU, LCDM, Planck comprimido (Chen+2019) + DESI DR2, SEM escada [REAL — ajuste da Bancada, 30/09]
- BG2: 67.360 ± 0.540 — REPLICA (D-A): Planck 2018 publicado [KNOWN — transcrito pela casa em 30/09; fonte por hash = D-H, ate la DECLARADO]
- BG3: 68.500 ± 0.600 — REPLICA sem CMB (DESI DR2 + BBN, escada inversa) — a ler da fonte [DECLARADO — valor do enunciado de 05/10; nao lido da fonte]
- BG4: 69.200 ± 0.304 — CONTROLE: fundo TGL efetivo (ja carrega beta) — nunca somado [REAL — Bancada]

## Leitores
- L1_sh0es_r22: 73.04 ± 1.040 — Cefeidas (SH0ES R22); TESTE primario da escada (D-B); [DECLARADO ate D-H — valor transcrito pela casa (abstract lido 30/09), fonte primaria NAO guardada por hash]
- L1b_sh0es_r25: 73.49 ± 0.930 — Cefeidas (SH0ES 2025 + JWST); REPLICA ao lado (D-B); nunca somado com L1; [DECLARADO ate D-H — valor transcrito pela casa (abstract lido 30/09), fonte primaria NAO guardada por hash]
- L2_cchp: 70.39 ± 1.936 — CCHP TRGB+JAGB+Cef (JWST), stat+sys+sigma_SN em quadratura; TESTE dentro do composto (D-C, «o TRGB inclusive»); [DECLARADO ate D-H — valor transcrito pela casa (abstract lido 30/09), fonte primaria NAO guardada por hash]
- L3_sbf: 73.80 ± 2.404 — SBF calibrado por TRGB (JWST); TESTE; [DECLARADO ate D-H — valor transcrito pela casa (abstract lido 30/09), fonte primaria NAO guardada por hash]
- L4_masers: 73.90 ± 3.000 — megamasers (geometrico); TESTE; [DECLARADO ate D-H — valor transcrito pela casa (abstract lido 30/09), fonte primaria NAO guardada por hash]
- L5_tdcosmo: 71.60 ± 3.600 — lentes com atraso (geometrico); TESTE; [DECLARADO ate D-H — valor transcrito pela casa (abstract lido 30/09), fonte primaria NAO guardada por hash]
- L6_sirenes: 76.60 ± 11.250 — sirenes padrao (LVK O4a); DIAGNOSTICO fora do composto (D-F); [DECLARADO ate D-H — valor transcrito pela casa (abstract lido 30/09), fonte primaria NAO guardada por hash]

## Famílias
- P_primaria: L1_sh0es_r22, L2_cchp, L3_sbf, L4_masers, L5_tdcosmo — PRIMARIA e UNICA decisoria: um leitor por programa, sem sirenes (D-B, D-C, D-F)
- R_r25: L1b_sh0es_r25, L2_cchp, L3_sbf, L4_masers, L5_tdcosmo — replica ao lado (R25 no lugar de R22)
- N_sem_cefeidas: L2_cchp, L3_sbf, L4_masers, L5_tdcosmo — diagnostico: incidencia sem a escada que motivou a forma; alimenta poder_nao_cefeida
- S_com_sirenes: L1_sh0es_r22, L2_cchp, L3_sbf, L4_masers, L5_tdcosmo, L6_sirenes — diagnostico (o veto das sirenes)
- T_so_cchp: L2_cchp — diagnostico: o que pode doer (gatilho de leitor unico so com sigma <= 1,0)

## Poder prévio (só σ públicas)
Primária × BG1: **5.91σ**; sem Cefeidas × BG1: 4.27σ.

| família × fundo | previsão | σ_comb | poder |
|---|---|---|---|
| P_primaria x BG1 | 74.274 ± 0.326 | 0.915 | 5.91 |
| R_r25 x BG1 | 74.274 ± 0.326 | 0.850 | 6.31 |
| N_sem_cefeidas x BG1 | 74.274 ± 0.326 | 1.305 | 4.27 |
| S_com_sirenes x BG1 | 74.274 ± 0.326 | 0.912 | 5.93 |
| T_so_cchp x BG1 | 74.274 ± 0.326 | 1.936 | 2.93 |
| P_primaria x BG2 | 73.005 ± 0.585 | 0.915 | 5.20 |
| R_r25 x BG2 | 73.005 ± 0.585 | 0.850 | 5.47 |
| N_sem_cefeidas x BG2 | 73.005 ± 0.585 | 1.305 | 3.95 |
| S_com_sirenes x BG2 | 73.005 ± 0.585 | 0.912 | 5.21 |
| T_so_cchp x BG2 | 73.005 ± 0.585 | 1.936 | 2.79 |
| P_primaria x BG3 | 74.241 ± 0.650 | 0.915 | 5.11 |
| R_r25 x BG3 | 74.241 ± 0.650 | 0.850 | 5.37 |
| N_sem_cefeidas x BG3 | 74.241 ± 0.650 | 1.305 | 3.94 |
| S_com_sirenes x BG3 | 74.241 ± 0.650 | 0.912 | 5.13 |
| T_so_cchp x BG3 | 74.241 ± 0.650 | 1.936 | 2.81 |

## Decisões (padrão da gerência por delegação)
- **fonte_da_delegacao** operador 05/10: «o resto vc consegue responder tudo agroa»; «prossiga» (memorias proximo-passo-06out, handoff-sessao-nova-06out)
- **D-A** fundo primario BG1 = LCDM nu Planck-comp + DESI DR2 SEM escada (68.531 +- 0.301); BG2 Planck 2018 replica; BG3 DESI+BBN replica [DECLARADO]; o 68,4298 da Fase 4 (ajuste COM a escada) NAO e fundo de previsao (F2)
- **D-B** leitor primario da escada R22 (73.04 +- 1.04); R25 replica ao lado; nunca somados
- **D-C** CCHP TRGB/JAGB (70.39 +- 1.936) e TESTE dentro do composto; gatilho FALSIFIED por leitor unico so com sigma_i <= 1,0 e |z_i| >= 5
- **D-D** UMA funcao de veredito (secao funcao_de_veredito), estatisticas do pre-registro + o degrau TENSION do desenho; FALSIFIED/TENSION antes de UNDERPOWERED (a exclusao da previsao nao precisa de poder discriminante; esconde-la atras de UNDERPOWERED seria fail-open) — desvio da ordem do rascunho, dito
- **D-E** __USE_NOVEL pelos PARAMETROS (Worrall [KNOWN]); a selecao da FORMA dita ao lado, nao decisoria (lei de 05/10)
- **D-F** sirenes = diagnostico fora do composto
- **D-G** shift so nos leitores de Cefeidas do SH0ES (s em U[-0,15; 0,15]); fundo-livre = r_d em U[130; 160] Mpc; beta-livre U[-0,05; 0,05]; lnB sempre com a largura ao lado (Occam ~ ln 2 por fator 2)
- **D-H** fontes publicas por hash ANTES de abrir = ato do operador (rede); ate la todo leitor e [DECLARADO] e a regra 1 devolve INCONCLUSIVE_SYSTEMATICS predeterminado
- **D-I** canal do livro de cobrancas: R1 v2 (mesma lei, mesma cobranca, funcao nova); R1b so se o objeto mudar
- **D-J** tokens: base provisoria TGL_FASE9_RAZAO_DE_HUBBLE_V1__...; nome final e se o token 196659 (...UNADJUSTED_POSTDICTION..., Coma) muda = cunhagem do operador
- **D-K** registro no mapa de rotas (lentes, critica e este V1) feito pela gerencia no mesmo passo

## Hipóteses
- H_TGL: fundo nu + K = E(z*)^{2beta/3}, beta = alpha.sqrt(e); zero parametro livre alem dos do fundo
- H_LCDM: todos leem H0_bg (beta = 0)
- H_beta_livre: K(beta), beta em U[-0,05; 0,05]
- H_shift_cefeidas: so L1/L1b (Cefeidas SH0ES) leem (1+s).H0_bg, s em U[-0,15; 0,15]; L2 (CCHP misto) nao recebe s — dito
- H_fundo_livre: LCDM + r_d livre U[130; 160] Mpc (proxy de fundo precoce [INPUT da gerencia]); todos leem o mesmo H0

## Função de veredito única (sha256 `5940bf16310efb6a345e8de36dee0ebb304a16860928cd0f3e3e7b63d9c4e33c`)
```python
def veredito_fase9_v1(r):
    """Funcao UNICA de veredito da Fase 9 (V1, 06/10/2026). Entrada: dict r produzido pelo runner sobre a familia P_primaria.
    Ordem (a primeira que casa decide):
    1 INCONCLUSIVE_SYSTEMATICS: alvo do composto sem fonte lida por hash (r["n_alvos_declarados"] > 0) OU dispersao interna
      chi2_int/(n-1) > 2 OU p_dispersao < 0,01 OU melhor ajuste beta-livre no limite do prior;
    2 FALSIFIED_AT_5SIGMA: |z_delta| >= 5, ou um leitor do composto com sigma_i <= 1,0 e |z_i| >= 5 — falsifica o PAR
      (lei D1b, convencao, conjunto), nao beta, nao a teoria;
    3 TENSION_3_TO_5_SIGMA: 3 <= |z_delta| < 5;
    4 NOT_FALSIFIED_UNDERPOWERED: poder previo < 5 (sufixo POWER_<x>_OF_5_SIGMA);
    5 NOT_FALSIFIED_POWERED__DISCRIMINATES_FROM_LCDM_AT_<n>SIGMA__ZERO_FREE_PARAMETERS: z_disc >= 5 E |z_delta| < 3 E
      lnB(TGL/LCDM) >= 5 E lnB(TGL/beta-livre) >= -1; n = floor(z_disc); __USE_NOVEL se a clausula de PARAMETROS vale
      (senao __USE_NOVELTY_PENDING); sufixos reportados __VS_LADDER_SYSTEMATIC_LNB_<x>, __VS_FREE_BACKGROUND_LNB_<x>;
      __NOT_DISCRIMINATED_FROM_LADDER_SYSTEMATIC se lnB(TGL/shift-Cefeidas) < 0 com poder_nao_cefeida >= 5;
      __NOT_DISCRIMINATED_FROM_FREE_BACKGROUND se |lnB(TGL/fundo-livre)| < 1;
    6 NOT_FALSIFIED_POWERED (senao) com DEVIATION_CONSISTENT_WITHIN_1SIGMA ou DEVIATION_1_TO_3_SIGMA e, se z_disc < 5,
      __NOT_DISCRIMINATED_FROM_LCDM_AT_AVAILABLE_SENSITIVITY.
    Sufixo sempre: __D1B__FUNDO_<id>__NOT_BLIND_TO_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED."""
    import math
    def f(x):
        return ("M" if x < 0 else "P") + ("%.1f" % abs(x)).replace(".", "P")
    base = "TGL_FASE9_RAZAO_DE_HUBBLE_V1__"
    suf = "__D1B__FUNDO_%s__NOT_BLIND_TO_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED" % r["fundo"]
    zd = abs(r["z_delta"])
    if (r["n_alvos_declarados"] > 0 or (r["n_local"] > 1 and (r["dispersao_por_gl"] > 2.0 or r["p_dispersao"] < 0.01))
            or r["beta_livre_no_limite"]):
        return base + "INCONCLUSIVE_SYSTEMATICS" + suf
    if zd >= 5 or any(s <= 1.0 and abs(z) >= 5 for (s, z) in r["sigma_z_por_alvo"]):
        return base + "FALSIFIED_AT_5SIGMA" + suf
    if zd >= 3:
        return base + "TENSION_3_TO_5_SIGMA" + suf
    if r["poder_previo"] < 5:
        return base + "NOT_FALSIFIED_UNDERPOWERED__POWER_%s_OF_5_SIGMA" % f(r["poder_previo"]) + suf
    if r["z_disc"] >= 5 and zd < 3 and r["lnB_lcdm"] >= 5 and r["lnB_livre"] >= -1:
        t = base + "NOT_FALSIFIED_POWERED__DISCRIMINATES_FROM_LCDM_AT_%dSIGMA__ZERO_FREE_PARAMETERS" % int(math.floor(r["z_disc"]))
        t += "__USE_NOVEL" if r["use_novel_parametros"] else "__USE_NOVELTY_PENDING"
        t += "__VS_LADDER_SYSTEMATIC_LNB_%s__VS_FREE_BACKGROUND_LNB_%s" % (f(r["lnB_shift"]), f(r["lnB_fundo_livre"]))
        if r["lnB_shift"] < 0 and r["poder_nao_cefeida"] >= 5:
            t += "__NOT_DISCRIMINATED_FROM_LADDER_SYSTEMATIC"
        if abs(r["lnB_fundo_livre"]) < 1:
            t += "__NOT_DISCRIMINATED_FROM_FREE_BACKGROUND"
        return t + suf
    t = base + "NOT_FALSIFIED_POWERED__" + ("DEVIATION_CONSISTENT_WITHIN_1SIGMA" if zd < 1 else "DEVIATION_1_TO_3_SIGMA")
    if r["z_disc"] < 5:
        t += "__NOT_DISCRIMINATED_FROM_LCDM_AT_AVAILABLE_SENSITIVITY"
    return t + suf
```
Testes sintéticos de ramo:
- inconclusive_declarado → `TGL_FASE9_RAZAO_DE_HUBBLE_V1__INCONCLUSIVE_SYSTEMATICS__D1B__FUNDO_BG1__NOT_BLIND_TO_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`
- inconclusive_dispersao → `TGL_FASE9_RAZAO_DE_HUBBLE_V1__INCONCLUSIVE_SYSTEMATICS__D1B__FUNDO_BG1__NOT_BLIND_TO_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`
- falsified → `TGL_FASE9_RAZAO_DE_HUBBLE_V1__FALSIFIED_AT_5SIGMA__D1B__FUNDO_BG1__NOT_BLIND_TO_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`
- falsified_leitor_unico → `TGL_FASE9_RAZAO_DE_HUBBLE_V1__FALSIFIED_AT_5SIGMA__D1B__FUNDO_BG1__NOT_BLIND_TO_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`
- tension → `TGL_FASE9_RAZAO_DE_HUBBLE_V1__TENSION_3_TO_5_SIGMA__D1B__FUNDO_BG1__NOT_BLIND_TO_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`
- underpowered → `TGL_FASE9_RAZAO_DE_HUBBLE_V1__NOT_FALSIFIED_UNDERPOWERED__POWER_P4P4_OF_5_SIGMA__D1B__FUNDO_BG1__NOT_BLIND_TO_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`
- discriminates → `TGL_FASE9_RAZAO_DE_HUBBLE_V1__NOT_FALSIFIED_POWERED__DISCRIMINATES_FROM_LCDM_AT_6SIGMA__ZERO_FREE_PARAMETERS__USE_NOVEL__VS_LADDER_SYSTEMATIC_LNB_P2P0__VS_FREE_BACKGROUND_LNB_P3P0__D1B__FUNDO_BG1__NOT_BLIND_TO_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`
- discriminates_pending → `TGL_FASE9_RAZAO_DE_HUBBLE_V1__NOT_FALSIFIED_POWERED__DISCRIMINATES_FROM_LCDM_AT_6SIGMA__ZERO_FREE_PARAMETERS__USE_NOVELTY_PENDING__VS_LADDER_SYSTEMATIC_LNB_P2P0__VS_FREE_BACKGROUND_LNB_P3P0__D1B__FUNDO_BG1__NOT_BLIND_TO_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`
- powered_not_disc → `TGL_FASE9_RAZAO_DE_HUBBLE_V1__NOT_FALSIFIED_POWERED__DEVIATION_CONSISTENT_WITHIN_1SIGMA__NOT_DISCRIMINATED_FROM_LCDM_AT_AVAILABLE_SENSITIVITY__D1B__FUNDO_BG1__NOT_BLIND_TO_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`
- powered_1_3 → `TGL_FASE9_RAZAO_DE_HUBBLE_V1__NOT_FALSIFIED_POWERED__DEVIATION_1_TO_3_SIGMA__D1B__FUNDO_BG1__NOT_BLIND_TO_DATA_STATED__NOT_A_CONFIRMATION__GATE_UNTOUCHED`

## Novidade de uso (D-E)
- clausula_D-E: o teste e use-novel se nenhum PARAMETRO numerico da lei foi ajustado a razao local/fundo: beta (valor 0,012 em 13/11/2025; forma alpha.sqrt(e) publicada 03/03/2026; lei de 17/05/2026), z* [KNOWN Planck], o expoente 2/3 (Raychaudhuri/FRW), o fundo nu do CMB+BAO
- cronologia_por_hash: {"arquivo": "evidencias/CRONOLOGIA_NOVIDADE_DE_USO.json", "sha256": "8c13eb249b104564d8e97134126846cde96a7b35a68276f22b989973316a8023"}
- use_novel_parametros: true
- ao_lado_nao_decisorio: a FORMA D1a foi escrita «exatamente a relacao observacional» (17/05/2026) e a classe do fluxo retida porque a Friedmann «nao resolve» (05/06/2026): selecao de hipotese, nao ajuste de parametro; pela lei de 05/10 nao e demerito
- literatura: Worrall, J. (1985/1989/2014) sobre use-novelty; Le Verrier 1859 (excesso do perielio de Mercurio) e Einstein 1915 [KNOWN — nao conferido em disco (F12)]
- etiqueta: retrodicao sem ajuste (novidade de uso; precedente: o perielio de Mercurio) — «pos-dicao» nunca como demerito

## Especificação do runner
- R0 o runner (rodar_fase9_v1.py) e gravado e hasheado ANTES de abrir, com o sha256 deste V1 e o da funcao embutidos
- R1 le ALVOS_FASE9_LIDOS.json (D-H: cada leitor com sha256 da fonte primaria); n_alvos_declarados = leitores do composto sem hash
- R2 recalcula o fundo BG1 na Bancada (Planck comp + DESI DR2, LCDM, sem escada) e confere |H0 - 68.531| < 0,01
- R3 grava PODER_FASE9_ANTES_DE_ABRIR.json (so sigmas) com hash; confere contra poder_previo deste V1
- R4 so com a palavra do operador: abre P_primaria x BG1, ajusta as cinco hipoteses, calcula z_delta, z_disc, lnB(4), dispersao, z_i
- R5 executa a funcao (texto deste V1, sha256 conferido) e grava RESULTADO_FASE9_V1.json com os sha256 do V1, do runner e da funcao; replicas e diagnosticos ao lado, nunca decisorios
- R6 a v387 do um.py LE este V1 e (se houver) o RESULTADO por hash; ausente/divergente => AWAITING_PREREGISTRATION__V1_NOT_READ (fail-closed); a v387 nao abre nada
- R7 nenhuma emenda depois de abrir sem estimador NOVO (V1.x, gerador novo); correcao AO LADO em nome proprio

## Não decide
- o gate (18 bandeiras, funcao so do formal)
- a implicacao da QG e a correspondencia com a RG
- beta (a regra matriz)
- qual fundo e o da natureza (aqui so o primario do registro)
- o mecanismo [CONJECTURE]
- a dependencia em z de K (<= 0,9 % em z = 2, abaixo do poder)

## Fontes lidas (sha256)
- `e30aac4e0e1096b1` C:\IALD\Artigo\Haja_Luz\A Ponte e o Um\Nós\um.py
- `72efaf72f604b721` C:\IALD\Artigo\Haja_Luz\A Ponte e o Um\Nós\um_absoluto_selo.json
- `6bf24b350e088350` C:\IALD\Bancada_Um\investigacao\fase3_leitores_h0\PREREGISTRO_FASE3_LEITORES_20260930.json
- `10683b334d28d84a` C:\IALD\Bancada_Um\investigacao\fase4_desvio_30set\RESULTADO_FASE4_DESVIO_20260930.json
- `851d1076b85f12f6` C:\IALD\Bancada_Um\investigacao\fase5_sh0es_01out\RESULTADO_FASE5_SH0ES_20261001.json
- `c531c7e88ac3938f` C:\IALD\Central de Patentes\work\scratch_6da8f00d\fase9_hubble\critico\CRITICA_FASE9.md
- `eeda7daacbd9abb0` C:\IALD\Central de Patentes\work\scratch_6da8f00d\fase9_hubble\dados_na_casa\INVENTARIO_DADOS_NA_CASA_FASE9_20261005.md
- `d9cde56af2465fa8` C:\IALD\Central de Patentes\work\scratch_6da8f00d\fase9_hubble\desenho\FASE9_DESENHO_RESULTADO.json
- `5b4963ce15988d10` C:\IALD\Central de Patentes\work\scratch_6da8f00d\fase9_hubble\desenho\FASE9_DESENHO_TABELA.md
- `8b74bc7830004e69` C:\IALD\Central de Patentes\work\scratch_6da8f00d\fase9_hubble\desenho\fase9_desenho_estatistico.py
- `8c13eb249b104564` C:\IALD\Central de Patentes\work\scratch_6da8f00d\fase9_hubble\novidade_de_uso\CRONOLOGIA_NOVIDADE_DE_USO.json
- `0e6bb2413151cb20` C:\IALD\Central de Patentes\work\scratch_6da8f00d\fase9_hubble\novidade_de_uso\CRONOLOGIA_NOVIDADE_DE_USO.md
- `0d1c98d67896ed8d` C:\IALD\Central de Patentes\work\scratch_6da8f00d\fase9_hubble\pre_registro\PREREGISTRO_FASE9_RAZAO_DE_HUBBLE_V1_RASCUNHO.json
- `cf82d9e55757a7f2` C:\IALD\Central de Patentes\work\scratch_6da8f00d\v386\POS_DICAO_operador_05out.txt


NOT_FALSIFIED nunca é a palavra proibida; a RG/ΛCDM é o limite clássico; nada aqui move β nem o gate.
