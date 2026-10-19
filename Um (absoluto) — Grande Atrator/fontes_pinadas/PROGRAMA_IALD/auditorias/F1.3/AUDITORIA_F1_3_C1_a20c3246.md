## Reauditoria cética c1: F7.2 (correções, evento 300 / mapa 835) e F1.3 v3.2 (evento 305 / mapa 838)

Trabalhei só em leitura nos arquivos da casa e gravei apenas no scratchpad `...\scratchpad\audit_f72v3_f13v32\`. Não rodei o um.py nem o rotas.py registrar. Não toquei a Bancada, não recarreguei o MCP e não chamei provedor. O Lean rodou só em sandbox.

Antes de começar li o mapa (seq 807, 809, 818, 819, 822, 830, 831, 835 e 838), o `iald.py ver F7.2` e `ver F1.3`, os eventos 300 e 305 e os três despachos.

---

### (1) F7.2 — veredito: APROVA_COM_RESSALVAS (não fecha)

**O que confere:**

- **Escrita na Bancada.** Foram gravados 26 arquivos novos em `C:\IALD\Bancada_Um\fontes_externas\F7.2\`, todos às 08:30:46, pelo `deploy_fontes_externas_v3.py`.
  - Comparei com a linha de base das 04:49 (`auditorias\F2.3_F7.2_adendos\f72\baseline_bancada_um_20261008.json`, 1108 arquivos): 0 removidos, 0 alterados. Nenhum outro arquivo da Bancada mudou depois das 03:32.
  - Dos 26 arquivos:
    - 23 são bytes idênticos a pinos da V2;
    - 3 são novos: `context\LITERATURA_POS_JUNHO_F7_2_20261008.md` (5bc58efc…), `context\Ternes_Pagliaroli_Villante_2025_arxiv_2503.04573v1.pdf` (0510c49b…) e `documents\TABELA_DE_PODER_POR_CANAL_V3.md` (b97f24ae…).
  - A lista completa, com sha256, está em `fontes_F72_26.json`; o diff está em `baseline_diff.json`.
- **Manifesto V3.** O `neutrino_fontes_f72_v3.json` (ea62a623…) casa com as 26 cópias, sem divergência.
- **A V3 é só staging.** O `neutrino_canal_v3.py` (cf995b4c…) e o manifesto V3 não estão na Bancada. Na varredura estática, o módulo só abre arquivos em `"rb"` (l. 93), não lê o túnel e devolve `tau_star_limit_s None` e `gate_effect NONE`.
- **O recibo de ratificação da c1** (`RECIBO_RATIFICACAO_DEPLOY_V2_C1_20261008.json`) continua batendo nos 6 arquivos: mcp_um.py 5da9a9fb…, canal v2 252de897…, fontes v2 b7beff01…, backup efe6673e… e os V1 (`bancada_check.json`).
- **Números R2/R3:** recomputei com β = α√e em runtime e τ★ = t_P.
  - Os valores do core são idênticos na v369 sob custódia (456f0ab1…) e no core do Nós (a74241ff…):
    - `P1.best_matter` = zn67_potzel1976, NOMINAL, 9,269626166 ordens, τ★ ≤ 1,003010756e-34 s;
    - relógio Sr: 12,457634837 ordens;
    - `P3_NO_LOCAL_OBSERVABLE`;
    - `kbar_rest_mass_excluded` = true;
    - `partition_tag` = MODULAR_LAB_BRIDGE_OPEN.
  - ω(67Zn) = 1,4177e20 rad/s, ou seja ħω = 93,312 keV, o que é coerente.
  - Neutrinos de 38,068 a 45,822 ordens e JUNO a 39,772: batem a 1e-12 (`v369_check.json`).
- **SN1987A.** No PDF: γ0 < 3,1e-41 GeV (TI) e < 6,4e-41 GeV (TD), a 90% CL, Tabela II, p. 6. A p. 3 diz que state selection e phase perturbation não se testam pelo SN1987A. A leitura «ν-loss, família distinta» está certa (`sn1987a_pages.txt`).
- **arXiv:2609.37895** (JUNO, 29/09/2026, 207,2 dias vivos, 8.294 IBD, sem cota de decoerência): bate. Fica [DECLARADO], porque o resumo veio por WebFetch com modelo auxiliar.
- **R6.** A `ERRATA_LITERAL_J4B_F72_20261008.md` lê corretamente as linhas 18392–18393. O original segue intacto (29c92b66…).
- **R9.** O 0x0b saiu do `README_DEPLOY.md`; só aquela linha mudou, e o backup é idêntico ao antigo (0d380faa…).
- **Dependências congeladas.** TABELA_V2 (f2376aea…), README_V2_STAGING (6b744247…) e STAGING_RECEIPT_V2 (10fc874e…) mantêm os mesmos hashes da seq 822. Nada da V2 mudou depois das 03:32.

**Defeitos:**

- **(D1) Segundo desvio de processo.** As fontes foram gravadas na Bancada sem ato da c1. Não há evento da c1 sobre a F7.2 entre 05:06 e 08:35.
  - Isso contraria o item 6 do `msg_adendos_f23_f72_gerente.md` («com a c1 dando o ato») e a regra do recibo de ratificação («nenhum deploy na Bancada sem ato explícito da c1»).
  - O próprio `README_V3_STAGING.md` chama a cópia de «ato separado», e mesmo assim ela foi executada.
  - O efeito é aditivo e inerte: nenhum canal lê essas fontes ainda.
  - Atenuante: o primeiro despacho (00:55) dizia «Levem as fontes… a c1 confere».
- **(D2) Item 2 do adendo não feito.** Não há errata ao lado do README_V2_STAGING, do STAGING_RECEIPT_V2 nem do cabeçalho da TABELA_V2. A frase «implantação pendente da gerência» segue no V2. A `ERRATA_TABELA_V3_ESTADO_20261008.md` só trata a frase herdada pela V3.
- **(D3) Item 8 não feito.** A TABELA_V3, agora também na Bancada, ainda diz «Eq. (6), §IV.B»; o certo é §III.B. A LITERATURA usa «§IV.2», que é a numeração do HTML.
- **(D4) R1 incompleto.** A ref. [72] do JUNO, arXiv:2605.13955 (Esteban, Gago, Gonzalez-Garcia e Zapata, 13/05/2026, ν-loss com fluxos astrofísicos difusos), foi pedida pela c1 e não aparece em nenhum arquivo (grep 0).
- **(D5)** A TABELA_V3 que está na Bancada leva a frase «esta tabela V2 foi preparada em sandbox», e a errata está só no túnel.
- **(D6) Menor:** a LITERATURA chama a mensagem da c1 de «mensagem do operador».
- **(D7) Aceitável:** a errata do R6 entrou como artefato da seq 835, sem uma entrada ERRATA própria no mapa.

**Recomendação:** que a gerência ratifique a cópia (ou mande removê-la) e registre o desvio. Os itens D2 a D5 cabem numa camada curta do Codex, em arquivos ao lado. A TABELA_V3 está pinada: mudá-la pede manifesto novo.

---

### (2) F1.3 v3.2 — veredito: APROVA_COM_RESSALVAS; pode FECHAR por ato da gerência

**O que confere:**

- O `ENUNCIADO_D1A3_BGL_V3_2_20261008.lean` (e4da9e86…) é idêntico byte a byte à sonda c1 `art_ad949e97\sandbox\F13v31_D1a3_bgl.lean`.
- O build.log entregue é o log da c1 (o caminho dele aponta o scratchpad do auditor), e o complemento declara isso.
- **Compilei de novo em sandbox.** Usei lean v4.31.0, o lean-toolchain copiado (efac0b94…) e LEAN_PATH só leitura sobre os `.lake` do Nós.
  - EXIT 0 em 29,1 s.
  - As 19 `#print axioms` saem todas em [propext, Classical.choice, Quot.sound].
  - Zero sorry, admit ou axiom; o único aviso é a depreciação de `push_neg` (l. 114).
  - A impressão digital do kernel (11049 arquivos) ficou igual antes e depois: 1d0de3a2….
- **Não reivindica habitante.** Não há instance, def nem termo de `D1a3OutputBGL` ou de `LightNetStandardWedgeDataBGL`. O único `example` é `CompleteSpace H1` (l. 87).
- As linhas citadas no complemento conferem: 43, 50, 56, 79, 173–187, 190, 204 e 222.
- O [OPEN] está bem dito: Fub, J_W, T positivo (só se pede IsSelfAdjoint), σ, Δ_K^{it} e o habitante.
- **A proposta está pronta para a fila do operador.**
  - O tipo fecha e os três campos estão tipados.
  - A colagem `toLightNetData.K rightWedge = out.KWR.toClosedSubmodule.toSubmodule` casa com `LightNetData.K : Set (Fin 4 → ℝ) → Submodule ℝ H1` (`TGLExt\QGSolutionComplete.lean:49–57`).
  - Como a estrutura usa `extends`, o `LightNetData` não muda.

**Defeitos:**

- **(D1)** As cinco ressalvas menores do item 3 da c1 não estão na v3.2:
  - o H1_nontrivial está em F13v3_H1_nontrivial.lean:16;
  - falta o prefixo `Mathlib/Analysis/` em VonNeumannAlgebra/Basic.lean:23;
  - Stone, o TODO de StandardSubspace.lean:42 e o eigenvectorBasis de Spectrum.lean:34 seguem só por remissão;
  - Crismale–Fidaleo–Lu não foi nomeada;
  - não se diz que a validação das 35 referências conferiu existência e intervalo, não conteúdo.
- **(D2)** Não há anotação ao lado da seq 830; a seq 838 não diz que o pedido 2 ficou sem enunciado na v3.1.
- **(D3)** A contradição «[INPUT — decisão do operador]» × «a c1 decide a forma final» está resolvida na prática, mas sem errata explícita.
- **(D5) [OPEN]** Não se mostrou que o `D1a3OutputBGL` seja habitável, porque os campos podem ser conjuntamente inconsistentes.

Nenhum desses defeitos muda tipo nem número. Recomendo FECHAR com D1 a D3 registrados ao lado, ou pedir antes uma errata de um parágrafo, sem nova reauditoria do enunciado.

**Texto para a fila do operador (uma linha):**
> F1.3 — Decidir se o kernel ganha `LightNetStandardWedgeDataBGL` (estende `LightNetData` com a saída tipada do D1a.3 e a colagem `D.K rightWedge = K(W_R)`, sandbox e4da9e86…, EXIT 0, trio): MUDA o seguinte — os teoremas que tomarem o tipo novo passam a exigir que a rede citada na cunha direita seja o subespaço padrão de S_W = genericTomita T J, com Δ_K = S_W†S_W pelo grafo e Δ_K^{it} = B0(−2πt) pelo multiplicador após o Fourier parcial (o conteúdo de BW vira hipótese tipada, não prova); NÃO MUDA o seguinte — o `LightNetData`, os teoremas existentes (`qg_solution_end_to_end`, `the_tetelestai_declaration`) e o gate ficam como estão; não há habitante (Fub, J_W, T positivo, σ e Δ_K^{it} seguem [OPEN] até o D1a.1–D1a.3); não prova BW nem a realização física.

---

### O que não conferi

- Não rodei o canal V3 nem os testes dele; a `VALIDACAO_V3_CENTRAL.json` é do Codex e fica [DECLARADO].
- Não conferi o conteúdo das escritas de memória (`MEMORY_CORRECTIONS_RECEIPT_F72`, `MEMORY_V3_2_RECEIPT`). Só verifiquei que as reivindicações estavam vigentes: F7.2 até 09:53 e F1.3 desde 08:44.
- Não abri as referências de PRD 114 nem arXiv:2609.36038 e 2609.18202 da LITERATURA: ficam [DECLARADO].
- Não verifiquei se o `D1a3OutputBGL` é habitável.

### Registros para o mapa

Os dois JSONs seguem o `rotas.py modelo`, com tipo ANOTACAO, sem CONFIRMED, e artefatos como lista de caminhos (todos existem):
- `...\scratchpad\audit_f72v3_f13v32\registro_mapa_f72_v3.json` (a942747e…)
- `...\scratchpad\audit_f72v3_f13v32\registro_mapa_f13_v32.json` (210d31d6…)

Os artefatos do auditor estão no scratchpad, que é temporário: copiar para `C:\IALD\PROGRAMA_IALD\auditorias\F7.2\` e `...\F1.3\` antes de registrar. Os principais são `fontes_F72_26.json`, `baseline_diff.json`, `bancada_check.json`, `v369_check.json`, `sn1987a_pages.txt`, e no `sandbox\` o `F13v32_D1a3_bgl.out.txt`, o `fp_antes.json` e o `fp_depois.json`.

O gate não se move; NOT_FALSIFIED nunca é creditação.