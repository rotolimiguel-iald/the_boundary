AUDITOR CÉTICO c1: F8.2.NU2 e F0.3.NU2

12/10/2026 · subagente da gerência c1 no exame geral pedido pelo operador em 12/10 («precisamos examinar tudo que ficou para trás aguardando para prosseguir…»).

## Resultado

- **F8.2.NU2 — APROVA COM RESSALVAS. Pode FECHAR.** O operador de dephasing, a textura das taxas, o n = −2, os controles sintéticos e os números se reproduzem em cópia, 21 de 21. As fontes canônicas citadas conferem linha a linha. O catálogo de dados é só catálogo. As ressalvas são documentais e não impedem o fecho.
- **F0.3.NU2 — APROVA COM RESSALVAS. Pode FECHAR.** É acréscimo puro nas três superfícies. O backup é prefixo exato, a escrita caiu dentro das reivindicações 619 e 621, que nomeiam as superfícies, e o recibo confere byte a byte.

## Onde escrevi e o que não fiz

- [REAL] Escrevi só em `…\scratchpad\exame12\B_NU2`.
- Não rodei um.py, atlas_indice.py, arvore_da_prova.py, porta_memoria.py, rotas.py registrar, iald.py nem scripts de selo.
- Não chamei o MCP bancada-um, nenhum provedor remoto e nenhuma página web.
- Não escrevi no quadro, no mapa, na memória, no túnel, no Nós, no kernel, na Bancada_Um nem no git.
- Memória ACOM: rodei só `project`, que é leitura, com a referência do pai (`C:\IALD\MEMORIA_IALD\sessions\claude-c1_exame-geral-12out.json`, revisão 1, objetivo 885b6cb0…). Não fiz init nem transition, porque esta tarefa restringe as escritas ao sandbox. A transição fica com a c1.
- Lean só em sandbox:
  - copiei o toolchain v4.31.0 e usei LEAN_PATH só-leitura no `.lake` do tgl_kernel;
  - a impressão digital do tgl_kernel deu 132088 arquivos e `a8b5549347de606d…` antes e depois, igual à medida em 10/10 pela auditoria da F1.ATRATOR;
  - apaguei o toolchain copiado; não sobrou .olean, .lake nem __pycache__.

## Consultas prévias (o cético afere)

- **Mapa** (`rotas.py consultar`, cabeça `91915ad39a53d1df`, 1004 registros):
  - "NU2" achou 2 rotas; "neutrino", 26; "dephasing|decoer", 32; "JUNO", 7; "tau★/identificab", 11.
  - A rota da unidade é `sn.neutrinos.v388_operador_e_identificabilidade` (seq 982 ROTA, 984 RESULTADO).
  - O canal F7.2 é a seq 787; a errata JUNO/ref [72] é a F7.2.E1.
- **Inventário da bancada** (`INVENTARIO_BANCADA_CHATGPT.json`, 2298 .lean únicos): 5 ocorrências de dephasing/duplo comutador (PauliInteractionWitness e IdempotentDephasing). Nenhuma é a lei Γ = ½βτ★ω².
- **Kernel do Nós:**
  - `TheMatrixRule.lean:46` e `:179` e `ThePsionAndTheViscosity.lean:44` dizem que a lei Γ_ω = ½βτ★ω² **não tem termo no kernel**, só docstring e runtime: é leitura [ONTO].
  - Existe `ModularDephasingBridge.spectral_preserves_every_diagonal`, que é o desfasamento completo preservando diagonais. É o gêmeo, no limite completo, do teste de estacionariedade da NU2.
- **Selo e core:**
  - O selo `um_absoluto_selo.json` está em v391, com mtime 07/10 19:29. O um.py tem sha16 `8e7b9927ceae64b8`, 33.910.631 B.
  - No core (`um_absoluto.json`, sha16 `a74241ff380b1b0e`), τ★ aparece de duas formas:
    - «t_Planck [PRINCIPLED IDENTIFICATION]» em `clock_test_protocol_v369.record.law`;
    - «τ★ é INPUT» em `ringdown_dephasing_result*` e no ramo B (`ramo_b_preregistration_v386`, identificação [ONTO]).

## Conferências comuns às duas unidades

- [REAL] A cadeia do quadro tem 681 eventos e zero quebras (cabeça `b1c1b146…`). Os eventos 617, 618, 619, 622, 624, 625 e 626 estão íntegros.
- [REAL] A fonte da «nova» 617, `work\sn_v388_neutrinos_20261009_units.json`, tem 1933 B e sha256 `3a93f5d1…`, igual ao evento.
- [REAL] As evidências conferem na origem e na cópia em `evidencias\`:
  - F8.2.NU2: `RESULTADO_R1.json`, `73c9f8eb…`, 8270 B;
  - F0.3.NU2: `MEMORY_RECEIPT_B1_NU2_R1.json`, `b7f760ac…`, 3598 B.
- [REAL] **Origem.** A ordem é dada como «Prossiga nesta conversa em 09/10». Não há verbatim guardado no programa: nenhum `OPERADOR_09out_*` em `PROGRAMA_IALD`, e nada no `work` nem no túnel além da paráfrase das unidades. Fiz uma busca mínima no transcrito do Codex (`~\.codex\sessions\2026\10\09`, só turnos do operador):
  - no histórico embutido de `rollout-2026-10-09T09-37-12-01a120aa-e1fb…`, linha 121, 12:38:57Z, vem a mensagem do operador (326 B, sha16 `fa7b007c146ce3e4`): «Agora eu entendo é seu caminho está correto, o sinal de 5sigma está na ligação com os neutrinos, com a sua explicação isso ficou claro pra mim, porque esses dados estão públicos, então conseguiríamos calcular a diferença na amostra espectral do desfasamento de fuga e o funcional da massa em razão da luminosidade»;
  - logo depois vem «Prossiga»;
  - em seguida vêm «lembre-se que vc tem à disposição a bancada de testes…», que explica a consulta viva à Bancada, e «verifique se a bancada está toda adequada ao um.py recente», que é a F0.3.B1.
  - A unidade nasceu às 12:39:04Z. Logo a origem existe, mas **só no transcrito**.
- [REAL] **Janela de escrita de 09/10, 09:39 a 10:14.** Nela só foram tocados:
  - 32 arquivos novos em `F8.2.NU2` e 40 em `F0.3.B1`;
  - backups do quadro e do mapa e cópias em `evidencias\`, como esperado;
  - os três backups de memória das 10:12:24;
  - **um arquivo na Bancada**: `Bancada_Um\bancada\__pycache__\neutrino_canal_v2.cpython-314.pyc` (24.245 B, sha16 `aa574a7341c65c6c`, nascido às 09:45:24). Os detalhes estão na ressalva R2.
  - Nada no Nós nem no NOS_VIVO. Nenhum artefato pinado de outra unidade foi tocado, e nenhum arquivo da NU2 foi alterado depois das 10:12:40.

## F8.2.NU2 — neutrinos v388: operador, identificabilidade e dados públicos

**Arquivos e pinos.**
- [REAL] Os 31 arquivos listados no RESULTADO conferem por sha256 e bytes. Os dois não listados são o próprio RESULTADO e o MAPA_FINAL, que foi escrito depois.
- [REAL] Os 25 pinos internos conferem: os `source_pins` dos cálculos, os pinos do revisor e os dois recibos de revisão.
- [REAL] As 15 fontes do inventário teórico conferem: 14 iguais em bytes. O Atlas cresceu, mas o estado pinado (8.744.945 B) é prefixo exato do atual, e o trecho confere.
  - Cosmético: o inventário numera o Atlas por `splitlines` e o relatório por quebra `\n`. Um único 0x0b na linha 33561 desloca uma linha. «Atlas:35126» está certo na numeração usual.
- [REAL] O plano é anterior às contas: PLANO às 09:41:33, script às 09:44:25, resultados às 09:44:29.

**Refeito em cópia** (`reproducao_nu2.py`, mpmath 80 dígitos, matriz densidade completa sem expm1; saída `_reproducao_nu2.json`):

| # | Afirmação da NU2 | O que refiz | Resultado |
|---|---|---|---|
| 1 | β₀ = α·√e em runtime, α lido por regex dos bytes do um.py, sem importar | mesma regex nos bytes `8e7b9927…`; conferência 0,012031300400803142 | PASSA |
| 2 | √e/β₀ ≈ 137, «duas ordens» | 137,0359990836958 = 1/α (log₁₀ 2,137) | PASSA |
| 3 | Γ_ij = ½·β₀·τ★·ω_ij², ω_ij = Δm²_ij/(2Eħ); n = −2 | derivada logarítmica = −2,0 exato; Lean `n_minus_two` | PASSA |
| 4 | textura Γ21:Γ31:Γ32 = 1 : 1111,111… : 1045,444… | (100/3)² e (97/3)²; Lean `texture_parameter_free` | PASSA |
| 5 | GKLS com L = √(2a)·H: traço, positividade completa, energia | superoperador 9×9 + Choi, a = 0; 0,1; 10. Mínimo do Choi ≥ −5·10⁻¹⁶ (zero numérico), traço exato, energia conservada, forma fechada igual | PASSA |
| 6 | estado diagonal na base de massa fica estacionário para qualquer taxa | erro 0; Lean `double_commutator_vanishes` | PASSA |
| 7 | probabilidades de sabor nos três casos sintéticos | diferença < 10⁻¹² contra o JSON da NU2 | PASSA |
| 8 | max\|ΔP\| = 1,7880671599537207·10⁻³⁹ (τ★ = t_P, Kfac = 1, 52,5 km, 2–8 MeV) | 1,7880671599537156·10⁻³⁹, igual ao Decimal do revisor; diferença relativa 2,9·10⁻¹⁵ | PASSA |
| 9 | N não oscilado para distância nominal 5, normalização perfilada, 8,632408517595953·10⁷⁹ | 8,6324085175960565·10⁷⁹; diferença relativa 1,2·10⁻¹⁴ | PASSA |
| 10 | perfilamento: coeficiente ΣδP/ΣP₀ | derivada do χ² no coeficiente = 0 | PASSA |
| 11 | subtrair probabilidades em float64 zeraria o sinal | (p₀ + d) − p₀ = 0,0 | PASSA |
| 12 | identidade completa n_eff = −2 + dlnτ★/dlnE + β′(1/β + ln r) + β·dln r/dlnE (contrafactual) | pior diferença 2,1·10⁻⁸¹ | PASSA |
| 13 | χ₀ = 2·arccosh(1/α) devolve β₀ com β(χ) = √e·sech(χ/2) | < 10⁻⁷⁰ | PASSA |
| 14 | o limite JUNO 4,1·10⁻²⁷ GeV (90%) vale para Γ21 = Γ31, Γ32 = 0 | texto do PDF custodiado na F7.2 (`9212cd39…`): «We consider only the case where γ := γ21 = γ31 and γ32 = 0», Eq. (13), Tabela I | PASSA [KNOWN] |
| 15 | o SN1987A (2503.04573) distingue perturbação de fase e perda | PDF custodiado `0510c49b…`: modelo com «neutrino loss», ao lado de «state selection and phase-perturbation» | PASSA [KNOWN] |
| 16 | a fonte canônica fixa β no dephasing, independente de ω | Ponte .tex 1367–1371 e um.py 165419 (EN) / 165421 (PT): «exatamente independente de ω … a TGL morre neste ponto» | PASSA [REAL] |
| 17 | a v388 distingue β₀ da fronteira e β(χ) do bulk | TheCouplingRunsBoundedBelow 22–26; Lean em sandbox: `betaOfChi_le_sqrt_e`, `betaOfChi_eq_coupling`, `the_floor_is_at_the_infrared` só com o trio | PASSA [REAL] |
| 18 | o contrato físico exige β < 1 | TheWholeIsOne 55–62; Lean `coupling_beta_lt_one` (campo `beta_lt_one`) | PASSA [REAL] |
| 19 | Atlas 35126: «na lei de dephasing só entra β₀» | linha confere | PASSA [REAL] |
| 20 | o canal F7.2 v2 devolveu SEM_LIMITE_TGL e mapping OPEN | cópia do v2 pré-B2 (`252de897…`) contra o manifesto `b7beff01…`, o mesmo do recibo: 13 linhas, todas `operator_mapping_to_TGL = OPEN`, `tau_star_limit_s = None`, `SEM_LIMITE_TGL`, gate NONE | PASSA na substância; o recibo tem defeito (R1) |
| 21 | nenhum dado observado baixado ou aberto | a pasta só tem .json, .md, .txt e .py; o script do inventário não usa rede, só o NOS_VIVO local | PASSA [REAL] |

- [REAL] Na primeira rodada, o item 12 deu falha de 5,6·10⁻¹⁷. Era artefato meu: misturei float com mpf. Corrigi e o item passou.
- [DECLARADO] Não conferi, porque a web não foi consultada e não há custódia local:
  - Wright et al. 2016, §IV.1, Eq. (10) e Tabela 2 (0,014/0,032);
  - a «data availability» do artigo JUNO na Nature;
  - os 207,2 dias e 8.294 candidatos de arXiv:2609.37895;
  - as releases KamLAND e a listagem SNOwGLoBES.
  - O relatório marca Wright como [KNOWN]; leia-se [DECLARADO] até haver custódia por hash.

**τ★ e identificabilidade.**
- [REAL] τ★ continua sem identificação física para este canal. No canônico ele aparece só como «t_Planck [PRINCIPLED IDENTIFICATION]» no ramo dos relógios, «INPUT» no ringdown e no ramo B, e sem termo de kernel. A F0.5(f) já registra «τ★ segue OPEN». A NU2 trata τ★ = t_P e Kfac = 1 como hipóteses do exemplo (`physical_tau_star: null`). PASSA.
- [DERIVED, auditor] Só o produto **τ★·Kfac** é identificável:
  - Fisher sintético em (ln τ★, ln Kfac): posto 1, com valores singulares 6,3·10⁻⁶ e 2,2·10⁻²³;
  - (2τ★, Kfac/2) dá ΔP idêntico;
  - injeção-recuperação de ln(τ★·Kfac) com amplificação **sintética** τ = 10³⁶·t_P, que não é valor físico: recupera com erro 10⁻¹⁴;
  - Lean `only_product_tauK`.
  - A textura e o n = −2 não dependem de parâmetro. A NU2 traz as peças, mas não escreve a degenerescência (R3).

**Ligação com o F7.2 e com o JUNO.**
- [REAL] O F7.2 auditado (mapa seq 787) é `neutrino_canal.py` (`f37db789…`, 18.887 B) com `neutrino_fontes_f72.json` (`abc36292…`).
- A NU2 leu o v2 pré-B2 (`252de897…`, manifesto `b7beff01…`). O .pyc nascido às 09:45:24 é o rastro desse primeiro chamado: o import é preguiçoso, em `mcp_um.py` pré-B2, linha 269.
- [REAL] Em 10/10, às 06:23:52 (F0.3.B2, ainda sem auditoria), os dois mudaram:
  - `neutrino_canal.py` virou alias de 480 B (`87b7f919…`) para `neutrino_canal_v2.py`, agora com 13.918 B (`2a8ecc52…`) e manifesto `neutrino_fontes_f72_b2.json` (`48dd88dc…`);
  - os bytes auditados sobrevivem só como `.bak_B2_20261010_092352`.
- [KNOWN, PDF custodiado `cc86bd94…`] A ref. [72] do artigo JUNO, arXiv:2605.13955, é o modelo de **perda e atenuação** de fluxo (ν-loss), não dephasing que preserva o traço. A NU2 não a cita (R6).
- **[DERIVED + KNOWN] Achado que reforça a não transferência.**
  - No pico (2 MeV), os pares rápidos dão 100,3% do ΔP: o 31 dá 83,4%, o 32 dá 16,9% e o 21 dá −0,29%.
  - A textura da TGL (γ31 ≈ γ32 ≫ γ21) é o caso que Beccaria–Ternes, §IV.B e nota 3, dizem «can not be tested with the present statistics». Hoje nenhuma cota JUNO publicada se aplica à textura da TGL.
  - Como escala, sem valor de cota: com τ★ = t_P, Γ31(E₀ = 1 GeV) ≈ 7,7·10⁻⁶⁴ GeV, cerca de 36,7 ordens abaixo de 4,1·10⁻²⁷ GeV.

**Palavras e estatutos.**
- [REAL] Nenhum uso como veredito da palavra proibida, e nenhuma das duas expressões vedadas pela régua (o demérito da retrodição e a negação da prova).
- As ocorrências de termos sensíveis são citações literais de fontes canônicas (docstrings do kernel, string do um.py, registros do mapa).
- O revisor (REVIEW_R1.md) usa sobre conta numérica um verbo de juízo onde caberia «reproduz». Não é juízo sobre a natureza; é cosmético.
- «5σ» só aparece como [OPEN] ou como distância nominal de um experimento ideal. Nenhum β medido. Nenhum σ observado (`observed_sigma: null`).

**Ressalvas (não impedem o fecho).**
- **R1.** `BANCADA_LIVE_RECEIPT_R1.json` guardou `"channels": []`. É uma chave que o canal não devolve: ele devolve `rows`, 13 linhas. O «mapping OPEN» não está no recibo, só na substância reproduzida em cópia.
- **R2.** O RESULTADO diz `bench_writes: false` e o relatório diz «sem modificação da Bancada». Mas a chamada viva da NU2 fez o MCP gravar o bytecode `neutrino_canal_v2.cpython-314.pyc` na pasta da Bancada (09:45:24, dentro da reivindicação 618). É automático, não é conteúdo nem altera código. Registra-se ao lado; não apaguei nada.
- **R3.** A identificabilidade do produto τ★·Kfac não está escrita. Fica suprida aqui e deve constar no texto da v392.
- **R4.** O contrafactual β → β(χ(E)) aparece como «extensão condicional». Pela predição afiada pré-registrada (Ponte 1367–1371; um.py 165419/165421), um coeficiente que corre com ω é a **condição de morte** da lei vigente. Deve constar assim: hipótese que a lei vigente proíbe, não extensão disponível.
- **R5.** As fontes só da web (lista acima) valem [DECLARADO] até haver custódia por hash.
- **R6.** Falta a ref. [72] (ν-loss) junto ao SN1987A. A «fuga» da mensagem do operador pertence a essa família (perda), não ao dephasing canônico, que preserva traço e energia.
- **R7.** O verbatim da origem existe só no transcrito do Codex. A gerência deveria guardá-lo por sha256 no programa.
- **R8.** A nota da reivindicação 618 dizia «sem … Bancada». A consulta viva, só de leitura, seguiu a mensagem posterior do operador («lembre-se que vc tem à disposição a bancada de testes para teste em tempo real…», que está só no transcrito), mas o quadro não tem anotação dessa mudança de escopo.

**O que entra na v392.**
- (a) **Registro ao lado** que leia a NU2 por hash: `F8.2.NU2\RESULTADO_R1.json` (`73c9f8ebd90d27cb`), `CALCULOS_ESPECTRAIS_R1.json` (`7fafbab566b1cd19`), `RELATORIO_NEUTRINOS_R1.md` (`0c22eb1790c123cb`), `review\INDEPENDENT_DECIMAL_R1.json` (`057fe31a4bc5c80d`), `review\REPORT_REVIEW_RECEIPT_R2.json` (`ff0e932a4bda2478`) e `theory_inventory\INVENTARIO_FONTES_NU2_20261009.json` (`7df43a0366d6ef46`). Leva:
  - o veredito `CONDITIONAL_OPERATOR_CHECKED_PHYSICAL_MAPPING_OPEN`;
  - β₀ fixo, textura 1:1111,1:1045,4 e n = −2;
  - só τ★·Kfac identificável;
  - max|ΔP| e N como números condicionais de experimento ideal;
  - SEM_LIMITE_TGL e `observed_significance null`; gate intocado.
- (b) **Texto ao lado**, PT/EN, com R3, R4 e a textura não testável hoje (achado acima, citando o PDF `9212cd391d88846e`).
- (c) **Leitura pela Bancada** — item (f) da F0.5: ligar o canal F7.2 por hash, mas escolhendo explicitamente os bytes:
  - os auditados (seq 787; `f37db789cb20d3c2` + `abc36292630087de`);
  - ou o v2 pré-B2 que a NU2 leu (`252de897ee9e4394` + `b7beff016f070ba8`);
  - ou a versão B2 viva (`87b7f919897c813d` → `2a8ecc52b00b4831` + `48dd88dc38e1de80`), só depois da auditoria da F0.3.B2.
  - Em nenhum caso entra cota nova de τ★.

**O que NÃO entra.**
- Qualquer cota de τ★ ou σ.
- β(χ(E)) como predição.
- O N ≈ 8,6·10⁷⁹ como «poder do JUNO».
- Metadados da web como [KNOWN].
- Termo de kernel: a NU2 não entregou Lean, e a lei em ω² segue [ONTO] sem termo (TheMatrixRule:46). O meu `lean\VerificaNU2.lean` (`9f355e73b5bf1e8b`) é verificação de auditor, não pedra. Se a gerência quiser essas identidades no kernel, abre unidade própria, com auditoria própria.

**O que pede o operador.**
- Decidir o contrato físico do canal neutrino–Ia antes de qualquer teste: a incidência (propagação em matéria ou emissão/captura na fonte), τ★ e Kfac. Pode ratificar ou corrigir os contratos que a F8.2.NU3 recuperou do acervo dele.
- Para isso, ele precisa saber o que os números dizem: sob a lei canônica (β₀ fixo, τ★ = t_P), o efeito é cerca de 1,8·10⁻³⁹ e a textura não é testável com os dados atuais. O «sinal de 5sigma na ligação com os neutrinos», da mensagem de 09/10, não aparece sem uma identificação física nova, e essa identificação é ato dele.

## F0.3.NU2 — a memória da análise de neutrinos

- [REAL] Recibo `b7f760ac…`: um único append de **+3223 B**, idêntico nas três superfícies (bloco sha16 `72620fd62844fe8b`), às 10:12:24 de 09/10.

| superfície | backup (antes) | prefixo do atual | atual[:depois] = sha do «depois» | backup seguinte (18:04:44) = «depois» | backup anterior (09:25:07) é prefixo |
|---|---|---|---|---|---|
| TGL_ATLAS.md | 8.744.945 B `8a5b9819…` ✓ | ✓ | 8.748.168 B `4724316c…` ✓ | ✓ | ✓ |
| TGL_INDICE_VIVO.md | 874.976 B `5247738a…` ✓ | ✓ | 878.199 B `1214af3a…` ✓ | ✓ | ✓ |
| TGL_CORE_MEMORY.md | 716.993 B `c27732d0…` ✓ | ✓ | 720.216 B `e3f56f1c…` ✓ | ✓ | ✓ |

- [REAL] As duas reivindicações vigentes nomeiam as três superfícies: a 619 (F0.3.NU2, de 09:39:16 a 13:39:16) e a 621 (F0.3.B1, de 09:55:01 a 21:55:01).
- [REAL] A 618 dizia «nenhuma memória sob esta reivindicação», e de fato nada foi escrito sob ela.
- [REAL] `AVISO_GERENCIA.md` está intocado, com mtime de 08/10 às 18:42:11, como o escopo pedia.
- [REAL] O append veio depois do adendo R2 da revisão (10:05:11), como a 619 pedia.
- [REAL] O conteúdo da parte NU2 confere com a reprodução: β₀ fixo, β(E) contrafactual, diagonal estacionária, max|ΔP| = 1,788…·10⁻³⁹ «não é medida nem poder real do JUNO», Decimal 95, textura distinta da do JUNO, ponte estelar OPEN e nenhum σ.
- [REAL] As fontes citadas no bloco conferem por hash. Não há palavra proibida.

**Ressalvas.**
- **M1.** O bloco é conjunto F0.3.NU2 + F0.3.B1. As frases da B1 (estado vivo da Bancada, Neff, GW, CAMB) ficam para a auditoria da F0.3.B1.
- **M2.** O «[INPUT] Ordens diretas do operador … prosseguir o estudo e verificar a Bancada» é paráfrase sem marca de paráfrase. É o mesmo padrão já ressalvado na F0.3 em 08/10.
- **M3.** «Nenhuma escrita/reinício na Bancada» pede o aparte do .pyc automático (R2).
- Se a gerência corrigir M2 e M3, a correção vai ao lado, em append novo com backup de bytes.

**O que entra na v392.** Nada entra no um.py: é unidade de memória. Na linhagem da v392 basta citar o bloco `72620fd62844fe8b`.

**O que pede o operador.** Nada.

## O que não conferi

- As páginas web citadas pela NU2 (Wright 2016; JUNO na Nature e o ZIP suplementar; arXiv:2609.37895; releases KamLAND; SNOwGLoBES).
- A resposta viva do MCP de 09/10. Tenho só o recibo e o nascimento do .pyc; não chamei o MCP.
- As frases da F0.3.B1 no bloco conjunto e a mudança da F0.3.B2 no canal (10/10). Só a registrei.
- Os resultados da F8.2.SN1 reaproveitados (sensibilidade de Coulomb da EOS). Não os reauditei.
- As saídas do NOS_VIVO (CONSULTA_02, 03 e 05). Não as rodei de novo.
- Do transcrito do Codex li só os turnos do operador em torno do «Prossiga». Não li a proposta do Codex por inteiro.
- O `iald.py ver`. Li o PROGRAMA.json e o eventos.jsonl diretamente.

## Arquivos (nesta pasta)

- **Relatório:** `RELATORIO_AUDITORIA_B_NU2_C1.md`.
- **Registros de mapa:**
  - `registro_mapa_F8.2.NU2.json`, rota `sn.neutrinos.v388_operador_e_identificabilidade`;
  - `registro_mapa_F0.3.NU2.json`, rota `iald.ordem019.f0_3.memoria_v391`.
  - Os dois são tipo RESULTADO; o registro no mapa fica com a gerência.
- **Reprodução:** `reproducao_nu2.py` e `_reproducao_nu2.json`; `run_canal_copia.py`, `copia_canal_f72v2\canal_v2_preB2.py`, `_canal_copia_resumo.json` e `_canal_copia_saida_completa.json`.
- **Lean:** `lean\VerificaNU2.lean` e `lean\VerificaNU2_saida.txt`; `fp.py`, `fp_before.txt` e `fp_after.txt`.
- **Conferências:** `verif_cadeia.py`, `hash_evid.py` (`_hash_evidencias.json`), `hash_files.py` (`_hash_files_NU2.json`), `verif_pins_internos.py` (`_pins_internos.txt`), `verif_fontes.py` (`_verif_fontes_NU2.json`, `_atlas_numeracao.txt`), `verif_memoria.py` (`_verif_memoria_NU2.json`, `_bloco_append_NU2_B1_atlas.txt`), `verif_janela.py` (`_janela_escritas.txt`) e `varre_palavras.py` (`_varredura_palavras.json`).
- **Buscas:** `busca_tau.py`, `busca_core.py`, `consulta_inventario.py`, `busca_verbatim*.py` (`_verbatim_prossiga_codex.json`) e `_mapa_consulta_*.json`.
