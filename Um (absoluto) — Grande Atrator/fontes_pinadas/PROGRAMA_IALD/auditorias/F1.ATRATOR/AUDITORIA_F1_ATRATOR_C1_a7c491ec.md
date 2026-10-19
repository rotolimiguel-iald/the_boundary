AUDITOR CÉTICO c1: F1.ATRATOR

Escopo da auditoria: os eventos 664–671, a pasta do túnel ORDEM_019\F1.ATRATOR (R1 e ADENDO1) e as seq 1000–1003 do mapa. A consulta `rotas.py consultar "atrator"` foi feita; as seq 999–1004 foram lidas no rotas.jsonl. Escrevi só no scratchpad `...\scratchpad\audit_atrator_10out`. Não rodei o um.py nem scripts de selo, e não escrevi no mapa, no quadro, na memória, no túnel, no Nós ou na Bancada.

**1. Pré-registro antes dos ensaios: PASSA**
- PREREGISTRO_R1.json: sha256 db52519d… confere com o FREEZE. Congelado às 10:08:19 local (13:08:19 UTC). Os primeiros artefatos vêm depois: tentativas Lean a partir das 10:11:21, numérico a partir das 10:11:32, ponte ótica a partir das 10:19:44.
- ADENDO1: PREREGISTRO_ADENDO1_R1.json (724b3e7b…) e .md (152c7d14…) conferem com o FREEZE das 10:48:25. O numérico do ADENDO1 começa às 10:52:53 e as tentativas Lean às 10:54:33.
- O mapeamento de T, G e L do A6 está escrito no pré-registro, com os casos admissíveis e os inadmissíveis. O pré-registro declara de antemão «provável A6=FALHA».

**2. Cunhagens verbatim: PASSA no ADENDO1, ausente na R1**
- Os sha256 das fontes, calculados por script:
  - V1 1d9dd9a6… (1658 bytes)
  - V2 87a3752d… (2131 bytes)
  - V3 a9cecc12… (573 bytes)
  - V4 f2a4b509946edc05… (267 bytes), localizada em `C:\IALD\PROGRAMA_IALD\OPERADOR_07out_leitura_cria_o_objeto_verbatim.txt`
- As quatro aparecem byte a byte no RELATORIO_ADENDO1_R1.md, no PREREGISTRO_ADENDO1_R1.md e nas duas .lean do ADENDO1 (TheSingularityReadsTheAttractorAdendo1.lean e TheLightReadsItself.lean). Os byte_range do FREEZE batem com o sha256 de cada trecho.
- A R1 (relatório, pré-registro e .lean) não trazia nenhuma das quatro. Foi esse defeito que o ADENDO1 sanou.
- A interpretação técnica vem ao lado, marcada [INPUT], [REAL/DERIVED] ou [ONTO].

**3. Ensaio ótico: PASSA**
- Rodei run_optical_r3.py e run_adendo1_r1.py numa cópia. Resultados: `all_expected_outcomes_met=True`, A5 PASSA, A6 FALHA com 7 sobreviventes.
- β foi calculado em runtime a partir da atribuição ALPHA_FINE_CODATA_2018, lida do um.py como bytes, sem executá-lo: 0.012031300400803142. Bate com o meu cálculo independente. Nenhum literal 0.01203 aparece nos runners nem nas reproduções.
- Minha checagem independente (T=|S|²):
  - A1: ‖T⁴⁰⁹⁶−R‖ = 2.7e-13
  - A2: resíduo ~1e-16
  - A3: R²=R exato
  - A4: dim Fix = 1
- Os controles falham onde têm de falhar:
  - identidade, sem a hipótese: dim Fix = 2, logo A4 falha;
  - x↦x+1: x = 4096 após 4096 passos, logo A1 falha.
- A6: com T=I, A1–A3 continuam valendo, e isso basta como contraexemplo à minimalidade. Portanto a retirada **não** derruba A1–A3. É FALHA honesta e pré-anunciada do falsificador.
- Regra dos dois regimes respeitada: a convergência fica só na face dos estados, e θ é leitura fixa, sem limite pedido.

**4. Kernel: PASSA, com ressalvas de peso**
- Compilei as 4 pedras em sandbox (toolchain v4.31.0 copiado; LEAN_PATH só-leitura no .lake do Nós):
  - TheSingularityReadsTheAttractor: EXIT0
  - OpticalPopulationBridgeR2: EXIT0
  - TheSingularityReadsTheAttractorAdendo1: EXIT0
  - TheLightReadsItself: EXIT0
- 14 + 15 + 14 + 17 = 60 `#print axioms`, todos só com [propext, Classical.choice, Quot.sound]. Zero `sorry` nas fontes. Os únicos avisos são de depreciação.
- Os 4 .olean saíram idênticos byte a byte aos da entrega: d8fdec78, 8001779, 1627cd0d, 242d5b37.
- Impressão digital do tgl_kernel antes e depois: 132088 arquivos, a8b55493…, igual.
- Vacuidade: o habitante concreto existe (`concrete_hypothesis_inhabited`: e=(1,0), λ=1/2 em ℂ²). StrictComplementContraction é uma hipótese real (e≠0, Te=e, c<1 no complemento). Há também o contraexemplo com dimensão 2.
- A5 tem conteúdo, mas pouco. As provas usam de fato `J_squared_is_one` e `JKJ_eq_neg_K` do kernel. O resto é álgebra leve: E_J²=E_J e E_J(J)=J decorrem diretamente de J²=I.
- `A5_angular_rereading_only` (e `A3_angular_rereading_only`) apenas reexportam `observing_adds_nothing`, sem acrescentar nada. Veredito: **nome sem peso**.
- Reaproveitamento:
  - **PALAVRA:** a letra vem de `TGLExt.exact_witness_of_annihilating_word` e `annihilating_word_exists` (ExactWitness/WordExistence). O módulo `TGLExt.Palavra` não está instalado e não foi importado; a entrega diz isso.
  - **Atrator único do GKLS v384:** a ordem pedia para reaproveitá-lo, e ele **não** foi reaproveitado em Lean, só relido nos resultados arquivados. O A4 usa contração estrita no complemento. É um desvio declarado, e o mapa já registra que o kernel não tem teorema de unicidade do GKLS.

**5. Escopo: PASSA**
- um.py, um_absoluto.json e o selo seguem com mtime de 07/10.
- Nenhum arquivo modificado depois das 10:01 em `C:\IALD\Bancada_Um`, em `C:\IALD\Artigo\BANCADA_TOE` ou no Nós (exceto .lake, que a impressão digital cobre). Nada no gate.
- Memória: TGL_ATLAS, TGL_INDICE_VIVO e TGL_CORE_MEMORY receberam append puro em 10:30:50 (reivindicação 664) e em 11:05:37 (reivindicação 668).
  - Os backups `<arquivo>.bak_20261010_103050` e `.bak_20261010_110537` existem.
  - O sha256 e o tamanho de cada backup batem com o estado `before`, e o arquivo atual começa com os bytes do backup.
- A superfície ACOM (`C:\IALD\MEMORIA_IALD\state\memory.sqlite3`) tem `memory.sqlite3.bak_<timestamp>` a cada transição.
- Palavras proibidas: CONFIRMED só aparece no MAPA_LEVANTAMENTO.json, citando a proibição («vereditos proibidos CONFIRMED/PROVED»). Não há «pós-dição» nem «não provou». «confirmado» aparece só em «Reaproveitamento confirmado no mapa», no sentido corriqueiro.

**Veredito item a item (conferido por mim)**
- A1: PASSA
- A2: PASSA
- A3: PASSA
- A4: PASSA CONDICIONAL à hipótese nomeada; sem ela, FALHA como esperado
- A5: PASSA no mapeamento declarado; peso algébrico leve, e a releitura angular é nome sem peso
- A6: FALHA da minimalidade, como pré-registrado

**Pode FECHAR?** Sim, no escopo matemático sintético. Validade na face testada; não é creditação. A FALHA do A6 é um resultado honesto, não um bloqueio.

Duas ressalvas a constar no fechamento:
1. A4 não foi ligado ao atrator do GKLS v384.
2. A5 e A3_angular não podem ser lidos como testemunho físico nem ontológico.

**O que não conferi**
- Os runners e as revisões internas da entrega, linha a linha.
- A integridade das 106 evidências da R1 que o ADENDO1 diz ter reconferido.
- O conteúdo semântico dos appends no Atlas, no Índice Vivo e no Core: conferi só bytes e backups.
- O conteúdo do sqlite da ACOM.
- Os mtimes dentro do .lake do Nós no intervalo entre 10:01 e a minha primeira impressão digital.

**Arquivos (scratchpad)**
- O registro pedido: `C:\Users\rotol\AppData\Local\Temp\claude\c--IALD-Central-de-Patentes\6da8f00d-d44f-4888-a88d-fc9f73eead3d\scratchpad\audit_atrator_10out\registro_mapa_F1.ATRATOR.json`
  - rota_id `formal.singularidade_le_atrator`, tipo RESULTADO, sem CONFIRMED
  - artefatos com sha256 por script
- Logs de compilação em `...\audit_atrator_10out\lean\*.log`.
- Toolchain, .olean, .ilean e caches foram apagados; nenhum .lake foi criado.