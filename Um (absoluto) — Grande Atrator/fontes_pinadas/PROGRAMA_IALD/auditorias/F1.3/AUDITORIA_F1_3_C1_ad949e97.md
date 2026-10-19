# Auditoria cética c1 do complemento F1.3 v3.1 (08/10/2026)

## Veredito: **APROVA_COM_RESSALVAS — a F1.3 ainda não pode FECHAR**

- **Atendidos:** pedidos 1, 4, 5 e 6.
- **Parcial:** pedido 3.
- **Não atendido:** pedido 2. O complemento não escreveu a saída do D1a.3 como enunciado Lean e justificou que «a segunda camada só pode ser tipada após construir». Compilei no sandbox e a justificativa não se sustenta: o enunciado tipa hoje (detalhe abaixo).
- **Integridade sem defeito:** é acréscimo puro, os hashes conferem, não há palavras proibidas, a memória foi só acrescentada e o gate está intocado.

**Consultas feitas antes:**
- `rotas.py ver`: 8 registros (740 a 830); o 830 é a anotação da v3.1 feita pelo Codex.
- `iald.py ver F1.3`: unidade AUDITADA; reivindicação do Codex até 08:45:45; o evento 294 aparece no histórico.

## Hashes e aditividade [REAL, por script]

- Arquivos novos da v3.1:
  - `COMPLEMENTO_V3_1…md`: `3b4986c319f77bc1`, 9966 B;
  - `RESULTADO_V3_1…json`: `5263f4aac4063173`;
  - `MAPA_F13_V3_1…json`: `3a140217a7ceb629`;
  - `append_memoria_v3_1.py`: `e094f4b7a6fd9241`.
- Os hashes batem com o mapa (seq 830), o quadro (evento 294) e a entrada de memória.
- Os **26 arquivos anteriores** são iguais em bytes à lista da reauditoria c1. Só há 4 arquivos novos; a v3 (`599f7bcc…`) está intacta.
- O relatório citado da c1 confere: `89979aec…`, 9490 B.
- O pin confere: Lean v4.31.0 e Mathlib `fabf563a…`, no manifest, no toolchain e no HEAD do git.
- O gate não se moveu:
  - `um.py` `8e7b9927ceae64b8`, core `a74241ff380b1b0e` e selo `95061f18c7102248` estão inalterados;
  - o veredito do gate é o mesmo;
  - nenhum `.lean` do kernel foi alterado depois de 08/10 01:00.

## Os seis pedidos

| # | Situação | Fonte e linha |
|---|---|---|
| 1 | **atendido** | A tabela D0/D0.5–D7 tem 37 referências (35 únicas). Todas existem, estão no intervalo e o conteúdo bate com o que a tabela diz, salvo uma (ressalva menor abaixo). A parede de Fubini L² vetorial entra no D1a.1 como `[OPEN]` (COMPLEMENTO:17), não com a palavra «FALTA», mas é equivalente. A ausência no pin se mantém por busca ampla: só `MemLp.comp_fst/comp_snd` (`LpSeminorm/Prod.lean:26,34`); `WithLp`/`PiLp` são produtos finitos; `FourierTransform.lean:202` está dentro de `integral_fourierIntegral_swap`, que é Fubini de integrais. |
| 2 | **não atendido** | COMPLEMENTO:30 deixa a saída em prosa. Aceitável: o `∃ξ≠0` é rebaixado a consequência, citando Nontrivial H1, que era uma das saídas que a c1 permitia. |
| 3 | **parcial** | COMPLEMENTO:32 propõe `LightNetStandardWedgeData` em prosa: 2 dos 3 campos têm tipo; o «certificado modular derivado do Tomita» fica sem tipo. A proposta não está na fila do operador (`PROGRAMA.json` → `pendencias_externas` X2–X18; `PENDENCIAS_DO_OPERADOR.md` de 06:07). Quem põe item ali, pelo histórico, é a gerência. O texto se contradiz: marca «[INPUT — decisão do operador]» e logo depois diz «a c1 deve decidir a forma final dos campos». |
| 4 | **atendido** | As linhas 42/73/98/114/126/163 de `GenericAntilinearAdjoint.lean` conferem. Instanciei `generic_adjoint_pairing` em H1 no sandbox, no trio. |
| 5 | **atendido** | EO 1973 confere no Crossref: J. Funct. Anal. 13(1), 1–12, pii 0022123673900621. BGL v2 imprime «1-22» na ref. [15] e, na [33], cita LRT «Duality for the free electromagnetic field, Marseille preprint 1976, unpublished». Ann. SNS (5) XVII (2017) 113–141 cita LRT «Duality for quantum free fields», CPT-78/P-1016, 1978. `doi.org/10.1063/1.1703912` resolve para J. Math. Phys. 4(11), 1343. A divergência 1976/1978 está dita honestamente como [DECLARADO]. |
| 6 | **atendido** | Inventário `e448eae5`: a linha 57 é `IsotonyImpliesStripMultiplierCriterion` (K_w com U(a)K_w⊆K_w); as linhas 61–66 têm exatamente os seis nomes; a família curva ocupa as linhas 72–75. |

## O achado principal: a saída do D1a.3 tipa hoje [REAL, sandbox]

Escrevi duas formas do enunciado. Ambas compilam com EXIT 0; as 19 declarações ficam no trio [propext, Classical.choice, Quot.sound], sem `sorryAx`. A forma BGL, `D1a3OutputBGL`, é a mais fiel ao plano da v3:
- **Campos ainda abertos (D1a.1–D1a.2):** `Fub` (a ponte de Fubini L²), `J` (J_W, involutiva), `T` (Δ_W^{1/2}, auto-adjunto), `σ` (o símbolo), e as condições de involução fechada e densa de S_W = `genericTomita T J`.
- **K(W_R) é construído, não é campo livre:** sai de `closedAntilinearStandardSubspace`, e `mem_KWR` vale por `Iff.rfl`.
- **S_W é o Tomita de K:** h+ik ↦ h−ik, por `fixed_sum_tomita`.
- **Δ_K := S_W†S_W** vem pela caracterização do grafo no padrão das linhas 87–112, e Δ fica único (`tomitaSquare_unique`).
- **Δ_K^{it} = B0(−2πt)** vem do multiplicador e^{itσ} depois de `partialFourier Fub` = `Lp.fourierTransformₗᵢ ∘ Fub ∘ toRapidity`.
- **O `∃ξ≠0` é consequência:** `exists_nonzero`, com `H1_nontrivial` reproduzido no trio.
- **A interface com três campos tipados** (`LightNetStandardWedgeDataBGL`) também compila.

Logo, tipar não exige construir: as peças abertas entram como campos, e o próprio complemento admite no pedido 4 que o Tomita e o seu adjunto genéricos servem a H1. As sondas só **tipam** o enunciado; não constroem o D1a nem mostram que ele é habitado [OPEN]. É proposta minha [DERIVED], não pedra. Como quem constrói não aprova, a adoção cabe ao Codex ou à gerência, e a reauditoria a outro auditor.

## Ressalvas menores
- `H1_nontrivial` é atribuído a `F13v3_probe.lean:50–96`, mas está em `F13v3_H1_nontrivial.lean:16`.
- `VonNeumannAlgebra/Basic.lean:23` vem sem o prefixo `Mathlib/Analysis/`; o conteúdo confere.
- Três itens da c1 §3 ficaram só por remissão: Stone, o TODO de Tomita (`StandardSubspace.lean:42`) e `eigenvectorBasis` (`Spectrum.lean:34`).
- A bibliografia de 2017 que cita o preprint LRT 1978 (Crismale–Fidaleo–Lu) não é nomeada, nem a cópia do preprint no INIS.
- A «validação» das 35 referências conferiu só existência e intervalo, não o conteúdo.
- A entrada de memória e o registro seq 830 não dizem que o pedido 2 ficou sem enunciado. Convém pôr uma anotação ao lado ao registrar esta auditoria.

## Memória [REAL]
- As 4 superfícies (`TGL_ATLAS`, `TGL_INDICE_VIVO`, `TGL_CORE_MEMORY`, `AVISO_GERENCIA`) receberam o acréscimo às 06:29:05.
- Em cada uma, o backup `.bak_20261008_062905_*` é prefixo exato do arquivo atual, a entrada de 1793 B vem logo depois e o marcador aparece uma vez.
- A escrita cai dentro da reivindicação da F1.3.
- A varredura de palavras proibidas e de β literal deu zero ocorrências nos 4 arquivos novos.

## O que não conferi
- Não abri o preprint LRT 1978 (INIS) nem a página da AIP. O resumo do Crossref passou por um modelo auxiliar [DECLARADO].
- Não construí o D1a nem verifiquei se o enunciado é habitado.
- Não recompilei o kernel; usei os .olean com LEAN_PATH só leitura.
- Não refiz a tabela §3 da c1 inteira: só a parede de Fubini e amostras de ausência (Reeh–Schlieder, BW, polar, Stone de grupos unitários e bicomutante ausentes; KMS e Tomita só na docstring/TODO).
- Não abri o `tasks.sqlite3`; não rodei `um.py` nem `rotas.py registrar`; não chamei nenhum provedor.

## Registro para o mapa e artefatos (todos no Temp; copiar para pasta persistente antes de registrar)

O registro está em `C:\Users\rotol\AppData\Local\Temp\claude\c--IALD-Central-de-Patentes\6da8f00d-d44f-4888-a88d-fc9f73eead3d\scratchpad\audit_f13_v31\registro_mapa_f13_v31.json` (sha16 `7ae0e349cbec0303`):
- JSON válido, com todas as chaves do modelo; tipo RESULTADO na rota `formal.f1_3.reconcile_v376_mathlib_plan`.
- Status EXAMINADA_COM_DEFEITO (escolha minha; a gerência pode mudar).
- Zero ocorrências de «confirm»; 19 artefatos como lista de caminhos, todos existentes.

Os artefatos ficam na mesma pasta:
- `sandbox\F13v31_D1a3_bgl.lean` (`e4da9e865d00bdcb`), saída `551efbbb4210041e`;
- `sandbox\F13v31_D1a3_statement.lean` (`0337a6d7fe80b597`), saída `9b14053d381bbadc`;
- `sandbox\run_lean.py` e `lean-toolchain`, idênticos aos originais;
- os scripts de conferência `refcheck_v31`, `memcheck_v31`, `integridade_v31`, `invcheck_v31`, `estado_kernel_v31` e `wordscan_v31`, cada um com o seu `.out.txt`;
- `web\refs_web_v31.out.txt`, com os dois PDFs públicos (BGL v2 `10e51f87…`; Ann. SNS 2017 `157ece63…`).

**Próximo passo sugerido:** o Codex escreve uma v3.2 de um parágrafo, sem provedor, adotando (ou contestando com razão tipada) o `D1a3OutputBGL` como saída do D1a.3 e o `LightNetStandardWedgeDataBGL` como texto do item para a fila do operador. A gerência põe o item em `pendencias_externas` e manda reauditar só esses dois pontos antes de decidir FECHADA. A construção começa no D1a.1, em unidade própria: primeiro a ponte de Fubini L² vetorial, depois o Fourier parcial.