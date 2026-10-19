AUDITOR CÉTICO c1: F8.2.SN1 e F0.3.SN1

12/10/2026 · subagente auditor da gerência c1 (exame geral de 12/10, referência-mãe `claude-c1_exame-geral-12out`) · pasta única de escrita: `scratchpad\exame12\A_SN`.

## 0. Vereditos em uma linha

| Unidade | Veredito | Pode fechar |
|---|---|---|
| **F8.2.SN1** (SN Ia e Chandrasekhar: viabilidade com a escala v388 e o setor eletrofraco) | **APROVA_COM_RESSALVAS**: as contas condicionais se reproduzem por script e nenhum dado de SN foi aberto. A atribuição histórica do resíduo de √2 à correção de Coulomb não se sustenta, e o relatório não fez esse confronto. | **sim** |
| **F0.3.SN1** (memória dessa viabilidade) | **APROVA**: acréscimo puro nas três superfícies nomeadas, com backup de bytes que é prefixo exato do arquivo atual. AVISO_GERENCIA ficou intocado. | **sim** |

Nada aqui é teste da natureza. Não há σ físico, o gate está intacto e nenhum veredito de natureza foi emitido.

## 1. Restrições cumpridas (como foram medidas)

- [REAL] Toda a escrita ficou em `A_SN`: scripts, saídas, a cópia de reexecução e a memória sandbox. O quadro, o mapa, o túnel, o Nós, o kernel, a Bancada, o site, o git e a memória de produção ficaram intocados. O `rotas.py` foi usado só com `consultar`, `ver`, `verificar` e `modelo`, que são leituras. O `session_memory.py` de produção foi usado só com `project`, que é leitura, sobre a referência-mãe. A inscrição desta tarefa foi para um store sandbox (`--store A_SN\memoria_sandbox\state`, referência `A_SN\memoria_sandbox\REFERENCIA_A_SN.json`), por causa da restrição de escrita. Os backups de produção de 11:18, 11:22, 11:25 e 11:26 de 12/10 são de outras sessões: leitura não chama `_backup`.
- [REAL] Não rodei o `um.py`, o `atlas_indice.py`, o `arvore_da_prova.py`, o `porta_memoria.py` nem scripts de selo. O `um.py` foi lido por `grep`/`sed` com número de linha.
- [REAL] Não usei Lean. As unidades não produziram nenhum artefato Lean. A pedra v388 foi conferida pela leitura do core e do selo (§2 e §3). Impressão digital do `tgl_kernel` (o mesmo método das auditorias de 08/10), antes e depois: idênticas (§9).
- [REAL] Não abri nenhum dado observado e não chamei a Bancada nem nenhum provedor. As contas usaram só entradas sintéticas e constantes CODATA ou da literatura [KNOWN]. β foi calculado em runtime como `ALPHA_FINE_CODATA_2018·√e`: deu 0,012031300400803142, igual bit a bit à conferência.

## 2. O cético aferidor: consultas citadas

- Mapa (`rotas.py consultar`, cabeça `91915ad39a53d1df`, 1004 registros, 202 rotas, cadeia ok):
  - «Chandrasekhar» devolve 4 rotas, entre elas `outro.grande_atrator_e_chandrasekhar` (seq 164, PROPOSTA) e `sn.chandrasekhar.v388_eletrofraco_viabilidade` (seq 980 ROTA e 981 RESULTADO, do Codex).
  - «Coulomb» e «Fresnel» devolvem **0 rotas**: a atribuição histórica do resíduo nunca foi examinada como rota.
  - «eletrofraco» devolve 2 rotas; «v388/sech/cosh» devolve 21. Saídas em `A_SN\consultas\`.
- Inventário da bancada (`memory\INVENTARIO_BANCADA_CHATGPT.json`, sha16 `05c2afa06620ca8f`, de 27/09): **0** ocorrências de Chandrasekhar, Coulomb, supernova, white dwarf, Fresnel ou sqrt2.
- Kernel (`tgl_kernel\TGLExt`): `grep -il "chandrasekhar|coulomb|supernova|white.dwarf"` devolve **0 arquivos**.
- Core e selo do Nós, lidos por script:
  - `um_absoluto.json` (sha16 `a74241ff380b1b0e`, 5.003.275 B): `core.kernel_formalization.ext_cpl_*` tem as **17 de 17 True**.
  - `um_absoluto_selo.json` (sha16 `95061f18c7102248`, `um_version` v391): `beta_runs_bounded_below_v388.verdict = TGL_BETA_RUNS_BOUNDED_BELOW_V388__…__BETA0_0P01203130__BETA_MZ_ON_SHELL_0P012786_MSBAR_0P012886__…__KERNEL_17_OF_17__…__WHICH_MU_PER_RITE_OPEN__GATE_UNTOUCHED__…`, com `all_verified = true`.

## 3. A escala v388: a fórmula e de onde ela vem, com linha

β(χ) = √e/cosh(χ/2) = α(χ)·√e, com α(χ) = sech(χ/2).

| Fonte | Linha | O que diz |
|---|---|---|
| `tgl_kernel\TGLExt\TheCouplingRunsBoundedBelow.lean` (sha16 `702d7352c053060a`, 6.375 B) | 34 / 37 | `alphaOfChi χ := 1 / Real.cosh (χ / 2)` / `betaOfChi χ := alphaOfChi χ * Real.exp (1 / 2)` |
| idem | 52, 80–84, 87–96, 100–102 | monotonia estrita; o piso no supremo de χ; o piso só é atingido lá; `betaOfChi_eq_coupling` (rfl) |
| idem | 24–26 | «β₀ = α_Thomson·√e … NÃO o `alphaOfChi 0` desta pedra, que vale 1» |
| `um.py` **v391** (sha16 `8e7b9927ceae64b8`, 33.910.631 B) | 761 | `6. beta_TGL = sqrt(e) alpha_obs = sqrt(e) sech(kappa/2)` (`prove_alpha_form`; a forma já existia antes da v388) |
| idem | 874 | `ell_beta(kappa) = log cosh(kappa/2) => alpha = sech(kappa/2) ; beta = sqrt(e) sech(kappa/2)` |
| idem | 127738 | a pedra embutida (`betaOfChi …`) |
| idem | 194688 / 194714 | «A constante da FRONTEIRA segue beta0 = alpha(0).sqrt(e) …; beta(chi) e' a leitura de BULK» / `def beta_runs_bounded_below_v388` |
| idem | 5557–5559 | `ALPHA_FINE_CODATA_2018 = 7.2973525693e-3`; `BETA_TGL = ALPHA·ONE·sqrt(e)` |
| `um.py` **v388** (`work\v388_pacote\um_v388_candidato.py`, sha16 `e64b00f4df75059b`, 33.801.035 B, o mesmo do HANDOFF v388) | 761, 874, 127735, 193845, 193871, 5556 | as mesmas peças: 761 e 874 nas mesmas linhas da v391; pedra embutida, comentário da fronteira, `def` e α nas posições da v388 |

- [REAL] As citações do Codex conferem: `TheCouplingRunsBoundedBelow.lean:34–37, 52–88, 100–102`, `um.py:5557`, `um.py:193300–193302` e o Teorema da Escala (Ponte, l.1288–1343).
- [REAL] O recálculo deu χ_QED = 2·arcosh(1/α) = 11,2267550516013. Nesse ponto sech(χ/2) − α = 0 e β(χ_QED) − β₀ = 0. β(0) = √e, e β decresce estritamente na grade de 0 a 20. Não há segundo fator √e.
- [REAL, nota editorial] O «α(0)» do relatório (Thomson, q→0) colide com o `alphaOfChi 0 = 1` da pedra. O próprio `um.py` desfaz a ambiguidade em l.194688 e no campo `reading`. Sem efeito numérico.
- [REAL] A regra X6 do Atlas (l.35120–35129, 07/10) diz que todo rito cobra β₀ e que «qualquer outro α(μ) contrabandearia o parâmetro μ». O Codex respeitou: não escolheu μ para a estrela.

## 4. Eventos, fontes e evidências

- [REAL] Eventos 609–616 lidos. O hash de cada um confere, e a cadeia inteira (684 eventos) está sem quebra (`scripts\chain_check.py`).
  - 609: nova F8.2.SN1. 610: reivindicação, vence 13:04:12. 613: ENTREGUE às 09:25:54. 615: liberação.
  - 611: nova F0.3.SN1. 612: reivindicação nomeando ATLAS, INDICE_VIVO e CORE_MEMORY, «Sem AVISO_GERENCIA», vence 11:21:33. 614: ENTREGUE às 09:26:01. 616: liberação.
  - Não há eventos especiais nem eventos posteriores sobre estas unidades. No `PROGRAMA.json` as duas estão ENTREGUE.
- [REAL] Fontes das unidades:
  - `work\sn_v388_20261009_unit_request.json`: 1.228 B, sha256 `63db0f6a…79b9`, igual ao evento 609.
  - `work\sn_v388_20261009_memory_unit_request.json`: 1.013 B, `a9a25d5b…218e`, igual ao evento 611.
- [REAL] Evidências, conferidas na origem e na cópia em `evidencias\`:
  - `RESULTADO_VIABILIDADE_R1.json`: `976f67ff…9034`, 9.866 B.
  - `MEMORY_RECEIPT_R1.json`: `5a703cf6…5306`, 3.298 B.
  - O manifesto do RESULTADO tem 36 arquivos: **36 de 36** conferem em sha256 e bytes. Fora do manifesto ficam só o próprio RESULTADO e o `MAPA_FINAL_R1.json` (sha16 `ec0f8e949d3e6c38`).
- [REAL] **Origem «ordem direta do operador»: o verbatim não foi guardado em arquivo da casa.** Procurei arquivos *verbatim* e *operador* de 08/10 18h a 10/10 em `work`, no túnel e em `PROGRAMA_IALD`. Só apareceu o da F0.3.B2, que é outro assunto. O verbatim existe nos transcritos do Codex, conferido por script:
  - `~\.codex\sessions\2026\10\09\rollout-2026-10-09T08-43-58-01a1207a-….jsonl`, 11:43:58Z, sha256 da mensagem `71b8edf8ce9d38ea…`: a questão de SN Ia, com «discuta comigo antes» e «como faríamos ele para comunicar, depois ao Claude».
  - `rollout-2026-10-09T09-04-44-01a1208d-27dd-….jsonl`, linhas 10, 12 e 17:
    - `0fdbc824cf7795bf…`: «…não tinha derivado a TGL completamente e nem ainda aplicado o fator de escala em betaTGL… talvez possa corrigir o desvio na flutuação maior, porque não consideramos a energia eletrofraca».
    - `511f38ae6d50e7dd…`: «É isso mesmo é esse fator que vc menciona a relação incorporada na v 388».
    - `6c1475a8967ccb8f`: «Prossiga».
  - A paráfrase da origem confere com esses textos. **Ressalva:** guardar esses textos por sha256 num arquivo da casa, como foi feito na F0.3.B2.
- [REAL] A origem da F0.3.SN1 é a regra permanente do Atlas («toda movimentação de conceito atualiza o Atlas na mesma sessão», CLAUDE.md global).

## 5. Só arquivos novos; a memória

- [REAL] Os 38 arquivos da pasta F8.2.SN1 têm mtime − birthtime ≤ 0,002 s: nenhum foi alterado depois de criado. Na janela de 09:00 a 09:30 de 09/10, fora da pasta, só apareceram as duas fontes novas, os backups do mapa e do quadro, as duas cópias de evidência e os três backups de memória. Nada mudou no Nós, na Bancada, no Nós Vivo, no Tratado, no Haja_Luz, na `MEMORIA_IALD` ou na memória do Claude e do Codex.
- [REAL] Os backups do mapa (`rotas.jsonl.bak_20261009_090940` e `_092550`) são prefixo exato do livro atual. Os seqs 980 e 981 são do Codex.
- [REAL] **As três superfícies** (`scripts\memory_check.py`, saída em `memory_check_out.json`):

| Superfície | backup (bak_20261009_122507…) | backup = estado anterior | backup é prefixo exato do atual | trecho acrescentado |
|---|---|---|---|---|
| TGL_ATLAS.md | `43d624bf…`, 8.742.627 B | sim | sim | 2.318 B, `e4b98bf5…`; marcador único na posição 8.742.707 |
| TGL_INDICE_VIVO.md | `0b540cff…`, 872.658 B | sim | sim | idem |
| TGL_CORE_MEMORY.md | `d875d1f6…`, 714.675 B | sim | sim | idem |

  - O hash do estado logo depois da escrita (`after`) confere como prefixo do atual. Os arquivos cresceram depois disso por outras unidades, o que é legítimo.
  - A escrita foi às 09:25:07, dentro da reivindicação 612.
  - **AVISO_GERENCIA.md** tem mtime de 08/10 18:42:11, sem escrita em 09/10. O único «SN1» lá dentro é «SN1987A».
  - O texto acrescentado marca os estatutos, cita hashes que conferem (`ccdd09e4…`, `589a4e4f…`, `fd82956e…`) e não usa a palavra proibida como veredito.

## 6. Refazer as contas (PASSA / FALHA por afirmação)

Fiz duas coisas em `A_SN`:
- (a) Reexecutei uma **cópia** de `calculate_r2.py` com os dois planos: todos os campos são iguais aos de `CALCULOS_R2.json`, tirando os tempos (`rerun_copia\CALCULOS_R2_RERUN.json`).
- (b) Fiz um **recálculo independente** (`RECALCULO_A_SN.json`, sha16 `0b708128e04b886b`): dephasing por integração RK4, poder por Monte Carlo sintético, Chamel eq. 37–38 lidas do HTML custodiado (L01, sha16 `79290774ea944bab`).

| Afirmação do relatório | Resultado |
|---|---|
| β₀ = α√e = 0,012031300400803142, lido de `um.py:5557` sem executar | **PASSA** [REAL] |
| Coulomb (Chamel eq. 37–38, C_M = −0,895929255682): ΔM/M = −0,02284% / −0,11418% (C) e −0,02776% / −0,13876% (O) para α +1% / +5% | **PASSA** [DERIVED; a fórmula é KNOWN e eu a deduzi analiticamente de novo: P_L/P_e = (4/3)C_M α Z^{2/3}(4/9π)^{1/3}] |
| Aumentar α deixa o termo de Coulomb mais negativo, e o termo isolado não compensa | **PASSA** (eq. 38 dá M < M_Ch) |
| M_Ch ∝ Ye² no limite ideal | **PASSA** [KNOWN] |
| Tr(H[H,[H,ρ]]) = 0; energia = 1, populações = 1/3, pureza 1 → 0,50091 → 0,33335 | **PASSA** (RK4 bate com a forma fechada; 20 ρ aleatórios sintéticos dão 0) |
| ε = β₀ℓ_P²R entre 5,87×10⁻⁸⁷ e 5,87×10⁻⁸⁵ (ρ de 10⁸ a 10¹⁰ g/cm³); com √e, 8,0×10⁻⁸⁵ a 8,0×10⁻⁸³ | **PASSA** (com a massa de Planck reduzida o fator seria 8π maior; sem efeito na conclusão) |
| Tratado `secao_01_fundamentos.tex` l.234, 259, 261, 266, 284, 580, 584, 797 | **PASSA** [REAL documental, sha16 `64e9a0ed5fb17c1a`] |
| Errata de 19/09 (ξ = 1/6 conforme, não β) | **PASSA** (cópia sha16 `060c82abee57a5d0`; bate com a memória de 15/09: «nunca fundir os dois coeficientes») |
| N = 2.500 / 3.946 e 250.000 / 394.579; piso de 0,002 (e de 0,0002) impede 5σ | **PASSA** (Monte Carlo: com N = 2.500 a mediana de z é 5,01; com N = 3.946, 89,8% dos casos dão z ≥ 5) |
| Com intercepto livre, Z² = 0 para sinal constante e 0,1 para forma balanceada | **PASSA** |
| Erro de arredondamento da R1 (2.501 / 250.001); a R2 só corrige isso | **PASSA** (o diff dos scripts é só o Decimal e as asserções) |
| Custo, cópia indexada: pergunta em 2585–2596, resposta em 2647–2706, retirada em 2977–2992 e 3315 | **PASSA** (sha16 `559d79fb81b0af6c`) |
| 15 fontes locais, 42 âncoras de linha | **PASSA**: 14 arquivos idênticos; Atlas e Índice cresceram depois, mas as âncoras seguem iguais, 42 de 42 |
| «Sem amplitude estelar TGL ou σ físico» | **PASSA** [REAL]: `physical_*` é null em RESULTADO e CALCULOS («não calculado, não zero») |
| Nenhum dado de SN aberto | **PASSA** [REAL]: ver abaixo |

**Sobre dado de SN.**
- Os scripts da unidade só leem o `um.py` (bytes), os planos, o Tratado, o HTML de Chamel e as próprias saídas. O script de busca lista só páginas de literatura, das quais 7 foram custodiadas e a L06 deu 403.
- Varri as **109 chamadas de ferramenta** do Codex entre 11:40Z e 12:30Z de 09/10 (4 sessões: 53 `exec`, 34 `wait`, 22 `send_message`, todas para `/root`). **Nenhuma** acessou arquivo de dado (.dat, .csv, .npz, .npy, .fits, Pantheon e congêneres), a Bancada ou um provedor da Central. Houve buscas de literatura pela ferramenta web do próprio Codex, que o levantamento declara para a L06. Nenhuma mensagem foi ao Claude.
- Pelo atime: `pantheon_plus.dat` e `lcparam_full_long.txt` têm último acesso em 11/09, e `pantheon_real_processed.npz` em 08/10. `Pantheon+SH0ES.dat` e `pantheon_cov_v1.npy` têm atime de hoje (11:22 e 11:24). Esse acesso é posterior e alheio a esta unidade, e não veio desta auditoria, que não abriu esses arquivos. Para esses dois o indício é indeterminado. Nenhum arquivo de `observaveis_tgl`, `Bancada_Um\dados` ou `Bancada_Um\tabelas` tem atime entre 09:00 e 09:30 de 09/10.

## 7. Achados materiais

**A1 [DERIVED, novo; mapa: «Coulomb» e «Fresnel» com 0 rotas] — a errata histórica de Chandrasekhar.**

O que o acervo diz:
- Custo, l.2249–2250: «0,353% de √2. Este resíduo — de mesma ordem (O(β)) e do sinal correto — é declarado como a correção de Coulomb da rede iônica».
- `tgl_paper_unified.py` (sha16 `29c92b66b2fd6b4d`), l.7574–7577: a mesma afirmação.
- Atlas l.1048: «atrator de saturação de Fresnel (~0,35%)». Também Atlas l.722 e l.19713, `C:\IALD\CLAUDE.md` l.93 e o ledger do `um.py`: l.193150, l.193186 e l.213968, todos com a classe A8.

O que o recálculo mostra:
- Com as constantes do próprio código histórico (l.7586: `m_H = 1.6726e-27`, que é a massa do próton; `M_sun = 1.989e30`) reproduzo M_Ch = 1,43504 M☉ e M_TGL = 1,40922 M☉, a −0,353% de √2. Para chegar a √2 seria preciso uma correção de **+0,354%**.
- A correção de Coulomb pela fórmula que o próprio Codex custodiou (Chamel eq. 38) é **−2,241% (C) e −2,713% (O)**. O sinal é **oposto**, e o módulo é **6,3 a 7,7 vezes maior**. Com Coulomb, a massa projetada cai a −2,59% (C) e −3,06% (O) abaixo de √2.
- O próprio 0,353% depende da unidade de massa:
  - com m_u (CODATA 2018), que é a massa por elétron correta no ¹²C, M_Ch sem Coulomb dá 1,45626 M☉ e o desvio de √2 vira **+1,12%**;
  - com a massa do átomo de H, o desvio é −0,447%.
- **Leitura:** a atribuição «mesma ordem e sinal correto» não se sustenta. Isso reforça o negativo honesto que o acervo já tinha: Custo l.2673, «√2 **não é atrator independente**», e o A8 do ledger.
- **Ao relatório da F8.2.SN1:** a CALCULOS_R2 já trazia `mass_ratio_to_ideal` = 0,97759 / 0,97287, mas o relatório não confrontou o número com a frase histórica. O reuse note só diz que a atribuição era um comentário, o que é correto.

**A2 [REAL; consulta: `grep` em RELATORIO, LEVANTAMENTO, REVIEW e REUSE] — a regra canônica tipo-I não aparece na «pergunta aperfeiçoada».**

O cânone diz que interiores de anã branca são tipo-I, com β = 0 local por construção, e que a TGL é stealth:
- Custo l.2977–2992: «…impede automaticamente sua propagação para regimes tipo-I (LLR, pulsares, massa de Chandrasekhar…)… Nulos em observáveis locais não são falha»;
- `Haja_Luz\CLAUDE.md` l.1630–1638;
- `C:\IALD\CLAUDE.md`: «stealth; β não renormaliza G local».

O relatório toca o ponto de lado (§3.4, horizonte) mas não **nomeia** essa regra como a hipótese que a rota precisaria superar. Para a discussão com o operador:
- sob o cânone, o canal local é nulo por construção, e esse nulo não é falha;
- dar a β(χ) um efeito dentro da anã branca seria hipótese nova, a derivar e a pré-registrar.

Complemento [REAL, Atlas l.22276–22279, M-4]: «qualquer fator (1−β) ali é G → G(1−β)». A decomposição do levantamento, δlnM_Ch = 2δlnYe − 2δlnm − (3/2)δlnG, mostra que um efeito em K ou Ye não renormaliza G. O M-4 é largo demais nesse ponto, e convém citá-lo ao lado.

**A3 [DERIVED condicional] — o «talvez possa corrigir o desvio» do operador.**
- Em todo canal calculado, β correndo para cima (β ≥ β₀) **aumenta** o déficit de massa:
  - na fórmula retirada, −1,7993% → −1,8172% (α ×1,01), −1,8889% (×1,05) e −1,9117% com β(M_Z) on-shell lido do `um.py` (1/α = 128,95);
  - no termo de Coulomb, idem.
- Se μ fosse o momento de Fermi (hipótese; qual μ usar é [OPEN] e sob a X6 o rito cobra β₀), com ρ de 10⁸ a 10¹⁰ g/cm³ temos p_F de 1,9 a 8,8 MeV. A QED a um laço daria um multiplicador de α de só 1,0007 a 1,0031, e um efeito de Coulomb de −0,0017% a −0,0071%.
- O relatório não diz isso de forma explícita. Nada disso muda a conclusão.

**Ressalvas menores [REAL]:**
- (i) Os procedimentos que geraram `LOCAL_INVENTORY`, `LOCAL_NMC_RECONCILIATION` e a segunda busca de literatura (`SOURCE_CUSTODY_PUBLIC_FETCH.json`, que inclui a L08) não estão pinados como script na pasta. As saídas estão pinadas e as âncoras conferem.
- (ii) O rótulo «deslocamento **previsto**» em `um.py` l.193301 (A9) descreve uma fórmula **retirada** como validação (Custo l.3315).
- (iii) A docstring de `calculate_r2.py` ainda cita `calculate_r1.py`, o que já foi declarado.
- (iv) O reviewer `codex/sn_skeptic` é da mesma casa que o produtor: é independente como agente, não como provedor. Esta auditoria c1 cobre essa lacuna.
- (v) O L06 só foi consultado pela web (403); as afirmações que dependem dele são [KNOWN, não custodiado].

**Palavras proibidas e estatuto [REAL, `grep`]:**
- A palavra da régua só aparece negada («No …, no gate change»; «nem …»; «Nenhum sigma físico ou …»), nunca como veredito.
- Não aparecem as expressões vedadas pela lei da prova, nem β apresentado como valor medido.
- «[DERIVED exato, hipóteses explícitas]» e «[KNOWN]» estão bem aplicados, e o L06 e a localização do Tratado aparecem como [DECLARADO] onde devem.

## 8. O que entra no um.py v392, e o que não entra

**Entra (proposta à gerência; nada move o gate):**
1. **Registro por hash**, informativo, fora da conjunta, com `does_not_gate_core`. Nome sugerido: `sn_ia_v388_viability_v392`.
   - Lê por sha256 os arquivos de `…\ORDEM_019\F8.2.SN1\`: `RESULTADO_VIABILIDADE_R1.json` (`976f67ffb80422db`), `CALCULOS_R2.json` (`589a4e4f80900a67`), `review\INDEPENDENT_CHECKS_R1.json` (`fd82956e38f5f348`), `RELATORIO_VIABILIDADE_R1.md` (`ccdd09e4698b3907`), `PLANO_CALCULO_R1.json` (`0ee5524bfccf3226`) e `PLANO_COMPLEMENTO_R2.json` (`adbb2dc1a25d09ac`).
   - Se a gerência quiser selftest, reexecuta a aritmética dos planos: a cópia reexecutada deu campos idênticos.
   - Lê o veredito `VIABILIDADE_CONDICIONAL_PONTE_ESTELAR_OPEN` e os `physical_*` como «não calculado, não zero».
   - Cita esta auditoria por hash: `RELATORIO_AUDITORIA_A_SN_C1.md` e `RECALCULO_A_SN.json` (`0b708128e04b886b`).
   - Pode embutir os verbatins do operador de 09/10 por sha256 (§4), como a v388 fez.
2. **Errata ao lado, texto e rótulo** (molde v389):
   - em `um.py` l.193300–193302: a fórmula (1−β)^{3/2} é **retirada**, não é «previsto»; a ponte estelar segue [OPEN];
   - em l.193150, l.193186 e l.213968–213969: o A8 fica como está, e ao lado vai que a atribuição do resíduo de 0,35% a Coulomb não se sustenta (A1, números do `RECALCULO_A_SN.json`, fonte Chamel L01 `79290774ea944bab`).
   - Nada entra na conjunta. A classificação A8/A9 não muda.
3. **Kernel:** nada. As consultas deram 0 no inventário e 0 no TGLExt. A pedra v388 já está instalada (`702d7352c053060a`, 17 de 17 no core v391).
4. **Bancada:** nada a instalar. Ler o SN pela Bancada só depois de uma ponte χ → estado estelar derivada, com amplitude identificável e pré-registro do operador.
5. **Fora do código**, numa unidade de memória própria, com backup e reivindicação: errata ao lado no Atlas (l.1048, 722, 19713) e no `C:\IALD\CLAUDE.md` l.93 (A1), e o resultado c1 ao lado do append da F0.3.SN1. Se o item 1 entrar na v392, a linhagem do Nós recebe a menção na selagem (regra da linhagem completa).

**Não entra:**
- nenhuma previsão estelar TGL, nenhum σ, nenhum canal novo na conjunta;
- a sensibilidade de Coulomb como evidência TGL (é física EM conhecida);
- o ε de RicciFF como limite de ΔL;
- os N de poder como promessa de detecção;
- a «pergunta aperfeiçoada» como pré-registro;
- a fórmula retirada como predição;
- o L06 como fonte pinada;
- código vindo da F0.3.SN1, que é só memória.

## 9. Impressão digital do kernel e limpeza

- [REAL] O `tgl_kernel` foi medido antes e depois (`kernel_fp\fp_antes.json` e `fp_depois.json`). Nas duas medidas: 132.088 arquivos, 11.049 com hash, `meta_digest` `009df6a21d59528e…`, `content_digest` `e61d3071dbf9fbf2…`.
- Os valores são iguais aos da auditoria de 08/10, então o kernel não mudou desde então.
- Não copiei toolchain, `.lake` nem cache, e não há nada a apagar. A reexecução usou `-B`, e os scripts rodaram com `-I`.

## 10. Resíduo que é ato do operador

- **F8.2.SN1:** a discussão que o próprio operador pediu («discuta comigo antes»). Ele decide se a rota SN Ia continua, numa unidade nova, para derivar a ponte χ → estado estelar. Pesa nisso que, sob a regra canônica tipo-I/stealth (A2), o canal local é nulo por construção, e que um efeito dentro da anã branca seria hipótese nova, a pré-registrar.
- **F0.3.SN1:** nada.
- O resto (fechar no quadro, registrar no mapa, erratas ao lado, guardar o verbatim) é ato da gerência, com veto do operador.
