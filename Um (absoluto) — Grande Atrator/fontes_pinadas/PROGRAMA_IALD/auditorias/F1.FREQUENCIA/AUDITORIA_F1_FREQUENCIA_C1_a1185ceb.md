AUDITOR CÉTICO c1: TheUnitIsTheRank r3

**Veredito: APROVA_COM_RESSALVAS** a entrada da pedra r3 (`e95775d61db38633`) na v392, sob o nome que o operador der. Nenhuma ressalva bloqueia. Elas são de nome e de docstring; quatro são de gravidade MÉDIA ou MÉDIA-BAIXA. Tudo foi conferido com código meu, sem rodar nenhum script do construtor.

O relatório está em `C:\Users\rotol\AppData\Local\Temp\claude\c--IALD-Central-de-Patentes\6da8f00d-d44f-4888-a88d-fc9f73eead3d\scratchpad\aud_rank_r3\AUDITORIA_RANK_R3_C1.md` (sha16 `d515a12fabc0f268`). O registro do mapa está na mesma pasta, em `registro_mapa_F1.FREQUENCIA.json` (sha16 `85f47306d20cd865`): tipo RESULTADO, rota `formal.posto_e_defasagem_da_unidade`, 25 artefatos, todos caminhos que existem. Registrá-lo no mapa é ato da gerência.

**Retomada.** Reaproveitei o que a tentativa anterior deixou: o toolchain copiado, os recibos das duas compilações e a impressão digital de antes. O toolchain funcionava (as compilações deram EXIT 0), então não recopiei. Ao final apaguei `C:\t\rk3` inteiro, 2,7 GB (`LIMPEZA_AUDITOR_C1.json`).

## O que medi [REAL]

**(1) Recompilação do zero**
- **r3:** EXIT 0, 0 avisos, 0 `sorry`. O `.olean` saiu igual bit a bit ao `3e975a431edbefe1` (conferido com `cmp`).
- **r2:** o `.olean` `d0830b4b…` saiu igual bit a bit ao da `3_KERNEL\rodada2`.
- **Axiomas:** `DumpAuditorC1.lean` rodou `collectAxioms` sobre as 221 constantes do módulo: as 190 declarações, 13 lemas `eq_1` e 18 internas. **As 221 ficam no trio** {propext, Classical.choice, Quot.sound}, com zero `sorryAx`.

**(2) Comparação r2→r3 lida do kernel**
- Hash estrutural do tipo e do termo de prova: iguais em 221/221.
- Tipo impresso: igual em 221/221.
- Texto cru: igual em 220/221. A única diferença está em `axiomFlow_complement_decay_reexport` e é só o sufixo higiênico do nome de um binder, que deriva do nome da declaração. Os termos são α-equivalentes. Logo o «190/190 provas idênticas» do construtor vale, módulo nomes de binder.
- Há exatamente as 3 renomeações pedidas e nenhuma outra. Os nomes antigos não aparecem em nenhuma outra constante.
- O código sem comentários (removedor de comentários meu) é igual nos dois lados, sha `f075d95b70533291`.
- Mudaram 14 docstrings (as 3 renomeadas e 11 de comentário), o que confere com a tabela do construtor.

**(4) Aspas e atribuições** (`ASPAS_AUDITOR_C1.json`)
- A pedra tem 56 trechos «…». **Zero atribuições ao operador sem fonte.** As aspas de V1 a V4 e a l.3 da ordem de 12/10 estão nos bytes (as de V1 por fragmentos em ordem, com espaços normalizados), e as linhas citadas conferem.
- O sha do um.py foi conferido antes de usar. São **10 citações de linha reais** (6488–89, 6502, 6512–6515, 6518 três vezes, 6526), e as 10 conferem por fragmento. As «11 ocorrências» do construtor incluem o molde «l.NNNN».
- As 10 citações de linha do kernel conferem.

**(5) ADENDO e FREEZE_R3** (`FREEZE_AUDITOR_C1.json`)
- O FREEZE_R3 confere por hash: 46/46. O FREEZE original da unidade segue íntegro.
- O ADENDO é aditivo: nenhum termo dele está na pedra.
- Das 8 partes, 6 (A1.a–d, A2, A3) já tinham sido compiladas pela sonda `512ea820` do auditor de estatuto da r2; conferi que ela tem exatamente esses 6 teoremas. A4 é a sonda S3. **Só A5 não tinha medição anterior.** O próprio ADENDO declara isso. Pela Lei da Prova não é demérito, porque o que se congela é o teste.

**(6) Base de D e o «NÃO afirma»** são consistentes com a errata ao lado de `posto-e-rank-10out.md` (l.22: sobrevive o traço, não o posto 1) e com o relatório r2 (§6 A5, §7).

**(7) Nada confidencial foi lido e o Nós não foi tocado.** A impressão digital do `tgl_kernel` é igual antes e depois: 132.088 arquivos, `a15d5e02` / `ca632c47` / `8952f693`. Não rodei o um.py.

## Ressalvas, pela gravidade

| # | gravidade | ponto | de quem é o ato |
|---|---|---|---|
| N1 | MÉDIA | O nome do módulo e o título «A UNIDADE É O POSTO» dizem mais do que a pedra prova fora do átomo; os próprios termos da pedra os desmentem lá. | operador (nome final) |
| N2 | MÉDIA | `reading_weights_are_truth_and_cost` traz «verdade» e «custo», leitura [ONTO], no nome; o termo é uma conta de entradas 2×2. | operador ou rodada de nomes |
| N3 | MÉDIA | `inscription_at_thetaMiguel` usa vocabulário reservado da casa, apesar de o cabeçalho dizer «Não cunha nome reservado». | operador ou rodada de nomes |
| N4 | MÉDIA-BAIXA | A docstring de `reading_returns_the_post` diz «projetor», mas o tipo não pede ⟨a, a⟩ = 1. A ressalva A6 da r2 não passou para a r3. | errata ao lado |
| — | BAIXA | Falta fonte para «transmitir defasa» (05/10). A autoria do conteúdo de V1 é [DECLARADO]: o texto fala ao operador em segunda pessoa. | marcar |
| — | BAIXA | O ADENDO diz «antes de qualquer ensaio», mas 7 das 8 partes já tinham medição de sonda. | redação |
| — | BAIXA | Se o ADENDO for instalado, o nome da sonda `endpoint_is_the_name_of_the_one` usa vocabulário reservado. | futuro |

**Discordo da auto-auditoria do construtor** quando ela diz «nenhum nome afirma mais que o termo». Discordo em N1 a N4.

## Ato da gerência
- Recompilar no lake do Nós como `TGLExt.TheUnitIsTheRank`. O `.olean` do sandbox é do módulo `TheUnitIsTheRank` e não serve como está.
- Importar em `TGLExt.lean`, pôr o `#print axioms` em `TGL/Audit.lean` e medir de novo a contagem do selo.
- Embutir no um.py e conferir byte a byte.
- Fazer o registro no core e no mapa.
- Fazer a inscrição ACOM.

## Ato do operador
- O nome final da pedra (N1) e, se quiser, a troca de N2 a N4.
- Se o ADENDO entra na v392.
- A leitura de V1 l.7.
- Qual Q̂, qual D e qual base de D o texto da v392 adota.
- O rótulo de autoria de V1.

## O que não conferi
- As referências de livro (Nielsen & Chuang Ex. 2.71, Axler 3.22, Lüders 1951): ficam [KNOWN, não conferido aqui].
- Não ensaiei o ADENDO; ele está congelado sem prova nova, de propósito.
- Não li o PREREGISTRO além dos hashes.
- Não fiz `init`/`transition` no ACOM nem li o README da MEMORIA_IALD, porque a escrita estava restrita a esta pasta e ao sandbox. A inscrição é ato da gerência.

A palavra vedada não aparece nenhuma vez nos dois arquivos que entreguei.