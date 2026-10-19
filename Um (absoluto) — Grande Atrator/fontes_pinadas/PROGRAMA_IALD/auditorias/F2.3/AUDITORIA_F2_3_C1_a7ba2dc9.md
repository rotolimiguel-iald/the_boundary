AUDITOR CÉTICO c1: camadas 354/355/360

Fiz só no scratchpad `C:\Users\rotol\AppData\Local\Temp\claude\c--IALD-Central-de-Patentes\6da8f00d-d44f-4888-a88d-fc9f73eead3d\scratchpad\audit_camadas3_08out\`. O relatório completo está em `RELATORIO_AUDITORIA_CAMADAS3_C1.md` (sha256 8954fa9c…). Os três registros propostos (`registro_mapa_F1.1.E1.json`, `registro_mapa_F2.3.json`, `registro_mapa_F1.3.E1.json`) estão lá, nenhum registrado no mapa por mim. Usam tipo RESULTADO, os rota_id das seq 870–872 e não contêm CONFIRMED. Consultei o mapa antes. `rotas.py ver` só aceita rota_id, não seq, então li as seq 830 e 868–872 direto do `rotas.jsonl`.

## Comum às três camadas
- **Quadro:** a cadeia está íntegra (363 eventos, zero quebra). As 18 evidências dos eventos 354, 355 e 360 conferem em sha256 e bytes, na origem e na cópia em `evidencias\`. Os artefatos das seq 870, 871 e 872 também conferem (5, 14 e 7).
- **Memória:** houve só duas escritas, às 11:37:02 e às 12:04:27. Nas quatro superfícies, o `.bak_113702` é prefixo exato do `.bak_150427_*_UTC`, que é prefixo exato do arquivo atual. Os deltas medem 3550 B (sha 99760f66…, igual ao recibo) e 2147 B, e os before/after dos recibos batem com o que medi.
- **Cobertura das escritas:** a das 11:37 está coberta pelas reivindicações 347 e 349, e a das 12:04 pela 359. As três nomeiam as quatro superfícies e estavam vigentes. Nenhuma escrita ficou sem cobertura.
- **Palavras proibidas:** nenhuma. Só aparecem «confirma», dentro da citação verbatim da docstring, e «NOT_FALSIFIED não é CONFIRMED».
- **Nós:** a impressão digital do tgl_kernel é igual antes e depois das minhas compilações (meta 009df6a2…, conteúdo e61d3071…), e é a mesma das auditorias c1 da manhã.
- **F7.2:** não chegou camada nova depois de 11:38. Não há evento F7.2 após o 351, nem arquivo novo no túnel F7.2 ou na Bancada_Um.

## (1) F1.1.E1 (evento 354) — APROVA. Pode FECHAR.
- **Acréscimo puro:** `INVENTARIO.md.bak_20261008_113445` é prefixo exato do INVENTARIO atual, e o delta é byte a byte a ERRATA de 4533 B.
- **R2:** o grep confere `O16.lean` nas linhas 19, 48 e 96. O fecho transitivo de imports de `QGSolutionComplete` (755 módulos locais) não contém nenhuma pedra polar, Wick, CalibrationReuse ou HorizonDataReuse, e tem zero menções aos namespaces e aos dois lemas.
- **R3:** `polar_period_matches_contract` está em ContractCalibrationReuse.lean:104–107 e `data_temperature_from_same_period` em HorizonDataReuse_Integration_v2.lean:23–29. Ambos estão descritos como condicionais.
- **R4:** conferido byte a byte. «RELER NÃO ACRESCENTA» e «A primeira leitura identifica; a segunda confirma que a operação não altera.» existem em TheObserverReadsTheAngle.lean (c764e907…, linhas 110–113). A forma composta não existe no fonte, e a errata a declara composta.
- **R5:** o nome PALAVRA está certo (Palavra.lean 33184b49…).
- **Ressalva menor:** a seq 870 pina o INVENTARIO no estado atual, não a camada isolada.

## (2) F2.3 Temperatura (evento 355) — APROVA_COM_RESSALVAS. A F2.3 PODE FECHAR com só LiganteLeiCampoCurvo na v392; a Temperatura fica fora.
- **Build em sandbox:** toolchain v4.31.0, LEAN_PATH só-leitura ao .lake do Nós. EXIT 0, 10 `#print axioms`, todos só com propext, Classical.choice e Quot.sound, sem sorry nem aviso. O `.olean` saiu byte a byte igual ao deles (21ab80ff…, 568192 B). Os 24 pins do RESULTADO conferem.
- **T1/T2/T6:** atendido pela via de remover. `observer_boost` e `observer_boost_eq` saíram, e todos os campos que restam na leitura são consumidos pelas provas.
- **Sondas de vacuidade:** continuam passando (re-sondas compiladas, EXIT 0, todas no trio).
  - T1: dá para fabricar a leitura com qualquer κ>0.
  - T2: a taxa é livre na mesma cunha.
  - T3: a ponte térmica é ociosa.
  - T4: o teorema é corolário da pedra auditada.
  - T9 (sonda nova): a leitura é equivalente ao par cunha × κ positivo.
  - A Codex declara isso na docstring e no ADENDO: o peso falso foi retirado e o vazio ficou dito.
- **Nota de fiação no ADENDO:** feita, em acréscimo puro (backups são prefixos). Cita os cinco termos e `kappa_is_input`, com linhas conferidas por grep, e retrata a frase «exigiria hipótese tipada».
- **Processo:** atendido. Estão registrados o evento 278, os três logs de 1372 B (que são o mesmo transcrito), os pins das três tentativas e do 454fb4f8, e a «auditoria semântica independente» como [DECLARADO].
- **Ressalvas:**
  - (a) **NÃO ATENDIDO:** a frase «No rotation of the light net, all-wedge family, or H3 is constructed here.» não voltou à docstring de `LiganteLeiCampoCurvoRamoCAuditoriaC1.lean`. O arquivo é de 10:48 e não há versão nova depois do despacho msg_camadas2. Não bloqueia, porque o ramo C não entra na v392; fica como pendência ao lado.
  - (b) A unidade F2.3.T1 não foi criada. A reivindicação 347 foi liberada no evento 357, o que atende à alternativa «liberem».
  - (c) Os identificadores `faithful_boost_reading…` ainda dizem «boost» e «faithful» sem peso.
  - (d) O build da Codex usou uma junção NTFS dentro do túnel (`F2.3\lean_cache_readonly_20261008`) que aponta para o Nós. O Nós não mudou, mas é um risco de processo; recomendo remover a junção.
  - (e) A memória das 11:37 descreve a versão Standalone, anterior à SemBoostOcioso.

## (3) F1.3.E1 (evento 360) — APROVA_COM_RESSALVAS. Pode FECHAR.
- Só há arquivos novos na pasta, e a v3.1 e a v3.2 estão intactas.
- **As cinco ressalvas foram atendidas e conferidas:**
  - `F13v3_H1_nontrivial.lean:16`;
  - `Mathlib/Analysis/VonNeumannAlgebra/Basic.lean:23`;
  - StandardSubspace.lean:42 e Spectrum.lean:34, com Stone marcado [FALTA];
  - Crismale–Fidaleo–Lu (2017) nomeada;
  - as 35 referências descritas como checagem de existência e intervalo, não de conteúdo.
- **ANOTACAO ao lado da seq 830:** está na seq 872, mas a ligação à 830 é só por texto.
- **Ressalva:** a errata da contradição do pedido 3 é implícita. Dá a leitura certa, mas não cita nem retrata a frase da v3.1 (COMPLEMENTO_V3_1:32, «A c1 deve decidir a forma final dos campos depois da construção de D1a.1–D1a.3»).

## O que não conferi
- O conteúdo das 35 referências e a bibliografia de Crismale–Fidaleo–Lu.
- A sonda `fecho_qg.log` em Lean: fiz o fecho por grep de imports, como o pedido especificava.
- A reprodução do `.olean` do RamoCAuditoriaC1.
- Os arquivos MAPA_*_INPUT.json.
- O estado da F2.3 no PROGRAMA.json: li só os eventos.

Nada disto move o gate.

Arquivos principais no scratchpad:
- `RELATORIO_AUDITORIA_CAMADAS3_C1.md`
- `registro_mapa_*.json`
- `log_hashes_evidencias.txt`
- `log_memoria_prefixo.txt`
- `log_fecho_imports_QG.txt`
- `log_R4_verbatim.txt`
- `log_pins_resultado_f23.txt`
- `build_sem_boost_ocioso.log`
- `sondas_sem_boost.log`
- `sandbox/AuditTemp3/SondasSemBoost.lean`
- `fp_antes.json`
- `fp_depois.json`
- `log_palavras_proibidas.txt`