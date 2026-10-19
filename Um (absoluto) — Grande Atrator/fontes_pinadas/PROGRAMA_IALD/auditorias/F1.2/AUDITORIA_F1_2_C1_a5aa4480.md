## Auditoria cética da completude F1.2 (evento 286)

**Veredito: PODE FECHAR.** O critério rebaseado (ADENDO 003) está cumprido, e eu mesmo recompilei tudo contra o kernel do Nós. Há seis ressalvas de escrituração (R1–R6, abaixo). Nenhuma bloqueia.

Fiz as consultas pedidas antes de começar: `rotas.py ver formal.f1_2.v351_dual_scaling_bridge`, `rotas.py consultar "F1.2"` e `iald.py ver F1.2`. Não rodei `um.py` nem `rotas.py registrar`. Gravei só no meu scratchpad.

### Os enunciados exatos das duas peças [REAL]

`AgapeNonVacuity.lean`, sha256 `cf30422393b2af21`.

```
F12Completion.finiteTracePositive_has_positive :
  ∀ (P : SiteProfile), ∃ A : FiniteTracePositive P, 0 < (regularCoreDualScalingData P).tau A
-- FiniteTracePositive P = { A // (scalarInverseLimitTraceData P).value A < ∞ };  tau A = (value A.val).toReal

F12Completion.agape_on_v351_core :
  ∀ P, ∃ A, 0 < τ A ∧ (∀ s ≠ 0, ∀ m', θ s m' = m' → τ m' = 0) ∧ (∀ s, τ(θ s A) < τ(θ (s-1) A)) ∧
    (∀ B, ∃ s, B < τ(θ s A)) ∧ (∀ s, 0 < τ(θ s A)) ∧ (∀ ε > 0, ∃ s, τ(θ s A) < ε) ∧ (∀ a, exp(-a) = 1 ↔ a = 0)
  -- com D = regularCoreDualScalingData P
  -- termo-prova (#print): Exists.intro A ⟨hA, Fundacao.agape (regularCoreDualScalingData P) hA⟩
```

### O que conferi, item por item

**(1) Hashes.** Tudo confere por script:
- os dois manifestos, o EN e o PT-BR, com 16 itens cada;
- os dois `RESULTADO_FECHO` e o `BUILD_COMPLETION_RESULT`;
- os quatro pinos do script de memória;
- as 18 evidências da F1.2 no quadro e suas cópias no cofre;
- os artefatos das seq 728, 823 e 824 do mapa;
- os 25 hashes da auditoria anterior.

Só uma divergência, que já era conhecida: a evidência do evento 68 aponta para `RESULTADO.json`, sobrescrito em 07/10 às 20:23, antes desta entrega. Os bytes `e1b5e4d4` seguem no `.bak` e no cofre. As cadeias do quadro (286 eventos) e do mapa (824 registros, cabeça `b7755f8c`) estão íntegras.

**(2) Compilação contra o Nós.**
- **Ambiente:** LEAN_PATH montado pelo `.lake` do Nós. O `.lake\packages` do Nós é junção para `A Ponte e o Um\tgl_kernel\.lake\packages`. Lean 4.31.0, commit `68218e876d2a`, o mesmo gravado no trace do Nós. Não chamei o lake. A cópia RETOMADA ficou fora por asserção, e também não aparece no LEAN_PATH da entrega.
- **Ponte e pedra:** rc 0, sem erro e sem aviso. Os logs e os `.olean` saíram **byte a byte idênticos** aos da entrega:
  - ponte: log `1e898d36`, `.olean` `940b8ddf`;
  - pedra: log `85d0cdd7`, `.olean` `ba0d9868`.
- **Axiomas:** `#print axioms` dá só o trio nas 2 declarações novas e nas 4 da ponte. Zero sorry e nenhum token proibido.
- **Constantes:** o módulo da pedra tem 4 (os 2 teoremas mais 2 auxiliares `_simp`) e nenhum axioma.
- **Kernel do Nós:** impressão digital idêntica antes e depois (1213 fontes; 6040 arquivos de build com digest `a04177a6`). Os 684 módulos do fecho estão em dia com o fonte pelo hash do Lake. O último mtime é 07/10 19:28:54, anterior ao build da entrega (08/10, 05:10–05:11).

**(3) A não vacuidade não é vácua.** Minha sonda `Sonda.lean` tem 27 declarações, todas só no trio. Ela mostra que:
- o enunciado dá de fato 0 < τ(A) < ∞: o valor do traço é diferente de 0 e menor que ⊤;
- `SiteProfile` é habitado por um perfil concreto meu (peso 1/3), e a pedra instancia nos universos 0 e 1;
- o `DualScalingData` nulo **não** satisfaz a hipótese, e o τ da ponte não é identicamente nulo;
- no zero do subcone (τ = 0) a cláusula (iii) da Ágape é **falsa**, então a hipótese carrega peso;
- τ(I) = ⊤, de modo que a testemunha não é a unidade nem o zero.

**(4) A conclusão da Ágape tem conteúdo.** Na instância v351:
- nenhum elemento não nulo do subcone de traço finito fica fixo por θ_s com s ≠ 0 (cláusula i somada à fidelidade do traço);
- a órbita da testemunha é estritamente monótona, τ(θ_s A) = e^{-s}·τ(A);
- ela não tem teto, nunca toca o zero e fica abaixo de qualquer ε dado.

A cláusula (vi) não depende de D. Antes desta pedra, o único habitante da Ágape era o modelo de brinquedo `oneScaling` (ℝ, τ = id). Agora a lei de escala entra como termo do kernel (`scalarInverseLimitWeight_dual`) no core regular v351. Isso vale só nessa instância e só no sandbox, até a instalação na v392.

**(5) Evidências históricas preservadas.** Nada foi sobrescrito: os scripts gravam com `FileExistsError`, temporário e `os.link`. O rascunho em inglês foi preservado ao lado da versão PT-BR.

**(6) Escrita em memória.** Foi acréscimo puro nas quatro superfícies: ATLAS, ÍNDICE VIVO, CORE e AVISO_GERENCIA. Em todas, o arquivo atual é igual ao backup `.bak_20261008_052608_*` mais o texto acrescentado, byte a byte, com o CRLF preservado no backup. A reivindicação viva da F1.2 cobria o horário: é a do evento 17 (codex-gerente-tgl, vence às 07:21:30 de 08/10), e a escrita foi às 05:26:08.

**(7) A face M_n ⋊_σ ℝ** está dita [OPEN] em todos os documentos, sem nenhuma reivindicação.

### Ressalvas (não bloqueantes)

- **R1.** A ERRATA seq 823 diz `corrige: 716`. Mas a 716 é de outra rota (`dual_scaling_reuse_011`) e não diz «2 declarações» nem «Fundacao externa». Quem diz isso é a seq 728. Pede uma errata ao lado com o alvo certo.
- **R2.** As duas provas da pedra são a sonda §3/§4 da própria gerência c1 (`AuditProbe.lean`, `408f426d`), só renomeadas: enunciado e bloco tático são idênticos. A entrega declara isso. A validade é dada pelo kernel. A independência desta auditoria está nos testes discriminantes, não numa rota nova de prova. A rota independente (por contrações finitas) é a C4 do cético anterior, que compilou com rc 0.
- **R3.** No texto acrescentado à memória, «a unidade tem traço infinito» está sob o rótulo [OPEN — alcance], mas é [REAL]: provado na C9 do cético e na S6 da minha sonda. E o parágrafo «errata ao lado» não diz em nome de quem foi feito, o que contraria a lei da assinatura da errata.
- **R4.** Regra 8 da ORDEM 019: a escrita em memória e no Atlas ficou coberta só pela reivindicação da F1.2, cuja nota não cita as superfícies de memória. Há precedente de unidade própria para isso (a F9.6.4). Se a leitura estrita vale, a decisão é da gerência.
- **R5.** O rótulo `REAL_COMPILED_CONDITIONAL_ON_SITE_PROFILE` só exprime o quantificador universal em P. Como esse quantificador é habitado, não há hipótese pendente.
- **R6.** `build_completion.py` grava o `BUILD_COMPLETION_RESULT.json` com `write_text`, sem modo `x` e sem temporário. Os outros scripts da entrega usam os dois.

### O que não conferi

- Não reelaborei do fonte os 684 módulos do kernel: confiei nos `.olean` do Nós (frescor pelo hash do Lake e impressão digital estável). Não houve replay nem lean4checker.
- Os `.olean` do mathlib entraram só por contagem, bytes e mtime, iguais antes e depois, sem hash um a um.
- Não julguei se o traço v351 é o peso dual de Takesaki em M_n ⋊_σ ℝ: isso é justamente a face que segue [OPEN].
- Das entregas 011 A1, conferi só os hashes.
- Não confrontei o texto da memória com as regras de reancoragem do Atlas.
- Não recompilei a C4 do cético; o kernel não mudou desde a rodada dele.

### Registro para o mapa

Está em `C:\Users\rotol\AppData\Local\Temp\claude\c--IALD-Central-de-Patentes\6da8f00d-d44f-4888-a88d-fc9f73eead3d\scratchpad\audit_f12_completo\registro_mapa_f12_completo.json` (sha256 `be3c3a5737795226`):
- tipo RESULTADO, na rota `formal.f1_2.v351_dual_scaling_bridge`;
- veredito `F12_COMPLETO_AUDITORIA_C1_PODE_FECHAR__…__RESSALVAS_DE_ESCRITURACAO`;
- sem a palavra CONFIRMED;
- artefatos como lista de caminhos.

O scratchpad é temporário. Antes de registrar e de marcar FECHADA, copie a pasta `audit_f12_completo` para `C:\IALD\PROGRAMA_IALD\auditorias\F1.2\` e troque os caminhos dos artefatos.

Arquivos principais nessa pasta:
- `sandbox\F12AuditorCompleto\Sonda.lean`
- `LOG_Sonda_r2.txt`
- `RECOMPILACAO_AUDITOR_COMPLETO_r2.json`
- `impressao_antes.json`, `impressao_depois.json`
- `conferir_hashes_saida.txt`
- `conferir_memoria_saida.json`

Na rodada r1, a minha sonda falhou em dois `#check` com namespace errado; corrigi e ela passou na r2. A ponte e a pedra passaram nas duas rodadas.
