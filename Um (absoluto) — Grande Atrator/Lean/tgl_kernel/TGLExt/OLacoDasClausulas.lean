import TGLExt.ACicloDoLiteral

set_option autoImplicit false
set_option maxHeartbeats 800000
set_option maxRecDepth 8192

/-!
# A simulação da máquina TM2: o laço das cláusulas e o verificador com testemunha em tempo polinomial

## Contexto

O módulo usa a `AFasesDaMaquina` (partida e parada) e o `ACicloDoLiteral` (o ciclo do literal) e trata o laço
das cláusulas: a varredura de uma cláusula (literal a literal, com saída antecipada no primeiro acerto), o
`saltarResto` (o resto da cláusula satisfeita se descarta até o terminador) e a composição sobre a lista de
cláusulas. Com isso habita a `ASimulacaoCompleta` da `AMaquinaTM2DaVerificacao` e, por ela, o
`ObjetivoTM2Polinomial` da `APonteAoTM2`: a máquina TM2 concreta, no modelo `Turing.TM2ComputableInPolyTime` do
Mathlib, que, dadas a fórmula e a testemunha codificadas, calcula `verificaNaFita` em tempo polinomial.

## O que fica provado [REAL]

1. Os passos de fronteira: `passo_lerClausula_fimF`, `passo_lerClausula_fimC`, `passo_lerClausula_literal`,
   `passo_proximoLiteralCheck_fimC`, `passo_proximoLiteralCheck_literal`, `passo_saltarResto_fim`,
   `passo_saltarResto_move`.
2. Os auxiliares de boa formação: `proximo_nao_tick` (depois de um literal nunca vem tick) e
   `fimC_nao_esta_nos_literais` (o `saltarResto` para no terminador certo).
3. ★ As varreduras da cláusula: `varredura_falha_total` (nenhum literal serve: a máquina para com `false`) e
   `varredura_acerto` (algum serve: a máquina volta a `lerClausula` com o resto da fórmula, a testemunha
   intacta), cada uma dentro de uma cota explícita de passos.
4. ★ `fase_dasClausulas`: o laço inteiro, de `lerClausula` com a fórmula na pilha ao `haltList` com a resposta
   `Cs.all (clausulaSatNaFita E)` (`true` SSE toda cláusula tem literal que serve), dentro de uma cota explícita
   de passos.
5. ★★ `a_resposta_e_a_verificacao` (a resposta do laço, na testemunha codificada, é igual a `verificaNaFita`),
   `a_simulacao_completa` (habita `ASimulacaoCompleta`: para todo par (fórmula, testemunha) a máquina imprime
   `codificaSaida (verificaNaFita p.1 p.2)` em no máximo `contaFases p` passos) e `o_objetivo_cumprido`
   (habita `ObjetivoTM2Polinomial` pelo `o_objetivo_se_reduz_a_simulacao` da irmã). O tipo de
   `ObjetivoTM2Polinomial` afirma que existe um polinômio; na construção da irmã, o tempo é o `contaFases`, que
   `a_forma_do_custo_e_polinomial` mantém em no máximo `20·L + 5` no comprimento `L` da entrada codificada.
   A injetividade da codificação da entrada é o teorema `codificaEntrada_injetiva` da `APonteAoTM2`. O mesmo
   `ObjetivoTM2Polinomial` também é habitado, por outro caminho, em `SimTM2.verificacao_tm2_polinomial`.

## A contabilidade (cotas superiores, dentro do `contaFases` da `AMaquinaTM2DaVerificacao`)

Por literal de índice `i`, no máximo `4·i + 10` passos (`custoLiteral`: o ciclo do literal de `ACicloDoLiteral`
e o roteamento); por cláusula, a fronteira, o resto saltado e o terminador; a partida e a parada vêm de
`AFasesDaMaquina`. O total fica dentro do `contaFases` (em `a_simulacao_completa`, com
`custo_total_das_clausulas` e `comprimento_das_clausulas`) [REAL]. Todos os custos são cotas superiores
(`EvalsToInTime` do Mathlib limita o número de passos por cima); nenhuma igualdade de passos é afirmada.

## O que NÃO se afirma

* Não se prova aqui a NP-dificuldade de SAT (a redução de Cook–Levin de toda linguagem de NP a SAT). O que fica
  provado é o verificador: a máquina TM2 padrão que, dadas a fórmula e a testemunha codificadas, devolve
  `verificaNaFita` em tempo polinomial no comprimento da entrada inteira (fórmula e testemunha). Leitura [INPUT]:
  é a forma de verificador de «SAT está em NP».
* Não se afirma que alguma máquina decida SAT sem testemunha em tempo polinomial. Decidir SAT sem testemunha,
  em tempo polinomial, na máquina de fita segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem `sorry` e sem `axiom` no código; os axiomas de cada declaração estão no relatório do kernel.
-/

namespace TGLExt.OLacoDasClausulas

open TGLExt.APonteAoTM2 TGLExt.AMaquinaTM2DaVerificacao TGLExt.AFasesDaMaquina TGLExt.ACicloDoLiteral
  StateTransition

/-- O teste de um literal na fita, como Bool: o que a máquina computa. -/
def testLiteral (E : List Simbolo) (p : ℕ × Bool) : Bool := decide (E[p.1]? = esperado p.2)

/-- A satisfação de uma cláusula na fita. -/
def clausulaSatNaFita (E : List Simbolo) (C : List (ℕ × Bool)) : Bool :=
  C.any (testLiteral E)

/-- O custo máximo de um literal no ciclo, com o roteamento (cobre os três desfechos). -/
def custoLiteral (i : ℕ) : ℕ := 4 * i + 10

/-! ## Os passos de fronteira -/

/-- lerClausula ao ver o fim da fórmula: bandeira verdadeira, vai limpar. -/
theorem passo_lerClausula_fimF (v : Estado) (e rest t d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .lerClausula v e (Simbolo.fimF :: rest) t d sa)
      = some (cfg .limparFormula (v.1, true, v.2.2) e (Simbolo.fimF :: rest) t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_cons]
  rfl

/-- lerClausula ao ver o fim de cláusula (cláusula vazia na boca): bandeira falsa, vai limpar. -/
theorem passo_lerClausula_fimC (v : Estado) (e rest t d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .lerClausula v e (Simbolo.fimC :: rest) t d sa)
      = some (cfg .limparFormula (v.1, false, v.2.2) e (Simbolo.fimC :: rest) t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_cons]
  simp only [show ((some Simbolo.fimC : Option Simbolo) = some Simbolo.fimF) ↔ False from by simp]
  rfl

/-- lerClausula ao ver um marcador de literal: vai ler o literal (o marcador fica na pilha para o pop). -/
theorem passo_lerClausula_literal (v : Estado) (e : List Simbolo) (s : Simbolo) (rest t d : List Simbolo)
    (sa : List Bool) (hsF : s ≠ Simbolo.fimF) (hsC : s ≠ Simbolo.fimC) :
    aMaquinaTM2.step (cfg .lerClausula v e (s :: rest) t d sa)
      = some (cfg .lerLiteral (v.1, false, v.2.2) e (s :: rest) t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_cons]
  simp only [show ((some s : Option Simbolo) = some Simbolo.fimF) ↔ False from by simp [hsF],
    show ((some s : Option Simbolo) = some Simbolo.fimC) ↔ False from by simp [hsC]]
  rfl

/-- proximoLiteralCheck ao ver o fim de cláusula: a cláusula morreu sem acerto — bandeira falsa, vai limpar. -/
theorem passo_proximoLiteralCheck_fimC (v : Estado) (e rest t d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .proximoLiteralCheck v e (Simbolo.fimC :: rest) t d sa)
      = some (cfg .limparFormula (v.1, false, v.2.2) e (Simbolo.fimC :: rest) t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_cons]
  rfl

/-- proximoLiteralCheck ao ver um marcador: vai ler o próximo literal. -/
theorem passo_proximoLiteralCheck_literal (v : Estado) (e : List Simbolo) (s : Simbolo) (rest t d : List Simbolo)
    (sa : List Bool) (hs : s ≠ Simbolo.fimC) :
    aMaquinaTM2.step (cfg .proximoLiteralCheck v e (s :: rest) t d sa)
      = some (cfg .lerLiteral (v.1, false, v.2.2) e (s :: rest) t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_cons]
  simp only [show ((some s : Option Simbolo) = some Simbolo.fimC) ↔ False from by simp [hs]]
  rfl

/-- O saltarResto ao ver o terminador: consome-o e volta ao laço das cláusulas. -/
theorem passo_saltarResto_fim (v : Estado) (e rest t d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .saltarResto v e (Simbolo.fimC :: rest) t d sa)
      = some (cfg .lerClausula (v.1, true, v.2.2) e rest t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_cons, List.tail_cons]
  apply algum_inj
  apply cfg_fecha rfl rfl
  intro k; fin_cases k <;> simp [cfgStacks, Function.update]

/-- O saltarResto com símbolo de literal: descarta-o e segue. -/
theorem passo_saltarResto_move (v : Estado) (e : List Simbolo) (s : Simbolo) (rest t d : List Simbolo)
    (sa : List Bool) (hs : s ≠ Simbolo.fimC) :
    aMaquinaTM2.step (cfg .saltarResto v e (s :: rest) t d sa)
      = some (cfg .saltarResto (v.1, false, v.2.2) e rest t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_cons, List.tail_cons]
  simp only [show ((some s : Option Simbolo) = some Simbolo.fimC) ↔ False from by simp [hs]]
  apply algum_inj
  apply cfg_fecha rfl rfl
  intro k; fin_cases k <;> simp [cfgStacks, Function.update]

/-- Depois de um literal nunca vem tick: vem marcador ou terminadores. -/
theorem proximo_nao_tick (C : List (ℕ × Bool)) (rest : List Simbolo) :
    (C.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest).head? ≠ some Simbolo.tick := by
  cases C with
  | nil => simp
  | cons l C' =>
    obtain ⟨i, b⟩ := l
    cases b <;> simp [codificaLiteral, List.flatMap_cons, List.cons_append, List.head?_cons]

/-- O terminador de cláusula não aparece dentro dos literais. -/
theorem fimC_nao_esta_nos_literais (C : List (ℕ × Bool)) :
    Simbolo.fimC ∉ C.flatMap codificaLiteral := by
  simp only [List.mem_flatMap, codificaLiteral, List.mem_cons, List.mem_replicate, not_exists]
  intro p hp
  obtain ⟨i, b⟩ := p
  cases b
  · simp at hp
  · simp at hp

/-- A forma canônica da pilha da fórmula na boca de um literal: o marcador à frente. -/
theorem forma_na_leitura (i : ℕ) (b : Bool) (X rest : List Simbolo) :
    codificaLiteral (i, b) ++ X ++ [Simbolo.fimC] ++ rest
      = (if b then Simbolo.litT else Simbolo.litF)
          :: (List.replicate i Simbolo.tick ++ X ++ [Simbolo.fimC] ++ rest) := by
  simp [codificaLiteral, List.cons_append, List.append_assoc]

/-! ## O laço do saltarResto -/

/-- ★ O SALTARRESTO: o resto da cláusula satisfeita se descarta até o terminador (consumido), e a máquina volta
a `lerClausula`. Cota: no máximo um passo por símbolo descartado, mais o do terminador (`X.length + 1`). -/
def fase_saltarResto (v : Estado) (e rest t d : List Simbolo) (sa : List Bool) (X : List Simbolo)
    (hX : Simbolo.fimC ∉ X) :
    EvalsToInTime aMaquinaTM2.step (cfg .saltarResto v e (X ++ [Simbolo.fimC] ++ rest) t d sa)
      (some (cfg .lerClausula (v.1, true, v.2.2) e rest t d sa)) (X.length + 1) := by
  induction X generalizing v e with
  | nil =>
    exact passo_unico (passo_saltarResto_fim v e rest t d sa)
  | cons s X' ih =>
    have hs : s ≠ Simbolo.fimC := fun hss => hX (hss ▸ List.mem_cons_self)
    have hX' : Simbolo.fimC ∉ X' := fun hm => hX (List.mem_cons_of_mem s hm)
    have h1 := passo_saltarResto_move v e s (X' ++ [Simbolo.fimC] ++ rest) t d sa hs
    have h2 := ih (v.1, false, v.2.2) e hX'
    exact EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h1) h2

/-! ## As pontes de forma -/

/-- A cabeça do conteúdo de um literal é o seu marcador. (Auxiliar das junções.) -/
theorem a_cabeca_e_o_marcador (i : ℕ) (b : Bool) (X : List Simbolo) :
    codificaLiteral (i, b) ++ X
      = (if b then Simbolo.litT else Simbolo.litF) :: (List.replicate i Simbolo.tick ++ X) := by
  simp [codificaLiteral, List.cons_append]

/-- A ponte de configuração: mesmo rótulo, mesmo estado, a fórmula igual. (Auxiliar das junções.) -/
theorem ponte_cfg {l : Rotulo} {v : Estado} {e X Y t d : List Simbolo}
    (h : X = Y) : cfg l v e X t d [] = cfg l v e Y t d [] := by
  subst h; rfl

/-! ## As varreduras da cláusula -/

/-- ★ A VARREDURA SEM ACERTO: nenhum literal da cláusula serve — a máquina para com `false` na saída (a
cláusula morreu, e com ela a fórmula). -/
def varredura_falha_total (v : Estado) (E : List Simbolo) (C : List (ℕ × Bool)) (rest : List Simbolo)
    (hnenhum : ∀ p ∈ C, testLiteral E p = false) (hC : C ≠ []) :
    EvalsToInTime aMaquinaTM2.step
      (cfg .lerLiteral v E (C.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest) [] [] [])
      (some (Turing.haltList aMaquinaTM2 [false]))
      ((C.map fun p => custoLiteral p.1).sum + rest.length + E.length + 5) := by
  induction C generalizing v with
  | nil => exact absurd rfl hC
  | cons l C' ih =>
    obtain ⟨i, b⟩ := l
    have hne : E[i]? ≠ esperado b := by
      have h0 := hnenhum (i, b) (List.mem_cons_self)
      simp only [testLiteral, decide_eq_false_iff_not] at h0
      exact h0
    have hn' : ∀ p ∈ C', testLiteral E p = false := fun p hp => hnenhum p (List.mem_cons_of_mem _ hp)
    rw [List.flatMap_cons, show (codificaLiteral (i, b) ++ C'.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest)
        = codificaLiteral (i, b) ++ (C'.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest) from by
        simp [List.append_assoc]]
    by_cases hi : i ≤ E.length
    · have hc := ciclo_literal_falha v i b (C'.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest) E []
        (proximo_nao_tick C' rest) hi hne
      cases C' with
      | nil =>
        simp only [List.flatMap_nil, List.nil_append] at hc
        have hfim := passo_proximoLiteralCheck_fimC ((E.take i).head?, true, false) E rest [] [] []
        have hp := a_parada ((E.take i).head?, false, false) E (Simbolo.fimC :: rest)
        refine custo_mono (EvalsToInTime.trans _ _ _ _ _ _ hc
          (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico hfim) hp)) ?_
        simp [custoLiteral, List.map_cons, List.sum_cons, List.map_nil, List.sum_nil, List.length_cons,
          List.length_nil]
        omega
      | cons l' C''' =>
        obtain ⟨i', b'⟩ := l'
        have hforma : ((i', b') :: C''').flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest
            = (if b' then Simbolo.litT else Simbolo.litF) :: (List.replicate i' Simbolo.tick
              ++ (C'''.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest)) := by
          simp [List.flatMap_cons, a_cabeca_e_o_marcador, List.append_assoc]
        have hc' := evals_congr (algum_inj (ponte_cfg hforma)) hc
        have hprox := passo_proximoLiteralCheck_literal ((E.take i).head?, true, false) E
          (if b' then Simbolo.litT else Simbolo.litF)
          (List.replicate i' Simbolo.tick ++ (C'''.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest)) [] [] []
          (by cases b' <;> simp)
        have hih := evals_congr_inicio (ponte_cfg hforma)
          (ih ((E.take i).head?, false, false) hn' (by simp))
        refine custo_mono (EvalsToInTime.trans _ _ _ _ _ _ hc'
          (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico hprox) hih)) ?_
        · simp [custoLiteral, List.map_cons, List.sum_cons, List.length_cons, List.length_flatMap,
            List.length_append, List.length_singleton]
          omega
    · have hi' : E.length < i := Nat.lt_of_not_le hi
      have hc := ciclo_literal_fora v i b (C'.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest) E []
        (proximo_nao_tick C' rest) hi'
      cases C' with
      | nil =>
        simp only [List.flatMap_nil, List.nil_append] at hc
        have hfim := passo_proximoLiteralCheck_fimC (E.head?, true, false) E rest [] [] []
        have hp := a_parada (E.head?, false, false) E (Simbolo.fimC :: rest)
        refine custo_mono (EvalsToInTime.trans _ _ _ _ _ _ hc
          (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico hfim) hp)) ?_
        · simp [custoLiteral, List.map_cons, List.sum_cons, List.map_nil, List.sum_nil, List.length_cons,
            List.length_nil]
          omega
      | cons l' C''' =>
        obtain ⟨i', b'⟩ := l'
        have hforma : ((i', b') :: C''').flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest
            = (if b' then Simbolo.litT else Simbolo.litF) :: (List.replicate i' Simbolo.tick
              ++ (C'''.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest)) := by
          simp [List.flatMap_cons, a_cabeca_e_o_marcador, List.append_assoc]
        have hc' := evals_congr (algum_inj (ponte_cfg hforma)) hc
        have hprox := passo_proximoLiteralCheck_literal (E.head?, true, false) E
          (if b' then Simbolo.litT else Simbolo.litF)
          (List.replicate i' Simbolo.tick ++ (C'''.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest)) [] [] []
          (by cases b' <;> simp)
        have hih := evals_congr_inicio (ponte_cfg hforma)
          (ih (E.head?, false, false) hn' (by simp))
        refine custo_mono (EvalsToInTime.trans _ _ _ _ _ _ hc'
          (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico hprox) hih)) ?_
        · simp [custoLiteral, List.map_cons, List.sum_cons, List.length_cons, List.length_flatMap,
            List.length_append, List.length_singleton]
          omega

/-- ★ A VARREDURA COM ACERTO: algum literal serve — a máquina volta a `lerClausula` com o resto da fórmula e a
testemunha intacta. O estado de chegada carrega o lixo do percurso (o registrador), devolvido com a corrida. -/
def varredura_acerto (v : Estado) (E : List Simbolo) (C : List (ℕ × Bool)) (rest : List Simbolo)
    (hsucesso : C.any (testLiteral E) = true) (hC : C ≠ []) :
    Σ' (junk : Estado), EvalsToInTime aMaquinaTM2.step
      (cfg .lerLiteral v E (C.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest) [] [] [])
      (some (cfg .lerClausula junk E rest [] [] []))
      ((C.map fun p => custoLiteral p.1).sum + (C.flatMap codificaLiteral).length + 2) := by
  induction C generalizing v with
  | nil => exact absurd rfl hC
  | cons l C' ih =>
    obtain ⟨i, b⟩ := l
    rw [List.flatMap_cons, show (codificaLiteral (i, b) ++ C'.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest)
        = codificaLiteral (i, b) ++ (C'.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest) from by
        simp [List.append_assoc]]
    by_cases h0 : testLiteral E (i, b) = true
    · have hbit : E[i]? = esperado b := by
        simp only [testLiteral, decide_eq_true_eq] at h0; exact h0
      have h1 := ciclo_literal_satisfeito v i b (C'.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest) E []
        (proximo_nao_tick C' rest) hbit
      have h2 := fase_saltarResto ((E.take i).head?, true, true) E rest [] [] [] (C'.flatMap codificaLiteral)
        (fimC_nao_esta_nos_literais C')
      refine ⟨((E.take i).head?, true, true), custo_mono (EvalsToInTime.trans _ _ _ _ _ _ h1 h2) ?_⟩
      simp [custoLiteral]
      omega
    · have hne : E[i]? ≠ esperado b := by
        simp only [testLiteral] at h0
        have h0f : decide (E[i]? = esperado b) = false := by
          cases hD : decide (E[i]? = esperado b) with
          | false => rfl
          | true => exact absurd hD h0
        exact (decide_eq_false_iff_not).mp h0f
      have hs' : C'.any (testLiteral E) = true := by
        simp only [List.any_cons, h0, Bool.false_or] at hsucesso
        exact hsucesso
      by_cases hi : i ≤ E.length
      · have hc := ciclo_literal_falha v i b (C'.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest) E []
          (proximo_nao_tick C' rest) hi hne
        cases C' with
        | nil => simp [List.any_nil] at hs'
        | cons l' C''' =>
          obtain ⟨i', b'⟩ := l'
          have hforma : ((i', b') :: C''').flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest
              = (if b' then Simbolo.litT else Simbolo.litF) :: (List.replicate i' Simbolo.tick
                ++ (C'''.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest)) := by
            simp [List.flatMap_cons, a_cabeca_e_o_marcador, List.append_assoc]
          have hc' := evals_congr (algum_inj (ponte_cfg hforma)) hc
          have hprox := passo_proximoLiteralCheck_literal ((E.take i).head?, true, false) E
            (if b' then Simbolo.litT else Simbolo.litF)
            (List.replicate i' Simbolo.tick ++ (C'''.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest)) [] [] []
            (by cases b' <;> simp)
          obtain ⟨junk, hjunk⟩ := ih ((E.take i).head?, false, false) hs' (by simp)
          have hjunk' := evals_congr_inicio (ponte_cfg hforma) hjunk
          refine ⟨junk, custo_mono (EvalsToInTime.trans _ _ _ _ _ _ hc'
            (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico hprox) hjunk')) ?_⟩
          simp [custoLiteral, List.map_cons, List.sum_cons, List.length_cons, List.length_flatMap,
            List.length_append, List.length_singleton]
          omega
      · have hi' : E.length < i := Nat.lt_of_not_le hi
        have hc := ciclo_literal_fora v i b (C'.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest) E []
          (proximo_nao_tick C' rest) hi'
        cases C' with
        | nil => simp [List.any_nil] at hs'
        | cons l' C''' =>
          obtain ⟨i', b'⟩ := l'
          have hforma : ((i', b') :: C''').flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest
              = (if b' then Simbolo.litT else Simbolo.litF) :: (List.replicate i' Simbolo.tick
                ++ (C'''.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest)) := by
            simp [List.flatMap_cons, a_cabeca_e_o_marcador, List.append_assoc]
          have hc' := evals_congr (algum_inj (ponte_cfg hforma)) hc
          have hprox := passo_proximoLiteralCheck_literal (E.head?, true, false) E
            (if b' then Simbolo.litT else Simbolo.litF)
            (List.replicate i' Simbolo.tick ++ (C'''.flatMap codificaLiteral ++ [Simbolo.fimC] ++ rest)) [] [] []
            (by cases b' <;> simp)
          obtain ⟨junk, hjunk⟩ := ih (E.head?, false, false) hs' (by simp)
          have hjunk' := evals_congr_inicio (ponte_cfg hforma) hjunk
          refine ⟨junk, custo_mono (EvalsToInTime.trans _ _ _ _ _ _ hc'
            (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico hprox) hjunk')) ?_⟩
          simp [custoLiteral, List.map_cons, List.sum_cons, List.length_cons, List.length_flatMap,
            List.length_append, List.length_singleton]
          omega

/-! ## O laço das cláusulas e o fechamento -/

/-- A ponte da falha: a cláusula sem acerto tem todos os literais falsos. (Auxiliar do laço.) -/
theorem de_toda_falha (E : List Simbolo) (C : List (ℕ × Bool)) (h : clausulaSatNaFita E C = false) :
    ∀ p ∈ C, testLiteral E p = false := by
  induction C with
  | nil => intro p hp; exact absurd hp (by simp)
  | cons l ls ih =>
    intro p hp
    have h' : (testLiteral E l || ls.any (testLiteral E)) = false := h
    have hl : testLiteral E l = false ∧ ls.any (testLiteral E) = false := by
      cases hll : testLiteral E l with
      | false =>
        simp only [hll, Bool.false_or] at h'
        exact ⟨rfl, h'⟩
      | true => simp [hll] at h'
    rw [List.mem_cons] at hp
    cases hp with
    | inl hpeq => subst hpeq; exact hl.1
    | inr hpin => exact ih hl.2 p hpin

/-- O decide do teste é o bit da fita. (Auxiliar da ponte semântica.) -/
theorem decide_converte (P : Prop) [Decidable P] (q : Bool) (h : P ↔ q) : decide P = q := by
  cases q with
  | true => simp [decide_eq_true_eq, h]
  | false => simp [decide_eq_false_iff_not, h]

/-- O teste do literal na fita codificada é o `==` da `verificaNaFita`. (Auxiliar da ponte semântica.) -/
theorem teste_literal_converte (bits : List Bool) (p : ℕ × Bool) :
    testLiteral (codificaBits bits) p = (bits[p.1]? == some p.2) := by
  obtain ⟨i, b⟩ := p
  exact decide_converte _ _ (o_teste_e_o_da_fita bits i b)

/-- A cláusula satisfeita na fita codificada é o any da `verificaNaFita`. (Auxiliar da ponte semântica.) -/
theorem clausula_converte (bits : List Bool) (C : List (ℕ × Bool)) :
    clausulaSatNaFita (codificaBits bits) C = C.any fun p => (bits[p.1]? == some p.2) := by
  induction C with
  | nil => rfl
  | cons l ls ih =>
    have ih' : ls.any (testLiteral (codificaBits bits))
        = ls.any (fun p => bits[p.1]? == some p.2) := ih
    simp only [clausulaSatNaFita, List.any_cons]
    rw [teste_literal_converte bits l, ih']

/-- ★★ **A RESPOSTA DA MÁQUINA É A VERIFICAÇÃO DA FITA.** -/
theorem a_resposta_e_a_verificacao (φf : List (List (ℕ × Bool))) (bits : List Bool) :
    φf.all (clausulaSatNaFita (codificaBits bits)) = verificaNaFita φf bits := by
  induction φf with
  | nil => rfl
  | cons C Cs ih =>
    rw [List.all_cons, clausula_converte bits C, ih]
    rfl

/-- O custo dos literais de uma cláusula, desdobrado. (Auxiliar da contabilidade final.) -/
theorem soma_custoLiteral (C : List (ℕ × Bool)) :
    (C.map fun p => custoLiteral p.1).sum = 4 * (C.map Prod.fst).sum + 10 * C.length := by
  induction C with
  | nil => rfl
  | cons l ls ih =>
    obtain ⟨i, b⟩ := l
    rw [List.map_cons, List.sum_cons, ih]
    simp [custoLiteral, List.map_cons, List.sum_cons, List.length_cons]
    omega

/-- O comprimento de uma cláusula na fita: os ticks, os marcadores. (Auxiliar da contabilidade final.) -/
theorem comprimento_da_clausula (C : List (ℕ × Bool)) :
    (C.flatMap codificaLiteral).length = (C.map Prod.fst).sum + C.length := by
  induction C with
  | nil => rfl
  | cons l ls ih =>
    obtain ⟨i, b⟩ := l
    rw [List.flatMap_cons, List.length_append, ih]
    simp [codificaLiteral, List.length_replicate, List.map_cons, List.sum_cons, List.length_cons]
    omega

/-- O custo total das cláusulas na forma fechada. (Auxiliar da contabilidade final.) -/
theorem custo_total_das_clausulas (Cs : List (List (ℕ × Bool))) :
    (Cs.map fun C => (C.map fun p => custoLiteral p.1).sum + (C.flatMap codificaLiteral).length + 2).sum
      = 5 * (Cs.flatten.map Prod.fst).sum + 11 * Cs.flatten.length + 2 * Cs.length := by
  induction Cs with
  | nil => rfl
  | cons C Cs' ih =>
    rw [List.map_cons, List.sum_cons, soma_custoLiteral C, comprimento_da_clausula C, ih]
    simp only [List.flatten_cons, List.map_append, List.sum_append, List.length_append, List.length_cons]
    omega

/-- O comprimento da fórmula pelas cláusulas, na forma fechada. (Auxiliar da contabilidade final.) -/
theorem comprimento_das_clausulas (Cs : List (List (ℕ × Bool))) :
    (Cs.flatMap codificaClausula).length
      = (Cs.flatten.map Prod.fst).sum + Cs.flatten.length + Cs.length := by
  induction Cs with
  | nil => rfl
  | cons C Cs' ih =>
    simp [List.flatMap_cons, List.length_append, codificaClausula, comprimento_da_clausula C, ih,
      List.flatten_cons, List.map_append, List.sum_append, List.length_append, List.length_cons]
    omega

/-- ★ O LAÇO DAS CLÁUSULAS: de `lerClausula` com a fórmula na pilha ao `haltList` com a resposta — a
resposta é `true` SSE toda cláusula tem literal que serve. -/
def fase_dasClausulas (v : Estado) (E : List Simbolo) (Cs : List (List (ℕ × Bool))) :
    EvalsToInTime aMaquinaTM2.step
      (cfg .lerClausula v E (Cs.flatMap codificaClausula ++ [Simbolo.fimF]) [] [] [])
      (some (Turing.haltList aMaquinaTM2 [Cs.all (clausulaSatNaFita E)]))
      ((Cs.map fun C => (C.map fun p => custoLiteral p.1).sum + (C.flatMap codificaLiteral).length + 2).sum
        + ((Cs.flatMap codificaClausula).length + E.length + 5)) := by
  induction Cs generalizing v with
  | nil =>
    have h1 := passo_lerClausula_fimF v E [] [] [] []
    have hp := a_parada (v.1, true, v.2.2) E [Simbolo.fimF]
    refine custo_mono (evals_congr ?_ (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h1) hp)) ?_
    · rfl
    · simp; omega
  | cons C Cs' ih =>
    rw [List.flatMap_cons, show (codificaClausula C ++ Cs'.flatMap codificaClausula ++ [Simbolo.fimF])
        = codificaClausula C ++ (Cs'.flatMap codificaClausula ++ [Simbolo.fimF]) from by
        simp [List.append_assoc]]
    cases C with
    | nil =>
      rw [show codificaClausula ([] : List (ℕ × Bool)) = [Simbolo.fimC] from rfl]
      rw [show ([Simbolo.fimC] ++ (Cs'.flatMap codificaClausula ++ [Simbolo.fimF]))
          = Simbolo.fimC :: (Cs'.flatMap codificaClausula ++ [Simbolo.fimF]) from rfl]
      have h1 := passo_lerClausula_fimC v E (Cs'.flatMap codificaClausula ++ [Simbolo.fimF]) [] [] []
      have hp := a_parada (v.1, false, v.2.2) E
        (Simbolo.fimC :: (Cs'.flatMap codificaClausula ++ [Simbolo.fimF]))
      refine custo_mono (evals_congr ?_ (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h1) hp)) ?_
      · show some (Turing.haltList aMaquinaTM2 [false])
          = some (Turing.haltList aMaquinaTM2 [(([] : List (ℕ × Bool)) :: Cs').all (clausulaSatNaFita E)])
        simp [clausulaSatNaFita, List.all_cons, List.any_nil]
      · simp [custoLiteral, codificaClausula, List.map_cons, List.sum_cons, List.flatMap_cons,
          List.flatMap_nil, List.nil_append, List.length_cons, List.length_append, List.length_singleton]
        omega
    | cons l C'' =>
      obtain ⟨i, b⟩ := l
      have hforma : codificaClausula ((i, b) :: C'') ++ (Cs'.flatMap codificaClausula ++ [Simbolo.fimF])
          = (if b then Simbolo.litT else Simbolo.litF) :: (List.replicate i Simbolo.tick
            ++ (C''.flatMap codificaLiteral ++ [Simbolo.fimC]
              ++ (Cs'.flatMap codificaClausula ++ [Simbolo.fimF]))) := by
        simp [codificaClausula, codificaLiteral, List.flatMap_cons, List.cons_append, List.append_assoc]
      rw [hforma]
      have h1 := passo_lerClausula_literal v E (if b then Simbolo.litT else Simbolo.litF)
        (List.replicate i Simbolo.tick ++ (C''.flatMap codificaLiteral ++ [Simbolo.fimC]
          ++ (Cs'.flatMap codificaClausula ++ [Simbolo.fimF]))) [] [] [] (by cases b <;> simp)
        (by cases b <;> simp)
      have hstart : cfg Rotulo.lerLiteral (v.1, false, v.2.2) E
          ((if b then Simbolo.litT else Simbolo.litF) :: (List.replicate i Simbolo.tick
            ++ (C''.flatMap codificaLiteral ++ [Simbolo.fimC]
              ++ (Cs'.flatMap codificaClausula ++ [Simbolo.fimF])))) [] [] []
          = cfg Rotulo.lerLiteral (v.1, false, v.2.2) E
          (((i, b) :: C'').flatMap codificaLiteral ++ [Simbolo.fimC]
            ++ (Cs'.flatMap codificaClausula ++ [Simbolo.fimF])) [] [] [] := by
        simp [codificaLiteral, List.flatMap_cons, List.cons_append, List.append_assoc]
      by_cases hsat : clausulaSatNaFita E ((i, b) :: C'') = true
      · obtain ⟨junk, hrun0⟩ := varredura_acerto (v.1, false, v.2.2) E ((i, b) :: C'')
          (Cs'.flatMap codificaClausula ++ [Simbolo.fimF]) hsat (by simp)
        have hrun := evals_congr_inicio hstart.symm hrun0
        have hih := ih junk
        refine custo_mono (evals_congr ?_ (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h1)
          (EvalsToInTime.trans _ _ _ _ _ _ hrun hih))) ?_
        · show some (Turing.haltList aMaquinaTM2 [Cs'.all (clausulaSatNaFita E)])
            = some (Turing.haltList aMaquinaTM2 [(((i, b) :: C'') :: Cs').all (clausulaSatNaFita E)])
          simp [List.all_cons, hsat]
        · simp [custoLiteral, codificaClausula, List.map_cons, List.sum_cons, List.flatMap_cons,
            List.length_cons, List.length_append, List.length_singleton]
          omega
      · have hsatf : clausulaSatNaFita E ((i, b) :: C'') = false := by
          cases hc : clausulaSatNaFita E ((i, b) :: C'') with
          | false => rfl
          | true => exact absurd hc hsat
        have hvf := varredura_falha_total (v.1, false, v.2.2) E ((i, b) :: C'')
          (Cs'.flatMap codificaClausula ++ [Simbolo.fimF]) (de_toda_falha E _ hsatf) (by simp)
        refine custo_mono (evals_congr ?_ (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h1)
          (evals_congr_inicio hstart.symm hvf))) ?_
        · show some (Turing.haltList aMaquinaTM2 [false])
            = some (Turing.haltList aMaquinaTM2 [(((i, b) :: C'') :: Cs').all (clausulaSatNaFita E)])
          simp [List.all_cons, hsatf]
        · simp [custoLiteral, codificaClausula, List.map_cons, List.sum_cons, List.flatMap_cons,
            List.length_cons, List.length_append, List.length_singleton]
          omega

/-! ## O fechamento -/

/-- ★★ **A SIMULAÇÃO COMPLETA** — habita a `ASimulacaoCompleta` da `AMaquinaTM2DaVerificacao`: para todo par
(fórmula, testemunha), a máquina TM2 da verificação imprime o veredito da fita em no máximo `contaFases` passos. -/
def a_simulacao_completa : ASimulacaoCompleta := fun p => by
  refine ⟨?_⟩
  have h1 := a_partida p
  have h2 := fase_dasClausulas
    ((codificaFormula p.1).head? <|> ((codificaFormula p.1).reverse.head? <|> none), true, false)
    (codificaBits p.2) p.1
  refine custo_mono (evals_congr ?_ (EvalsToInTime.trans _ _ _ _ _ _ h1 h2)) ?_
  · show some (Turing.haltList aMaquinaTM2 [p.1.all (clausulaSatNaFita (codificaBits p.2))])
      = Option.map (Turing.haltList aMaquinaTM2) (some (codificaSaida (verificaNaFita p.1 p.2)))
    rw [a_resposta_e_a_verificacao]
    rfl
  · have hLf : (codificaFormula p.1).length = (p.1.flatMap codificaClausula).length + 1 := by
      simp [codificaFormula, List.length_append]
    have hB : (codificaBits p.2).length = p.2.length := by simp [codificaBits]
    have hS := custo_total_das_clausulas p.1
    have hC := comprimento_das_clausulas p.1
    simp only [contaFases]
    omega

/-- ★★ **O OBJETIVO CUMPRIDO**: habita `ObjetivoTM2Polinomial` — a máquina TM2 concreta, no modelo padrão do
Mathlib, que, dadas a fórmula e a testemunha codificadas, calcula `verificaNaFita` em tempo limitado pelo
polinômio fixo `20·L + 5` no comprimento da entrada. É o verificador de SAT com testemunha; a NP-dificuldade de
SAT (a outra metade de Cook–Levin) não se prova aqui. -/
theorem o_objetivo_cumprido : ObjetivoTM2Polinomial :=
  o_objetivo_se_reduz_a_simulacao a_simulacao_completa

end TGLExt.OLacoDasClausulas

