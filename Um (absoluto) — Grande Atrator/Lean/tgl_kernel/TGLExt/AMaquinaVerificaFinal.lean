import TGLExt.AMaquinaVerifica3

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A máquina verifica: o literal, o veredito e a limpeza final

## O que fica provado [REAL]

1. ★ `correr_add`: correr a + b passos é correr a passos e depois b passos do ponto alcançado.
2. ★ `append_tick_replicate` (auxiliar de lista) e `leMarcador_ticks_fecha`: o marcador move os ticks da fórmula para a contagem, e o
   estado sai com o último símbolo lido.
3. ★★ `o_cauda_do_literal`: contada a contagem, marcado o marcador, procurado o bit, comparada a polaridade e restaurada a testemunha,
   o casamento da máquina é o bit da variável batendo a polaridade do marcador.
4. ★★ `o_literal_verifica`: com os ticks à frente do marcador, a testemunha cheia e o índice no alcance, o literal processa-se e a
   máquina decide pelo casamento.
5. ★★ `o_cauda_do_literal_alem` e `o_literal_verifica_alem`: com o índice além do alcance, o bit lido é ausente, a guarda da presença
   não casa e a testemunha volta íntegra.
6. ★★★ `o_literal_cobre`: para todo índice, dentro do alcance ou além, o literal processa-se, e o casamento é a guarda da presença
   com o bit batendo a polaridade.
7. ★ `passo_escreveVerdade`, `passo_escreveMentira`, `passo_parada`: a máquina escreve o veredito e passa à limpeza.
8. ★ `passo_limpaFormula_cons`, `passo_limpaFormula_nil`, `passo_limpaTestemunha_cons`, `passo_limpaTestemunha_nil`,
   `passo_paradaFinal`, `limpaFormula_fecha` e `limpaTestemunha_fecha`: a limpeza drena a fórmula e a testemunha, zera o estado e
   apaga o rótulo.
9. ★★ `a_limpeza_fecha`: da parada, em 1 + (|f| + 1) + (|t| + 1) + 1 passos, a máquina chega à configuração final limpa, com a saída
   intacta.
10. ★ `passo_parada_halt`: com o rótulo apagado, a máquina não anda.
11. ★ `inicioClausula_vazio_fecha`: a fórmula vazia decide pela verdade em três passos.

## O que NÃO se afirma

* A correção geral e o tempo da verificadora, para toda fórmula e toda testemunha: estão em `AMaquinaVerificaCompleta`
  (`a_maquina_verifica`, com combustível de no máximo L² + 20L + 10, sendo L o comprimento da entrada codificada) e em
  `AMaquinaVerificaHabitante` (`a_verificadora_em_tempo_polinomial`).
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AMaquinaVerifica

open TGLExt TGLExt.TraducaoTuring TGLExt.APonteAoTM2 TGLExt.ATensaoMinima Turing.TM2.Stmt Turing
open TGLExt.AVerificadoraTM2

/-- ★ **O CORREDOR DECOMPÕE O COMBUSTÍVEL**: correr a+b é correr a e depois b do ponto alcançado. -/
theorem correr_add (tm : Turing.FinTM2) (c : tm.Cfg) :
    ∀ (a b : ℕ), correr tm c (a + b) = (correr tm c a).bind (correr tm · b) := by
  intro a
  induction a generalizing c with
  | zero =>
    intro b
    rw [Nat.zero_add]
    rfl
  | succ a ih =>
    intro b
    have h : a + 1 + b = (a + b) + 1 := by omega
    rw [h, correr]
    show (match Turing.FinTM2.step tm c with
      | none => some c
      | some c' => correr tm c' (a + b)) = _
    cases hstep : Turing.FinTM2.step tm c with
    | none =>
      show some c = (correr tm c (a + 1)).bind (correr tm · b)
      have h1 : ∀ n : ℕ, correr tm c n = some c := by
        intro n
        induction n with
        | zero => rfl
        | succ n _ => rw [correr, hstep]
      simp only [h1, Option.bind_some]
    | some c' =>
      show correr tm c' (a + b) = (correr tm c (a + 1)).bind (correr tm · b)
      have h2 : correr tm c (a + 1) = correr tm c' a := by
        rw [correr, hstep]
      rw [h2]
      exact ih c' b

/-- ★ **A PONTE DOS TICKS** (auxiliar de lista): o tick acumulado à esquerda é o replicate crescido. -/
theorem append_tick_replicate (i : ℕ) (c : List Simbolo) :
    List.replicate i Simbolo.tick ++ (Simbolo.tick :: c)
      = List.replicate (i + 1) Simbolo.tick ++ c := by
  induction i generalizing c with
  | zero => rfl
  | succ i ih =>
    have h1 : List.replicate (i + 1 + 1) Simbolo.tick ++ c
        = Simbolo.tick :: (List.replicate (i + 1) Simbolo.tick ++ c) := by
      rw [List.replicate_succ, List.cons_append]
    have h2 : List.replicate (i + 1) Simbolo.tick ++ (Simbolo.tick :: c)
        = Simbolo.tick :: (List.replicate i Simbolo.tick ++ (Simbolo.tick :: c)) := by
      rw [List.replicate_succ, List.cons_append]
    rw [h2, ih, h1]

/-- ★ **O LAÇO DOS TICKS**: o marcador move os ticks da fórmula para a contagem; o estado sai com o
último lido no tick. -/
theorem leMarcador_ticks_fecha (v : EstadoV) (t c r : List Simbolo) (out : List Bool) :
    ∀ (j : ℕ) (f : List Simbolo),
      correr aVerificadora
        (aConfig (some Fase.leMarcador) v [] (List.replicate (j + 1) Simbolo.tick ++ f) t c r out)
        (j + 1)
        = some (aConfig (some Fase.leMarcador) { v with lido := some Simbolo.tick } [] f t
            (List.replicate (j + 1) Simbolo.tick ++ c) r out) := by
  intro j
  induction j generalizing v c with
  | zero =>
    intro f
    show correr aVerificadora (aConfig (some Fase.leMarcador) v [] (Simbolo.tick :: f) t c r out) 1
        = some (aConfig (some Fase.leMarcador) { v with lido := some Simbolo.tick } [] f t
            (Simbolo.tick :: c) r out)
    rw [correr, passo_leMarcador_tick]
    rfl
  | succ j ih =>
    intro f
    have hrep : List.replicate (j + 1 + 1) Simbolo.tick ++ f
        = Simbolo.tick :: (List.replicate (j + 1) Simbolo.tick ++ f) := by
      rw [List.replicate_succ, List.cons_append]
    rw [hrep, correr, passo_leMarcador_tick]
    show correr aVerificadora (aConfig (some Fase.leMarcador) { v with lido := some Simbolo.tick } []
        (List.replicate (j + 1) Simbolo.tick ++ f) t (Simbolo.tick :: c) r out) (j + 1) = _
    rw [ih { v with lido := some Simbolo.tick } (Simbolo.tick :: c) f, append_tick_replicate]

/-- ★★ **A CAUDA DO LITERAL**: contada a contagem, marcado o marcador, procurado o bit, comparada a
polaridade e restaurada a testemunha — o casamento da máquina é o bit da variável batendo o
marcador; a testemunha volta íntegra. -/
theorem o_cauda_do_literal (v : EstadoV) (w f : List Simbolo) (out : List Bool)
    (i : ℕ) (pol : Bool) (hle : i ≤ w.length) :
    correr aVerificadora
      (aConfig (some Fase.leMarcador) v []
        ((if pol then Simbolo.litT else Simbolo.litF) :: f)
        w (List.replicate i Simbolo.tick) [] out) (1 + (i + 1) + (i + 1))
      = some (if ((w.drop i).head?.isSome && (((w.drop i).head? == some Simbolo.um) == pol)) then
          aConfig (some Fase.saltaLiterais)
            ⟨Fase.procuraBit, false, pol, none, (w.drop i).head?⟩ [] f w [] [] out
        else aConfig (some Fase.leMarcador)
            ⟨Fase.procuraBit, v.falha, pol, none, (w.drop i).head?⟩ [] f w [] [] out) := by
  have hcomb : 1 + (i + 1) + (i + 1) = ((i + 1) + (i + 1)) + 1 := by omega
  have hlen : ((w.take i).reverse ++ ([] : List Simbolo)).length = i := by
    rw [List.length_append, List.length_reverse, List.length_take, List.length_nil, Nat.add_zero,
      Nat.min_eq_left hle]
  have hfuel : i + 1 = ((w.take i).reverse ++ ([] : List Simbolo)).length + 1 := by rw [hlen]
  cases pol with
  | true =>
    show correr aVerificadora (aConfig (some Fase.leMarcador) v [] (Simbolo.litT :: f)
        w (List.replicate i Simbolo.tick) [] out) (1 + (i + 1) + (i + 1)) = _
    rw [hcomb, correr, passo_leMarcador_litT]
    show correr aVerificadora (aConfig (some Fase.procuraBit)
        ⟨Fase.procuraBit, v.falha, true, some Simbolo.litT, v.bitLido⟩ [] f w
        (List.replicate i Simbolo.tick) [] out) ((i + 1) + (i + 1)) = _
    rw [correr_add,
      procuraBit_fecha ⟨Fase.procuraBit, v.falha, true, some Simbolo.litT, v.bitLido⟩ f [] out i w
        hle]
    simp only [Option.bind_some]
    rw [hfuel, restauraTestemunha_fecha]
    rw [List.append_nil, List.reverse_reverse, List.take_append_drop]
  | false =>
    show correr aVerificadora (aConfig (some Fase.leMarcador) v [] (Simbolo.litF :: f)
        w (List.replicate i Simbolo.tick) [] out) (1 + (i + 1) + (i + 1)) = _
    rw [hcomb, correr, passo_leMarcador_litF]
    show correr aVerificadora (aConfig (some Fase.procuraBit)
        ⟨Fase.procuraBit, v.falha, false, some Simbolo.litF, v.bitLido⟩ [] f w
        (List.replicate i Simbolo.tick) [] out) ((i + 1) + (i + 1)) = _
    rw [correr_add,
      procuraBit_fecha ⟨Fase.procuraBit, v.falha, false, some Simbolo.litF, v.bitLido⟩ f [] out i w
        hle]
    simp only [Option.bind_some]
    rw [hfuel, restauraTestemunha_fecha]
    rw [List.append_nil, List.reverse_reverse, List.take_append_drop]

/-- ★★ **O LITERAL VERIFICA**: com os ticks à frente do marcador (a fita invertida), a testemunha
cheia e o índice no alcance, o literal processa-se por inteiro: o casamento é o bit batendo a
polaridade. -/
theorem o_literal_verifica (v : EstadoV) (w f : List Simbolo) (out : List Bool)
    (i : ℕ) (pol : Bool) (hle : i ≤ w.length) :
    correr aVerificadora
      (aConfig (some Fase.leMarcador) v []
        (List.replicate i Simbolo.tick ++ (if pol then Simbolo.litT else Simbolo.litF) :: f)
        w [] [] out) (i + 1 + (i + 1) + (i + 1))
      = some (if ((w.drop i).head?.isSome && (((w.drop i).head? == some Simbolo.um) == pol)) then
          aConfig (some Fase.saltaLiterais)
            ⟨Fase.procuraBit, false, pol, none, (w.drop i).head?⟩ [] f w [] [] out
        else aConfig (some Fase.leMarcador)
            ⟨Fase.procuraBit, v.falha, pol, none, (w.drop i).head?⟩ [] f w [] [] out) := by
  cases i with
  | zero =>
    exact o_cauda_do_literal v w f out 0 pol (Nat.zero_le _)
  | succ j =>
    have hcomb : j + 1 + 1 + (j + 1 + 1) + (j + 1 + 1)
        = (j + 1) + (1 + (j + 1 + 1) + (j + 1 + 1)) := by omega
    rw [hcomb, correr_add, leMarcador_ticks_fecha]
    simp only [Option.bind_some]
    rw [List.append_nil]
    exact o_cauda_do_literal { v with lido := some Simbolo.tick } w f out (j + 1) pol hle

/-- ★★ **A CAUDA DO LITERAL ALÉM DA TESTEMUNHA**: marcado o marcador e esgotada a procura sem
inventar, o bit lido é o silêncio — a guarda da presença não casa nunca; a testemunha volta
íntegra. -/
theorem o_cauda_do_literal_alem (v : EstadoV) (w f : List Simbolo) (out : List Bool)
    (i : ℕ) (pol : Bool) (hle : w.length < i) :
    correr aVerificadora
      (aConfig (some Fase.leMarcador) v []
        ((if pol then Simbolo.litT else Simbolo.litF) :: f)
        w (List.replicate i Simbolo.tick) [] out) (1 + (i + 2) + (w.length + 1))
      = some (aConfig (some Fase.leMarcador)
          ⟨Fase.procuraBit, v.falha, pol, none, none⟩ [] f w [] [] out) := by
  have hcomb : 1 + (i + 2) + (w.length + 1) = ((i + 2) + (w.length + 1)) + 1 := by omega
  cases pol with
  | true =>
    show correr aVerificadora (aConfig (some Fase.leMarcador) v [] (Simbolo.litT :: f)
        w (List.replicate i Simbolo.tick) [] out) (1 + (i + 2) + (w.length + 1)) = _
    rw [hcomb, correr, passo_leMarcador_litT]
    show correr aVerificadora (aConfig (some Fase.procuraBit)
        ⟨Fase.procuraBit, v.falha, true, some Simbolo.litT, v.bitLido⟩ [] f w
        (List.replicate i Simbolo.tick) [] out) ((i + 2) + (w.length + 1)) = _
    rw [correr_add,
      procuraBit_alem ⟨Fase.procuraBit, v.falha, true, some Simbolo.litT, v.bitLido⟩ f [] out i w hle]
    simp only [Option.bind_some]
    have hlen : (w.reverse ++ ([] : List Simbolo)).length = w.length := by
      rw [List.length_append, List.length_reverse, List.length_nil, Nat.add_zero]
    have hfuel : w.length + 1 = (w.reverse ++ ([] : List Simbolo)).length + 1 := by rw [hlen]
    rw [hfuel, restauraTestemunha_fecha]
    simp only [List.append_nil, List.reverse_reverse]
    rfl
  | false =>
    show correr aVerificadora (aConfig (some Fase.leMarcador) v [] (Simbolo.litF :: f)
        w (List.replicate i Simbolo.tick) [] out) (1 + (i + 2) + (w.length + 1)) = _
    rw [hcomb, correr, passo_leMarcador_litF]
    show correr aVerificadora (aConfig (some Fase.procuraBit)
        ⟨Fase.procuraBit, v.falha, false, some Simbolo.litF, v.bitLido⟩ [] f w
        (List.replicate i Simbolo.tick) [] out) ((i + 2) + (w.length + 1)) = _
    rw [correr_add,
      procuraBit_alem ⟨Fase.procuraBit, v.falha, false, some Simbolo.litF, v.bitLido⟩ f [] out i w
        hle]
    simp only [Option.bind_some]
    have hlen : (w.reverse ++ ([] : List Simbolo)).length = w.length := by
      rw [List.length_append, List.length_reverse, List.length_nil, Nat.add_zero]
    have hfuel : w.length + 1 = (w.reverse ++ ([] : List Simbolo)).length + 1 := by rw [hlen]
    rw [hfuel, restauraTestemunha_fecha]
    simp only [List.append_nil, List.reverse_reverse]
    rfl

/-- ★★ **O LITERAL ALÉM DA TESTEMUNHA**: com os ticks à frente do marcador e o índice além do
alcance, o bit lido é o silêncio e a guarda não casa — o literal não serve. -/
theorem o_literal_verifica_alem (v : EstadoV) (w f : List Simbolo) (out : List Bool)
    (i : ℕ) (pol : Bool) (hle : w.length < i) :
    correr aVerificadora
      (aConfig (some Fase.leMarcador) v []
        (List.replicate i Simbolo.tick ++ (if pol then Simbolo.litT else Simbolo.litF) :: f)
        w [] [] out) (i + 1 + (i + 2) + (w.length + 1))
      = some (aConfig (some Fase.leMarcador)
          ⟨Fase.procuraBit, v.falha, pol, none, none⟩ [] f w [] [] out) := by
  cases i with
  | zero => simp at hle
  | succ j =>
    have hcomb : j + 1 + 1 + (j + 1 + 2) + (w.length + 1)
        = (j + 1) + (1 + (j + 1 + 2) + (w.length + 1)) := by omega
    rw [hcomb, correr_add, leMarcador_ticks_fecha]
    simp only [Option.bind_some]
    rw [List.append_nil]
    exact o_cauda_do_literal_alem { v with lido := some Simbolo.tick } w f out (j + 1) pol hle

/-- ★★★ **O LITERAL COBRE TODO ÍNDICE, COM A COTA**: dentro do alcance ou além, o literal
processa-se com combustível limitado — a cota do literal é linear no índice e na testemunha. -/
theorem o_literal_cobre (v : EstadoV) (w f : List Simbolo) (out : List Bool)
    (i : ℕ) (pol : Bool) :
    ∃ fuel : ℕ, fuel ≤ 3 * (i + 1) + (w.length + 1) + 3 ∧
      correr aVerificadora
        (aConfig (some Fase.leMarcador) v []
          (List.replicate i Simbolo.tick ++ (if pol then Simbolo.litT else Simbolo.litF) :: f)
          w [] [] out) fuel
        = some (if ((w.drop i).head?.isSome && (((w.drop i).head? == some Simbolo.um) == pol)) then
            aConfig (some Fase.saltaLiterais)
              ⟨Fase.procuraBit, false, pol, none, (w.drop i).head?⟩ [] f w [] [] out
          else aConfig (some Fase.leMarcador)
              ⟨Fase.procuraBit, v.falha, pol, none, (w.drop i).head?⟩ [] f w [] [] out) := by
  cases Nat.lt_or_ge i w.length with
  | inl h =>
    refine ⟨i + 1 + (i + 1) + (i + 1), by omega,
      o_literal_verifica v w f out i pol (Nat.le_of_lt h)⟩
  | inr h =>
    cases Nat.eq_or_lt_of_le h with
    | inl heq =>
      refine ⟨i + 1 + (i + 1) + (i + 1), by omega,
        o_literal_verifica v w f out i pol (Nat.le_of_eq heq.symm)⟩
    | inr hlt =>
      have hdrop : w.drop i = [] := by
        have hlen : (w.drop i).length = 0 := by
          rw [List.length_drop]
          omega
        exact List.eq_nil_of_length_eq_zero hlen
      have hcond : ((w.drop i).head?.isSome && (((w.drop i).head? == some Simbolo.um) == pol))
          = false := by
        rw [hdrop]
        rfl
      rw [hcond, hdrop, if_neg (by decide : ¬(false = true))]
      exact ⟨i + 1 + (i + 2) + (w.length + 1), by omega,
        o_literal_verifica_alem v w f out i pol hlt⟩

/-- ★ **O PASSO DA VERDADE**: escreve a luz na saída e vai à parada. -/
theorem passo_escreveVerdade (v : EstadoV) (f t c r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.escreveVerdade) v [] f t c r out)
      = some (aConfig (some Fase.parada) v [] f t c r (true :: out)) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe]
  congr
  try (funext k; cases k <;> rfl)
  try rfl

/-- ★ **O PASSO DA MENTIRA**: escreve a ausência na saída e vai à parada. -/
theorem passo_escreveMentira (v : EstadoV) (f t c r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.escreveMentira) v [] f t c r out)
      = some (aConfig (some Fase.parada) v [] f t c r (false :: out)) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe]
  congr
  try (funext k; cases k <;> rfl)
  try rfl

/-- ★ **O PASSO DA PARADA**: a parada vai à limpeza da fórmula. -/
theorem passo_parada (v : EstadoV) (f t c r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.parada) v [] f t c r out)
      = some (aConfig (some Fase.limpaFormula) v [] f t c r out) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig]
  rfl

/-- ★ **O PASSO DA LIMPA-FÓRMULA, COM SÍMBOLO**: drena um símbolo da fórmula. -/
theorem passo_limpaFormula_cons (v : EstadoV) (s₀ : Simbolo) (f' t c r : List Simbolo)
    (out : List Bool) :
    Turing.FinTM2.step aVerificadora
      (aConfig (some Fase.limpaFormula) v [] (s₀ :: f') t c r out)
      = some (aConfig (some Fase.limpaFormula) { v with lido := some s₀ } [] f' t c r out) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
    List.head?_cons]
  congr
  try (funext k; cases k <;> rfl)
  try rfl

/-- ★ **O PASSO DA LIMPA-FÓRMULA, VAZIA**: a fórmula limpa, à limpa-testemunha. -/
theorem passo_limpaFormula_nil (v : EstadoV) (t c r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora
      (aConfig (some Fase.limpaFormula) v [] [] t c r out)
      = some (aConfig (some Fase.limpaTestemunha) { v with lido := none } [] [] t c r out) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe]
  congr
  try (funext k; cases k <;> rfl)
  try rfl

/-- ★ **O PASSO DA LIMPA-TESTEMUNHA, COM SÍMBOLO**: drena um símbolo da testemunha. -/
theorem passo_limpaTestemunha_cons (v : EstadoV) (s₀ : Simbolo) (t' f c r : List Simbolo)
    (out : List Bool) :
    Turing.FinTM2.step aVerificadora
      (aConfig (some Fase.limpaTestemunha) v [] f (s₀ :: t') c r out)
      = some (aConfig (some Fase.limpaTestemunha) { v with lido := some s₀ } [] f t' c r out) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
    List.head?_cons]
  congr
  try (funext k; cases k <;> rfl)
  try rfl

/-- ★ **O PASSO DA LIMPA-TESTEMUNHA, VAZIA**: a testemunha limpa, o estado zera-se — ao halt final. -/
theorem passo_limpaTestemunha_nil (v : EstadoV) (f c r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora
      (aConfig (some Fase.limpaTestemunha) v [] f [] c r out)
      = some (aConfig (some Fase.paradaFinal)
          ⟨Fase.copiaEntrada, false, false, none, none⟩ [] f [] c r out) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe]
  congr
  try (funext k; cases k <;> rfl)
  try rfl

/-- ★ **O PASSO DA PARADA FINAL**: o halt — o rótulo apaga-se com a casa limpa. -/
theorem passo_paradaFinal (v : EstadoV) (f t c r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.paradaFinal) v [] f t c r out)
      = some (aConfig none v [] f t c r out) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig]
  rfl

/-- ★ **O LAÇO DA LIMPA-FÓRMULA**: drena a fórmula inteira. -/
theorem limpaFormula_fecha (v : EstadoV) (t c r : List Simbolo) (out : List Bool) :
    ∀ (f : List Simbolo),
      correr aVerificadora (aConfig (some Fase.limpaFormula) v [] f t c r out) (f.length + 1)
        = some (aConfig (some Fase.limpaTestemunha) { v with lido := none } [] [] t c r out) := by
  intro f
  induction f generalizing v with
  | nil =>
    show correr aVerificadora (aConfig (some Fase.limpaFormula) v [] [] t c r out) (0 + 1) = _
    rw [correr, passo_limpaFormula_nil]
    rfl
  | cons s₀ f' ih =>
    show correr aVerificadora (aConfig (some Fase.limpaFormula) v [] (s₀ :: f') t c r out)
        ((s₀ :: f').length + 1) = _
    rw [List.length_cons, correr, passo_limpaFormula_cons]
    show correr aVerificadora (aConfig (some Fase.limpaFormula) { v with lido := some s₀ } [] f'
        t c r out) (f'.length + 1) = _
    rw [ih]

/-- ★ **O LAÇO DA LIMPA-TESTEMUNHA**: drena a testemunha inteira; no fim, o estado zera-se. -/
theorem limpaTestemunha_fecha (v : EstadoV) (f c r : List Simbolo) (out : List Bool) :
    ∀ (t : List Simbolo),
      correr aVerificadora (aConfig (some Fase.limpaTestemunha) v [] f t c r out) (t.length + 1)
        = some (aConfig (some Fase.paradaFinal)
            ⟨Fase.copiaEntrada, false, false, none, none⟩ [] f [] c r out) := by
  intro t
  induction t generalizing v with
  | nil =>
    show correr aVerificadora (aConfig (some Fase.limpaTestemunha) v [] f [] c r out) (0 + 1) = _
    rw [correr, passo_limpaTestemunha_nil]
    rfl
  | cons s₀ t' ih =>
    show correr aVerificadora (aConfig (some Fase.limpaTestemunha) v [] f (s₀ :: t') c r out)
        ((s₀ :: t').length + 1) = _
    rw [List.length_cons, correr, passo_limpaTestemunha_cons]
    show correr aVerificadora (aConfig (some Fase.limpaTestemunha) { v with lido := some s₀ } [] f
        t' c r out) (t'.length + 1) = _
    rw [ih]

/-- ★★ **A LIMPEZA FECHA**: da parada à configuração final limpa — a fórmula e a testemunha
drenadas, o estado zerado, o rótulo apagado: a casa em ordem para o `haltList` da Mathlib. -/
theorem a_limpeza_fecha (v : EstadoV) (f t c r : List Simbolo) (out : List Bool) :
    correr aVerificadora (aConfig (some Fase.parada) v [] f t c r out)
        (1 + (f.length + 1) + (t.length + 1) + 1)
      = some (aConfig none ⟨Fase.copiaEntrada, false, false, none, none⟩ [] [] [] c r out) := by
  rw [correr, passo_parada]
  show correr aVerificadora (aConfig (some Fase.limpaFormula) v [] f t c r out)
      (1 + (f.length + 1) + (t.length + 1)) = _
  have hcomb : 1 + (f.length + 1) + (t.length + 1)
      = (f.length + 1) + (1 + (t.length + 1)) := by omega
  rw [hcomb, correr_add, limpaFormula_fecha]
  simp only [Option.bind_some]
  have hcomb2 : 1 + (t.length + 1) = (t.length + 1) + 1 := by omega
  rw [hcomb2, correr_add, limpaTestemunha_fecha]
  simp only [Option.bind_some]
  rw [correr, passo_paradaFinal]
  rfl

/-- ★ **O PASSO PARADO**: com o rótulo apagado, a máquina não anda — o `none` da Mathlib. -/
theorem passo_parada_halt (v : EstadoV) (f t c r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora (aConfig none v [] f t c r out) = none := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aConfig]
  rfl

/-- ★ **OS TRÊS PASSOS DO VAZIO**: a fórmula vazia decide a verdade — o terminador da fórmula, o
fim da entrada, a luz escrita, a parada. -/
theorem inicioClausula_vazio_fecha (v : EstadoV) (t c r : List Simbolo) (out : List Bool) :
    correr aVerificadora (aConfig (some Fase.inicioClausula) v [] [Simbolo.fimF] t c r out) 3
      = some (aConfig (some Fase.parada) { v with lido := none } [] [] t c r (true :: out)) := by
  rw [correr, passo_inicioClausula]
  show correr aVerificadora (aConfig (some Fase.inicioClausula) { v with lido := some Simbolo.fimF }
      [] [] t c r out) 2 = _
  rw [correr, passo_inicioClausula]
  show correr aVerificadora (aConfig (some Fase.escreveVerdade) { v with lido := none } [] [] t c r
      out) 1 = _
  rw [correr, passo_escreveVerdade]
  rfl

end TGLExt.AMaquinaVerifica

