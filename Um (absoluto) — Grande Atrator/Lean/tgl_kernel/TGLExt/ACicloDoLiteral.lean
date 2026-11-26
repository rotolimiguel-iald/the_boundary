import TGLExt.AFasesDaMaquina

set_option autoImplicit false
set_option maxHeartbeats 800000
set_option maxRecDepth 8192

/-!
# A simulação da máquina TM2: o ciclo do literal — o coração da verificação

## O que é

Sobre a máquina `aMaquinaTM2` e as configurações `cfg` de `AFasesDaMaquina`, o ciclo de UM literal: ler o
marcador com a polaridade, caminhar os ticks (o índice em unário), caminhar os bits da testemunha até a posição,
ler o bit, comparar com a polaridade, restaurar a testemunha na ordem original — e rotear: satisfeito leva a
`saltarResto`, falha leva a `proximoLiteralCheck`. É o trecho em que a máquina lê o bit da testemunha.

Os custos abaixo são LIMITES SUPERIORES: `EvalsToInTime f a b m` afirma que `b` é alcançado a partir de `a` em no
máximo `m` passos (por isso vale `custo_mono`, que só aumenta `m`).

## O que fica provado

1. Os passos elementares do ciclo: `passo_lerLiteral`, `passo_andarTicks_move`, `passo_andarTicks_fim`,
   `passo_andarBits_move`, `passo_andarBits_fimTicks`, `passo_andarBits_fimBits`, `passo_lerBitAtual`,
   `passo_esvaziarTicks_move`, `passo_esvaziarTicks_fim`, `passo_comparar_satisfeito`, `passo_comparar_falha`,
   `passo_clausulaSatisfeita`, `passo_proximoLiteral`, `passo_restaurar_move`, `passo_restaurar_fim`.
2. Os laços: `fase_andarTicks` (os ticks à pilha de trabalho), `fase_andarBits_coberto` (a caminhada dos bits
   quando o índice cabe na testemunha, `i ≤ |E|`), `fase_andarBits_alem` (a testemunha acaba antes, `|E| < i`:
   o bit é ausente), `fase_esvaziarTicks` (descartar os ticks que sobram), `fase_restaurar` (o depósito `X` volta
   à entrada como `X.reverse ++ e`: a testemunha devolvida na ordem original).
3. ★ Os três ciclos completos do literal, com limite de custo: `ciclo_literal_satisfeito` (no máximo `3·i + 7`
   passos, destino `saltarResto`), `ciclo_literal_falha` (o bit na posição não é o esperado, inclusive em
   `i = |E|`, onde não há bit: no máximo `3·i + 7`, destino `proximoLiteralCheck`), `ciclo_literal_fora` (índice
   além da testemunha, `|E| < i`: no máximo `2·i + |E| + 7`, destino `proximoLiteralCheck` — o `none` da borda
   falha o literal; quando `E` é a testemunha codificada, é o teste `bits[i]? == some b` da `verificaNaFita`,
   ver `o_teste_e_o_da_fita`). Nos três, partindo com as pilhas de trabalho e de depósito vazias e com o símbolo
   seguinte da fórmula diferente de tick, a testemunha volta intacta e essas pilhas terminam vazias. A ordem dos
   argumentos de `cfg` é: rótulo, estado, entrada, fórmula, trabalho, depósito, saída.
4. A ponte semântica do bit: `bit_na_fita` (o símbolo na posição `i` da testemunha codificada) e
   `o_teste_e_o_da_fita` (`(codificaBits bits)[i]? = esperado b ↔ (bits[i]? == some b)`: o teste da máquina é o
   teste por literal da `verificaNaFita`).

## O que NÃO se afirma

* O laço das cláusulas (a composição dos ciclos com `lerClausula`, `saltarResto`, `proximoLiteralCheck`) não é
  provado neste módulo.
* Este módulo não afirma que SAT se decide em tempo polinomial: prova os passos e os limites de custo do ciclo de
  um literal, com a testemunha dada na fita.
* Nada sobre a Palavra de toda CNF plana. Nada sobre a natureza. O gate não se move.

Sem `sorry`, sem `axiom` declarado neste módulo.
-/

namespace TGLExt.ACicloDoLiteral

open TGLExt.APonteAoTM2 TGLExt.AMaquinaTM2DaVerificacao TGLExt.AFasesDaMaquina StateTransition

/-- O bit esperado pela polaridade, como símbolo da fita. -/
def esperado (b : Bool) : Option Simbolo := if b then some Simbolo.um else some Simbolo.zero

/-- `x <|> none` é o próprio `x`. (Auxiliar dos estados dos laços.) -/
theorem ou_none (x : Option Simbolo) : (x <|> none) = x := by cases x <;> rfl

/-- Igualdade no início: os passos são os mesmos, a partida é reescrita. (Auxiliar da composição.) -/
def evals_congr_inicio {a₁ a₂ : Turing.TM2.Cfg stackAlphabet Rotulo Estado}
    {c : Option (Turing.TM2.Cfg stackAlphabet Rotulo Estado)} {m : ℕ}
    (h : a₁ = a₂) (hE : EvalsToInTime aMaquinaTM2.step a₁ c m) : EvalsToInTime aMaquinaTM2.step a₂ c m :=
  h ▸ hE

/-! ## Os passos do ciclo -/

/-- Ler o marcador: a polaridade vai à bandeira; o índice (os ticks) fica na pilha da fórmula. -/
theorem passo_lerLiteral (v : Estado) (e : List Simbolo) (i : ℕ) (b : Bool) (rest t d : List Simbolo)
    (sa : List Bool) :
    aMaquinaTM2.step (cfg .lerLiteral v e (codificaLiteral (i, b) ++ rest) t d sa)
      = some (cfg .andarTicks (v.1, b, v.2.2) e (List.replicate i Simbolo.tick ++ rest) t d sa) := by
  cases b <;>
    simp only [codificaLiteral, aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma,
      Turing.TM2.stepAux, cfg, cfgStacks, List.head?_cons, List.tail_cons] <;>
    apply algum_inj <;>
    apply cfg_fecha rfl rfl <;>
    intro k <;> fin_cases k <;> simp [cfgStacks, Function.update]

/-- A caminhada dos ticks com tick à vista: move-o à pilha de trabalho. -/
theorem passo_andarTicks_move (v : Estado) (e : List Simbolo) (rest t d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .andarTicks v e (Simbolo.tick :: rest) t d sa)
      = some (cfg .andarTicks (some Simbolo.tick, v.2.1, true) e rest (Simbolo.tick :: t) d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_cons, List.tail_cons, Option.getD_some]
  apply algum_inj
  apply cfg_fecha rfl rfl
  intro k; fin_cases k <;> simp [cfgStacks, Function.update]

/-- A caminhada dos ticks no fim do índice (o próximo símbolo não é tick): vai caminhar os bits. -/
theorem passo_andarTicks_fim (v : Estado) (e rest t d : List Simbolo) (sa : List Bool)
    (ht : rest.head? ≠ some Simbolo.tick) :
    aMaquinaTM2.step (cfg .andarTicks v e rest t d sa)
      = some (cfg .andarBits (v.1, v.2.1, false) e rest t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks]
  simp only [show (rest.head? = some Simbolo.tick) ↔ False from by simp [ht]]
  rfl

/-- A caminhada dos bits com tick e bit à vista: o tick sai da pilha de trabalho, o bit vai ao depósito. -/
theorem passo_andarBits_move (v : Estado) (s : Simbolo) (restE rest t d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .andarBits v (s :: restE) rest (Simbolo.tick :: t) d sa)
      = some (cfg .andarBits (some s, v.2.1, false) restE rest t (s :: d) sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_cons, List.tail_cons, Option.getD_some]
  simp only [show ((some Simbolo.tick : Option Simbolo) = none) ↔ False from by simp,
    show ((some s : Option Simbolo) = none) ↔ False from by simp]
  apply algum_inj
  apply cfg_fecha rfl rfl
  intro k; fin_cases k <;> simp [cfgStacks, Function.update]

/-- A caminhada dos bits com os ticks pagos: vai ler o bit da posição. -/
theorem passo_andarBits_fimTicks (v : Estado) (e rest d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .andarBits v e rest [] d sa)
      = some (cfg .lerBitAtual (v.1, v.2.1, true) e rest [] d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_nil]
  rfl

/-- A caminhada dos bits com a testemunha esgotada antes dos ticks: vai esvaziar os ticks (o bit é ausente). -/
theorem passo_andarBits_fimBits (v : Estado) (rest t d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .andarBits v [] rest (Simbolo.tick :: t) d sa)
      = some (cfg .esvaziarTicks (v.1, v.2.1, true) [] rest (Simbolo.tick :: t) d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_nil, List.head?_cons]
  simp only [show ((some Simbolo.tick : Option Simbolo) = none) ↔ False from by simp]
  rfl

/-- Ler o bit da posição: o carregador recebe a cabeça da testemunha restante (ou `none` na borda). -/
theorem passo_lerBitAtual (v : Estado) (e rest d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .lerBitAtual v e rest [] d sa)
      = some (cfg .comparar (e.head?, v.2.1, v.2.2) e rest [] d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks]

/-- Esvaziar os ticks com tick à vista: descarta-o. -/
theorem passo_esvaziarTicks_move (v : Estado) (e rest t d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .esvaziarTicks v e rest (Simbolo.tick :: t) d sa)
      = some (cfg .esvaziarTicks (v.1, v.2.1, false) e rest t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_cons, List.tail_cons]
  simp only [show ((some Simbolo.tick : Option Simbolo) = none) ↔ False from by simp]
  apply algum_inj
  apply cfg_fecha rfl rfl
  intro k; fin_cases k <;> simp [cfgStacks, Function.update]

/-- Esvaziar os ticks no fim: o carregador fica `none` (o bit ausente) e vai comparar. -/
theorem passo_esvaziarTicks_fim (v : Estado) (e rest d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .esvaziarTicks v e rest [] d sa)
      = some (cfg .comparar (none, v.2.1, true) e rest [] d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_nil]
  rfl

/-- Comparar com acerto: a bandeira fica verdadeira e vai marcar a cláusula como satisfeita. -/
theorem passo_comparar_satisfeito (v : Estado) (e rest t d : List Simbolo) (sa : List Bool)
    (h : v.1 = (if v.2.1 then some Simbolo.um else some Simbolo.zero)) :
    aMaquinaTM2.step (cfg .comparar v e rest t d sa)
      = some (cfg .clausulaSatisfeita (none, true, v.2.2) e rest t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks]
  simp only [show (v.1 = if v.2.1 then some Simbolo.um else some Simbolo.zero) ↔ True from iff_true_intro h]
  rfl

/-- Comparar sem acerto: a bandeira fica falsa e vai ao próximo literal. -/
theorem passo_comparar_falha (v : Estado) (e rest t d : List Simbolo) (sa : List Bool)
    (h : ¬ (v.1 = (if v.2.1 then some Simbolo.um else some Simbolo.zero))) :
    aMaquinaTM2.step (cfg .comparar v e rest t d sa)
      = some (cfg .proximoLiteral (none, false, v.2.2) e rest t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks]
  simp only [show (v.1 = if v.2.1 then some Simbolo.um else some Simbolo.zero) ↔ False from iff_false_intro h]
  rfl

/-- Marcar a rota «satisfeito»: vai restaurar. -/
theorem passo_clausulaSatisfeita (v : Estado) (e rest t d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .clausulaSatisfeita v e rest t d sa)
      = some (cfg .restaurar (v.1, v.2.1, true) e rest t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks]

/-- Marcar a rota «falhou»: vai restaurar. -/
theorem passo_proximoLiteral (v : Estado) (e rest t d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .proximoLiteral v e rest t d sa)
      = some (cfg .restaurar (v.1, v.2.1, false) e rest t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks]

/-- O restaurar com bit no depósito: devolve-o à entrada. -/
theorem passo_restaurar_move (v : Estado) (s : Simbolo) (e rest t d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .restaurar v e rest t (s :: d) sa)
      = some (cfg .restaurar (some s, false, v.2.2) (s :: e) rest t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_cons, List.tail_cons, Option.getD_some]
  simp only [show ((some s : Option Simbolo) = none) ↔ False from by simp]
  apply algum_inj
  apply cfg_fecha rfl rfl
  intro k; fin_cases k <;> simp [cfgStacks, Function.update]

/-- O restaurar no fim: segue a rota marcada. -/
theorem passo_restaurar_fim (v : Estado) (e rest t : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .restaurar v e rest t [] sa)
      = some (cfg (if v.2.2 then Rotulo.saltarResto else Rotulo.proximoLiteralCheck)
          (v.1, true, v.2.2) e rest t [] sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_nil]
  rfl

/-! ## Os laços do ciclo -/

/-- A acumulação dos ticks: mais um tick no fim da invertida é um tick no início da direta. (Auxiliar.) -/
theorem replicate_snoc (i : ℕ) (T : List Simbolo) :
    List.replicate i Simbolo.tick ++ (Simbolo.tick :: T) = (Simbolo.tick :: List.replicate i Simbolo.tick) ++ T := by
    induction i generalizing T with
    | zero => rfl
    | succ i ih => simp [List.replicate_succ, List.cons_append, ih]

/-- ★ O LAÇO DOS TICKS: o índice sai da pilha da fórmula para a pilha de trabalho. No máximo `i + 1` passos
(um por tick, mais a detecção do fim). -/
def fase_andarTicks (v : Estado) (e : List Simbolo) (i : ℕ) (rest T d : List Simbolo) (sa : List Bool)
    (ht : rest.head? ≠ some Simbolo.tick) :
    EvalsToInTime aMaquinaTM2.step (cfg .andarTicks v e (List.replicate i Simbolo.tick ++ rest) T d sa)
      (some (cfg .andarBits ((List.replicate i Simbolo.tick).reverse.head? <|> v.1, v.2.1, false) e rest
        (List.replicate i Simbolo.tick ++ T) d sa)) (i + 1) := by
  induction i generalizing v T with
  | zero =>
    have h1 := passo_andarTicks_fim v e rest T d sa ht
    refine evals_congr ?_ (passo_unico h1)
    simp [List.replicate_zero, List.nil_append]
  | succ i ih =>
    have h1 := passo_andarTicks_move v e (List.replicate i Simbolo.tick ++ rest) T d sa
    have h2 := ih (some Simbolo.tick, v.2.1, true) (Simbolo.tick :: T)
    refine evals_congr ?_ (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h1) h2)
    apply algum_inj
    apply cfg_fecha rfl
    · show ((List.replicate i Simbolo.tick).reverse.head? <|> some Simbolo.tick,
          v.2.1, false)
        = ((List.replicate (i + 1) Simbolo.tick).reverse.head? <|> v.1, v.2.1, false)
      rw [cabeca_reverse_cons v.1 Simbolo.tick (List.replicate i Simbolo.tick)]
      rw [← List.replicate_succ]
    · intro k; fin_cases k <;> simp [cfgStacks, List.replicate_succ, List.reverse_replicate,
        List.singleton_append, List.append_assoc, List.replicate_add, replicate_snoc]

/-- ★ O LAÇO DOS BITS, COM O ÍNDICE NA TESTEMUNHA: `i` bits vão ao depósito (invertidos), a testemunha fica a
partir da posição `i`. No máximo `i + 1` passos (um por bit, mais a detecção). -/
def fase_andarBits_coberto (v : Estado) (i : ℕ) (E rest D : List Simbolo) (sa : List Bool) (hi : i ≤ E.length) :
    EvalsToInTime aMaquinaTM2.step (cfg .andarBits v E rest (List.replicate i Simbolo.tick) D sa)
      (some (cfg .lerBitAtual ((E.take i).reverse.head? <|> v.1, v.2.1, true) (E.drop i) rest []
        ((E.take i).reverse ++ D) sa)) (i + 1) := by
  induction i generalizing v E D with
  | zero =>
    have h1 := passo_andarBits_fimTicks v E rest D sa
    refine evals_congr ?_ (passo_unico h1)
    simp [List.take_zero, List.drop_zero]
  | succ i ih =>
    cases E with
    | nil => simp at hi
    | cons b0 E' =>
      have h1 := passo_andarBits_move v b0 E' rest (List.replicate i Simbolo.tick) D sa
      have hi' : i ≤ E'.length := by simp only [List.length_cons] at hi; omega
      have h2 := ih (some b0, v.2.1, false) E' (b0 :: D) hi'
      refine evals_congr ?_ (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h1) h2)
      apply algum_inj
      apply cfg_fecha rfl
      · simp [cabeca_reverse_cons, List.take_succ_cons]
      · intro k; fin_cases k <;>
          simp [cfgStacks, List.take_succ_cons, List.drop_succ_cons, List.reverse_cons, List.append_assoc,
            List.singleton_append]

/-- ★ O LAÇO DOS BITS, COM O ÍNDICE ALÉM DA TESTEMUNHA: a testemunha esvazia-se inteira ao depósito, e os
ticks que sobram ficam para o esvaziamento. No máximo `|E| + 1` passos (um por bit, mais a detecção). -/
def fase_andarBits_alem (v : Estado) (i : ℕ) (E rest D : List Simbolo) (sa : List Bool) (hi : E.length < i) :
    EvalsToInTime aMaquinaTM2.step (cfg .andarBits v E rest (List.replicate i Simbolo.tick) D sa)
      (some (cfg .esvaziarTicks (E.reverse.head? <|> v.1, v.2.1, true) [] rest
        (List.replicate (i - E.length) Simbolo.tick) (E.reverse ++ D) sa)) (E.length + 1) := by
  induction E generalizing v i D with
  | nil =>
    cases i with
    | zero => simp at hi
    | succ i' =>
      have h1 := passo_andarBits_fimBits v rest (List.replicate i' Simbolo.tick) D sa
      refine evals_congr_inicio (by simp [List.replicate_succ]) (evals_congr ?_ (passo_unico h1))
      apply algum_inj
      apply cfg_fecha rfl
      · simp [List.reverse_nil]
      · intro k; fin_cases k <;> simp [cfgStacks, List.replicate_succ, List.reverse_nil, List.nil_append,
          Nat.sub_zero]
  | cons b0 E' ih =>
    cases i with
    | zero => simp at hi
    | succ i' =>
      have h1 := passo_andarBits_move v b0 E' rest (List.replicate i' Simbolo.tick) D sa
      have hi' : E'.length < i' := by simp only [List.length_cons] at hi; omega
      have h2 := ih (some b0, v.2.1, false) i' (b0 :: D) hi'
      refine evals_congr_inicio (by simp [List.replicate_succ])
        (evals_congr ?_ (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h1) h2))
      apply algum_inj
      apply cfg_fecha rfl
      · simp [cabeca_reverse_cons]
      · intro k; fin_cases k <;>
          simp [cfgStacks, List.reverse_cons, List.append_assoc, List.singleton_append, List.length_cons,
            Nat.succ_sub_one]

/-- ★ O ESVAZIAMENTO DOS TICKS: descarta os que sobram, com o carregador `none` (o bit ausente). No máximo
`j + 1` passos. -/
def fase_esvaziarTicks (v : Estado) (j : ℕ) (e rest D : List Simbolo) (sa : List Bool) :
    EvalsToInTime aMaquinaTM2.step (cfg .esvaziarTicks v e rest (List.replicate j Simbolo.tick) D sa)
      (some (cfg .comparar (none, v.2.1, true) e rest [] D sa)) (j + 1) := by
  induction j generalizing v with
  | zero =>
    exact evals_congr_inicio (by simp [List.replicate_zero]) (passo_unico (passo_esvaziarTicks_fim v e rest D sa))
  | succ j ih =>
    have h1 := passo_esvaziarTicks_move v e rest (List.replicate j Simbolo.tick) D sa
    have h2 := ih (v.1, v.2.1, false)
    exact evals_congr_inicio (by simp [List.replicate_succ])
      (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h1) h2)

/-- ★ O RESTAURAR: o depósito `X` volta à entrada como `X.reverse ++ e` (a ordem original) e a máquina segue a
rota marcada. No máximo `|X| + 1` passos (um por símbolo, mais a detecção). -/
def fase_restaurar (v : Estado) (X e rest t : List Simbolo) (sa : List Bool) :
    EvalsToInTime aMaquinaTM2.step (cfg .restaurar v e rest t X sa)
      (some (cfg (if v.2.2 then Rotulo.saltarResto else Rotulo.proximoLiteralCheck)
        (X.reverse.head? <|> v.1, true, v.2.2) (X.reverse ++ e) rest t [] sa)) (X.length + 1) := by
  induction X generalizing v e with
  | nil =>
    have h1 := passo_restaurar_fim v e rest t sa
    refine evals_congr ?_ (passo_unico h1)
    simp [List.reverse_nil, List.nil_append]
  | cons s X' ih =>
    have h1 := passo_restaurar_move v s e rest t X' sa
    have h2 := ih (some s, false, v.2.2) (s :: e)
    refine evals_congr ?_ (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h1) h2)
    apply algum_inj
    apply cfg_fecha rfl
    · simp [cabeca_reverse_cons]
    · intro k; fin_cases k <;> simp [cfgStacks, List.reverse_cons, List.append_assoc, List.singleton_append]

/-! ## O ciclo completo do literal -/

/-- A testemunha tem a posição lida. (Auxiliar da composição.) -/
theorem alcance_da_leitura {E : List Simbolo} {i : ℕ} {w : Simbolo} (h : E[i]? = some w) :
    i < E.length :=
  (List.getElem?_eq_some_iff.mp h).1

/-- ★ **O CICLO DO LITERAL COM ACERTO**: o bit na posição do índice é o esperado pela polaridade — a máquina vai
a `saltarResto`, com a testemunha intacta e as pilhas de trabalho vazias. No máximo `3·i + 7` passos. -/
def ciclo_literal_satisfeito (v : Estado) (i : ℕ) (b : Bool) (rest E : List Simbolo) (sa : List Bool)
    (ht : rest.head? ≠ some Simbolo.tick) (hbit : E[i]? = esperado b) :
    EvalsToInTime aMaquinaTM2.step (cfg .lerLiteral v E (codificaLiteral (i, b) ++ rest) [] [] sa)
      (some (cfg .saltarResto ((E.take i).head?, true, true) E rest [] [] sa)) (3 * i + 7) := by
  have h1 := passo_lerLiteral v E i b rest [] [] sa
  have h2 := fase_andarTicks (v.1, b, v.2.2) E i rest [] [] sa ht
  simp only [List.append_nil] at h2
  have hi : i ≤ E.length := by
    cases b
    · exact (List.getElem?_eq_some_iff.mp (by simpa [esperado] using hbit)).1.le
    · exact (List.getElem?_eq_some_iff.mp (by simpa [esperado] using hbit)).1.le
  have h3 := fase_andarBits_coberto ((List.replicate i Simbolo.tick).reverse.head? <|> v.1, b, false)
    i E rest [] sa hi
  simp only [List.append_nil] at h3
  have h4 := passo_lerBitAtual ((E.take i).reverse.head? <|> ((List.replicate i Simbolo.tick).reverse.head? <|> v.1), b, true) (E.drop i) rest ((E.take i).reverse) sa
  dsimp only [] at h4
  have hbit' : (E.drop i).head? = (if b then some Simbolo.um else some Simbolo.zero) := by
    rw [List.head?_drop]; exact hbit
  have h5 := passo_comparar_satisfeito ((E.drop i).head?, b, true) (E.drop i) rest [] ((E.take i).reverse) sa
    hbit'
  dsimp only [] at h5
  have h6 := passo_clausulaSatisfeita ((none, true, true) : Estado) (E.drop i) rest [] ((E.take i).reverse) sa
  dsimp only [] at h6
  have h7 := fase_restaurar ((none, true, true) : Estado) ((E.take i).reverse) (E.drop i) rest [] sa
  dsimp only [] at h7
  have hcomp := EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h1)
    (EvalsToInTime.trans _ _ _ _ _ _ h2
      (EvalsToInTime.trans _ _ _ _ _ _ h3
        (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h4)
          (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h5)
            (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h6) h7)))))
  refine custo_mono (evals_congr ?_ hcomp) ?_
  · apply algum_inj
    apply cfg_fecha rfl
    · show (((E.take i).reverse).reverse.head? <|> none, true, true)
        = ((E.take i).head?, true, true)
      simp [List.reverse_reverse, ou_none]
    · intro k; fin_cases k <;>
        simp [cfgStacks, List.reverse_reverse, List.take_append_drop, List.length_reverse,
          List.length_take, min_eq_left hi]
  · simp only [List.length_reverse, List.length_take, min_eq_left hi]; omega

/-- ★ **O CICLO DO LITERAL SEM ACERTO, NO ALCANCE** (`i ≤ |E|`): o bit na posição não é o esperado (em
`i = |E|` não há bit) — a máquina vai a `proximoLiteralCheck`, com a testemunha intacta. No máximo `3·i + 7`
passos. -/
def ciclo_literal_falha (v : Estado) (i : ℕ) (b : Bool) (rest E : List Simbolo) (sa : List Bool)
    (ht : rest.head? ≠ some Simbolo.tick) (hi : i ≤ E.length) (hne : E[i]? ≠ esperado b) :
    EvalsToInTime aMaquinaTM2.step (cfg .lerLiteral v E (codificaLiteral (i, b) ++ rest) [] [] sa)
      (some (cfg .proximoLiteralCheck ((E.take i).head?, true, false) E rest [] [] sa)) (3 * i + 7) := by
  have h1 := passo_lerLiteral v E i b rest [] [] sa
  have h2 := fase_andarTicks (v.1, b, v.2.2) E i rest [] [] sa ht
  simp only [List.append_nil] at h2
  have h3 := fase_andarBits_coberto ((List.replicate i Simbolo.tick).reverse.head? <|> v.1, b, false)
    i E rest [] sa hi
  simp only [List.append_nil] at h3
  have h4 := passo_lerBitAtual ((E.take i).reverse.head? <|> ((List.replicate i Simbolo.tick).reverse.head? <|> v.1), b, true) (E.drop i) rest ((E.take i).reverse) sa
  dsimp only [] at h4
  have hne' : (E.drop i).head? ≠ (if b then some Simbolo.um else some Simbolo.zero) := by
    rw [List.head?_drop]; exact hne
  have h5 := passo_comparar_falha ((E.drop i).head?, b, true) (E.drop i) rest [] ((E.take i).reverse) sa hne'
  dsimp only [] at h5
  have h6 := passo_proximoLiteral ((none, false, true) : Estado) (E.drop i) rest [] ((E.take i).reverse) sa
  dsimp only [] at h6
  have h7 := fase_restaurar ((none, false, false) : Estado) ((E.take i).reverse) (E.drop i) rest [] sa
  dsimp only [] at h7
  have hcomp := EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h1)
    (EvalsToInTime.trans _ _ _ _ _ _ h2
      (EvalsToInTime.trans _ _ _ _ _ _ h3
        (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h4)
          (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h5)
            (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h6) h7)))))
  refine custo_mono (evals_congr ?_ hcomp) ?_
  · apply algum_inj
    apply cfg_fecha rfl
    · show (((E.take i).reverse).reverse.head? <|> none, true, false)
        = ((E.take i).head?, true, false)
      simp [List.reverse_reverse, ou_none]
    · intro k; fin_cases k <;>
        simp [cfgStacks, List.reverse_reverse, List.take_append_drop, List.length_reverse,
          List.length_take, min_eq_left hi]
  · simp only [List.length_reverse, List.length_take, min_eq_left hi]; omega

/-- ★ **O CICLO DO LITERAL ALÉM DA TESTEMUNHA**: o índice não tem bit — o `none` da borda falha o literal, como
na `verificaNaFita` — e a máquina vai a `proximoLiteralCheck`, com a testemunha intacta. No máximo
`2·i + |E| + 7` passos. -/
def ciclo_literal_fora (v : Estado) (i : ℕ) (b : Bool) (rest E : List Simbolo) (sa : List Bool)
    (ht : rest.head? ≠ some Simbolo.tick) (hi : E.length < i) :
    EvalsToInTime aMaquinaTM2.step (cfg .lerLiteral v E (codificaLiteral (i, b) ++ rest) [] [] sa)
      (some (cfg .proximoLiteralCheck (E.head?, true, false) E rest [] [] sa)) (2 * i + E.length + 7) := by
  have h1 := passo_lerLiteral v E i b rest [] [] sa
  have h2 := fase_andarTicks (v.1, b, v.2.2) E i rest [] [] sa ht
  simp only [List.append_nil] at h2
  have h3 := fase_andarBits_alem ((List.replicate i Simbolo.tick).reverse.head? <|> v.1, b, false)
    i E rest [] sa hi
  simp only [List.append_nil] at h3
  have h4 := fase_esvaziarTicks (E.reverse.head? <|> ((List.replicate i Simbolo.tick).reverse.head? <|> v.1), b, true) (i - E.length) [] rest (E.reverse) sa
  dsimp only [] at h4
  have h5 := passo_comparar_falha ((none, b, true) : Estado) [] rest [] (E.reverse) sa (by cases b <;> simp [esperado])
  dsimp only [] at h5
  have h6 := passo_proximoLiteral ((none, false, true) : Estado) [] rest [] (E.reverse) sa
  dsimp only [] at h6
  have h7 := fase_restaurar ((none, false, false) : Estado) E.reverse [] rest [] sa
  dsimp only [] at h7
  have hcomp := EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h1)
    (EvalsToInTime.trans _ _ _ _ _ _ h2
      (EvalsToInTime.trans _ _ _ _ _ _ h3
        (EvalsToInTime.trans _ _ _ _ _ _ h4
          (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h5)
            (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h6) h7)))))
  refine custo_mono (evals_congr ?_ hcomp) ?_
  · apply algum_inj
    apply cfg_fecha rfl
    · show (E.reverse.reverse.head? <|> none, true, false)
        = (E.head?, true, false)
      simp [List.reverse_reverse, ou_none]
    · intro k; fin_cases k <;>
        simp [cfgStacks, List.reverse_reverse, List.nil_append, List.append_nil]
  · simp only [List.length_reverse]; omega

/-! ## A ponte semântica do bit -/

/-- O bit na posição `i` da testemunha na fita é o mapeamento do bit da testemunha. (Auxiliar da ponte.) -/
theorem bit_na_fita (bits : List Bool) (i : ℕ) :
    (codificaBits bits)[i]? = (bits[i]?).map fun b => if b then Simbolo.um else Simbolo.zero := by
  simp [codificaBits, List.getElem?_map]

/-- ★ **O TESTE DA MÁQUINA É O TESTE DA FITA**: conferir o símbolo esperado na posição é o `==` da
`verificaNaFita`. -/
theorem o_teste_e_o_da_fita (bits : List Bool) (i : ℕ) (b : Bool) :
    ((codificaBits bits)[i]? = esperado b) ↔ (bits[i]? == some b) := by
  rw [bit_na_fita, esperado]
  cases hb : bits[i]? with
  | none => cases b <;> simp
  | some w => cases w <;> cases b <;> simp

end TGLExt.ACicloDoLiteral

