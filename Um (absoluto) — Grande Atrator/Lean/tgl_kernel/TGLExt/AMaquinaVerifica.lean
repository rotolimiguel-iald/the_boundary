import TGLExt.AVerificadoraTM2

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A máquina verifica: a cópia, a colheita e o início

## O que é

A correção da verificadora TM2 (`aVerificadora`) prova-se fase a fase: cada laço da máquina tem a sua forma fechada. Este módulo
fecha os laços mecânicos do começo da corrida.

## O que fica provado [REAL]

1. `stkDe`, `aConfig` [DEF]: as pilhas e a configuração, por nome.
2. ★ `passo_copiaEntrada` e `copiaEntrada_fecha`: a cópia inverte a entrada e passa à colheita.
3. ★ `passo_colheTestemunha_nil`, `passo_colheTestemunha_sep`, `passo_colheTestemunha_cons` e `colheTestemunha_fecha`: a colheita
   separa os bits da testemunha.
4. ★ `passo_inicioClausula` e `inicioClausula_fecha`: o início pula o terminador da fórmula e abre a cláusula com a falha armada.

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

/-- [DEF] As pilhas da configuração, por nome. -/
def stkDe (e f t c r : List Simbolo) (out : List Bool) : ∀ k : PV, List (aVerificadora.Γ k)
  | PV.entrada => e
  | PV.saida => out
  | PV.formula => f
  | PV.testemunha => t
  | PV.contagem => c
  | PV.restaura => r

/-- [DEF] A configuração por nome. -/
def aConfig (l : Option Fase) (v : EstadoV) (e f t c r : List Simbolo) (out : List Bool) :
    aVerificadora.Cfg :=
  ⟨l, v, stkDe e f t c r out⟩

/-- ★ **O PASSO DA CÓPIA**: com a entrada não-vazia, o passo move um símbolo; vazia, passa à colheita. -/
theorem passo_copiaEntrada (v : EstadoV) (p t c r : List Simbolo) (out : List Bool) (w : List Simbolo) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.copiaEntrada) v w p t c r out)
      = some (match w with
        | [] => aConfig (some Fase.colheTestemunha) { v with lido := none } [] p t c r out
        | s₀ :: w' => aConfig (some Fase.copiaEntrada) { v with lido := some s₀ } w' (s₀ :: p) t c r out) := by
  cases w with
  | nil =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | cons s₀ w' =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons]
    congr
    try (funext k; cases k <;> rfl)
    try rfl

/-- ★ **A CÓPIA INVERTE**: o laço da `copiaEntrada` termina com a pilha `w.reverse ++ p` e vai para a
colheita. -/
theorem copiaEntrada_fecha (v : EstadoV) (t c r : List Simbolo) (out : List Bool) :
    ∀ (w p : List Simbolo),
      correr aVerificadora (aConfig (some Fase.copiaEntrada) v w p t c r out) (w.length + 1)
        = some (aConfig (some Fase.colheTestemunha) { v with lido := none }
            [] (w.reverse ++ p) t c r out) := by
  intro w
  induction w with
  | nil =>
    intro p
    rw [correr, passo_copiaEntrada]
    rfl
  | cons s₀ w' ih =>
    intro p
    have h := ih (s₀ :: p)
    show correr aVerificadora (aConfig (some Fase.copiaEntrada) v (s₀ :: w') p t c r out)
        ((s₀ :: w').length + 1) = _
    rw [List.length_cons, correr, passo_copiaEntrada, List.reverse_cons]
    show correr aVerificadora (aConfig (some Fase.copiaEntrada)
        { v with lido := some s₀ } w' (s₀ :: p) t c r out) (w'.length + 1) = _
    rw [List.append_assoc]
    exact h

/-- ★ **O PASSO DA COLHEITA, NO VAZIO**: a colheita vazia vai ao início. -/
theorem passo_colheTestemunha_nil (v : EstadoV) (t c r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.colheTestemunha) v [] [] t c r out)
      = some (aConfig (some Fase.inicioClausula) { v with lido := none } [] [] t c r out) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe]
  congr
  try (funext k; cases k <;> rfl)
  try rfl

/-- ★ **O PASSO DA COLHEITA, NA DIVISÓRIA**: a divisória é consumida e vai ao início. -/
theorem passo_colheTestemunha_sep (v : EstadoV) (f' t c r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.colheTestemunha) v [] (Simbolo.sep :: f') t c r out)
      = some (aConfig (some Fase.inicioClausula) { v with lido := some Simbolo.sep } [] f' t c r out) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
    List.head?_cons]
  congr
  try (funext k; cases k <;> rfl)
  try rfl

/-- ★ **O PASSO DA COLHEITA, NO BIT**: fora da divisória, o símbolo desce para a testemunha. -/
theorem passo_colheTestemunha_cons (v : EstadoV) (s₀ : Simbolo) (f' t c r : List Simbolo)
    (out : List Bool) (hne : s₀ ≠ Simbolo.sep) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.colheTestemunha) v [] (s₀ :: f') t c r out)
      = some (aConfig (some Fase.colheTestemunha) { v with lido := some s₀ } [] f' (s₀ :: t) c r out) := by
  cases s₀ with
  | tick =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons, (by decide : (Simbolo.tick == Simbolo.sep) = false)]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | litT =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons, (by decide : (Simbolo.litT == Simbolo.sep) = false)]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | litF =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons, (by decide : (Simbolo.litF == Simbolo.sep) = false)]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | fimC =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons, (by decide : (Simbolo.fimC == Simbolo.sep) = false)]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | fimF =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons, (by decide : (Simbolo.fimF == Simbolo.sep) = false)]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | sep =>
    exact absurd rfl hne
  | um =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons, (by decide : (Simbolo.um == Simbolo.sep) = false)]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | zero =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons, (by decide : (Simbolo.zero == Simbolo.sep) = false)]
    congr
    try (funext k; cases k <;> rfl)
    try rfl

/-- ★ **A COLHEITA SEPARA**: o laço da `colheita`, com os bits w sem divisória à frente dela e o
resto r' depois, termina com a testemunha `w.reverse ++ t` e a fórmula r'. -/
theorem colheTestemunha_fecha (v : EstadoV) (c r : List Simbolo) (out : List Bool) :
    ∀ (w t r' : List Simbolo), (∀ s₀ ∈ w, s₀ ≠ Simbolo.sep) →
      correr aVerificadora (aConfig (some Fase.colheTestemunha) v [] (w ++ Simbolo.sep :: r') t c r out)
        (w.length + 1)
        = some (aConfig (some Fase.inicioClausula) { v with lido := some Simbolo.sep }
            [] r' (w.reverse ++ t) c r out) := by
  intro w
  induction w with
  | nil =>
    intro t r' _
    rw [List.nil_append, correr, passo_colheTestemunha_sep]
    rfl
  | cons s₀ w' ih =>
    intro t r' hsep
    have hne : s₀ ≠ Simbolo.sep := hsep s₀ List.mem_cons_self
    have h := ih (s₀ :: t) r' (fun s₀ hs => hsep s₀ (List.mem_cons_of_mem _ hs))
    show correr aVerificadora (aConfig (some Fase.colheTestemunha) v [] ((s₀ :: w') ++ Simbolo.sep :: r') t c r out)
        ((s₀ :: w').length + 1) = _
    rw [List.length_cons, List.cons_append, correr, passo_colheTestemunha_cons _ _ _ _ _ _ _ hne,
      List.reverse_cons]
    show correr aVerificadora (aConfig (some Fase.colheTestemunha)
        { v with lido := some s₀ } [] (w' ++ Simbolo.sep :: r') (s₀ :: t) c r out) (w'.length + 1) = _
    rw [List.append_assoc]
    exact h

/-- ★ **O PASSO DO INÍCIO**: pula o terminador da fórmula; no terminador de cláusula, arma a falha e
vai ao marcador; no conteúdo, ao marcador; no vazio, escreve a verdade. -/
theorem passo_inicioClausula (v : EstadoV) (f t c r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.inicioClausula) v [] f t c r out)
      = some (match f with
        | [] => aConfig (some Fase.escreveVerdade) { v with lido := none } [] f t c r out
        | Simbolo.fimF :: f' => aConfig (some Fase.inicioClausula) { v with lido := some Simbolo.fimF }
            [] f' t c r out
        | Simbolo.fimC :: f' => aConfig (some Fase.leMarcador)
            { v with lido := some Simbolo.fimC, falha := true } [] f' t c r out
        | s₀ :: f' => aConfig (some Fase.leMarcador) { v with lido := some s₀ } [] (s₀ :: f') t c r out) := by
  cases f with
  | nil =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | cons s₀ f' =>
    cases s₀ with
    | tick =>
      simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
        List.head?_cons, (by decide : (Simbolo.tick == Simbolo.fimF) = false),
        (by decide : (Simbolo.tick == Simbolo.fimC) = false)]
      congr
      try (funext k; cases k <;> rfl)
      try rfl
    | litT =>
      simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
        List.head?_cons, (by decide : (Simbolo.litT == Simbolo.fimF) = false),
        (by decide : (Simbolo.litT == Simbolo.fimC) = false)]
      congr
      try (funext k; cases k <;> rfl)
      try rfl
    | litF =>
      simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
        List.head?_cons, (by decide : (Simbolo.litF == Simbolo.fimF) = false),
        (by decide : (Simbolo.litF == Simbolo.fimC) = false)]
      congr
      try (funext k; cases k <;> rfl)
      try rfl
    | fimC =>
      simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
        List.head?_cons, (by decide : (Simbolo.fimC == Simbolo.fimF) = false)]
      congr
      try (funext k; cases k <;> rfl)
      try rfl
    | fimF =>
      simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
        List.head?_cons]
      congr
      try (funext k; cases k <;> rfl)
      try rfl
    | sep =>
      simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
        List.head?_cons, (by decide : (Simbolo.sep == Simbolo.fimF) = false),
        (by decide : (Simbolo.sep == Simbolo.fimC) = false)]
      congr
      try (funext k; cases k <;> rfl)
      try rfl
    | um =>
      simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
        List.head?_cons, (by decide : (Simbolo.um == Simbolo.fimF) = false),
        (by decide : (Simbolo.um == Simbolo.fimC) = false)]
      congr
      try (funext k; cases k <;> rfl)
      try rfl
    | zero =>
      simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
        List.head?_cons, (by decide : (Simbolo.zero == Simbolo.fimF) = false),
        (by decide : (Simbolo.zero == Simbolo.fimC) = false)]
      congr
      try (funext k; cases k <;> rfl)
      try rfl

/-- ★ **O INÍCIO ABRE A CLÁUSULA**: o `inicioClausula` pula o terminador da fórmula (uma vez) e, no
terminador de cláusula, arma a falha e vai ao marcador. -/
theorem inicioClausula_fecha (v : EstadoV) (t c r : List Simbolo) (out : List Bool) (r' : List Simbolo) :
    correr aVerificadora (aConfig (some Fase.inicioClausula) v []
        (Simbolo.fimF :: Simbolo.fimC :: r') t c r out) 2
      = some (aConfig (some Fase.leMarcador) { v with lido := some Simbolo.fimC, falha := true }
          [] r' t c r out) := by
  rw [correr, passo_inicioClausula]
  show correr aVerificadora (aConfig (some Fase.inicioClausula)
      { v with lido := some Simbolo.fimF } [] (Simbolo.fimC :: r') t c r out) 1 = _
  rw [correr, passo_inicioClausula]
  rfl

end TGLExt.AMaquinaVerifica

