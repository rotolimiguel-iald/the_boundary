import TGLExt.AMaquinaVerifica2

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A máquina verifica: os laços da procura, da restauração e do salto

## O que fica provado [REAL]

1. ★ `procuraBit_fecha`: com a contagem cheia de ticks e o índice no alcance, a máquina desloca os bits e lê o bit da variável; a
   contagem zera.
2. ★ `esgotaContagem_fecha`: o esgotamento consome os ticks sem tocar em nada.
3. ★★ `procuraBit_alem`: com o índice além do alcance, a máquina desloca a testemunha inteira, esgota os ticks sem inventar bit, e a
   testemunha volta íntegra, com o bit lido ausente.
4. ★ `restauraTestemunha_fecha`: a restauração devolve os bits à testemunha e, no fim, compara com a guarda da presença e decide.
5. ★ `passo_saltaLiterais_cons` e `saltaLiterais_fecha`: o salto consome os símbolos até o próximo terminador, ou até o fim.

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

/-- ★ **O LAÇO DA PROCURA**: com a contagem cheia de ticks e o índice no alcance, a máquina desloca
os bits e lê o bit da variável — a contagem zera e a restaura cresce. -/
theorem procuraBit_fecha (v : EstadoV) (f r : List Simbolo) (out : List Bool) :
    ∀ (i : ℕ) (t : List Simbolo), i ≤ t.length →
      correr aVerificadora
        (aConfig (some Fase.procuraBit) v [] f t (List.replicate i Simbolo.tick) r out) (i + 1)
        = some (aConfig (some Fase.restauraTestemunha)
            { v with lido := none, bitLido := (t.drop i).head? } [] f (t.drop i) []
            ((t.take i).reverse ++ r) out) := by
  intro i
  induction i generalizing v r with
  | zero =>
    intro t _
    show correr aVerificadora (aConfig (some Fase.procuraBit) v [] f t (List.replicate 0 Simbolo.tick) r out)
        (0 + 1) = _
    have h0 : List.replicate 0 Simbolo.tick = ([] : List Simbolo) := rfl
    rw [h0, correr, passo_procuraBit_nil]
    rfl
  | succ i ih =>
    intro t hle
    show correr aVerificadora (aConfig (some Fase.procuraBit) v [] f t (List.replicate (i + 1) Simbolo.tick) r out)
        (i + 1 + 1) = _
    have hrep : List.replicate (i + 1) Simbolo.tick = Simbolo.tick :: List.replicate i Simbolo.tick :=
      List.replicate_succ
    rw [hrep]
    cases t with
    | nil =>
      simp at hle
    | cons b t' =>
      rw [correr, passo_procuraBit_cons]
      show correr aVerificadora (aConfig (some Fase.procuraBit)
          { v with lido := some b, bitLido := some b } [] f t' (List.replicate i Simbolo.tick)
          (b :: r) out) (i + 1) = _
      have hle' : i ≤ t'.length := by
        rw [List.length_cons] at hle
        omega
      rw [ih { v with lido := some b, bitLido := some b } (b :: r) t' hle']
      show some (aConfig (some Fase.restauraTestemunha)
          { { v with lido := some b, bitLido := some b } with
            lido := none, bitLido := (t'.drop i).head? } [] f (t'.drop i) []
          ((t'.take i).reverse ++ (b :: r)) out) = _
      rw [List.drop_succ_cons, List.take_succ_cons, List.reverse_cons, List.append_assoc]
      rfl

/-- ★ **O LAÇO DO ESGOTAMENTO**: a drenagem consome os ticks sem tocar em nada — nada se fabrica. -/
theorem esgotaContagem_fecha (v : EstadoV) (f t r : List Simbolo) (out : List Bool) :
    ∀ (i : ℕ),
      correr aVerificadora
        (aConfig (some Fase.esgotaContagem) v [] f t (List.replicate i Simbolo.tick) r out) (i + 1)
        = some (aConfig (some Fase.restauraTestemunha) { v with lido := none } [] f t [] r out) := by
  intro i
  induction i generalizing v with
  | zero =>
    show correr aVerificadora (aConfig (some Fase.esgotaContagem) v [] f t
        (List.replicate 0 Simbolo.tick) r out) (0 + 1) = _
    have h0 : List.replicate 0 Simbolo.tick = ([] : List Simbolo) := rfl
    rw [h0, correr, passo_esgotaContagem_nil]
    rfl
  | succ i ih =>
    show correr aVerificadora (aConfig (some Fase.esgotaContagem) v [] f t
        (List.replicate (i + 1) Simbolo.tick) r out) (i + 1 + 1) = _
    have hrep : List.replicate (i + 1) Simbolo.tick = Simbolo.tick :: List.replicate i Simbolo.tick :=
      List.replicate_succ
    rw [hrep, correr, passo_esgotaContagem_cons]
    show correr aVerificadora (aConfig (some Fase.esgotaContagem) { v with lido := some Simbolo.tick } []
        f t (List.replicate i Simbolo.tick) r out) (i + 1) = _
    rw [ih]

/-- ★★ **A PROCURA ALÉM DA TESTEMUNHA**: com o índice estritamente além do alcance, a máquina
desloca a testemunha inteira, esgota os ticks sem inventar, e o bit lido é o silêncio (`none`) —
nada se fabrica; a restaura devolve a testemunha íntegra. -/
theorem procuraBit_alem (v : EstadoV) (f r : List Simbolo) (out : List Bool) :
    ∀ (i : ℕ) (t : List Simbolo), t.length < i →
      correr aVerificadora
        (aConfig (some Fase.procuraBit) v [] f t (List.replicate i Simbolo.tick) r out) (i + 2)
        = some (aConfig (some Fase.restauraTestemunha)
            { v with lido := none, bitLido := none } [] f [] [] (t.reverse ++ r) out) := by
  intro i t
  induction t generalizing v r i with
  | nil =>
    intro h
    cases i with
    | zero => simp at h
    | succ j =>
      show correr aVerificadora (aConfig (some Fase.procuraBit) v [] f []
          (List.replicate (j + 1) Simbolo.tick) r out) (j + 1 + 2) = _
      have hrep : List.replicate (j + 1) Simbolo.tick = Simbolo.tick :: List.replicate j Simbolo.tick :=
        List.replicate_succ
      rw [hrep, show j + 1 + 2 = (j + 2) + 1 from by omega, correr, passo_procuraBit_esgota]
      show correr aVerificadora (aConfig (some Fase.esgotaContagem)
          { v with lido := some Simbolo.tick, bitLido := none } [] f []
          (Simbolo.tick :: List.replicate j Simbolo.tick) r out) (j + 1 + 1) = _
      rw [← List.replicate_succ,
        esgotaContagem_fecha { v with lido := some Simbolo.tick, bitLido := none } f [] r out (j + 1)]
      rfl
  | cons b t' ih =>
    intro h
    cases i with
    | zero => simp at h
    | succ j =>
      have h' : t'.length < j := by
        rw [List.length_cons] at h
        omega
      show correr aVerificadora (aConfig (some Fase.procuraBit) v [] f (b :: t')
          (List.replicate (j + 1) Simbolo.tick) r out) (j + 1 + 2) = _
      have hrep : List.replicate (j + 1) Simbolo.tick = Simbolo.tick :: List.replicate j Simbolo.tick :=
        List.replicate_succ
      rw [hrep, show j + 1 + 2 = (j + 2) + 1 from by omega, correr, passo_procuraBit_cons]
      show correr aVerificadora (aConfig (some Fase.procuraBit)
          { v with lido := some b, bitLido := some b } [] f t' (List.replicate j Simbolo.tick)
          (b :: r) out) (j + 2) = _
      rw [ih { v with lido := some b, bitLido := some b } (b :: r) j h']
      show some (aConfig (some Fase.restauraTestemunha)
          { { v with lido := some b, bitLido := some b } with lido := none, bitLido := none } [] f []
          [] (t'.reverse ++ (b :: r)) out) = _
      rw [List.reverse_cons, List.append_assoc]
      rfl

/-- ★ **O LAÇO DA RESTAURA**: devolve os bits à testemunha; no fim, a guarda da presença compara e
decide. -/
theorem restauraTestemunha_fecha (v : EstadoV) (f c : List Simbolo) (out : List Bool) :
    ∀ (r t : List Simbolo),
      correr aVerificadora
        (aConfig (some Fase.restauraTestemunha) v [] f t c r out) (r.length + 1)
        = some (if v.bitLido.isSome && ((v.bitLido == some Simbolo.um) == v.marcador) then
            aConfig (some Fase.saltaLiterais) { v with lido := none, falha := false } [] f (r.reverse ++ t) c [] out
          else aConfig (some Fase.leMarcador) { v with lido := none } [] f (r.reverse ++ t) c [] out) := by
  intro r
  induction r generalizing v with
  | nil =>
    intro t
    show correr aVerificadora (aConfig (some Fase.restauraTestemunha) v [] f t c [] out) (0 + 1) = _
    rw [correr, passo_restauraTestemunha_nil]
    rfl
  | cons b r' ih =>
    intro t
    show correr aVerificadora (aConfig (some Fase.restauraTestemunha) v [] f t c (b :: r') out)
        ((b :: r').length + 1) = _
    rw [List.length_cons, correr, passo_restauraTestemunha_cons]
    show correr aVerificadora (aConfig (some Fase.restauraTestemunha)
        { v with lido := some b } [] f (b :: t) c r' out) (r'.length + 1) = _
    rw [ih]
    cases hb : (v.bitLido.isSome && ((v.bitLido == some Simbolo.um) == v.marcador)) with
    | true => simp [hb, List.reverse_cons, List.append_assoc]
    | false => simp [hb, List.reverse_cons, List.append_assoc]

/-- ★ **O PASSO DO SALTO, FORA DO TERMINADOR**: consome um símbolo qualquer. -/
theorem passo_saltaLiterais_cons (v : EstadoV) (s₀ : Simbolo) (f' t c r : List Simbolo)
    (out : List Bool) (hne : s₀ ≠ Simbolo.fimC) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.saltaLiterais) v [] (s₀ :: f') t c r out)
      = some (aConfig (some Fase.saltaLiterais) { v with lido := some s₀ } [] f' t c r out) := by
  cases s₀ with
  | tick =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons, (by decide : (Simbolo.tick == Simbolo.fimC) = false)]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | litT =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons, (by decide : (Simbolo.litT == Simbolo.fimC) = false)]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | litF =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons, (by decide : (Simbolo.litF == Simbolo.fimC) = false)]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | fimC =>
    exact absurd rfl hne
  | fimF =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons, (by decide : (Simbolo.fimF == Simbolo.fimC) = false)]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | sep =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons, (by decide : (Simbolo.sep == Simbolo.fimC) = false)]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | um =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons, (by decide : (Simbolo.um == Simbolo.fimC) = false)]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | zero =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons, (by decide : (Simbolo.zero == Simbolo.fimC) = false)]
    congr
    try (funext k; cases k <;> rfl)
    try rfl

/-- ★ **O LAÇO DO SALTO**: consome os símbolos até o próximo terminador (ou o fim). -/
theorem saltaLiterais_fecha (v : EstadoV) (t c r : List Simbolo) (out : List Bool) :
    ∀ (w f' : List Simbolo), (∀ s₀ ∈ w, s₀ ≠ Simbolo.fimC) →
      correr aVerificadora (aConfig (some Fase.saltaLiterais) v [] (w ++ f') t c r out) (w.length + 1)
        = some (match f' with
          | [] => aConfig (some Fase.escreveVerdade) { v with lido := none } [] [] t c r out
          | Simbolo.fimC :: f'' => aConfig (some Fase.inicioClausula) { v with lido := some Simbolo.fimC }
              [] (Simbolo.fimC :: f'') t c r out
          | s₀ :: f'' => aConfig (some Fase.saltaLiterais) { v with lido := some s₀ } [] f'' t c r out) := by
  intro w
  induction w generalizing v with
  | nil =>
    intro f' _
    show correr aVerificadora (aConfig (some Fase.saltaLiterais) v [] ([] ++ f') t c r out) (0 + 1) = _
    rw [List.nil_append, correr, passo_saltaLiterais]
    cases f' with
    | nil => rfl
    | cons s₀ f'' =>
      cases s₀ with
      | fimC => rfl
      | tick => rfl
      | litT => rfl
      | litF => rfl
      | fimF => rfl
      | sep => rfl
      | um => rfl
      | zero => rfl
  | cons s₀ w' ih =>
    intro f' hne
    have hs : s₀ ≠ Simbolo.fimC := hne s₀ List.mem_cons_self
    have h := ih v f' (fun s₀ hs => hne s₀ (List.mem_cons_of_mem _ hs))
    show correr aVerificadora (aConfig (some Fase.saltaLiterais) v [] ((s₀ :: w') ++ f') t c r out)
        ((s₀ :: w').length + 1) = _
    rw [List.length_cons, List.cons_append, correr]
    show (match Turing.FinTM2.step aVerificadora
        (aConfig (some Fase.saltaLiterais) v [] (s₀ :: (w' ++ f')) t c r out) with
      | none => some (aConfig (some Fase.saltaLiterais) v [] (s₀ :: (w' ++ f')) t c r out)
      | some c' => correr aVerificadora c' (w'.length + 1)) = _
    rw [passo_saltaLiterais_cons v s₀ (w' ++ f') t c r out hs]
    show correr aVerificadora (aConfig (some Fase.saltaLiterais)
        { v with lido := some s₀ } [] (w' ++ f') t c r out) (w'.length + 1) = _
    exact h

end TGLExt.AMaquinaVerifica

