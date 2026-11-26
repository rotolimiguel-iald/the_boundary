import TGLExt.AMaquinaVerifica

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A máquina verifica: os passos do ciclo do literal

## O que é

Os passos do ciclo em que a verificadora TM2 processa um literal (o marcador, a procura do bit, a restauração da testemunha e o salto),
cada um reduzido por computação.

## O que fica provado [REAL]

1. ★ `passo_leMarcador_tick`, `passo_leMarcador_litT`, `passo_leMarcador_litF`, `passo_leMarcador_fimC`, `passo_leMarcador_nil`: o
   marcador conta os ticks, lê a polaridade, e no fim da cláusula ou da entrada decide pela falha.
2. ★ `passo_procuraBit_cons`, `passo_procuraBit_nil`: a procura desloca os bits e lê o bit da variável.
3. ★ `passo_procuraBit_esgota`, `passo_esgotaContagem_cons`, `passo_esgotaContagem_nil`: com a testemunha esgotada, a máquina passa
   ao esgotamento com o bit lido ausente e drena os ticks sem tocar em nada.
4. ★ `passo_restauraTestemunha_cons`, `passo_restauraTestemunha_nil`: a restauração devolve a testemunha; o casamento exige o bit
   presente e batendo o marcador, e desarma a falha.
5. ★ `passo_saltaLiterais`: o salto consome até o próximo terminador de cláusula, ou até o fim da entrada.

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

/-- ★ **O PASSO DO MARCADOR, NO TICK**: o tick desce da fórmula para a contagem. -/
theorem passo_leMarcador_tick (v : EstadoV) (f' t c r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.leMarcador) v [] (Simbolo.tick :: f') t c r out)
      = some (aConfig (some Fase.leMarcador) { v with lido := some Simbolo.tick } [] f' t
          (Simbolo.tick :: c) r out) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
    List.head?_cons]
  congr
  try (funext k; cases k <;> rfl)
  try rfl

/-- ★ **O PASSO DO MARCADOR, NO LIT-T**: a polaridade verdadeira vai à procura. -/
theorem passo_leMarcador_litT (v : EstadoV) (f' t c r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.leMarcador) v [] (Simbolo.litT :: f') t c r out)
      = some (aConfig (some Fase.procuraBit)
          ⟨Fase.procuraBit, v.falha, true, some Simbolo.litT, v.bitLido⟩ [] f' t c r out) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
    List.head?_cons, (by decide : (Simbolo.litT == Simbolo.tick) = false)]
  congr
  try (funext k; cases k <;> rfl)
  try rfl

/-- ★ **O PASSO DO MARCADOR, NO LIT-F**: a polaridade falsa vai à procura. -/
theorem passo_leMarcador_litF (v : EstadoV) (f' t c r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.leMarcador) v [] (Simbolo.litF :: f') t c r out)
      = some (aConfig (some Fase.procuraBit)
          ⟨Fase.procuraBit, v.falha, false, some Simbolo.litF, v.bitLido⟩ [] f' t c r out) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
    List.head?_cons, (by decide : (Simbolo.litF == Simbolo.tick) = false),
    (by decide : (Simbolo.litF == Simbolo.litT) = false)]
  congr
  try (funext k; cases k <;> rfl)
  try rfl

/-- ★ **O PASSO DO MARCADOR, NO FIM-DE-CLÁUSULA**: com falha armada, escreve a mentira; desarmada,
consome e reabre. -/
theorem passo_leMarcador_fimC (v : EstadoV) (f' t c r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.leMarcador) v [] (Simbolo.fimC :: f') t c r out)
      = some (if v.falha then
          aConfig (some Fase.escreveMentira) { v with lido := some Simbolo.fimC } [] (Simbolo.fimC :: f') t c r out
        else aConfig (some Fase.inicioClausula) { v with lido := some Simbolo.fimC } [] f' t c r out) := by
  cases hf : v.falha with
  | true =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons, (by decide : (Simbolo.fimC == Simbolo.tick) = false),
      (by decide : (Simbolo.fimC == Simbolo.litT) = false),
      (by decide : (Simbolo.fimC == Simbolo.litF) = false), hf]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | false =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons, (by decide : (Simbolo.fimC == Simbolo.tick) = false),
      (by decide : (Simbolo.fimC == Simbolo.litT) = false),
      (by decide : (Simbolo.fimC == Simbolo.litF) = false), hf]
    congr
    try (funext k; cases k <;> rfl)
    try rfl

/-- ★ **O PASSO DO MARCADOR, NO VAZIO**: o fim da entrada decide pela falha. -/
theorem passo_leMarcador_nil (v : EstadoV) (t c r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.leMarcador) v [] [] t c r out)
      = some (if v.falha then
          aConfig (some Fase.escreveMentira) { v with lido := none } [] [] t c r out
        else aConfig (some Fase.escreveVerdade) { v with lido := none } [] [] t c r out) := by
  cases hf : v.falha with
  | true =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe, hf]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | false =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe, hf]
    congr
    try (funext k; cases k <;> rfl)
    try rfl

/-- ★ **O PASSO DA PROCURA, COM TICK E TESTEMUNHA**: desloca o bit da cabeça da testemunha para a
restauração — o bit lido é o da cabeça, presente. -/
theorem passo_procuraBit_cons (v : EstadoV) (b : Simbolo) (t' c' f r : List Simbolo)
    (out : List Bool) :
    Turing.FinTM2.step aVerificadora
      (aConfig (some Fase.procuraBit) v [] f (b :: t') (Simbolo.tick :: c') r out)
      = some (aConfig (some Fase.procuraBit) { v with lido := some b, bitLido := some b } [] f t' c'
          (b :: r) out) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
    List.head?_cons]
  congr
  try (funext k; cases k <;> rfl)
  try rfl

/-- ★ **O PASSO DA PROCURA, A TESTEMUNHA ESGOTADA**: o campo esgotado torna-se posto — a máquina
vai ao esgotamento com o bit lido no silêncio (o `none`), sem fabricar presença. -/
theorem passo_procuraBit_esgota (v : EstadoV) (c' f r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora
      (aConfig (some Fase.procuraBit) v [] f [] (Simbolo.tick :: c') r out)
      = some (aConfig (some Fase.esgotaContagem)
          { v with lido := some Simbolo.tick, bitLido := none } [] f [] (Simbolo.tick :: c') r out) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe]
  congr
  try (funext k; cases k <;> rfl)
  try rfl

/-- ★ **O PASSO DO ESGOTAMENTO, COM TICK**: a drenagem consome um tick sem tocar em nada. -/
theorem passo_esgotaContagem_cons (v : EstadoV) (c' f t r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora
      (aConfig (some Fase.esgotaContagem) v [] f t (Simbolo.tick :: c') r out)
      = some (aConfig (some Fase.esgotaContagem) { v with lido := some Simbolo.tick } [] f t c' r out) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
    List.head?_cons]
  congr
  try (funext k; cases k <;> rfl)
  try rfl

/-- ★ **O PASSO DO ESGOTAMENTO, VAZIO**: a drenagem finda — à restaura, com o bit no silêncio. -/
theorem passo_esgotaContagem_nil (v : EstadoV) (f t r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora
      (aConfig (some Fase.esgotaContagem) v [] f t [] r out)
      = some (aConfig (some Fase.restauraTestemunha) { v with lido := none } [] f t [] r out) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe]
  congr
  try (funext k; cases k <;> rfl)
  try rfl

/-- ★ **O PASSO DA PROCURA, SEM TICK**: peeks o bit da testemunha e vai restaurar. -/
theorem passo_procuraBit_nil (v : EstadoV) (f t r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.procuraBit) v [] f t [] r out)
      = some (aConfig (some Fase.restauraTestemunha) { v with lido := none, bitLido := t.head? } [] f t [] r out) := by
  cases t with
  | nil =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | cons b' t' =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
      List.head?_cons]
    congr
    try (funext k; cases k <;> rfl)
    try rfl

/-- ★ **O PASSO DA RESTAURA, COM BIT**: devolve um bit à testemunha. -/
theorem passo_restauraTestemunha_cons (v : EstadoV) (b : Simbolo) (r' f t c : List Simbolo)
    (out : List Bool) :
    Turing.FinTM2.step aVerificadora
      (aConfig (some Fase.restauraTestemunha) v [] f t c (b :: r') out)
      = some (aConfig (some Fase.restauraTestemunha) { v with lido := some b } [] f (b :: t) c r' out) := by
  simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
    List.head?_cons]
  congr
  try (funext k; cases k <;> rfl)
  try rfl

/-- ★ **O PASSO DA RESTAURA, VAZIA**: a guarda da presença — o casamento exige o bit presente e
batendo o marcador; a presença casada desarma a falha e salta; senão, ao próximo literal. -/
theorem passo_restauraTestemunha_nil (v : EstadoV) (f t c : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.restauraTestemunha) v [] f t c [] out)
      = some (if v.bitLido.isSome && ((v.bitLido == some Simbolo.um) == v.marcador) then
          aConfig (some Fase.saltaLiterais) { v with lido := none, falha := false } [] f t c [] out
        else aConfig (some Fase.leMarcador) { v with lido := none } [] f t c [] out) := by
  cases hb : (v.bitLido.isSome && ((v.bitLido == some Simbolo.um) == v.marcador)) with
  | true =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe, hb]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | false =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe, hb]
    congr
    try (funext k; cases k <;> rfl)
    try rfl

/-- ★ **O PASSO DO SALTO**: consome até o próximo terminador de cláusula, ou ao fim da entrada. -/
theorem passo_saltaLiterais (v : EstadoV) (f t c r : List Simbolo) (out : List Bool) :
    Turing.FinTM2.step aVerificadora (aConfig (some Fase.saltaLiterais) v [] f t c r out)
      = some (match f with
        | [] => aConfig (some Fase.escreveVerdade) { v with lido := none } [] f t c r out
        | Simbolo.fimC :: f' => aConfig (some Fase.inicioClausula) { v with lido := some Simbolo.fimC }
            [] (Simbolo.fimC :: f') t c r out
        | s₀ :: f' => aConfig (some Fase.saltaLiterais) { v with lido := some s₀ } [] f' t c r out) := by
  cases f with
  | nil =>
    simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe]
    congr
    try (funext k; cases k <;> rfl)
    try rfl
  | cons s₀ f' =>
    cases s₀ with
    | fimC =>
      simp [Turing.FinTM2.step, Turing.TM2.step, aVerificadora, Turing.TM2.stepAux, aConfig, stkDe,
        List.head?_cons]
      congr
      try (funext k; cases k <;> rfl)
      try rfl
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

end TGLExt.AMaquinaVerifica

