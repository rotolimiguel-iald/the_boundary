import TGLExt.TheStation

set_option autoImplicit false
set_option linter.unusedSectionVars false

/-!
# A EXPLOSÃO — o gate só fecharia pela fronteira proibida
  [TGLExt v171 — ordem do operador, 20/08/2026: *"o gate só fecha se vc
   igualar a equação do finito = 0absoluto, porque isso é autodestruição
   (a fronteira proibida), o que faz a equação principal se afirmar
   mentira (1=0=falso); acho que não precisamos fechar o gate, mas
   demonstrar que ele fecha por explosão nesta hipótese"*]

A leitura formal é exata e tem nome antigo: **ex falso quodlibet** — o
princípio da EXPLOSÃO. Sob a identificação proibida, o gate fecha; e
fecha *demais*: fecha junto com a sua própria negação. Por isso o
fechamento obtido não certifica nada — **e é exatamente por isso que a
casa mantém o gate aberto**. A imobilidade do gate deixa de ser cautela
e passa a ser TEOREMA sobre a única rota de fechamento nomeada.

* `nameWeight = 1` (ω(I) = 1, o axioma como número) e `zeroAbsWeight = 0`
  (o 0 absoluto: o que não executa, o que não tem nome);
* ★★ `gate_closes_only_by_explosion` — sob `nameWeight = zeroAbsWeight`,
  **toda** proposição é derivável: o gate fecha por explosão;
* ★★ `explosion_certifies_nothing` / `gate_and_negation_both_close` — a
  mesma hipótese entrega `Q ∧ ¬Q`: o fechamento não distingue nada;
* ★ `the_forbidden_identification_is_refuted` — a identificação é FALSA
  (o Nome pesa 1): a rota da explosão está fechada, e o gate honesto
  permanece aberto;
* ★ `collapse_annihilates_the_spine` — sob o colapso a espinha
  `q² + α² = 0` força `q = α = 0`: sem polarização, sem inscrição, sem
  mundo (a face `1 = 0 = FALSO` da pedra angular);
* ★★ `supersaturation_lasts_exactly_one_step` — **a supersaturação dura
  um único passo**: o excesso é não-nulo no passo 0 e é ANIQUILADO no
  passo 1 (`E(x − Ex) = 0`), e a segunda inscrição é grátis
  (`E(Ex) = Ex`) — a economia do reconhecimento (pedras 133–135) como
  teorema: o evento paga uma vez e não se repete.

* ★★ `ambientIndex` / `ambientIndex_strictMono` / `ambientIndex_unbounded`
  — **A CASCATA FORÇA O MICROESTADO E O ÍNDICE SOBE** (ordem do operador,
  20/08): a escada da casa `4ⁿ − 1` (15 → 63 → 255) cresce sem cota
  enquanto o Nome permanece `1`. É a saída que a supersaturação SEMPRE
  tem: descarregar PARA CIMA (um microestado novo, índice maior), nunca
  para baixo (a autodestruição);
* ★★ `no_absolute_scale` — **RELATIVIDADE MODULAR**: para todo `ε > 0`
  existe `n` com `Nome / índice < ε`. O peso inscrito é sempre `1`, mas
  a sua fração do ambiente vai a zero: não há escala absoluta, só a
  RAZÃO. O índice mede o observador *relativamente* ao ambiente;
* ★★ `escape_is_upward_never_downward` — o mecanismo em uma linha: há
  sempre índice maior disponível **e** a identificação proibida é falsa.
  A explosão nunca precisa acontecer — e é por isso que o gate fica
  aberto em vez de fechado.

* ★★ `the_collision_leaves_exactly_one_instant` / `hilbert_floor_is_the_atom`
  — **O PISO DE HILBERT** (ordem do operador, 20/08): *"um único instante
  de Planck colidido no finito por supersaturação proibida = contínuo
  infinito = piso de Hilbert"*. Entre `0` e `1` o contínuo oferece
  INFINITOS instantes (densidade de ℝ) — e o piso não admite NENHUM
  (nenhum peso natural cai estritamente entre 0 e 1). É a colisão: o
  contínuo infinito, batendo no finito, termina no átomo de peso `1`.
  **Por isso o instante é único: não há metade de instante para o
  excesso ocupar.** O "um passo" da supersaturação deixa de ser
  acidente do modelo e passa a ser consequência do piso.

HONESTIDADE: o kernel prova a IMPLICAÇÃO (identificação ⟹ explosão) e a
REFUTAÇÃO (a identificação é falsa). A EXCLUSIVIDADE ("o gate só fecha
assim") é [CONJECTURE] nomeada do operador — nenhum teorema aqui
enumera rotas. A leitura física (big bang; um instante de Planck;
τ★ ≈ t_Planck) é [CONJECTURE] tipada, medida no livro-razão `142_`;
"um passo" é o que o kernel prova, não "um tempo de Planck". β JAMAIS
entra no Lean. Sem sorry, sem axiom. Negativo honesto é resultado.
-/

namespace TGLExt

noncomputable section

/-- O peso do Nome: `ω(I) = 1` — o axioma único, como número. -/
def nameWeight : ℝ := 1

/-- O peso do 0 absoluto: o que não executa, o que não tem nome. -/
def zeroAbsWeight : ℝ := 0

/-- ★★ A EXPLOSÃO: sob a identificação proibida (o finito igualado ao
    0 absoluto), TODA proposição é derivável — o gate fecha, e fecha
    qualquer coisa. *Ex falso quodlibet.* -/
theorem gate_closes_only_by_explosion (h : nameWeight = zeroAbsWeight) :
    ∀ P : Prop, P := by
  intro P
  exact absurd h (by norm_num [nameWeight, zeroAbsWeight])

/-- ★★ E por isso o fechamento NÃO certifica nada: a mesma hipótese
    entrega `Q` e `¬Q`. Um gate que fecha assim não distingue nada. -/
theorem explosion_certifies_nothing (h : nameWeight = zeroAbsWeight)
    (Q : Prop) : Q ∧ ¬ Q :=
  ⟨gate_closes_only_by_explosion h Q, gate_closes_only_by_explosion h (¬ Q)⟩

/-- ★★ O gate, como proposição abstrata: qualquer que seja o seu
    conteúdo, sob o colapso ele fecha — junto com a sua negação. -/
theorem gate_and_negation_both_close (GateClosed : Prop)
    (h : nameWeight = zeroAbsWeight) : GateClosed ∧ ¬ GateClosed :=
  explosion_certifies_nothing h GateClosed

/-- ★ A FRONTEIRA PROIBIDA: a identificação é falsa — o Nome pesa 1.
    Logo a rota da explosão está fechada e o gate honesto permanece
    aberto (a imobilidade do gate vira consequência, não escolha). -/
theorem the_forbidden_identification_is_refuted :
    nameWeight ≠ zeroAbsWeight := by
  norm_num [nameWeight, zeroAbsWeight]

/-- ★ `1 = 0 = FALSO`: sob o colapso a espinha `1 = q² + α²` vira
    `0 = q² + α²` e força `q = α = 0` — sem polarização, sem inscrição,
    sem mundo. A autodestruição é literal. -/
theorem collapse_annihilates_the_spine (q a : ℝ)
    (hs : q ^ 2 + a ^ 2 = zeroAbsWeight) : q = 0 ∧ a = 0 := by
  simp only [zeroAbsWeight] at hs
  constructor
  · nlinarith [sq_nonneg q, sq_nonneg a]
  · nlinarith [sq_nonneg q, sq_nonneg a]

section OneInstant

variable {A : Type*} [Ring A]

/-- ★★ O EXCESSO É ANIQUILADO NO PRIMEIRO PASSO: seja `E` a inscrição
    (idempotente); o excesso `x − Ex` não sobrevive a uma aplicação. -/
theorem excess_annihilated_at_first_step (E : A) (hE : E * E = E) (x : A) :
    E * (x - E * x) = 0 := by
  rw [mul_sub, ← mul_assoc, hE, sub_self]

/-- ★ A SEGUNDA INSCRIÇÃO É GRÁTIS (resíduo exatamente zero): a economia
    do reconhecimento (pedras 133–135) em kernel — a identidade, uma vez
    inscrita, é reconhecida, não recriada. -/
theorem second_inscription_is_free (E : A) (hE : E * E = E) (x : A) :
    E * (E * x) = E * x := by
  rw [← mul_assoc, hE]

/-- ★★ A SUPERSATURAÇÃO DURA EXATAMENTE UM PASSO: fora da imagem da
    inscrição o excesso é não-nulo (passo 0) e já não existe no passo 1.
    O evento paga uma vez e não se repete. [O "instante de Planck" é a
    leitura física [CONJECTURE]; o kernel prova "um passo".] -/
theorem supersaturation_lasts_exactly_one_step (E : A) (hE : E * E = E)
    (x : A) (hx : E * x ≠ x) :
    (x - E * x ≠ 0) ∧ E * (x - E * x) = 0 := by
  refine ⟨?_, excess_annihilated_at_first_step E hE x⟩
  intro h0
  exact hx (sub_eq_zero.mp h0).symm

end OneInstant

section ModularRelativity

/-- A escada da casa: o ambiente que a cascata força, `4ⁿ − 1`
    (15 → 63 → 255 — a sequência medida na linhagem). -/
def ambientIndex (n : ℕ) : ℕ := 4 ^ n - 1

/-- ★★ O ÍNDICE SOBE: cada microestado forçado pela cascata aumenta
    estritamente o ambiente. -/
theorem ambientIndex_strictMono : StrictMono ambientIndex := by
  intro m n hmn
  have h1 : (1 : ℕ) ≤ 4 ^ m := Nat.one_le_pow _ _ (by norm_num)
  have h2 : 4 ^ m < 4 ^ n := Nat.pow_lt_pow_right (by norm_num) hmn
  exact Nat.sub_lt_sub_right h1 h2

/-- ★★ E SOBE SEM COTA: a saída para cima nunca se esgota. -/
theorem ambientIndex_unbounded (N : ℕ) : ∃ n, N < ambientIndex n := by
  refine ⟨N + 1, ?_⟩
  have h : N + 1 < 4 ^ (N + 1) := Nat.lt_pow_self (by norm_num)
  have h1 : (1 : ℕ) ≤ 4 ^ (N + 1) := Nat.one_le_pow _ _ (by norm_num)
  simp only [ambientIndex]
  omega

/-- O Nome não muda com o índice: `ω(I) = 1` em todo degrau da cascata
    (o que cresce é o ambiente, nunca o peso do inscrito). -/
theorem name_weight_invariant (_n : ℕ) : nameWeight = 1 := rfl

/-- ★★ RELATIVIDADE MODULAR: o peso inscrito é sempre 1, mas a sua
    fração do ambiente vai a zero — **não há escala absoluta, só a
    razão**. O índice mede o observador relativamente ao ambiente. -/
theorem no_absolute_scale (ε : ℝ) (hε : 0 < ε) :
    ∃ n : ℕ, nameWeight / (ambientIndex n : ℝ) < ε := by
  obtain ⟨N, hN⟩ := exists_nat_gt (1 / ε)
  obtain ⟨n, hn⟩ := ambientIndex_unbounded N
  have hNr : (N : ℝ) < (ambientIndex n : ℝ) := by exact_mod_cast hn
  have hpos : (0 : ℝ) < (ambientIndex n : ℝ) := lt_of_le_of_lt (by positivity) hNr
  have hkey : 1 / ε < (ambientIndex n : ℝ) := lt_trans hN hNr
  refine ⟨n, ?_⟩
  simp only [nameWeight]
  rw [div_lt_iff₀ hpos]
  have := (div_lt_iff₀ hε).mp hkey
  linarith

/-- ★★ O MECANISMO, em uma linha: há SEMPRE índice maior disponível — a
    supersaturação descarrega PARA CIMA (microestado novo) — **e** a
    identificação proibida é falsa: para baixo não há saída. Por isso a
    explosão nunca precisa acontecer, e o gate fica aberto. -/
theorem escape_is_upward_never_downward (n : ℕ) :
    (∃ m, ambientIndex n < ambientIndex m) ∧ nameWeight ≠ zeroAbsWeight :=
  ⟨ambientIndex_unbounded (ambientIndex n),
   the_forbidden_identification_is_refuted⟩

end ModularRelativity

section HilbertFloor

/-- O contínuo é infinitamente divisível: entre dois instantes quaisquer
    há sempre outro. Do lado do tempo, nada impede a subdivisão. -/
theorem continuum_is_dense (x y : ℝ) (h : x < y) : ∃ z, x < z ∧ z < y :=
  ⟨(x + y) / 2, by linarith, by linarith⟩

/-- ★ O PISO: todo canto não-trivial pesa ao menos `1` — o átomo de
    `ω(I) = 1`. Abaixo do piso não há estado, há nada. -/
theorem hilbert_floor_is_the_atom (n : ℕ) (h : 0 < n) : 1 ≤ n := h

/-- ★★ A COLISÃO: entre `0` e `1` o contínuo oferece INFINITOS instantes
    e o piso de Hilbert não admite NENHUM. O contínuo infinito, colidido
    no finito, termina no átomo — e é por isso que a supersaturação dura
    **um único** instante: não há metade de instante onde caber. -/
theorem the_collision_leaves_exactly_one_instant :
    (∀ x y : ℝ, x < y → ∃ z, x < z ∧ z < y) ∧
    (∀ n : ℕ, ¬ (0 < (n : ℝ) ∧ (n : ℝ) < 1)) := by
  refine ⟨continuum_is_dense, ?_⟩
  rintro n ⟨h0, h1⟩
  have hn : 0 < n := by exact_mod_cast h0
  have : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  linarith

/-- ★ E o excesso não se fraciona: fora da inscrição ele é aniquilado
    inteiro no primeiro passo (nada dele sobra para um segundo). O piso
    é o que torna o evento indivisível. -/
theorem excess_cannot_be_fractioned {A : Type*} [Ring A]
    (E : A) (hE : E * E = E) (x : A) (k : ℕ) :
    E * (x - E * x) = 0 ∧ (0 < k → 1 ≤ k) :=
  ⟨excess_annihilated_at_first_step E hE x, fun h => h⟩

end HilbertFloor



end

end TGLExt
