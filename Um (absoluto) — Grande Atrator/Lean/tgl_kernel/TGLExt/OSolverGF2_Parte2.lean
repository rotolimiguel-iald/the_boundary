import TGLExt.OSolverGF2

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# O solver GF(2): a preservação do servir pela redução

Um ponto `x : Fin n → Bool` serve a equação de paridade `(a, v)` quando `paridade2 x a = v`
(`serveEq`). A pedra `TGLExt.OSolverGF2` define a eliminação sobre GF(2) (`reduzirNova`,
`inserirEscalonado`) e prova `paridade2_mask_xor`: o xor de máscaras soma as paridades. Esta pedra prova
que os passos da eliminação preservam o servir, nas duas direções, cada um sob a sua hipótese:
`passoReduzir_preserva` e `reduzirNova_preserva` pedem que `x` sirva o pivô ou a lista de pivôs usada;
`pivos_reduzidos_preserva` pede que `x` sirva a equação nova `(a, v)`; `inserirEscalonado_preserva` não
pede nada sobre `x`, e sim a consistência existencial `hcons` (algum `y` serve `pivots` e `e`). A forma é
condicional: a soma das paridades não determina as parcelas em geral; com a parcela conhecida servida,
determina. Os pivôs existentes são reduzidos contra a equação nova, como na eliminação de Gauss–Jordan
(forma escalonada reduzida).

## O que fica provado nesta pedra [REAL]

1. `serve_xor_combinacao`: se `paridade2 x a₂ = v₂`, então `paridade2 x (xorPonto a₁ a₂) = v₁ + v₂` se e
   só se `paridade2 x a₁ = v₁`.
2. `serveEspectro2_cons'`: servir `p :: ps` é servir `p` e servir `ps`.
3. `paridade2_mascara_zero`: a paridade na máscara zero é zero; `mascara_zero_de_bitPivo_none`: se
   `bitPivo a = none`, então `a` é a máscara zero.
4. `passoReduzir_preserva`: se `x` serve o pivô `p`, servir `passoReduzir p av` é servir `av`.
5. `reduzirNova_preserva`: se `x` serve a lista `pivots`, servir `reduzirNova av.1 av.2 pivots` é servir
   `av`.
6. `pivos_reduzidos_preserva`: se `paridade2 x a = v`, então, para todo bit `q`, `x` serve os pivôs
   reduzidos contra `(a, v)` no bit `q` (o passo de Gauss–Jordan) se e só se serve os pivôs originais.
7. `inserirEscalonado_preserva`: sob a hipótese `hcons` (existe um ponto `y` que serve `pivots` e serve
   `e`), para todo `x`: `x` serve `inserirEscalonado e pivots` se e só se `x` serve `pivots` e serve `e`.
   A hipótese entra no caso redundante (`bitPivo` da equação reduzida é `none`), onde ela força o valor
   reduzido a ser zero.
8. `reduzirNova_cons`: a redução contra `p :: ps` desdobra na cabeça (o `foldl` abre e fecha por `rfl`): `passoReduzir` é
   exatamente o passo que `reduzirNova` usa.

## O que NÃO se afirma

* Não se afirma aqui que o ponto construído pela eliminação (`pontoConstruido`) serve o escalonado, nem
  que serve o espectro original, nem a montagem final do solver: esses enunciados ficam nomeados fora
  desta pedra, não admitidos aqui. A correção da eliminação inteira não é provada nesta pedra.
* `inserirEscalonado_preserva` pede a consistência conjunta `hcons`; sem ela, nada se afirma sobre o caso
  redundante.
* Resolver um sistema consistente de equações de paridade sobre GF(2) por eliminação de Gauss é
  polinomial [KNOWN]; isso não é decidir SAT. Nada aqui decide SAT, nem dá custo de tempo na fita.
* Esta pedra não toma como hipótese `hdiv`, `hpoco`, `GatesDaDescida`, `GatesDaMorte` nem `AVerdadeFinal`.
  Sobre esses contratos o kernel mede, em `TGLExt.AVacuidadeDosContratos` [REAL]: `AVerdadeFinal`,
  `GatesDaMorte` e `GatesDaDescida` não podem ser satisfeitos como estão escritos
  (`aVerdadeFinal_insatisfazivel`, `gatesDaMorte_insatisfazivel`, `gatesDaDescida_insatisfazivel`);
  `hdiv` e `hpoco` juntos não podem ser satisfeitos (`gates_do_processo_insatisfaziveis`; também
  `hdiv_falso_n0`, com zero variáveis, e `hpoco_falso`, com ao menos uma variável); `hfiel` pode
  (`hfiel_habitado`).
* O gate do Instituto segue de pé e nomeado: decidir SAT sem testemunha, em tempo polinomial, na máquina
  de fita segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.OSolverGF2_Parte2

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2

variable {n : ℕ}

/-- [DEF] Servir uma equação: `paridade2 x e.1 = e.2`. -/
def serveEq {n : ℕ} (x : Fin n → Bool) (e : (Fin n → Bool) × ZMod 2) : Prop :=
  paridade2 x e.1 = e.2

/-- Servir a lista é servir a cabeça e a cauda. -/
theorem serveEspectro2_cons' {n : ℕ} (x : Fin n → Bool) (p : (Fin n → Bool) × ZMod 2)
    (ps : Espectro2 n) :
    serveEspectro2 x (p :: ps) = true ↔ serveEq x p ∧ serveEspectro2 x ps = true := by
  simp only [serveEspectro2, List.all_cons, Bool.and_eq_true]
  constructor
  · intro h
    exact ⟨(beq_iff_eq).mp h.1, h.2⟩
  · intro h
    exact ⟨(beq_iff_eq).mpr h.1, h.2⟩

/-- A paridade na máscara zero é zero. -/
theorem paridade2_mascara_zero {n : ℕ} (x : Fin n → Bool) :
    paridade2 x (fun _ => false) = 0 := by
  unfold paridade2
  simp

/-- Sem bit de pivô, a máscara é a zero. -/
theorem mascara_zero_de_bitPivo_none {n : ℕ} (a : Fin n → Bool) (h : bitPivo a = none) :
    a = fun _ => false := by
  ext i
  show a i = false
  exact bitPivo_none_spec a h i

/-- O passo da redução, nas duas direções: se `x` serve `(a₂, v₂)`, servir a combinação
`(xorPonto a₁ a₂, v₁ + v₂)` é servir `(a₁, v₁)`. O xor de máscaras soma as paridades, e a parcela do pivô
cancela. -/
theorem serve_xor_combinacao {n : ℕ} (x : Fin n → Bool) (a₁ a₂ : Fin n → Bool) (v₁ v₂ : ZMod 2)
    (h2 : paridade2 x a₂ = v₂) :
    (paridade2 x (xorPonto a₁ a₂) = v₁ + v₂) ↔ (paridade2 x a₁ = v₁) := by
  rw [paridade2_mask_xor, h2]
  exact add_right_cancel_iff

/-- [DEF] O passo da redução contra um pivô: se o pivô tem bit `i` e a equação tem esse bit aceso, a
equação é trocada pela sua soma com o pivô; caso contrário, fica como está. -/
def passoReduzir {n : ℕ} (p : (Fin n → Bool) × ZMod 2) (av : (Fin n → Bool) × ZMod 2) :
    (Fin n → Bool) × ZMod 2 :=
  match bitPivo p.1 with
  | none => av
  | some i => if av.1 i then (xorPonto av.1 p.1, av.2 + p.2) else av

/-- O passo da redução preserva o servir, com o pivô servido. -/
theorem passoReduzir_preserva {n : ℕ} (x : Fin n → Bool) (p : (Fin n → Bool) × ZMod 2)
    (hxp : serveEq x p) (av : (Fin n → Bool) × ZMod 2) :
    serveEq x (passoReduzir p av) ↔ serveEq x av := by
  unfold passoReduzir serveEq
  cases hb : bitPivo p.1 with
  | none => simp [hb]
  | some i =>
    simp only [hb]
    by_cases hbit : av.1 i = true
    · rw [if_pos hbit]
      show (paridade2 x (xorPonto av.1 p.1) = av.2 + p.2) ↔ (paridade2 x av.1 = av.2)
      exact serve_xor_combinacao x av.1 p.1 av.2 p.2 hxp
    · rw [if_neg hbit]

/-- A redução desdobra na cabeça: `reduzirNova` sobre `p :: ps` é o passo contra `p` seguido da redução
contra `ps`. -/
theorem reduzirNova_cons {n : ℕ} (a : Fin n → Bool) (v : ZMod 2) (p : (Fin n → Bool) × ZMod 2)
    (ps : Espectro2 n) :
    reduzirNova a v (p :: ps)
      = reduzirNova (passoReduzir p (a, v)).1 (passoReduzir p (a, v)).2 ps := by
  unfold reduzirNova
  rw [List.foldl_cons]
  rfl

/-- A redução preserva o servir: se `x` serve os pivôs, servir a equação reduzida é servir a equação
original. -/
theorem reduzirNova_preserva {n : ℕ} (x : Fin n → Bool) :
    ∀ (pivots : Espectro2 n), serveEspectro2 x pivots = true →
      ∀ (av : (Fin n → Bool) × ZMod 2),
        serveEq x (reduzirNova av.1 av.2 pivots) ↔ serveEq x av := by
  intro pivots
  induction pivots with
  | nil => intro _ av; simp [reduzirNova]
  | cons p ps ih =>
    intro hx av
    have hxp : serveEq x p := ((serveEspectro2_cons' x p ps).mp hx).1
    have hxps : serveEspectro2 x ps = true := ((serveEspectro2_cons' x p ps).mp hx).2
    rw [reduzirNova_cons]
    rw [ih hxps (passoReduzir p (av.1, av.2))]
    have hpasso := passoReduzir_preserva x p hxp (av.1, av.2)
    unfold serveEq at hpasso ⊢
    simp only [] at hpasso ⊢
    rw [hpasso]

/-- A redução dos pivôs preserva o servir: se `x` serve `(a, v)`, então, para todo bit `q`, `x` serve os
pivôs reduzidos contra `(a, v)` no bit `q` se e só se serve os pivôs originais. -/
theorem pivos_reduzidos_preserva {n : ℕ} (x : Fin n → Bool) (a : Fin n → Bool) (v : ZMod 2)
    (hxa : paridade2 x a = v) (q : Fin n) :
    ∀ (pivots : Espectro2 n),
      serveEspectro2 x (pivots.map (fun p => if p.1 q then (xorPonto p.1 a, p.2 + v) else p)) = true
        ↔ serveEspectro2 x pivots = true := by
  intro pivots
  induction pivots with
  | nil => simp
  | cons p ps ih =>
    rw [List.map_cons]
    rw [serveEspectro2_cons', serveEspectro2_cons', ih]
    by_cases hq : p.1 q = true
    · rw [if_pos hq]
      have hcomb := serve_xor_combinacao x p.1 a p.2 v hxa
      show (paridade2 x (xorPonto p.1 a) = p.2 + v ∧ serveEspectro2 x ps = true) ↔
        (paridade2 x p.1 = p.2 ∧ serveEspectro2 x ps = true)
      rw [hcomb]
    · rw [if_neg hq]

/-- A inserção preserva o servir: sob a hipótese `hcons` (algum ponto serve os pivôs e a equação `e`),
`x` serve o escalonado com `e` inserida se e só se `x` serve os pivôs e serve `e`. No caso redundante a
hipótese força o valor reduzido a ser zero. -/
theorem inserirEscalonado_preserva {n : ℕ} (pivots : Espectro2 n) (e : (Fin n → Bool) × ZMod 2)
    (x : Fin n → Bool)
    (hcons : ∃ y : Fin n → Bool, serveEspectro2 y pivots = true ∧ serveEq y e) :
    serveEspectro2 x (inserirEscalonado e pivots) = true ↔
      serveEspectro2 x pivots = true ∧ serveEq x e := by
  unfold inserirEscalonado
  dsimp only []
  set av := reduzirNova e.1 e.2 pivots with hav
  cases hb : bitPivo av.1 with
  | none =>
    dsimp only []
    have hz : av.1 = fun _ => false := mascara_zero_de_bitPivo_none av.1 hb
    obtain ⟨y, hyp, hye⟩ := hcons
    have hyred : serveEq y av := (reduzirNova_preserva y pivots hyp (e.1, e.2)).mpr hye
    have hyred' : paridade2 y av.1 = av.2 := hyred
    have hv0 : av.2 = 0 := by
      rw [hz, paridade2_mascara_zero] at hyred'
      exact hyred'.symm
    constructor
    · intro hx
      refine ⟨hx, ?_⟩
      have hxred : serveEq x av := by
        show paridade2 x av.1 = av.2
        rw [hz, paridade2_mascara_zero, hv0]
      have hxe : serveEq x (e.1, e.2) :=
        (reduzirNova_preserva x pivots hx (e.1, e.2)).mp hxred
      exact hxe
    · intro h
      exact h.1
  | some q =>
    dsimp only []
    rw [serveEspectro2_cons']
    constructor
    · intro h
      obtain ⟨hxav, hxmap⟩ := h
      have hxpiv : serveEspectro2 x pivots = true :=
        (pivos_reduzidos_preserva x av.1 av.2 hxav q pivots).mp hxmap
      refine ⟨hxpiv, ?_⟩
      exact (reduzirNova_preserva x pivots hxpiv (e.1, e.2)).mp hxav
    · intro h
      obtain ⟨hxpiv, hxe⟩ := h
      have hxav : serveEq x av := (reduzirNova_preserva x pivots hxpiv (e.1, e.2)).mpr hxe
      exact ⟨hxav, (pivos_reduzidos_preserva x av.1 av.2 hxav q pivots).mpr hxpiv⟩

end TGLExt.OSolverGF2_Parte2

