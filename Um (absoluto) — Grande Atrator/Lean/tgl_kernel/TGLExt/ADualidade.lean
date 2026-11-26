import TGLExt.AFibraECoclasse
import Mathlib.LinearAlgebra.Dual.Lemmas
import Mathlib.LinearAlgebra.Pi

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A dualidade em GF(2): o eixo fora do span toma os dois valores dentro de um resíduo habitado

## O que é

O cubo booleano `Fin n → Bool` lido como o espaço vetorial `Fin n → ZMod 2`; o eixo como funcional linear (a forma
bilinear simétrica de GF(2), o produto escalar `∑ a i * x i`; não é produto interno no sentido estrito, porque em
GF(2) não é definida positiva); a paridade de um ponto num eixo como o valor desse funcional; e o teorema do
aniquilador duplo do Mathlib (`Subspace.dualCoannihilator_dualAnnihilator_eq`) [KNOWN, álgebra linear em dimensão
finita] trazido ao vocabulário de `AMorteEARessurreicao` e `AParedeDoSpan` (`paridade2`, `leitura2`, `xorComb`,
`eixoNoSpan`).

## O que fica provado [REAL]

1. `comoVetor`, `comoPonto` [INPUT, definições] e `comoVetor_ida_e_volta`, `comoPonto_ida_e_volta`: as duas
   traduções entre o cubo booleano e `Fin n → ZMod 2` são inversas uma da outra.
2. `funcDe` [INPUT, definição: o funcional do eixo, soma de `a i • proj i`], `funcDe_apply` (o valor é
   `∑ a i * x i`), `funcDe_zero`, `funcDeMap` (a ida do eixo ao funcional é linear), `funcDe_inj` (é injetiva) e
   ★ `paridade2_e_o_funcional`: `paridade2 x a = funcDe (comoVetor a) (comoVetor x)`, isto é, a paridade é o
   funcional do eixo `a` aplicado a `x`.
3. `funcDe_xorPonto`: a tradução leva o xor de pontos à soma em GF(2). `xorPonto_zero`: o ponto zero é neutro no
   xor de pontos. `xorComb_nil`, `xorComb_de_iniciado`, `xorComb_append`, `xorComb_un`: a aritmética do xor de
   listas. `leitura2_zeros_sse`: a leitura é toda zero sse cada eixo da lista lê zero.
4. `o_funcional_no_span_extrai_o_eixo`: se o funcional de `a` está no span dos funcionais dos eixos da lista,
   então `eixoNoSpan axes a`.
5. ★★ `a_dualidade`: se o eixo `a` lê zero em todo ponto cuja leitura pela lista `axes` é toda zero (o núcleo da
   leitura), então `eixoNoSpan axes a`. É a inclusão não trivial de «o aniquilador do núcleo é o span das linhas».
   Na tradução, o núcleo da leitura é o coaniquilador `Φ.dualCoannihilator` do span `Φ` dos funcionais dos eixos da
   lista: a prova põe o funcional de `a` em `Φ.dualCoannihilator.dualAnnihilator`, aplica a esse `Φ` o teorema
   `Subspace.dualCoannihilator_dualAnnihilator_eq` (que dá `Φ`) e volta ao eixo por
   `o_funcional_no_span_extrai_o_eixo`.
6. ★★ `o_eixo_transversal_divide`: com as hipóteses `¬ eixoNoSpan axes a` (o eixo fora do span) e
   `leitura2 axes x₀ = vals` para algum `x₀` (o resíduo de valores `vals` é habitado), existe no resíduo um ponto
   de paridade 0 no eixo `a` e outro de paridade 1.

Leitura [ONTO]: o eixo transversal lido é a poda que divide o resíduo; é o tetelestai: opera o recibo, a
transcrição. O teorema 6 dá condições suficientes para essa divisão.

## O que NÃO se afirma

* O teorema 6 prova só a direção suficiente: eixo fora do span e resíduo habitado implicam que o eixo toma os dois
  valores. A recíproca (com o eixo no span, a paridade é constante no resíduo `leitura2 axes x = vals`) não é
  provada neste módulo [DERIVED, fora deste módulo].
* Sem a hipótese do span, o eixo pode não dividir. O contraexemplo é `a_parede_do_eixo_no_span` (de
  `AParedeDoSpan`), enunciado sobre o resíduo de `serveEspectro2`: todo ponto que serve
  `S2 = [(eixo0, 1), (eixo1, 0)]` tem paridade 1 no eixo `eixo01`. A ponte para o resíduo usado aqui é
  `serveEspectro2_leitura2` (de `AMorteEARessurreicao`): servir `S` equivale a
  `leitura2 (S.map Prod.fst) x = S.map Prod.snd`; com `S2`, os eixos são `[eixo0, eixo1]` e os valores `[1, 0]`.
* O eixo pode estar no span sem estar na lista: `o_eixo01_esta_no_span` (`eixo01` está no span de
  `[eixo0, eixo1]`) e `o_eixo_no_span_nao_esta_na_lista` (`eixo01 ∉ [eixo0, eixo1]`), ambos de `AParedeDoSpan`; e
  aí a leitura não divide. Logo uma checagem por pertença à lista, como a de `hdiv` (`∉ S.map Prod.fst`), não basta
  para que o eixo divida o resíduo; por isso a hipótese do teorema 6 é pelo span, `¬ eixoNoSpan axes a`.
* Nada aqui diz, para fórmula alguma, que o eixo da morte (`eixoDaMorteDe`) fique fora do span. Esta pedra não
  prova `gateG2linha` (de `AParedeDoSpan`) para função alguma, nem o toma como hipótese. A refutação de
  `gateG2linha` como está escrito é esperada, e não é medida no kernel [DERIVED, fora do kernel; não há teorema
  `gateG2linha_insatisfazivel`]: os eixos unitários lidos no próprio ponto geram todo o cubo. Se a refutação vale,
  juntar o teorema 6 com `gateG2linha` dá um enunciado que vale no vazio.
* Esta pedra não toma como hipótese `GatesDaMorte`, `GatesDaDescida`, `AVerdadeFinal`, nem `hdiv` e `hpoco` juntas.
  Medido em `TGLExt.AVacuidadeDosContratos`: os três primeiros não podem ser cumpridos como estão escritos
  (`gatesDaMorte_insatisfazivel`, `gatesDaDescida_insatisfazivel`, `aVerdadeFinal_insatisfazivel`); `hdiv` e
  `hpoco` não valem juntas para nenhum `n` e nenhuma fórmula (`gates_do_processo_insatisfaziveis`); `hfiel` pode
  ser cumprida (`hfiel_habitado`).
* Não se afirma que, quando a fórmula tem testemunha, a descida a encontre (a completude) [OPEN], nem o transporte
  construtivo do processo para a máquina de fita [OPEN].
* A eliminação em GF(2) é polinomial [KNOWN]; isso não é decidir SAT. Nada aqui diz que algum processo, ou a
  máquina de Turing nua, decida SAT em tempo polinomial; isso segue [OPEN]. O gate do Instituto (P versus NP, do
  Instituto Clay) segue de pé e nomeado.
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ADualidade

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse

variable {n : ℕ}

/-! ## O cubo é o espaço GF(2) -/

/-- [DEF] O ponto como vetor GF(2). -/
def comoVetor {n : ℕ} (x : Fin n → Bool) : Fin n → ZMod 2 := fun i => if x i then 1 else 0

/-- [DEF] O vetor como ponto. -/
def comoPonto {n : ℕ} (v : Fin n → ZMod 2) : Fin n → Bool := fun i => v i == 1

/-- A ida e a volta do ponto. (Auxiliar da ponte.) -/
theorem comoVetor_ida_e_volta {n : ℕ} (x : Fin n → Bool) : comoPonto (comoVetor x) = x := by
  ext i
  unfold comoPonto comoVetor
  cases x i <;> decide

/-- A ida e a volta do vetor. (Auxiliar da ponte.) -/
theorem comoPonto_ida_e_volta {n : ℕ} (v : Fin n → ZMod 2) : comoVetor (comoPonto v) = v := by
  ext i
  unfold comoPonto comoVetor
  cases zmod2_exaustivo (v i) with
  | inl h => simp [h]
  | inr h => simp [h]

/-! ## O eixo como funcional -/

/-- [DEF] O funcional do eixo: a forma bilinear de GF(2) (o produto escalar `∑ a i * x i`) com o eixo fixado. -/
def funcDe {n : ℕ} (a : Fin n → ZMod 2) : Module.Dual (ZMod 2) (Fin n → ZMod 2) :=
  ∑ i : Fin n, (a i) • LinearMap.proj i

/-- O funcional aplicado é a soma dos produtos. (Auxiliar da ponte.) -/
theorem funcDe_apply {n : ℕ} (a x : Fin n → ZMod 2) : funcDe a x = ∑ i : Fin n, a i * x i := by
  unfold funcDe
  rw [LinearMap.sum_apply]
  apply Finset.sum_congr rfl
  intro i _
  rw [LinearMap.smul_apply, LinearMap.proj_apply, smul_eq_mul]

/-- ★ A paridade é o funcional: `paridade2 x a` é o funcional do eixo `a` (a forma bilinear de GF(2)) aplicado
ao ponto `x`, pelas traduções `comoVetor`. -/
theorem paridade2_e_o_funcional {n : ℕ} (x a : Fin n → Bool) :
    paridade2 x a = funcDe (comoVetor a) (comoVetor x) := by
  rw [funcDe_apply]
  unfold paridade2
  apply Finset.sum_congr rfl
  intro i _
  unfold comoVetor
  cases a i <;> cases x i <;> decide

/-- O funcional do eixo zero é zero. (Auxiliar do span.) -/
theorem funcDe_zero {n : ℕ} : funcDe (0 : Fin n → ZMod 2) = 0 := by
  unfold funcDe
  simp

/-- O mapa do eixo ao funcional é linear (no eixo). -/
def funcDeMap {n : ℕ} : (Fin n → ZMod 2) →ₗ[ZMod 2] Module.Dual (ZMod 2) (Fin n → ZMod 2) where
  toFun := funcDe
  map_add' := fun a b => by
    apply LinearMap.ext
    intro x
    rw [funcDe_apply, LinearMap.add_apply, funcDe_apply, funcDe_apply, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro i _
    rw [Pi.add_apply, add_mul]
  map_smul' := fun r a => by
    apply LinearMap.ext
    intro x
    rw [funcDe_apply, LinearMap.smul_apply, smul_eq_mul, funcDe_apply, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    simp [Pi.smul_apply, smul_eq_mul, mul_assoc, RingHom.id_apply]

/-- O mapa do eixo ao funcional é injetivo: o funcional determina o eixo (lendo a base). (Auxiliar da
extração.) -/
theorem funcDe_inj {n : ℕ} (a b : Fin n → ZMod 2) (h : funcDe a = funcDe b) : a = b := by
  ext i
  have h1 := congrFun (congrArg DFunLike.coe h) (Pi.single i 1)
  rw [funcDe_apply, funcDe_apply] at h1
  rw [Finset.sum_eq_single i (fun j _ hji => by
      rw [Pi.single_eq_of_ne hji, mul_zero]) (by simp)] at h1
  rw [Finset.sum_eq_single i (fun j _ hji => by
      rw [Pi.single_eq_of_ne hji, mul_zero]) (by simp)] at h1
  simpa [Pi.single_eq_same] using h1

/-- A tradução do xor de dois pontos é a soma, em GF(2), das traduções: a ponte da aditividade. (Auxiliar
da extração.) -/
theorem funcDe_xorPonto {n : ℕ} (x y : Fin n → Bool) :
    comoVetor (xorPonto x y) = comoVetor x + comoVetor y := by
  ext i
  simp only [Pi.add_apply]
  unfold comoVetor xorPonto
  cases hx : x i <;> cases hy : y i <;> decide

/-! ## A aritmética do span por xor -/

/-- O xor da lista vazia é o ponto zero. -/
theorem xorComb_nil {n : ℕ} : xorComb ([] : List (Fin n → Bool)) = fun _ => false := rfl

/-- O ponto zero é neutro no xor; e dobrar com início é dobrar com a dobra pura em xor. (Auxiliares da
extração.) -/
theorem xorPonto_zero {n : ℕ} (x : Fin n → Bool) : xorPonto x (fun _ => false) = x := by
  ext i
  show (x i != false) = x i
  cases hx : x i <;> decide

theorem xorComb_de_iniciado {n : ℕ} (b : Fin n → Bool) (l : List (Fin n → Bool)) :
    l.foldl xorPonto b = xorPonto b (l.foldl xorPonto (fun _ => false)) := by
  induction l generalizing b with
  | nil => exact (xorPonto_zero b).symm
  | cons a ax ih =>
    simp only [List.foldl_cons]
    rw [ih (xorPonto b a), ih (xorPonto (fun _ => false) a)]
    ext i
    show ((b i != a i) != (ax.foldl xorPonto (fun _ => false)) i)
      = (b i != ((false != a i) != (ax.foldl xorPonto (fun _ => false)) i))
    cases hb : b i <;> cases ha : a i <;> cases hF : (ax.foldl xorPonto (fun _ => false)) i <;> decide

/-- O xor da lista composta é o xor das dobras. (Auxiliar da extração.) -/
theorem xorComb_append {n : ℕ} (s₁ s₂ : List (Fin n → Bool)) :
    xorComb (s₁ ++ s₂) = xorPonto (xorComb s₁) (xorComb s₂) := by
  unfold xorComb
  rw [List.foldl_append]
  exact xorComb_de_iniciado _ _

/-- O xor de um eixo só é o próprio eixo. (Auxiliar da extração.) -/
theorem xorComb_un {n : ℕ} (m : Fin n → Bool) : xorComb [m] = m := by
  unfold xorComb
  ext i
  unfold xorPonto
  simp

/-- A leitura zero sse cada eixo lê zero. (Auxiliar da ponte do núcleo.) -/
theorem leitura2_zeros_sse {n : ℕ} (axes : List (Fin n → Bool)) (x : Fin n → Bool) :
    leitura2 axes x = List.replicate axes.length 0 ↔ ∀ a ∈ axes, paridade2 x a = 0 := by
  unfold leitura2
  induction axes with
  | nil => simp
  | cons a ax ih =>
    rw [List.length_cons, List.map_cons, List.replicate_succ, List.cons.injEq, List.forall_mem_cons]
    constructor
    · intro h
      exact ⟨h.1, ih.mp h.2⟩
    · intro h
      exact ⟨h.1, ih.mpr h.2⟩

/-- ★ A extração: o funcional no span dos funcionais dos eixos dá o eixo no span dos eixos (por indução no
span, e `funcDe_inj` lendo a base). -/
theorem o_funcional_no_span_extrai_o_eixo {n : ℕ} (axes : List (Fin n → Bool)) (a : Fin n → Bool)
    (h : funcDe (comoVetor a)
      ∈ Submodule.span (ZMod 2) ((fun m => funcDe (comoVetor m)) '' {m | m ∈ axes})) :
    eixoNoSpan axes a := by
  have hP : ∃ m : Fin n → Bool,
      (∃ sub : List (Fin n → Bool), sub ⊆ axes ∧ xorComb sub = m)
        ∧ funcDe (comoVetor a) = funcDe (comoVetor m) := by
    refine Submodule.span_induction (p := fun φ (_ : φ ∈ Submodule.span (ZMod 2) _) =>
      ∃ m : Fin n → Bool, (∃ sub : List (Fin n → Bool), sub ⊆ axes ∧ xorComb sub = m)
        ∧ φ = funcDe (comoVetor m)) ?_ ?_ ?_ ?_ h
    · intro φ hφ
      obtain ⟨m0, hm0, hφ0⟩ := hφ
      exact ⟨m0, ⟨[m0], by simpa using hm0, xorComb_un m0⟩, hφ0.symm⟩
    · exact ⟨fun _ => false, ⟨[], List.nil_subset _, rfl⟩, by
        show (0 : Module.Dual (ZMod 2) (Fin n → ZMod 2)) = funcDe (comoVetor fun _ => false)
        have h0 : comoVetor (fun _ => false) = (0 : Fin n → ZMod 2) := by
          ext i
          simp [comoVetor]
        rw [h0, funcDe_zero]⟩
    · intro φ₁ φ₂ _ _ hφ₁ hφ₂
      obtain ⟨m₁, ⟨sub1, hsub1, hx1⟩, hφ1⟩ := hφ₁
      obtain ⟨m₂, ⟨sub2, hsub2, hx2⟩, hφ2⟩ := hφ₂
      refine ⟨xorPonto m₁ m₂, ⟨sub1 ++ sub2, ?_, ?_⟩, ?_⟩
      · intro s hs
        rw [List.mem_append] at hs
        cases hs with
        | inl h1 => exact hsub1 h1
        | inr h2 => exact hsub2 h2
      · rw [xorComb_append, hx1, hx2]
      · rw [hφ1, hφ2]
        show funcDe (comoVetor m₁) + funcDe (comoVetor m₂) = funcDe (comoVetor (xorPonto m₁ m₂))
        rw [funcDe_xorPonto]
        exact (funcDeMap.map_add _ _).symm
    · intro r φ _ hφ
      obtain ⟨m₀, hsub0, hφ0⟩ := hφ
      cases zmod2_exaustivo r with
      | inl hr =>
        refine ⟨fun _ => false, ⟨[], List.nil_subset _, rfl⟩, ?_⟩
        rw [hr]
        show (0 : ZMod 2) • φ = funcDe (comoVetor fun _ => false)
        rw [zero_smul]
        have h0 : comoVetor (fun _ => false) = (0 : Fin n → ZMod 2) := by
          ext i
          simp [comoVetor]
        rw [h0, funcDe_zero]
      | inr hr =>
        rw [hr]
        exact ⟨m₀, hsub0, by rw [one_smul]; exact hφ0⟩
  obtain ⟨m, hsub, hfunc⟩ := hP
  have hm : m = a := by
    have h1 : comoVetor m = comoVetor a := funcDe_inj _ _ hfunc.symm
    ext i
    have h2 := congrFun h1 i
    unfold comoVetor at h2
    cases hm2 : m i <;> cases ha2 : a i
    · rfl
    · rw [hm2, ha2] at h2; simp at h2
    · rw [hm2, ha2] at h2; simp at h2
    · rfl
  obtain ⟨sub, hsub, hxor⟩ := hsub
  exact ⟨sub, hsub, by rw [hxor]; exact hm⟩

/-- ★★ **A DUALIDADE** — o eixo que se anula no núcleo da leitura está no span dos eixos (a inclusão não
trivial de «o aniquilador do núcleo é o span das linhas»). O núcleo da leitura é, na tradução, o coaniquilador
`Φ.dualCoannihilator` do span `Φ` dos funcionais dos eixos; a prova aplica a ele
`Subspace.dualCoannihilator_dualAnnihilator_eq`. -/
theorem a_dualidade {n : ℕ} (axes : List (Fin n → Bool)) (a : Fin n → Bool)
    (h : ∀ k : Fin n → Bool, leitura2 axes k = List.replicate axes.length 0 → paridade2 k a = 0) :
    eixoNoSpan axes a := by
  classical
  set Φ : Subspace (ZMod 2) (Module.Dual (ZMod 2) (Fin n → ZMod 2)) :=
    Submodule.span (ZMod 2) ((fun m => funcDe (comoVetor m)) '' {m | m ∈ axes}) with hΦ
  haveI : FiniteDimensional (ZMod 2) Φ := by
    apply FiniteDimensional.span_of_finite
    have h1 : {m : Fin n → Bool | m ∈ axes} = ↑(List.toFinset axes) := by
      ext m
      simp [List.mem_toFinset]
    rw [h1]
    exact (List.toFinset axes).finite_toSet.image _
  have hmem : funcDe (comoVetor a) ∈ Φ.dualCoannihilator.dualAnnihilator := by
    rw [Submodule.mem_dualAnnihilator]
    intro v hv
    rw [Submodule.mem_dualCoannihilator] at hv
    have hk : leitura2 axes (comoPonto v) = List.replicate axes.length 0 := by
      rw [leitura2_zeros_sse]
      intro a0 ha0
      rw [paridade2_e_o_funcional, comoPonto_ida_e_volta]
      exact hv _ (Submodule.subset_span ⟨a0, by simpa using ha0, rfl⟩)
    have h0 := h (comoPonto v) hk
    rw [paridade2_e_o_funcional, comoPonto_ida_e_volta] at h0
    exact h0
  rw [Subspace.dualCoannihilator_dualAnnihilator_eq] at hmem
  exact o_funcional_no_span_extrai_o_eixo axes a (hΦ ▸ hmem)

/-- ★★ **O EIXO TRANSVERSAL DIVIDE** — sob as hipóteses de que o eixo está fora do span (`hfora`) e de que o
resíduo é habitado (`hx₀`), o eixo toma os dois valores dentro do resíduo. Não diz que o eixo da morte de
alguma fórmula esteja fora do span. -/
theorem o_eixo_transversal_divide {n : ℕ} (axes : List (Fin n → Bool)) (vals : List (ZMod 2))
    (a : Fin n → Bool) (hfora : ¬ eixoNoSpan axes a)
    (x₀ : Fin n → Bool) (hx₀ : leitura2 axes x₀ = vals) :
    (∃ x : Fin n → Bool, leitura2 axes x = vals ∧ paridade2 x a = 0)
      ∧ (∃ x : Fin n → Bool, leitura2 axes x = vals ∧ paridade2 x a = 1) := by
  cases zmod2_exaustivo (paridade2 x₀ a) with
  | inl hv0 =>
    refine ⟨⟨x₀, hx₀, hv0⟩, ?_⟩
    by_contra hnao
    push_neg at hnao
    have hzero : ∀ k : Fin n → Bool, leitura2 axes k = List.replicate axes.length 0 →
        paridade2 k a = 0 := by
      intro k hk
      have hlen : vals.length = axes.length := by rw [← hx₀]; exact leitura2_length axes x₀
      have hcom : leitura2 axes (xorPonto k x₀) = vals := by
        rw [leitura2_xor, hk, hx₀, zipAdd_zero_esq _ _ hlen]
      have hpar := hnao (xorPonto k x₀) hcom
      have hpar0 : paridade2 (xorPonto k x₀) a = 0 := by
        cases zmod2_exaustivo (paridade2 (xorPonto k x₀) a) with
        | inl h1 => exact h1
        | inr h1 => exact absurd h1 hpar
      rw [paridade2_add, hv0, add_zero] at hpar0
      exact hpar0
    exact hfora (a_dualidade axes a hzero)
  | inr hv1 =>
    refine ⟨?_, ⟨x₀, hx₀, hv1⟩⟩
    by_contra hnao
    push_neg at hnao
    have hzero : ∀ k : Fin n → Bool, leitura2 axes k = List.replicate axes.length 0 →
        paridade2 k a = 0 := by
      intro k hk
      have hlen : vals.length = axes.length := by rw [← hx₀]; exact leitura2_length axes x₀
      have hcom : leitura2 axes (xorPonto k x₀) = vals := by
        rw [leitura2_xor, hk, hx₀, zipAdd_zero_esq _ _ hlen]
      have hpar := hnao (xorPonto k x₀) hcom
      have hpar1 : paridade2 (xorPonto k x₀) a = 1 := by
        cases zmod2_exaustivo (paridade2 (xorPonto k x₀) a) with
        | inl h1 => exact absurd h1 hpar
        | inr h1 => exact h1
      rw [paridade2_add, hv1] at hpar1
      have h2 : paridade2 k a + 1 = 1 := hpar1
      have h3 : paridade2 k a = 0 := by
        cases zmod2_exaustivo (paridade2 k a) with
        | inl hh => exact hh
        | inr hh => rw [hh] at h2; simp at h2
      exact h3
    exact hfora (a_dualidade axes a hzero)

end TGLExt.ADualidade

