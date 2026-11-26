import TGLExt.OSolverGF2_Parte3

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# O solver GF(2): o invariante preservado e a correção do resolvedor

## O que é

A eliminação de Gauss–Jordan sobre GF(2) (`reduzirNova`, `inserirEscalonado`, `oEscalonado`, em `OSolverGF2`) leva um
espectro (lista de equações de paridade) à forma isolada, e o ponto construído (`pontoConstruido`) lê dessa forma um ponto,
bit a bit. A ressurreição construtiva (`ressurreicaoConstrutiva`) é esse ponto, calculado do espectro, sem escolha.

## O que fica provado [REAL]

1. `bitPivo_iff`: o bit de pivô de uma máscara é um bit aceso, e nenhum bit aceso vem antes dele.
   `bitPivo_xor_preserva`: se o pivô de `p` é `i`, o pivô de `a` é `q` e `i < q`, o ou-exclusivo de `p` com `a` tem pivô `i`.
2. `reduzirNova_preserva_bit`: se toda equação da lista que acende o bit `i` tem pivô `i`, a redução de uma máscara com o bit
   `i` apagado o deixa apagado. `reduzirNova_livre`: com a lista na forma isolada, a máscara reduzida não tem nenhum bit de
   pivô da lista.
3. ★ `inserirEscalonado_preserva_isolada`: a inserção preserva a forma isolada. `oEscalonado_isolada`: o escalonado de
   qualquer espectro está na forma isolada.
4. ★ `oEscalonado_preserva`: quando o espectro tem um ponto que o serve, um ponto serve o escalonado se e só se serve o
   espectro original.
5. ★★ `o_solver_serve`: quando o espectro tem um ponto que o serve, o ponto construído do escalonado serve o espectro
   original. É a correção do resolvedor construtivo: a ressurreição fiel, calculada sem escolha clássica.

## O que NÃO se afirma

* O custo: o número de passos da eliminação não é contado aqui. Que a eliminação de Gauss–Jordan leve tempo polinomial é
  [KNOWN]. Do transporte T, como `ARessurreicao` o nomeia (um resolvedor construtivo e polinomial que cumpra o G1), fica
  provada aqui a correção; o custo segue fora do kernel.
* Nada aqui trata da divisão na morte (`hdiv`) nem do eco: que, havendo solução, o eco venha segue [OPEN].
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.OSolverGF2_Parte4

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3

variable {n : ℕ}

/-- ★ A caracterização do bit de pivô: aceso e mínimo. (Auxiliar do invariante.) -/
theorem bitPivo_iff {n : ℕ} (a : Fin n → Bool) (i : Fin n) :
    bitPivo a = some i ↔ a i = true ∧ ∀ j : Fin n, a j = true → i ≤ j := by
  constructor
  · intro h
    exact ⟨bitPivo_spec a i h, bitPivo_minimal a i h⟩
  · intro h
    unfold bitPivo
    have hne : ((Finset.univ : Finset (Fin n)).filter fun k => a k = true).Nonempty :=
      ⟨i, by simp [Finset.mem_filter, h.1]⟩
    rw [dif_pos hne]
    have hmem : i ∈ (Finset.univ : Finset (Fin n)).filter (fun k => a k = true) := by
      simp [Finset.mem_filter, h.1]
    have h1 : ((Finset.univ : Finset (Fin n)).filter (fun k => a k = true)).min' hne ≤ i :=
      Finset.min'_le _ _ hmem
    have h2 : i ≤ ((Finset.univ : Finset (Fin n)).filter (fun k => a k = true)).min' hne := by
      have hm := Finset.min'_mem _ hne
      simp [Finset.mem_filter] at hm
      exact h.2 _ hm
    have heq : ((Finset.univ : Finset (Fin n)).filter (fun k => a k = true)).min' hne = i :=
      le_antisymm h1 h2
    rw [heq]

/-- ★ O xor com a máscara além preserva o pivô. (Auxiliar do invariante.) -/
theorem bitPivo_xor_preserva {n : ℕ} (p a : Fin n → Bool) (i q : Fin n)
    (hp : bitPivo p = some i) (ha : bitPivo a = some q) (hiq : i < q) :
    bitPivo (xorPonto p a) = some i := by
  rw [bitPivo_iff]
  have hpi : p i = true := bitPivo_spec p i hp
  have hai : a i = false := by
    by_contra hc
    have hbit : a i = true := by cases h : a i <;> simp_all
    have hle := bitPivo_minimal a q ha i hbit
    have hle' : (q : ℕ) ≤ (i : ℕ) := hle
    have hiq' : (i : ℕ) < (q : ℕ) := hiq
    omega
  constructor
  · show (p i != a i) = true
    rw [hpi, hai]
    rfl
  · intro j hj
    show i ≤ j
    by_contra hji
    have hji' : (j : ℕ) < (i : ℕ) := by
      have h1 : ¬ (i : Fin n) ≤ j := hji
      have h2 : ¬ (i : ℕ) ≤ (j : ℕ) := fun h => h1 h
      omega
    have hpj : p j = false := by
      by_contra hc
      have hbit : p j = true := by cases h : p j <;> simp_all
      have hle := bitPivo_minimal p i hp j hbit
      have hle' : (i : ℕ) ≤ (j : ℕ) := hle
      omega
    have haj : a j = false := by
      by_contra hc
      have hbit : a j = true := by cases h : p j <;> simp_all
      have hle := bitPivo_minimal a q ha j hbit
      have hle' : (q : ℕ) ≤ (j : ℕ) := hle
      omega
    have hxor : (p j != a j) = true := hj
    rw [hpj, haj] at hxor
    simp at hxor

/-- ★ A redução não acende um bit morto. (Auxiliar do livre.) -/
theorem reduzirNova_preserva_bit {n : ℕ} (i : Fin n) :
    ∀ (pivots : Espectro2 n), (∀ q ∈ pivots, q.1 i = true → bitPivo q.1 = some i) →
      ∀ (a : Fin n → Bool) (v : ZMod 2), a i = false → (reduzirNova a v pivots).1 i = false := by
  intro pivots
  induction pivots with
  | nil => intro _ a v ha; simpa [reduzirNova] using ha
  | cons p ps ih =>
    intro hpivots a v ha
    rw [reduzirNova_cons]
    apply ih (fun q hq => hpivots q (List.mem_cons_of_mem _ hq))
    unfold passoReduzir
    cases hb : bitPivo p.1 with
    | none => simp [hb]; exact ha
    | some j =>
      dsimp only []
      by_cases hbit : a j = true
      · rw [if_pos hbit]
        show (xorPonto a p.1) i = false
        show (a i != p.1 i) = false
        by_cases hpi : p.1 i = true
        · have hji := hpivots p (List.mem_cons_self) hpi
          rw [hb] at hji
          have hji' : j = i := Option.some.inj hji
          subst hji'
          rw [hbit] at ha
          simp at ha
        · have hpf : p.1 i = false := by cases h : p.1 i <;> simp_all
          rw [ha, hpf]
          rfl
      · rw [if_neg hbit]; exact ha

/-- ★ A máscara reduzida não tem os bits de pivô da lista. (Auxiliar do invariante.) -/
theorem reduzirNova_livre {n : ℕ} :
    ∀ (pivots : Espectro2 n), formaIsolada pivots →
      ∀ (a : Fin n → Bool) (v : ZMod 2) (i : Fin n),
        (∃ p ∈ pivots, bitPivo p.1 = some i) → (reduzirNova a v pivots).1 i = false := by
  intro pivots
  induction pivots with
  | nil =>
    intro _ a v i hi
    obtain ⟨p, hp, _⟩ := hi
    exact absurd hp (List.not_mem_nil)
  | cons p ps ih =>
    intro hiso a v i hi
    obtain ⟨hiso1, hiso2, hiso3⟩ := hiso
    have hps : formaIsolada ps := by
      refine ⟨?_, ?_, ?_⟩
      · intro q hq
        exact hiso1 q (List.mem_cons_of_mem _ hq)
      · intro q₁ hq₁ q₂ hq₂ j
        exact hiso2 q₁ (List.mem_cons_of_mem _ hq₁) q₂ (List.mem_cons_of_mem _ hq₂) j
      · intro q₁ hq₁ q₂ hq₂ j₁ j₂
        exact hiso3 q₁ (List.mem_cons_of_mem _ hq₁) q₂ (List.mem_cons_of_mem _ hq₂) j₁ j₂
    rw [reduzirNova_cons]
    obtain ⟨p', hp', hpi⟩ := hi
    rw [List.mem_cons] at hp'
    cases hp' with
    | inl hpp =>
      cases hpp
      have h1 : (passoReduzir p (a, v)).1 i = false := by
        unfold passoReduzir
        rw [hpi]
        simp only []
        by_cases hbit : a i = true
        · rw [if_pos hbit]
          show (xorPonto a p.1) i = false
          show (a i != p.1 i) = false
          rw [hbit, bitPivo_spec p.1 i hpi]
          rfl
        · rw [if_neg hbit]
          cases h : a i <;> simp_all
      have hps' : ∀ q ∈ ps, q.1 i = true → bitPivo q.1 = some i := by
        intro q hq hqi
        obtain ⟨jq, hjq⟩ := hiso1 q (List.mem_cons_of_mem _ hq)
        by_cases hjeq : jq = i
        · subst hjeq; exact hjq
        · have hz := hiso3 q (List.mem_cons_of_mem _ hq) p (List.mem_cons_self) jq i hjq hpi hjeq
          rw [hz] at hqi
          simp at hqi
      exact reduzirNova_preserva_bit i ps hps' _ _ h1
    | inr hps_mem =>
      exact ih hps _ _ i ⟨p', hps_mem, hpi⟩

/-- ★ A inserção preserva a forma isolada. -/
theorem inserirEscalonado_preserva_isolada {n : ℕ} (pivots : Espectro2 n) (e : (Fin n → Bool) × ZMod 2)
    (hiso : formaIsolada pivots) :
    formaIsolada (inserirEscalonado e pivots) := by
  unfold inserirEscalonado
  dsimp only []
  obtain ⟨hiso1, hiso2, hiso3⟩ := hiso
  set av := reduzirNova e.1 e.2 pivots with hav
  cases hb : bitPivo av.1 with
  | none =>
    dsimp only []
    exact ⟨hiso1, hiso2, hiso3⟩
  | some q =>
    dsimp only []
    have livre : ∀ j : Fin n, (∃ p ∈ pivots, bitPivo p.1 = some j) → av.1 j = false :=
      fun j hj => reduzirNova_livre pivots ⟨hiso1, hiso2, hiso3⟩ _ _ j hj
    -- F2: o pivô da mapeada é o da original
    have hpiv : ∀ p₀ ∈ pivots, ∀ i₀ : Fin n, bitPivo p₀.1 = some i₀ →
        bitPivo (if p₀.1 q = true then (xorPonto p₀.1 av.1, p₀.2 + av.2) else p₀).1 = some i₀ := by
      intro p₀ hp₀ i₀ hi₀
      by_cases hf : p₀.1 q = true
      · rw [if_pos hf]
        show bitPivo (xorPonto p₀.1 av.1) = some i₀
        apply bitPivo_xor_preserva p₀.1 av.1 i₀ q hi₀ hb
        have hle := bitPivo_minimal p₀.1 i₀ hi₀ q hf
        have hne : i₀ ≠ q := by
          intro heq
          cases heq
          have h1 : av.1 q = false := livre q ⟨p₀, hp₀, hi₀⟩
          have h2 : av.1 q = true := bitPivo_spec av.1 q hb
          rw [h1] at h2
          simp at h2
        have hle' : (i₀ : ℕ) ≤ (q : ℕ) := hle
        have hne' : (i₀ : ℕ) ≠ (q : ℕ) := fun h => hne (Fin.ext h)
        show (i₀ : Fin n) < q
        omega
      · rw [if_neg hf]
        exact hi₀
    -- F3: o q é fresco
    have hfresco : ∀ p₀ ∈ pivots, ∀ i₀ : Fin n, bitPivo p₀.1 = some i₀ → i₀ ≠ q := by
      intro p₀ hp₀ i₀ hi₀ heq
      cases heq
      have h1 : av.1 q = false := livre q ⟨p₀, hp₀, hi₀⟩
      have h2 : av.1 q = true := bitPivo_spec av.1 q hb
      rw [h1] at h2
      simp at h2
    refine ⟨?_, ?_, ?_⟩
    · -- toda equação tem pivô
      intro p hp
      rw [List.mem_cons, List.mem_map] at hp
      cases hp with
      | inl h =>
        subst h
        exact ⟨q, hb⟩
      | inr h =>
        obtain ⟨p₀, hp₀, hpeq⟩ := h
        subst hpeq
        obtain ⟨i₀, hi₀⟩ := hiso1 p₀ hp₀
        exact ⟨i₀, hpiv p₀ hp₀ i₀ hi₀⟩
    · -- o mesmo bit é a mesma equação
      intro p₁ hp₁ p₂ hp₂ i hb1 hb2
      rw [List.mem_cons, List.mem_map] at hp₁ hp₂
      cases hp₁ with
      | inl h1 =>
        subst h1
        have hiq : i = q := Option.some.inj (hb1.symm.trans hb)
        cases hiq
        cases hp₂ with
        | inl h2 =>
          subst h2
          rfl
        | inr h2 =>
          obtain ⟨p₀₂, hp₀₂, hpeq₂⟩ := h2
          subst hpeq₂
          obtain ⟨i₂, hi₂⟩ := hiso1 p₀₂ hp₀₂
          have hb2' := hpiv p₀₂ hp₀₂ i₂ hi₂
          rw [hb2'] at hb2
          have hii : i₂ = q := Option.some.inj hb2
          exact absurd hii (hfresco p₀₂ hp₀₂ i₂ hi₂)
      | inr h1 =>
        obtain ⟨p₀₁, hp₀₁, hpeq₁⟩ := h1
        subst hpeq₁
        obtain ⟨i₁, hi₁⟩ := hiso1 p₀₁ hp₀₁
        have hb1' := hpiv p₀₁ hp₀₁ i₁ hi₁
        rw [hb1'] at hb1
        have hi1 : i₁ = i := Option.some.inj hb1
        cases hp₂ with
        | inl h2 =>
          subst h2
          have hiq : i = q := Option.some.inj (hb2.symm.trans hb)
          exact absurd (hi1.trans hiq) (hfresco p₀₁ hp₀₁ i₁ hi₁)
        | inr h2 =>
          obtain ⟨p₀₂, hp₀₂, hpeq₂⟩ := h2
          subst hpeq₂
          obtain ⟨i₂, hi₂⟩ := hiso1 p₀₂ hp₀₂
          have hb2' := hpiv p₀₂ hp₀₂ i₂ hi₂
          rw [hb2'] at hb2
          have hi2 : i₂ = i := Option.some.inj hb2
          have hii : i₁ = i₂ := hi1.trans hi2.symm
          have hb2'' : bitPivo p₀₂.1 = some i₁ := by rw [hii]; exact hi₂
          have hpar : p₀₁ = p₀₂ := hiso2 p₀₁ hp₀₁ p₀₂ hp₀₂ i₁ hi₁ hb2''
          cases hpar
          rfl
    · -- o isolamento
      intro p₁ hp₁ p₂ hp₂ i j hb1 hb2 hij
      rw [List.mem_cons, List.mem_map] at hp₁ hp₂
      cases hp₁ with
      | inl h1 =>
        subst h1
        have hiq : i = q := Option.some.inj (hb1.symm.trans hb)
        cases hiq
        cases hp₂ with
        | inl h2 =>
          subst h2
          exact absurd (Option.some.inj (hb1.symm.trans hb2)) hij
        | inr h2 =>
          obtain ⟨p₀₂, hp₀₂, hpeq₂⟩ := h2
          subst hpeq₂
          obtain ⟨j₂, hj₂⟩ := hiso1 p₀₂ hp₀₂
          have hb2' := hpiv p₀₂ hp₀₂ j₂ hj₂
          rw [hb2'] at hb2
          have hjj : j₂ = j := Option.some.inj hb2
          cases hjj
          exact livre j ⟨p₀₂, hp₀₂, hj₂⟩
      | inr h1 =>
        obtain ⟨p₀₁, hp₀₁, hpeq₁⟩ := h1
        subst hpeq₁
        obtain ⟨i₁, hi₁⟩ := hiso1 p₀₁ hp₀₁
        have hb1' := hpiv p₀₁ hp₀₁ i₁ hi₁
        rw [hb1'] at hb1
        have hi1 : i₁ = i := Option.some.inj hb1
        cases hi1
        cases hp₂ with
        | inl h2 =>
          subst h2
          have hjq : j = q := Option.some.inj (hb2.symm.trans hb)
          cases hjq
          by_cases hf : p₀₁.1 q = true
          · rw [if_pos hf]
            show (xorPonto p₀₁.1 av.1) q = false
            show (p₀₁.1 q != av.1 q) = false
            rw [hf, bitPivo_spec av.1 q hb]
            rfl
          · rw [if_neg hf]
            cases h : p₀₁.1 q <;> simp_all
        | inr h2 =>
          obtain ⟨p₀₂, hp₀₂, hpeq₂⟩ := h2
          subst hpeq₂
          obtain ⟨j₂, hj₂⟩ := hiso1 p₀₂ hp₀₂
          have hb2' := hpiv p₀₂ hp₀₂ j₂ hj₂
          rw [hb2'] at hb2
          have hjj : j₂ = j := Option.some.inj hb2
          cases hjj
          by_cases hf : p₀₁.1 q = true
          · rw [if_pos hf]
            show (xorPonto p₀₁.1 av.1) j = false
            show (p₀₁.1 j != av.1 j) = false
            have h1 : p₀₁.1 j = false :=
              hiso3 p₀₁ hp₀₁ p₀₂ hp₀₂ i j hi₁ hj₂ hij
            have h2 : av.1 j = false := livre j ⟨p₀₂, hp₀₂, hj₂⟩
            rw [h1, h2]
            rfl
          · rw [if_neg hf]
            exact hiso3 p₀₁ hp₀₁ p₀₂ hp₀₂ i j hi₁ hj₂ hij

/-- ★ O escalonado do espectro é isolado. -/
theorem oEscalonado_isolada {n : ℕ} (S : Espectro2 n) : formaIsolada (oEscalonado S) := by
  unfold oEscalonado
  suffices h : ∀ (S' : Espectro2 n) (acc : Espectro2 n), formaIsolada acc →
      formaIsolada (List.foldl (fun ps e => inserirEscalonado e ps) acc S') by
    exact h S [] formaIsolada_nil
  intro S'
  induction S' with
  | nil => intro acc hacc; simpa using hacc
  | cons e S'' ih =>
    intro acc hacc
    rw [List.foldl_cons]
    exact ih (inserirEscalonado e acc) (inserirEscalonado_preserva_isolada acc e hacc)

/-- ★ O escalonado preserva o servir. -/
theorem oEscalonado_preserva {n : ℕ} (S : Espectro2 n) (x : Fin n → Bool)
    (hcons : ∃ y : Fin n → Bool, serveEspectro2 y S = true) :
    serveEspectro2 x (oEscalonado S) = true ↔ serveEspectro2 x S = true := by
  unfold oEscalonado
  suffices h : ∀ (S' : Espectro2 n) (acc : Espectro2 n),
      (∃ y : Fin n → Bool, serveEspectro2 y acc = true ∧ serveEspectro2 y S' = true) →
      (serveEspectro2 x (List.foldl (fun ps e => inserirEscalonado e ps) acc S') = true ↔
        (serveEspectro2 x acc = true ∧ serveEspectro2 x S' = true)) by
    have h1 := h S [] (by obtain ⟨y, hy⟩ := hcons; exact ⟨y, by simp [serveEspectro2], hy⟩)
    have h2 : serveEspectro2 x ([] : Espectro2 n) = true := by simp [serveEspectro2]
    have h3 : (serveEspectro2 x ([] : Espectro2 n) = true ∧ serveEspectro2 x S = true) ↔
        serveEspectro2 x S = true := by
      constructor
      · intro hh; exact hh.2
      · intro hh; exact ⟨h2, hh⟩
    rw [h3] at h1
    exact h1
  intro S'
  induction S' with
  | nil =>
    intro acc hcon
    rw [List.foldl_nil]
    have h2 : serveEspectro2 x ([] : Espectro2 n) = true := by simp [serveEspectro2]
    rw [h2]
    simp
  | cons e S'' ih =>
    intro acc hcon
    rw [List.foldl_cons]
    obtain ⟨y, hyacc, hycons⟩ := hcon
    rw [serveEspectro2_cons'] at hycons
    have hcon' : ∃ y : Fin n → Bool, serveEspectro2 y acc = true ∧ serveEq y e := ⟨y, hyacc, hycons.1⟩
    have hins := inserirEscalonado_preserva acc e x hcon'
    have hcon'' : ∃ y : Fin n → Bool, serveEspectro2 y (inserirEscalonado e acc) = true ∧
        serveEspectro2 y S'' = true :=
      ⟨y, (inserirEscalonado_preserva acc e y hcon').mpr ⟨hyacc, hycons.1⟩, hycons.2⟩
    rw [ih (inserirEscalonado e acc) hcon'']
    rw [hins]
    constructor
    · intro hh
      obtain ⟨⟨h1, h2⟩, h3⟩ := hh
      exact ⟨h1, (serveEspectro2_cons' x e S'').mpr ⟨h2, h3⟩⟩
    · intro hh
      obtain ⟨h1, h2⟩ := hh
      rw [serveEspectro2_cons'] at h2
      exact ⟨⟨h1, h2.1⟩, h2.2⟩

/-- ★★ **A CORREÇÃO DO SOLVER** (o G1 construtivo): com o espectro consistente, o ponto construído do
escalonado serve o espectro original — a ressurreição construtiva é fiel. -/
theorem o_solver_serve {n : ℕ} (S : Espectro2 n)
    (hcons : ∃ y : Fin n → Bool, serveEspectro2 y S = true) :
    serveEspectro2 (ressurreicaoConstrutiva S) S = true := by
  unfold ressurreicaoConstrutiva
  have hiso := oEscalonado_isolada S
  have hserve := pontoConstruido_serve (oEscalonado S) hiso
  have h1 : serveEspectro2 (pontoConstruido (oEscalonado S)) (oEscalonado S) = true := by
    unfold serveEspectro2
    rw [List.all_eq_true]
    intro e he
    exact (beq_iff_eq).mpr (hserve e he)
  exact (oEscalonado_preserva S _ hcons).mp h1

end TGLExt.OSolverGF2_Parte4

