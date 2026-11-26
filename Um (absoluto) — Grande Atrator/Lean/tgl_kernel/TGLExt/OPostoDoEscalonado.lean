import TGLExt.ATensaoFundamental

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# O posto do escalonado: a base reduzida não passa de n pivôs

## O que é

A base escalonada (`oEscalonado`) que o processo da tensão guarda das equações registradas: cada entrada tem um pivô, os
pivôs são distintos e nenhum pivô aparece nas outras entradas (`formaIsolada`).

## O que fica provado [REAL]

1. `passo_reducao_mata_o_pivo`, `fold_preserva_morto`: o passo da redução zera o bit do pivô processado, e o bit zerado
   segue zero nos passos seguintes quando os outros pivôs não o tocam.
2. ★ `reduzirNova_sem_pivos`: numa base isolada, a máscara reduzida não tem o bit de pivô de nenhuma equação da base.
3. `xorPonto_cancela_esq`, `xorPonto_eq_dir`, `bitPivo_zero`: aritmética auxiliar.
4. ★ `inserirEscalonado_nodup`, `oEscalonado_nodup`: a inserção numa base isolada e sem repetição não repete entradas; o
   escalonado não repete entradas.
5. ★★ `oEscalonado_length_le`: o escalonado tem no máximo n entradas.
6. ★ `oEscalonado_cresce_um_com_retorno`: registrar uma literal de retorno útil (`retornoUtil`) cresce o escalonado de
   exatamente uma entrada.

## O que NÃO se afirma

* Que o processo da tensão responda com eco em toda fórmula que tem solução: a completude em geral segue [OPEN].
* A contagem dos passos do processo na fita, ou que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial:
  segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.OPostoDoEscalonado

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental

variable {n : ℕ}

/-- [DEF] O passo da redução (desdobrado do `reduzirNova`, para as induções). -/
def passoReducao {n : ℕ} (av : (Fin n → Bool) × ZMod 2) (p : (Fin n → Bool) × ZMod 2) :
    (Fin n → Bool) × ZMod 2 :=
  match bitPivo p.1 with
  | none => av
  | some i => if av.1 i then (xorPonto av.1 p.1, av.2 + p.2) else av

/-- O `reduzirNova` é a dobra do passo. -/
theorem reduzirNova_eq_fold {n : ℕ} (a : Fin n → Bool) (v : ZMod 2) (ps : Espectro2 n) :
    reduzirNova a v ps = ps.foldl passoReducao (a, v) := rfl

/-- A forma isolada da cauda. (Auxiliar da indução.) -/
theorem formaIsolada_cons {n : ℕ} (p : (Fin n → Bool) × ZMod 2) (ps : Espectro2 n)
    (h : formaIsolada (p :: ps)) : formaIsolada ps := by
  obtain ⟨h1, h2, h3⟩ := h
  refine ⟨?_, ?_, ?_⟩
  · intro q hq; exact h1 q (List.mem_cons_of_mem p hq)
  · intro q₁ hq₁ q₂ hq₂ i hp₁ hp₂
    exact h2 q₁ (List.mem_cons_of_mem p hq₁) q₂ (List.mem_cons_of_mem p hq₂) i hp₁ hp₂
  · intro q₁ hq₁ q₂ hq₂ i j hp₁ hp₂ hij
    exact h3 q₁ (List.mem_cons_of_mem p hq₁) q₂ (List.mem_cons_of_mem p hq₂) i j hp₁ hp₂ hij

/-- O passo da redução mata o bit do pivô processado. -/
theorem passo_reducao_mata_o_pivo {n : ℕ} (av : (Fin n → Bool) × ZMod 2)
    (p : (Fin n → Bool) × ZMod 2) (j : Fin n) (hj : bitPivo p.1 = some j) :
    (passoReducao av p).1 j = false := by
  unfold passoReducao
  split
  · rename_i hnone
    rw [hnone] at hj
    simp at hj
  · rename_i i hi
    rw [hi] at hj
    have hij : j = i := (Option.some.inj hj).symm
    subst hij
    by_cases hc : av.1 j = true
    · rw [if_pos hc]
      show (av.1 j != p.1 j) = false
      rw [hc, bitPivo_spec p.1 j hi]
      decide
    · have hc2 : av.1 j = false := by
        cases hc3 : av.1 j with
        | false => rfl
        | true => exact absurd hc3 hc
      rw [if_neg (by simp [hc2])]
      exact hc2

/-- ★ O bit morto segue morto na dobra: os outros pivôs não o tocam (a forma isolada), e o pivô de mesmo
bit não dispara (o bit já está morto). -/
theorem fold_preserva_morto {n : ℕ} (ps : Espectro2 n) (j : Fin n) :
    ∀ (av : (Fin n → Bool) × ZMod 2), av.1 j = false →
      (∀ p' ∈ ps, ∀ j', bitPivo p'.1 = some j' → j' ≠ j → p'.1 j = false) →
        (ps.foldl passoReducao av).1 j = false := by
  induction ps with
  | nil => intro av hdead _; exact hdead
  | cons p' ps' ih =>
    intro av hdead hnao
    rw [List.foldl_cons]
    apply ih
    · unfold passoReducao
      split
      · exact hdead
      · rename_i i hip
        by_cases hi : av.1 i = true
        · rw [if_pos hi]
          show (av.1 j != p'.1 j) = false
          rw [hdead]
          by_cases hij : i = j
          · subst hij
            rw [hi] at hdead
            simp at hdead
          · rw [hnao p' List.mem_cons_self i hip hij]
            decide
        · rw [if_neg (by simp [hi])]
          exact hdead
    · intro p'' hp'' j'' hj'' hj''j
      exact hnao p'' (List.mem_cons_of_mem p' hp'') j'' hj'' hj''j

/-- ★ **A REDUÇÃO LIMPA OS PIVÔS**: a máscara reduzida não tem bit de pivô de nenhuma equação da base. -/
theorem reduzirNova_sem_pivos {n : ℕ} (ps : Espectro2 n) (hiso : formaIsolada ps) :
    ∀ (av : (Fin n → Bool) × ZMod 2) (p₀ : (Fin n → Bool) × ZMod 2), p₀ ∈ ps →
      ∀ j : Fin n, bitPivo p₀.1 = some j →
        (ps.foldl passoReducao av).1 j = false := by
  induction ps with
  | nil => intro av p₀ h; simp at h
  | cons p ps ih =>
    intro av p₀ hp₀ j hj
    rw [List.foldl_cons]
    rw [List.mem_cons] at hp₀
    rcases hp₀ with hp₀ | hp₀
    · rw [hp₀] at hj
      apply fold_preserva_morto ps j (passoReducao av p) (passo_reducao_mata_o_pivo av p j hj)
      intro p' hp' j' hj' hj'j
      obtain ⟨h1, h2, h3⟩ := hiso
      exact h3 p' (List.mem_cons_of_mem p hp') p List.mem_cons_self j' j hj' hj hj'j
    · exact ih (formaIsolada_cons p ps hiso) (passoReducao av p) p₀ hp₀ j hj

/-- O xor cancela à esquerda. -/
theorem xorPonto_cancela_esq {n : ℕ} (a b c : Fin n → Bool)
    (h : xorPonto a c = xorPonto b c) : a = b := by
  ext i
  have h1 : (a i != c i) = (b i != c i) := congrFun h i
  cases ha : a i <;> cases hb : b i <;> cases hc : c i <;> simp [ha, hb, hc] at h1 ⊢

/-- Se `a ⊕ b = b`, então `a` é o ponto zero. -/
theorem xorPonto_eq_dir {n : ℕ} (a b : Fin n → Bool) (h : xorPonto a b = b) :
    a = fun _ => false := by
  ext i
  have h3 : (a i != b i) = b i := congrFun h i
  cases ha : a i <;> cases hb : b i <;> simp [ha, hb] at h3 ⊢

/-- O eixo zero não tem pivô. -/
theorem bitPivo_zero {n : ℕ} : bitPivo (fun _ => false : Fin n → Bool) = none := by
  unfold bitPivo
  split
  · rename_i hne
    obtain ⟨i, hi⟩ := hne
    simp [Finset.mem_filter] at hi
  · rfl

/-- ★ Numa base isolada e sem repetição, a inserção não repete entradas: a reduzida é nova e o ajuste das antigas é
injetivo. -/
theorem inserirEscalonado_nodup {n : ℕ} (ps : Espectro2 n) (e : (Fin n → Bool) × ZMod 2)
    (hiso : formaIsolada ps) (hnodup : ps.Nodup) :
    (inserirEscalonado e ps).Nodup := by
  obtain ⟨htem, hunico, hfora⟩ := hiso
  unfold inserirEscalonado
  dsimp only []
  split
  · exact hnodup
  · rename_i q hq
    set av := reduzirNova e.1 e.2 ps with hav
    have hsem : ∀ p₀ ∈ ps, ∀ j, bitPivo p₀.1 = some j → av.1 j = false := by
      intro p₀ hp₀ j hj
      rw [hav, reduzirNova_eq_fold]
      exact reduzirNova_sem_pivos ps ⟨htem, hunico, hfora⟩ (e.1, e.2) p₀ hp₀ j hj
    rw [List.nodup_cons]
    constructor
    · intro hmem
      rw [List.mem_map] at hmem
      obtain ⟨p, hp, hfp⟩ := hmem
      by_cases hpq : p.1 q = true
      · rw [if_pos hpq] at hfp
        have h1 : xorPonto p.1 av.1 = av.1 := congrArg Prod.fst hfp
        have h2 := xorPonto_eq_dir _ _ h1
        obtain ⟨j₀, hj₀⟩ := htem p hp
        rw [h2, bitPivo_zero] at hj₀
        simp at hj₀
      · rw [if_neg hpq] at hfp
        have h2 : p.1 = av.1 := congrArg Prod.fst hfp
        have hpivo : bitPivo p.1 = some q := by rw [h2]; exact hq
        exact hpq (bitPivo_spec p.1 q hpivo)
    · apply List.Nodup.map_on _ hnodup
      intro p₁ hp₁ p₂ hp₂ h
      by_cases h1 : p₁.1 q = true <;> by_cases h2 : p₂.1 q = true
      · rw [if_pos h1, if_pos h2] at h
        have hax : xorPonto p₁.1 av.1 = xorPonto p₂.1 av.1 := congrArg Prod.fst h
        have hval : p₁.2 + av.2 = p₂.2 + av.2 := congrArg Prod.snd h
        exact Prod.ext (xorPonto_cancela_esq _ _ _ hax) (add_right_cancel hval)
      · rw [if_pos h1, if_neg h2] at h
        have hax : xorPonto p₁.1 av.1 = p₂.1 := congrArg Prod.fst h
        obtain ⟨j₂, hj₂⟩ := htem p₂ hp₂
        have hbit2 : p₂.1 j₂ = true := bitPivo_spec _ _ hj₂
        have hjq : j₂ ≠ q := fun hjj => h2 (hjj ▸ hbit2)
        have hsem2 := hsem p₂ hp₂ j₂ hj₂
        have h3 : (p₁.1 j₂ != av.1 j₂) = true := by
          have h4 := congrFun hax j₂
          rw [hbit2] at h4
          exact h4
        rw [hsem2] at h3
        have hp1j2 : p₁.1 j₂ = true := by
          cases hp₁j₂ : p₁.1 j₂ with
          | false => simp [hp₁j₂] at h3
          | true => rfl
        obtain ⟨j₁, hj₁⟩ := htem p₁ hp₁
        have hj₁j₂ : j₁ ≠ j₂ := by
          intro hjj
          rw [hjj] at hj₁
          have h12 := hunico p₁ hp₁ p₂ hp₂ j₂ hj₁ hj₂
          rw [h12] at h1
          exact h2 h1
        have hzero := hfora p₁ hp₁ p₂ hp₂ j₁ j₂ hj₁ hj₂ hj₁j₂
        rw [hp1j2] at hzero
        exact Bool.noConfusion hzero
      · rw [if_neg h1, if_pos h2] at h
        have hax : p₁.1 = xorPonto p₂.1 av.1 := congrArg Prod.fst h
        obtain ⟨j₁, hj₁⟩ := htem p₁ hp₁
        have hbit1 : p₁.1 j₁ = true := bitPivo_spec _ _ hj₁
        have hjq : j₁ ≠ q := fun hjj => h1 (hjj ▸ hbit1)
        have hsem1 := hsem p₁ hp₁ j₁ hj₁
        have h3 : (p₂.1 j₁ != av.1 j₁) = true := by
          have h4 := congrFun hax j₁
          rw [hbit1] at h4
          exact h4.symm
        rw [hsem1] at h3
        have hp2j1 : p₂.1 j₁ = true := by
          cases hp₂j₁ : p₂.1 j₁ with
          | false => simp [hp₂j₁] at h3
          | true => rfl
        obtain ⟨j₂, hj₂⟩ := htem p₂ hp₂
        have hj₂j₁ : j₂ ≠ j₁ := by
          intro hjj
          rw [hjj] at hj₂
          have h21 := hunico p₂ hp₂ p₁ hp₁ j₁ hj₂ hj₁
          rw [h21] at h2
          exact h1 h2
        have hzero := hfora p₂ hp₂ p₁ hp₁ j₂ j₁ hj₂ hj₁ hj₂j₁
        rw [hp2j1] at hzero
        exact Bool.noConfusion hzero
      · rw [if_neg h1, if_neg h2] at h
        exact h

/-- ★ O escalonado não repete entradas. -/
theorem oEscalonado_nodup {n : ℕ} (S : Espectro2 n) : (oEscalonado S).Nodup := by
  unfold oEscalonado
  suffices h : ∀ (S' : Espectro2 n) (acc : Espectro2 n),
      formaIsolada acc → acc.Nodup →
      (List.foldl (fun ps e => inserirEscalonado e ps) acc S').Nodup by
    exact h S [] formaIsolada_nil (by simp)
  intro S'
  induction S' with
  | nil => intro acc _ h2; simpa using h2
  | cons e S'' ih =>
    intro acc h1 h2
    rw [List.foldl_cons]
    exact ih _ (inserirEscalonado_preserva_isolada acc e h1) (inserirEscalonado_nodup acc e h1 h2)

/-- ★★ **O POSTO NÃO PASSA DE N**: o escalonado tem no máximo n pivôs. -/
theorem oEscalonado_length_le {n : ℕ} (S : Espectro2 n) :
    (oEscalonado S).length ≤ n := by
  classical
  have hiso := oEscalonado_isolada S
  have hnodup := oEscalonado_nodup S
  obtain ⟨htem, hunico, _⟩ := hiso
  rcases Nat.eq_zero_or_pos n with hn | hn
  · subst hn
    have hvazio : oEscalonado S = ([] : Espectro2 0) := by
      rw [List.eq_nil_iff_forall_not_mem]
      intro p hp
      obtain ⟨i, _⟩ := htem p hp
      exact i.elim0
    rw [hvazio]
    simp
  · haveI : NeZero n := NeZero.of_pos hn
    rw [← List.toFinset_card_of_nodup hnodup]
    have hinj : Set.InjOn (fun p : (Fin n → Bool) × ZMod 2 => (bitPivo p.1).getD 0)
        (oEscalonado S).toFinset := by
      intro p₁ hp₁ p₂ hp₂ h
      simp only [Finset.mem_coe, List.mem_toFinset] at hp₁ hp₂
      dsimp only at h
      obtain ⟨i₁, hi₁⟩ := htem p₁ hp₁
      obtain ⟨i₂, hi₂⟩ := htem p₂ hp₂
      have hf₁ : (bitPivo p₁.1).getD 0 = i₁ := by rw [hi₁]; rfl
      have hf₂ : (bitPivo p₂.1).getD 0 = i₂ := by rw [hi₂]; rfl
      rw [hf₁, hf₂] at h
      subst h
      exact hunico p₁ hp₁ p₂ hp₂ i₁ hi₁ hi₂
    have hle := Finset.card_le_card_of_injOn _ (fun p _ => Finset.mem_univ _) hinj
    rw [Finset.card_univ, Fintype.card_fin] at hle
    exact hle

/-- O escalonado do acrescentado é a inserção no escalonado. (Auxiliar do crescimento.) -/
theorem oEscalonado_snoc {n : ℕ} (S : Espectro2 n) (e : (Fin n → Bool) × ZMod 2) :
    oEscalonado (S ++ [e]) = inserirEscalonado e (oEscalonado S) := by
  unfold oEscalonado
  rw [List.foldl_append, List.foldl_cons, List.foldl_nil]

/-- [DEF] A inserção por parâmetros separados (a mesma operação, amigável às provas). -/
def inserePar {n : ℕ} (a : Fin n → Bool) (v : ZMod 2) (ps : Espectro2 n) : Espectro2 n :=
  let av := reduzirNova a v ps
  match bitPivo av.1 with
  | none => ps
  | some q => av :: ps.map (fun p => if p.1 q = true then (xorPonto p.1 av.1, p.2 + av.2) else p)

/-- A inserção por par é a inserção do solver. -/
theorem inserePar_eq {n : ℕ} (a : Fin n → Bool) (v : ZMod 2) (ps : Espectro2 n) :
    inserePar a v ps = inserirEscalonado (a, v) ps := rfl

/-- O comprimento da inserção com pivô cresce um. -/
theorem inserePar_length {n : ℕ} (a : Fin n → Bool) (v : ZMod 2) (ps : Espectro2 n)
    (h : (bitPivo (reduzirNova a v ps).1).isSome = true) :
    (inserePar a v ps).length = ps.length + 1 := by
  unfold inserePar
  dsimp only []
  split
  · rename_i hnone
    rw [hnone] at h
    simp at h
  · rw [List.length_cons, List.length_map]

/-- ★ Registrar uma literal de retorno útil (`retornoUtil`) cresce o escalonado de exatamente uma entrada. -/
theorem oEscalonado_cresce_um_com_retorno {n : ℕ} (S : Espectro2 n) (l : Literal n)
    (h : retornoUtil S l = true) :
    (oEscalonado (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])).length
      = (oEscalonado S).length + 1 := by
  rw [oEscalonado_snoc]
  have h' : (bitPivo (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).1).isSome
      = true := h
  rw [← inserePar_eq]
  exact inserePar_length _ _ _ h'

end TGLExt.OPostoDoEscalonado

