import TGLExt.ACompletude

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A checagem do span é exata

## O que é

A checagem computável do span (`ACompletude.eixoNoSpanB`) procura, entre as sub-listas dos eixos inscritos, uma cujo
ou-exclusivo seja o eixo dado. O span (`AparedeDoSpan.eixoNoSpan`) admite qualquer sub-conjunto, em qualquer ordem e com
repetição. Esta pedra mostra que as duas coisas coincidem.

## O que fica provado [REAL]

1. `xorPonto_zero_esq`, `xorPonto_com`, `xorPonto_assoc`, `xorPonto_troca`, `xorPonto_self`: a aritmética do
   ou-exclusivo ponto a ponto (o eixo apagado é neutro; dobrar com o próprio eixo apaga). `xorComb_cons`: o ou-exclusivo
   de uma lista com cabeça.
2. ★ `xorComb_perm`: listas permutadas têm o mesmo ou-exclusivo.
3. ★ `existe_sublista_mesmo_xor`: todo sub-conjunto dos eixos, em qualquer ordem e com repetição, tem uma sub-lista dos
   eixos inscritos, na ordem deles, com o mesmo ou-exclusivo.
4. ★★ `eixoNoSpanB_completa` e `eixoNoSpanB_iff`: a checagem computável dispara se e só se o eixo está no span.

## O que NÃO se afirma

* O custo da checagem (ela percorre as sub-listas) não é contado aqui.
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AInvarianciaDaLeitura

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude

variable {n : ℕ}

/-! ## A aritmética do ponto -/

/-- O ponto zero é neutro à esquerda no xor. -/
theorem xorPonto_zero_esq {n : ℕ} (x : Fin n → Bool) : xorPonto (fun _ => false) x = x := by
  ext i
  show (false != x i) = x i
  cases hx : x i <;> decide

/-- O xor comuta. -/
theorem xorPonto_com {n : ℕ} (x y : Fin n → Bool) : xorPonto x y = xorPonto y x := by
  ext i
  show (x i != y i) = (y i != x i)
  cases hx : x i <;> cases hy : y i <;> decide

/-- O xor associa. -/
theorem xorPonto_assoc {n : ℕ} (x y z : Fin n → Bool) :
    xorPonto (xorPonto x y) z = xorPonto x (xorPonto y z) := by
  ext i
  show ((x i != y i) != z i) = (x i != (y i != z i))
  cases hx : x i <;> cases hy : y i <;> cases hz : z i <;> decide

/-- A troca do meio: ler y antes de x é ler x antes de y. (O passo do `swap` da permutação.) -/
theorem xorPonto_troca {n : ℕ} (x y z : Fin n → Bool) :
    xorPonto x (xorPonto y z) = xorPonto y (xorPonto x z) := by
  ext i
  show (x i != (y i != z i)) = (y i != (x i != z i))
  cases hx : x i <;> cases hy : y i <;> cases hz : z i <;> decide

/-- Dobrar com o próprio eixo apaga: a poda. -/
theorem xorPonto_self {n : ℕ} (x : Fin n → Bool) : xorPonto x x = fun _ => false := by
  ext i
  show (x i != x i) = false
  simp

/-- O xor da lista com cabeça é o xor da cabeça com o xor da cauda. -/
theorem xorComb_cons {n : ℕ} (x : Fin n → Bool) (l : List (Fin n → Bool)) :
    xorComb (x :: l) = xorPonto x (xorComb l) := by
  show (x :: l).foldl xorPonto (fun _ => false)
    = xorPonto x (l.foldl xorPonto (fun _ => false))
  rw [List.foldl_cons, xorComb_de_iniciado, xorPonto_zero_esq]

/-- ★ **A LEITURA NÃO DEPENDE DA ORDEM**: listas permutadas têm o mesmo xor. -/
theorem xorComb_perm {n : ℕ} {l₁ l₂ : List (Fin n → Bool)} (h : List.Perm l₁ l₂) :
    xorComb l₁ = xorComb l₂ := by
  induction h with
  | nil => rfl
  | cons x _ ih => rw [xorComb_cons, xorComb_cons, ih]
  | swap x y l =>
    rw [xorComb_cons, xorComb_cons, xorComb_cons, xorComb_cons]
    exact xorPonto_troca _ _ _
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

/-- ★ **A SUB-LISTA EXISTE**: todo subconjunto dos eixos — em qualquer ordem, com repetições — tem uma
sub-lista da fita (na ordem da fita) com o mesmo xor. A prova é a poda por ocorrência: na leitura, para
cada eixo, escolhe-se uma ocorrência na fita; se ela ainda não foi usada, insere-se (o xor ganha o eixo);
se já foi usada, retira-se (o xor perde o eixo) — e nos dois casos o xor flipa exatamente pelo eixo. -/
theorem existe_sublista_mesmo_xor {n : ℕ} (axes sub : List (Fin n → Bool)) :
    sub ⊆ axes → ∃ sub' : List (Fin n → Bool), sub'.Sublist axes ∧ xorComb sub' = xorComb sub := by
  induction sub with
  | nil => intro _; exact ⟨[], List.nil_sublist axes, rfl⟩
  | cons x xs ih =>
    intro hsub
    have hxs : xs ⊆ axes := fun y hy => hsub (List.mem_cons_of_mem x hy)
    obtain ⟨t, ht, htx⟩ := ih hxs
    obtain ⟨pfx, sfx, haxes⟩ := List.mem_iff_append.mp (hsub List.mem_cons_self)
    rw [haxes] at ht
    rw [List.sublist_append_iff] at ht
    obtain ⟨t₁, t₂, ht_eq, ht₁, ht₂⟩ := ht
    rw [← List.singleton_append] at ht₂
    rw [List.sublist_append_iff] at ht₂
    obtain ⟨tm, ts, ht₂eq, htm, hts⟩ := ht₂
    rw [List.sublist_singleton] at htm
    have htxs : xorComb t = xorPonto (xorComb t₁) (xorPonto (xorComb tm) (xorComb ts)) := by
      rw [ht_eq, ht₂eq, xorComb_append, xorComb_append]
    rcases htm with htm0 | htmx
    · -- a ocorrência está livre: insere o eixo
      refine ⟨t₁ ++ [x] ++ ts, ?_, ?_⟩
      · rw [haxes, ← List.singleton_append (l := sfx), ← List.append_assoc]
        exact List.Sublist.append (List.Sublist.append ht₁ (List.Sublist.refl [x])) hts
      · rw [xorComb_append, xorComb_append, xorComb_un, xorComb_cons, ← htx, htxs, htm0,
          xorComb_nil, xorPonto_zero_esq, xorPonto_assoc (xorComb t₁) x (xorComb ts)]
        exact xorPonto_troca _ _ _
    · -- a ocorrência já foi usada: retira-a — o eixo dobrado se anula
      refine ⟨t₁ ++ ts, ?_, ?_⟩
      · rw [haxes]
        exact (List.Sublist.append ht₁ hts).trans
          (List.Sublist.append (List.Sublist.refl pfx) (List.sublist_cons_self x sfx))
      · rw [xorComb_append, xorComb_cons, ← htx, htxs, htmx, xorComb_un,
          ← xorPonto_assoc x (xorComb t₁) (xorPonto x (xorComb ts)), xorPonto_com x (xorComb t₁),
          xorPonto_assoc (xorComb t₁) x (xorPonto x (xorComb ts)),
          ← xorPonto_assoc x x (xorComb ts), xorPonto_self, xorPonto_zero_esq]

/-- ★★ **A CHECAGEM DO SPAN É COMPLETA**: se o eixo está no span dos eixos (combinação linear por xor,
em qualquer ordem), a checagem computável por sub-listas dispara. -/
theorem eixoNoSpanB_completa {n : ℕ} (axes : List (Fin n → Bool)) (a : Fin n → Bool)
    (h : eixoNoSpan axes a) : eixoNoSpanB axes a = true := by
  obtain ⟨sub, hsub, hx⟩ := h
  obtain ⟨sub', hsub', hx'⟩ := existe_sublista_mesmo_xor axes sub hsub
  rw [eixoNoSpanB, List.any_eq_true]
  exact ⟨sub', List.mem_sublists.mpr hsub', by rw [decide_eq_true_eq]; exact hx'.trans hx⟩

/-- ★★ **A CHECAGEM DO SPAN É EXATA**: a checagem computável e o span coincidem. O silêncio do processo
decide corretamente a transversalidade — nunca cala um eixo transversal, nunca deixa passar um eixo já
inscrito. -/
theorem eixoNoSpanB_iff {n : ℕ} (axes : List (Fin n → Bool)) (a : Fin n → Bool) :
    eixoNoSpanB axes a = true ↔ eixoNoSpan axes a :=
  ⟨eixoNoSpanB_sadia axes a, eixoNoSpanB_completa axes a⟩

end TGLExt.AInvarianciaDaLeitura

