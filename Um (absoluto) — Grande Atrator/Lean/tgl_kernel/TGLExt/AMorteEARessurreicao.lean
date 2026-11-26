import TGLExt.ADescidaComEspectro
import Mathlib.Data.ZMod.Basic

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A morte e a ressurreição: a homogeneidade espectral e a partição do resíduo

## O que é

A leitura espectral de um ponto do cubo `Fin n → Bool` é a lista das suas paridades em `ZMod 2` num conjunto de
eixos. A paridade é um homomorfismo do cubo (o xor); por isso todas as fibras não vazias da leitura têm o mesmo
tamanho; e, quando uma paridade nova toma os dois valores dentro de uma fibra, os pontos da fibra em que ela vale 1
são exatamente metade da fibra (o teorema enuncia só o lado do valor 1).

Leitura do autor [ONTO]: o bit pago na morte é o próprio acoplamento não mínimo; o ponto do outro lado não é ponto,
é o espectro da projeção sendo lida; o ângulo lê o ponto de morte e o projeta no lado oposto.

## O que fica provado [REAL]

1. `paridade2_add`: a paridade é aditiva sob o xor do cubo, `paridade2 (xorPonto x d) a = paridade2 x a +
   paridade2 d a`.
2. `leitura2_xor`: a leitura do ponto traduzido é a soma, termo a termo, das leituras. `zipAdd_cancel` e
   `zipAdd_cancel_esq`: o cancelamento em `ZMod 2` de listas do mesmo comprimento.
3. `fibras_iguais`: dois valores atingidos pela leitura do mesmo conjunto de eixos têm fibras do mesmo cardinal
   (a translação pelo xor é a bijeção).
4. `zmod2_exaustivo`: todo elemento de `ZMod 2` é 0 ou 1.
5. `a_morte_parte_o_residuo`: se a paridade nova `a` toma os valores 0 e 1 dentro da fibra de `vals`, então
   `2 * card (fibra de vals ++ [1]) = card (fibra de vals)`. `a_morte_parte_o_residuo_espectral`: o mesmo na forma
   espectral, para a equação nova `(a, 1)`. O enunciado é para o valor 1; o lado do valor 0 segue pela mesma conta,
   mas não é enunciado.
6. As definições `eixoDaMorteDe`, `aMorte`, `DinamicaDaMorte` e o contrato `GatesDaMorte`, com os campos `fiel`
   (G1, a ressurreição fiel) e `divide` (G2, a leitura divide), que são hipóteses nomeadas, não teoremas.

## O que NÃO se afirma

* O contrato `GatesDaMorte`, como está escrito, não é cumprido por nenhuma dinâmica:
  `TGLExt.AVacuidadeDosContratos.gatesDaMorte_insatisfazivel` [REAL]. O campo `divide` é exigido para todo
  espectro, inclusive um contraditório (a equação de eixo vazio com valor 1), que nenhum ponto serve
  (`TGLExt.AVacuidadeDosContratos.ninguem_serve2`). Um teorema que tome
  `GatesDaMorte` como hipótese vale no vazio: não diz nada sobre fórmula alguma.
* A equação `aMorte φ w` pode ter valor 0; o teorema da partição está enunciado para a equação nova de valor 1.
* Esta pedra não conta o número de mortes nem dá um limite para ele: dá a razão da partição de uma fibra quando a
  paridade nova toma os dois valores.
* Nada aqui diz que algum procedimento ou a máquina de Turing decida SAT, em tempo algum. Decidir SAT sem
  testemunha, em tempo polinomial, na máquina de fita segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AMorteEARessurreicao

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro

variable {n : ℕ}

/-! ## A paridade como homomorfismo -/

/-- [INPUT] O xor de dois pontos (a translação do cubo). -/
def xorPonto {n : ℕ} (x y : Fin n → Bool) : Fin n → Bool := fun i => (x i != y i)

/-- [REAL] O xor é involução no segundo argumento. (Auxiliar da bijeção das fibras.) -/
theorem xorPonto_invol {n : ℕ} (x d : Fin n → Bool) : xorPonto (xorPonto x d) d = x := by
  ext i
  show ((x i != d i) != d i) = x i
  cases hx : x i <;> cases hd : d i <;> rfl

/-- [INPUT] A paridade em `ZMod 2`: a soma dos bits onde o eixo está aceso. -/
def paridade2 {n : ℕ} (x : Fin n → Bool) (a : Fin n → Bool) : ZMod 2 :=
  ∑ i : Fin n, (if a i && x i then (1 : ZMod 2) else 0)

/-- [REAL] A paridade é aditiva sob o xor: a leitura é um homomorfismo do cubo. -/
theorem paridade2_add {n : ℕ} (x d a : Fin n → Bool) :
    paridade2 (xorPonto x d) a = paridade2 x a + paridade2 d a := by
  unfold paridade2 xorPonto
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  cases a i <;> cases x i <;> cases d i <;> decide

/-- [INPUT] A leitura espectral: as paridades do ponto nos eixos. -/
def leitura2 {n : ℕ} (axes : List (Fin n → Bool)) (x : Fin n → Bool) : List (ZMod 2) :=
  axes.map fun a => paridade2 x a

/-- [REAL] A leitura do ponto traduzido é a soma das leituras. (Auxiliar da homogeneidade.) -/
theorem leitura2_xor {n : ℕ} (axes : List (Fin n → Bool)) (x d : Fin n → Bool) :
    leitura2 axes (xorPonto x d)
      = List.zipWith (· + ·) (leitura2 axes x) (leitura2 axes d) := by
  induction axes with
  | nil => rfl
  | cons a ax ih => simp [leitura2, List.map_cons, List.zipWith_cons_cons, paridade2_add, ih]

/-- [REAL] O cancelamento, à esquerda: `v₁ ⊞ (v₁ ⊞ v₂) = v₂`, com as duas listas do mesmo comprimento.
(Auxiliar da homogeneidade.) -/
theorem zipAdd_cancel : ∀ (v₁ v₂ : List (ZMod 2)), v₁.length = v₂.length →
    List.zipWith (· + ·) v₁ (List.zipWith (· + ·) v₁ v₂) = v₂ := by
  intro v₁
  induction v₁ with
  | nil =>
    intro v₂ hl
    cases v₂ with
    | nil => rfl
    | cons c cs => simp at hl
  | cons a ax ih =>
    intro v₂ hl
    cases v₂ with
    | nil => simp at hl
    | cons b bx =>
      have hhead : a + (a + b) = b := by
        rw [← add_assoc, CharTwo.add_self_eq_zero, zero_add]
      have hll : ax.length = bx.length := by simpa using hl
      simp [List.zipWith_cons_cons, hhead, ih bx hll]

/-- [REAL] O cancelamento, cruzado: `v₂ ⊞ (v₁ ⊞ v₂) = v₁`, com as duas listas do mesmo comprimento.
(Auxiliar da homogeneidade.) -/
theorem zipAdd_cancel_esq : ∀ (v₁ v₂ : List (ZMod 2)), v₁.length = v₂.length →
    List.zipWith (· + ·) v₂ (List.zipWith (· + ·) v₁ v₂) = v₁ := by
  intro v₁
  induction v₁ with
  | nil =>
    intro v₂ hl
    cases v₂ with
    | nil => rfl
    | cons c cs => simp at hl
  | cons a ax ih =>
    intro v₂ hl
    cases v₂ with
    | nil => simp at hl
    | cons b bx =>
      have hhead : b + (a + b) = a := by
        rw [add_comm b (a + b), add_assoc, CharTwo.add_self_eq_zero, add_zero]
      have hll : ax.length = bx.length := by simpa using hl
      simp [List.zipWith_cons_cons, hhead, ih bx hll]

/-- [REAL] **A homogeneidade espectral**: dois valores atingidos pela leitura do mesmo conjunto de eixos têm
fibras do mesmo cardinal; a translação pelo xor é a bijeção. Leitura [ONTO]: o espectro não privilegia sombra. -/
theorem fibras_iguais {n : ℕ} (axes : List (Fin n → Bool)) (v₁ v₂ : List (ZMod 2))
    (x₁ : Fin n → Bool) (h1 : leitura2 axes x₁ = v₁)
    (x₂ : Fin n → Bool) (h2 : leitura2 axes x₂ = v₂) :
    (Finset.univ.filter fun x => leitura2 axes x = v₁).card
      = (Finset.univ.filter fun x => leitura2 axes x = v₂).card := by
  have hlen : v₁.length = v₂.length := by
    rw [← h1, ← h2]
    simp [leitura2, List.length_map]
  simp only [← Fintype.card_subtype]
  apply Fintype.card_congr
  refine ⟨fun x => ⟨xorPonto x.1 (xorPonto x₁ x₂), ?_⟩, fun y => ⟨xorPonto y.1 (xorPonto x₁ x₂), ?_⟩,
    ?_, ?_⟩
  · have hd : leitura2 axes (xorPonto x₁ x₂) = List.zipWith (· + ·) v₁ v₂ := by
      rw [leitura2_xor, h1, h2]
    rw [leitura2_xor, x.2, hd, zipAdd_cancel v₁ v₂ hlen]
  · have hd : leitura2 axes (xorPonto x₁ x₂) = List.zipWith (· + ·) v₁ v₂ := by
      rw [leitura2_xor, h1, h2]
    rw [leitura2_xor, y.2, hd, zipAdd_cancel_esq v₁ v₂ hlen]
  · intro x; exact Subtype.ext (xorPonto_invol x.1 (xorPonto x₁ x₂))
  · intro y; exact Subtype.ext (xorPonto_invol y.1 (xorPonto x₁ x₂))

/-! ## A partição do resíduo -/

/-- [REAL] A leitura com um eixo a mais é a leitura com a paridade nova no fim. (Auxiliar da partição.) -/
theorem leitura2_snoc {n : ℕ} (axes : List (Fin n → Bool)) (a : Fin n → Bool) (x : Fin n → Bool) :
    leitura2 (axes ++ [a]) x = leitura2 axes x ++ [paridade2 x a] := by
  simp [leitura2, List.map_append]

/-- [REAL] A igualdade de cauda única: `L ++ [b] = M ++ [c]` dá `L = M` e `b = c`. (Auxiliar da partição.) -/
theorem append_snod_inj {α : Type} {L M : List α} {b c : α} (h : L ++ [b] = M ++ [c]) :
    L = M ∧ b = c := by
  have hr := congrArg List.reverse h
  simp only [List.reverse_append, List.reverse_singleton, List.singleton_append] at hr
  rw [List.cons.injEq] at hr
  exact ⟨List.reverse_injective hr.2, hr.1⟩

/-- [REAL] Todo elemento de `ZMod 2` é 0 ou 1. (Auxiliar da partição.) -/
theorem zmod2_exaustivo (z : ZMod 2) : z = 0 ∨ z = 1 := by
  obtain ⟨v, hv⟩ := z
  interval_cases v
  · exact Or.inl rfl
  · exact Or.inr rfl

/-- [REAL] **A partição do resíduo**: se a paridade nova `a` toma os valores 0 e 1 dentro da fibra de `vals`, a
fibra de `vals ++ [1]` tem exatamente metade dos pontos da fibra de `vals` (`2 * card = card`). O enunciado é para o
valor 1; o lado do valor 0 segue pela mesma conta, mas não é enunciado. Leitura [ONTO]: uma leitura que divide
custa um bit (o fator 2 da conta, isto é ln 2 nat). -/
theorem a_morte_parte_o_residuo {n : ℕ} (axes : List (Fin n → Bool)) (vals : List (ZMod 2))
    (a : Fin n → Bool)
    (h0 : ∃ x : Fin n → Bool, leitura2 axes x = vals ∧ paridade2 x a = 0)
    (h1 : ∃ x : Fin n → Bool, leitura2 axes x = vals ∧ paridade2 x a = 1) :
    2 * (Finset.univ.filter fun x => leitura2 (axes ++ [a]) x = vals ++ [1]).card
      = (Finset.univ.filter fun x => leitura2 axes x = vals).card := by
  obtain ⟨x0, hx0, hp0⟩ := h0
  obtain ⟨x1, hx1, hp1⟩ := h1
  have hfib := fibras_iguais (axes ++ [a]) (vals ++ [1]) (vals ++ [0]) x1
    (by rw [leitura2_snoc, hx1, hp1]) x0 (by rw [leitura2_snoc, hx0, hp0])
  have hpart : (Finset.univ.filter fun x => leitura2 axes x = vals)
      = (Finset.univ.filter fun x => leitura2 (axes ++ [a]) x = vals ++ [0])
        ∪ (Finset.univ.filter fun x => leitura2 (axes ++ [a]) x = vals ++ [1]) := by
    ext x
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union]
    rw [leitura2_snoc]
    constructor
    · intro hx
      cases zmod2_exaustivo (paridade2 x a) with
      | inl hz => exact Or.inl (by rw [hx, hz])
      | inr hz => exact Or.inr (by rw [hx, hz])
    · intro h
      cases h with
      | inl h0' => exact (append_snod_inj h0').1
      | inr h1' => exact (append_snod_inj h1').1
  have hdisj : Disjoint (Finset.univ.filter fun x => leitura2 (axes ++ [a]) x = vals ++ [0])
      (Finset.univ.filter fun x => leitura2 (axes ++ [a]) x = vals ++ [1]) := by
    rw [Finset.disjoint_left]
    intro x hx0' hx1'
    rw [Finset.mem_filter] at hx0' hx1'
    have h := hx0'.2.symm.trans hx1'.2
    have h01 : (0 : ZMod 2) = 1 := (append_snod_inj h).2
    exact (by decide : (0 : ZMod 2) ≠ 1) h01
  rw [hpart, Finset.card_union_of_disjoint hdisj]
  omega

/-! ## A forma espectral (o espectro como lista de equações) -/

/-- [INPUT] O espectro com valores em `ZMod 2`: a lista das equações lidas (eixo, valor). -/
abbrev Espectro2 (n : ℕ) := List ((Fin n → Bool) × ZMod 2)

/-- [INPUT] O ponto serve o espectro: cada equação lida vale nele. -/
def serveEspectro2 {n : ℕ} (x : Fin n → Bool) (S : Espectro2 n) : Bool :=
  S.all fun e => paridade2 x e.1 == e.2

/-- [REAL] Servir o espectro é a leitura dos eixos bater os valores. (Auxiliar da forma espectral.) -/
theorem serveEspectro2_leitura2 {n : ℕ} (x : Fin n → Bool) (S : Espectro2 n) :
    serveEspectro2 x S = true ↔ leitura2 (S.map Prod.fst) x = S.map Prod.snd := by
  induction S with
  | nil => simp [serveEspectro2, leitura2]
  | cons e S' ih =>
    obtain ⟨a, v⟩ := e
    have hcons : leitura2 (a :: S'.map Prod.fst) x
        = paridade2 x a :: leitura2 (S'.map Prod.fst) x := by
      simp [leitura2]
    have ih' : (S'.all fun e => paridade2 x e.1 == e.2) = true ↔
        leitura2 (S'.map Prod.fst) x = S'.map Prod.snd := ih
    simp only [serveEspectro2, List.all_cons, List.map_cons]
    rw [hcons, Bool.and_eq_true, beq_iff_eq, ih']
    constructor
    · intro h
      rw [h.1, h.2]
    · intro h
      rw [List.cons.injEq] at h
      exact ⟨h.1, h.2⟩

/-- [REAL] A partição do resíduo, na forma espectral: se a paridade em `a` toma os valores 0 e 1 entre os pontos
que servem `S`, os pontos que servem `S ++ [(a, 1)]` são exatamente metade dos que servem `S`. O enunciado é para a
equação nova de valor 1. -/
theorem a_morte_parte_o_residuo_espectral {n : ℕ} (S : Espectro2 n) (a : Fin n → Bool)
    (h0 : ∃ x : Fin n → Bool, serveEspectro2 x S = true ∧ paridade2 x a = 0)
    (h1 : ∃ x : Fin n → Bool, serveEspectro2 x S = true ∧ paridade2 x a = 1) :
    2 * (Finset.univ.filter fun x => serveEspectro2 x (S ++ [(a, 1)]) = true).card
      = (Finset.univ.filter fun x => serveEspectro2 x S = true).card := by
  obtain ⟨x0, hx0, hp0⟩ := h0
  obtain ⟨x1, hx1, hp1⟩ := h1
  have hbase := a_morte_parte_o_residuo (S.map Prod.fst) (S.map Prod.snd) a
    ⟨x0, (serveEspectro2_leitura2 x0 S).mp hx0, hp0⟩ ⟨x1, (serveEspectro2_leitura2 x1 S).mp hx1, hp1⟩
  have hb0 : (Finset.univ.filter fun x => leitura2 (S.map Prod.fst) x = S.map Prod.snd)
      = Finset.univ.filter fun x => serveEspectro2 x S = true := by
    ext x
    simp [Finset.mem_filter, serveEspectro2_leitura2]
  have hb1 : (Finset.univ.filter fun x => leitura2 (S.map Prod.fst ++ [a]) x = S.map Prod.snd ++ [1])
      = Finset.univ.filter fun x => serveEspectro2 x (S ++ [(a, 1)]) = true := by
    ext x
    simp [Finset.mem_filter, serveEspectro2_leitura2, List.map_append]
  rw [hb0, hb1] at hbase
  exact hbase

/-! ## A morte e o contrato -/

/-- [INPUT] O eixo da morte: a máscara das variáveis da primeira cláusula sem par no ponto `w` (`primeiraSemPar φ w`);
quando não há cláusula sem par, a máscara vazia (todo eixo apagado). Nada exige que `w` seja um poço. Leitura
[ONTO]: `w` lido como o poço; o eixo fixado pelo ponto morto, lido e não escolhido. -/
def eixoDaMorteDe {n : ℕ} (φ : CNF n) (w : Fin n → Bool) : Fin n → Bool :=
  match primeiraSemPar φ w with
  | none => fun _ => false
  | some C => fun i => decide (i ∈ sombraDe C)

/-- [INPUT] A equação que a morte lê no ponto `w`: o eixo da morte e a paridade de `w` nele. O valor pode ser 0
ou 1. Leitura [ONTO]: `w` lido como o poço; o bit pago. -/
def aMorte {n : ℕ} (φ : CNF n) (w : Fin n → Bool) : (Fin n → Bool) × ZMod 2 :=
  (eixoDaMorteDe φ w, paridade2 w (eixoDaMorteDe φ w))

/-- [INPUT] A dinâmica da morte: uma ressurreição, que devolve um ponto a partir da fórmula e do espectro
acumulado. Leitura [ONTO]: o ponto do outro lado, lido do espectro. -/
structure DinamicaDaMorte (n : ℕ) where
  ressurreicao : CNF n → Espectro2 n → (Fin n → Bool)

/-- [INPUT] **O contrato `GatesDaMorte`**, hipótese nomeada, não teorema.

- **G1 (`fiel`, a ressurreição fiel):** para toda fórmula `φ`, todo espectro `S` e todo ponto `w`, o ponto
  ressurgido serve `S ++ [aMorte φ w]`.
- **G2 (`divide`, a leitura divide):** para toda fórmula `φ`, todo espectro `S` e todo ponto `w`, entre os pontos
  que servem `S` a paridade no eixo da morte toma o valor 0 e o valor 1.

Como está escrito, nenhuma dinâmica cumpre este contrato:
`TGLExt.AVacuidadeDosContratos.gatesDaMorte_insatisfazivel` [REAL]. O campo `divide` pede um ponto que sirva um
espectro contraditório (a equação de eixo vazio com valor 1), e nenhum ponto o serve. Um teorema que tome
`GatesDaMorte` como hipótese vale no vazio e não diz nada sobre fórmula alguma. -/
structure GatesDaMorte {n : ℕ} (d : DinamicaDaMorte n) : Prop where
  fiel : ∀ (φ : CNF n) (S : Espectro2 n) (w : Fin n → Bool),
    serveEspectro2 (d.ressurreicao φ S) (S ++ [aMorte φ w]) = true
  divide : ∀ (φ : CNF n) (S : Espectro2 n) (w : Fin n → Bool),
    (∃ x : Fin n → Bool, serveEspectro2 x S = true ∧ paridade2 x (eixoDaMorteDe φ w) = 0)
      ∧ (∃ x : Fin n → Bool, serveEspectro2 x S = true ∧ paridade2 x (eixoDaMorteDe φ w) = 1)

end TGLExt.AMorteEARessurreicao

