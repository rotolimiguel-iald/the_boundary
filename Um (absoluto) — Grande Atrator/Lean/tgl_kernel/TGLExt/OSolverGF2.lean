import TGLExt.ADualidade

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# O solver GF(2): a eliminação pretendida e o ponto construído

[DEF] Este módulo define, sobre o cubo `Fin n → Bool` e com equações de paridade em `ZMod 2`, um
procedimento pretendido como eliminação de Gauss–Jordan (`oEscalonado`, uma dobra de `inserirEscalonado`
sobre o espectro) e o ponto que se lê da lista de pivôs que ele devolve (`pontoConstruido`).
Cada equação inserida é reduzida contra os pivôs, na ordem da lista: para cada pivô, se o bit de pivô dele
está aceso no valor acumulado até ali, soma-se o pivô (xor das máscaras, soma dos valores). Se sobra máscara
não nula, ela vira pivô e é somada a cada pivô antigo que tem o bit de pivô dela aceso.
Que a redução apague todos os bits de pivô da equação nova não se prova: isso vale quando os pivôs já estão
reduzidos entre si (terceira cláusula de `formaIsolada`, em `TGLExt.OSolverGF2_Parte3`); sem isso, o xor
com um pivô posterior pode reacender um bit apagado antes.
O ponto construído põe, no bit `j`, o valor da PRIMEIRA equação da lista cujo bit de pivô é `j` (`find?`),
e falso quando não há tal equação. Isso coincide com «o valor da equação do pivô `j`» quando os bits de pivô
são únicos (segunda cláusula de `formaIsolada`).
O ponto é calculado pelo procedimento; não é tirado de uma existência pela escolha clássica. A forma da
existência por escolha clássica é `ressurreicaoEscolha`, em `TGLExt.ARessurreicao`
(`a_ressurreicao_fiel_existe`).

## O que fica provado neste módulo [REAL]

1. `paridade2_symm`: a paridade é simétrica, `paridade2 x a = paridade2 a x`.
2. `paridade2_mask_xor`: a paridade é aditiva na máscara,
   `paridade2 x (xorPonto a b) = paridade2 x a + paridade2 x b` (é o que se usa para a redução somar
   valores ao somar máscaras).
3. `bitPivo_spec`: o bit de pivô, quando existe, está aceso na máscara.
4. `bitPivo_minimal`: o bit de pivô é o menor bit aceso (auxiliar pensado para o invariante da forma
   isolada, que não está provado para `oEscalonado`).
5. `bitPivo_none_spec`: sem bit de pivô, a máscara é a zero.

## O que se define [DEF]

`bitPivo` (o menor bit aceso), `reduzirNova` (a redução da equação nova contra os pivôs),
`inserirEscalonado` (a inserção na lista de pivôs), `oEscalonado` (a dobra da inserção, equação por
equação), `pontoConstruido` (a leitura direta do ponto) e `ressurreicaoConstrutiva` (o ponto construído
da lista que `oEscalonado` devolve para um espectro).

## O que NÃO se afirma

* Este módulo não prova que o ponto construído serve o espectro original, nem que o procedimento preserva o
  servir. Em outros módulos, sob hipóteses nomeadas (nada disso se usa nem se afirma aqui):
  - `TGLExt.OSolverGF2_Parte2`, `reduzirNova_preserva`: se `x` serve os pivôs, `x` serve a equação reduzida
    se e somente se serve a equação original (nas duas direções);
  - `TGLExt.OSolverGF2_Parte2`, `inserirEscalonado_preserva`: sob a hipótese de consistência `hcons`
    (existe `y` que serve os pivôs e a equação nova), `x` serve a lista após a inserção se e somente se serve
    os pivôs e a equação nova (nas duas direções);
  - `TGLExt.OSolverGF2_Parte3`, `pontoConstruido_serve`: sob a hipótese `formaIsolada` dos pivôs, o ponto
    construído serve cada equação da lista.
* [OPEN] A correção completa, isto é, que `ressurreicaoConstrutiva S` sirva todo espectro `S` consistente,
  não está provada em nenhum módulo do kernel. O elo que falta é que `oEscalonado` devolva uma lista em
  `formaIsolada` (forma escalonada reduzida): não há teorema sobre `oEscalonado`. Também não se prova que a
  saída seja escalonada; o nome «Gauss–Jordan» diz a intenção, não uma propriedade estabelecida.
* Resolver um sistema consistente de equações de paridade (XOR) sobre GF(2) por eliminação de Gauss é
  polinomial [KNOWN]; isso não é decidir SAT, e este módulo nada afirma sobre SAT ou sobre P e NP. O gate do
  Instituto Clay (P vs NP) segue hipótese nomeada; este módulo não o move.
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.OSolverGF2

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade

variable {n : ℕ}

/-! ## A paridade na máscara -/

/-- [REAL] A paridade é simétrica. -/
theorem paridade2_symm {n : ℕ} (x a : Fin n → Bool) : paridade2 x a = paridade2 a x := by
  unfold paridade2
  apply Finset.sum_congr rfl
  intro i _
  cases a i <;> cases x i <;> decide

/-- [REAL] A paridade é aditiva na máscara: o xor de máscaras soma as paridades. -/
theorem paridade2_mask_xor {n : ℕ} (x a b : Fin n → Bool) :
    paridade2 x (xorPonto a b) = paridade2 x a + paridade2 x b := by
  rw [paridade2_symm (x := x) (a := xorPonto a b), paridade2_add a b x, paridade2_symm (x := x) (a := a),
    paridade2_symm (x := x) (a := b)]

/-! ## O bit de pivô -/

/-- [DEF] O bit de pivô: o menor bit aceso da máscara (o mínimo do conjunto dos bits acesos). -/
def bitPivo {n : ℕ} (a : Fin n → Bool) : Option (Fin n) :=
  if h : ((Finset.univ : Finset (Fin n)).filter fun i => a i = true).Nonempty then
    some (((Finset.univ : Finset (Fin n)).filter fun i => a i = true).min' h)
  else none

/-- [REAL] O bit de pivô está aceso na máscara. -/
theorem bitPivo_spec {n : ℕ} (a : Fin n → Bool) (i : Fin n) (h : bitPivo a = some i) :
    a i = true := by
  unfold bitPivo at h
  split at h
  · rename_i hne
    have h2 := Option.some.inj h
    have hmem : ((Finset.univ : Finset (Fin n)).filter fun k => a k = true).min' hne ∈
        (Finset.univ : Finset (Fin n)).filter (fun k => a k = true) := Finset.min'_mem _ _
    rw [h2] at hmem
    rw [Finset.mem_filter] at hmem
    exact hmem.2
  · simp at h

/-- [REAL] O bit de pivô é o menor bit aceso: todo bit aceso fica nele ou além. -/
theorem bitPivo_minimal {n : ℕ} (a : Fin n → Bool) (i : Fin n) (h : bitPivo a = some i) :
    ∀ j : Fin n, a j = true → i ≤ j := by
  unfold bitPivo at h
  split at h
  · rename_i hne
    have h2 := Option.some.inj h
    intro j hj
    have hmem : j ∈ (Finset.univ : Finset (Fin n)).filter (fun k => a k = true) := by
      rw [Finset.mem_filter]
      exact ⟨Finset.mem_univ j, hj⟩
    have hle : ((Finset.univ : Finset (Fin n)).filter (fun k => a k = true)).min' hne ≤ j :=
      Finset.min'_le _ _ hmem
    rw [h2] at hle
    exact hle
  · simp at h

/-- [REAL] Sem bit de pivô, a máscara é a zero. -/
theorem bitPivo_none_spec {n : ℕ} (a : Fin n → Bool) (h : bitPivo a = none) :
    ∀ i : Fin n, a i = false := by
  intro i
  unfold bitPivo at h
  split at h
  · simp at h
  · rename_i hne
    by_contra hc
    have hbit : a i = true := by
      cases ha : a i with
      | false => exact absurd ha hc
      | true => rfl
    have hmem : i ∈ (Finset.univ : Finset (Fin n)).filter (fun k => a k = true) := by
      rw [Finset.mem_filter]
      exact ⟨Finset.mem_univ i, hbit⟩
    exact hne ⟨i, hmem⟩

/-! ## A eliminação -/

/-- [DEF] A redução da equação nova contra os pivôs, dobrando a lista na ordem: para cada pivô com bit de
pivô `i`, se `i` está aceso no valor acumulado até ali (não na equação nova original), soma-se o pivô ao
acumulado (xor das máscaras, soma dos valores). A ordem da dobra importa. -/
def reduzirNova {n : ℕ} (a : Fin n → Bool) (v : ZMod 2) (pivots : Espectro2 n) :
    (Fin n → Bool) × ZMod 2 :=
  pivots.foldl (fun av p =>
    match bitPivo p.1 with
    | none => av
    | some i => if av.1 i then (xorPonto av.1 p.1, av.2 + p.2) else av) (a, v)

/-- [DEF] A inserção de uma equação na lista de pivôs: reduz a nova; se sobra bit de pivô, ela entra como
pivô e é somada a cada pivô antigo que tem esse bit aceso. Se não sobra, os pivôs ficam como estão. -/
def inserirEscalonado {n : ℕ} (e : (Fin n → Bool) × ZMod 2) (pivots : Espectro2 n) : Espectro2 n :=
  let (a, v) := reduzirNova e.1 e.2 pivots
  match bitPivo a with
  | none => pivots
  | some q => (a, v) :: pivots.map (fun p => if p.1 q then (xorPonto p.1 a, p.2 + v) else p)

/-- [DEF] A dobra de `inserirEscalonado` sobre o espectro, equação por equação, a partir da lista vazia.
Pretendida como a eliminação de Gauss–Jordan; que a saída esteja em forma isolada é [OPEN]. -/
def oEscalonado {n : ℕ} (S : Espectro2 n) : Espectro2 n :=
  S.foldl (fun ps e => inserirEscalonado e ps) []

/-- [DEF] O ponto construído: no bit `j`, o valor da primeira equação da lista cujo bit de pivô é `j`
(`find?`); falso quando não há tal equação. Com bits de pivô únicos, é o valor da equação do pivô `j`. -/
def pontoConstruido {n : ℕ} (pivots : Espectro2 n) : Fin n → Bool :=
  fun j => match pivots.find? (fun p => decide (bitPivo p.1 = some j)) with
    | none => false
    | some p => decide (p.2 = 1)

/-- [DEF] A ressurreição construtiva: o ponto construído da lista que `oEscalonado` devolve. Que ele sirva
o espectro não se prova neste módulo nem em outro do kernel [OPEN]. -/
def ressurreicaoConstrutiva {n : ℕ} (S : Espectro2 n) : Fin n → Bool :=
  pontoConstruido (oEscalonado S)

end TGLExt.OSolverGF2

