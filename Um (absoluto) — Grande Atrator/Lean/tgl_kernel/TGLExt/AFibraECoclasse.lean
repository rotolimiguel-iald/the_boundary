import TGLExt.AParedeDoSpan

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A fibra é a coclasse: o resíduo é a translação do núcleo da leitura

## O que é

A leitura `leitura2 axes` leva um ponto `x : Fin n → Bool` à lista das paridades de `x` nos eixos `axes`, em
`ZMod 2`. Este módulo prova que, quando a fibra de um valor `vals` tem um ponto `x₀`, essa fibra é a translação
por `x₀` (pelo xor `xorPonto`) do núcleo da leitura, o conjunto dos pontos de leitura zero: a estrutura de
coclasse (afim) do resíduo.

## De onde vem

O módulo `AParedeDoSpan` refina o G2 (a leitura divide) no contrato nomeado `gateG2linha` (G2′): a morte lê um eixo
transversal, fora do span dos eixos lidos [INPUT, definição em `AParedeDoSpan`; contrato, não teorema]. O lema que
fecharia essa leitura é a dualidade: um eixo fora do span toma os dois valores dentro do resíduo [OPEN]. Este módulo
prova a peça que prepara o argumento da aniquilação: o resíduo habitado é a translação do núcleo da leitura, a
estrutura de coclasse a partir da qual a álgebra linear clássica leria esse resultado.

[INPUT/ONTO] Leitura do autor: a fibra habitada é o resíduo; o eixo se resolve com a morte e divide sempre porque
realiza a poda; é o tetelestai: opera o recibo, a transcrição. Esta leitura não é teorema deste módulo: aqui não se
prova que eixo algum divida o resíduo (ver «O que NÃO se afirma»).

## O que fica provado [REAL]

1. `zipAdd_self_zero`: a soma termo a termo de uma lista em `ZMod 2` consigo mesma é a lista de zeros do mesmo
   comprimento. `zipAdd_zero_esq`: a lista de `m` zeros é neutra à esquerda na soma termo a termo com uma lista de
   comprimento `m`.
2. `oNucleo` [INPUT, definição]: o núcleo da leitura, os pontos com leitura igual à lista de zeros.
   `leitura2_length`: o comprimento da leitura é o número de eixos.
3. ★ `a_fibra_e_coclasse`: dado `x₀` com `leitura2 axes x₀ = vals`, o conjunto dos pontos com leitura `vals` é
   igual à imagem de `oNucleo axes` pela translação `k ↦ xorPonto k x₀`. A prova usa a homomorfia da leitura
   (`leitura2_xor`: a leitura do xor é a soma das leituras) e a involução do xor (`xorPonto_invol`).

## O que NÃO se afirma

* Não se prova aqui a dualidade (que um eixo fora do span das linhas toma os dois valores dentro do resíduo). A
  forma que ela teria a partir da coclasse: a paridade do eixo novo é constante na coclasse sse o eixo se anula no
  núcleo, e o aniquilador do núcleo é o span das linhas. Esse conteúdo clássico de álgebra linear não está tipado
  neste módulo [OPEN].
* O teorema exige o ponto `x₀` como hipótese: não constrói nem encontra um ponto da fibra, e nada diz sobre fibras
  vazias.
* «Divide sempre», na leitura do autor, não é teorema. Como contrato, o campo `divide` (o G2) de `GatesDaMorte` e de
  `AVerdadeFinal` não é cumprido por nenhuma dinâmica, tal como está escrito: medido em
  `TGLExt.AVacuidadeDosContratos` (`gatesDaMorte_insatisfazivel`, `aVerdadeFinal_insatisfazivel`; o mesmo vale para
  `GatesDaDescida`, `gatesDaDescida_insatisfazivel`). Em `OProcessoDaLuz`, as hipóteses `hdiv` e `hpoco` não valem
  juntas para nenhum `n` e nenhuma fórmula (`gates_do_processo_insatisfaziveis`; `hpoco_falso`: a descida do laço não
  recebe o espectro); `hfiel` é cumprida (`hfiel_habitado`). Um teorema que tome um desses contratos, ou `hdiv` e
  `hpoco` juntas, como hipótese vale no vazio e não diz nada sobre fórmula alguma. Este módulo não toma nenhum deles
  como hipótese, nem `gateG2linha`; os seus teoremas não dependem deles.
* Nada aqui diz que algum processo decida SAT, em tempo algum. Decidir SAT sem testemunha, em tempo polinomial, na
  máquina de fita segue [OPEN].
* Nada sobre a natureza. O gate do Instituto (P versus NP) segue de pé e nomeado; o gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AFibraECoclasse

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan

variable {n : ℕ}

/-- A soma termo a termo de uma lista consigo mesma é a lista de zeros: `v ⊞ v = zeros`. (Auxiliar da coclasse.) -/
theorem zipAdd_self_zero : ∀ (v : List (ZMod 2)),
    List.zipWith (· + ·) v v = List.replicate v.length 0 := by
  intro v
  induction v with
  | nil => rfl
  | cons a ax ih =>
    simp [List.zipWith_cons_cons, List.replicate_succ, CharTwo.add_self_eq_zero, ih]

/-- A lista de zeros é neutra à esquerda na soma termo a termo: `zeros ⊞ v = v`. (Auxiliar da coclasse.) -/
theorem zipAdd_zero_esq : ∀ (m : ℕ) (v : List (ZMod 2)), v.length = m →
    List.zipWith (· + ·) (List.replicate m 0) v = v := by
  intro m
  induction m with
  | zero =>
    intro v hv
    cases v with
    | nil => rfl
    | cons c cs => simp at hv
  | succ k ih =>
    intro v hv
    cases v with
    | nil => simp at hv
    | cons b bx =>
      have hll : bx.length = k := by simpa using hv
      simp [List.replicate_succ, List.zipWith_cons_cons, zero_add, ih bx hll]

/-- [INPUT] Definição: o núcleo da leitura, os pontos com leitura zero. -/
def oNucleo {n : ℕ} (axes : List (Fin n → Bool)) : Finset (Fin n → Bool) :=
  Finset.univ.filter fun k => leitura2 axes k = List.replicate axes.length 0

/-- O comprimento da leitura é o número de eixos. (Auxiliar da coclasse.) -/
theorem leitura2_length {n : ℕ} (axes : List (Fin n → Bool)) (x : Fin n → Bool) :
    (leitura2 axes x).length = axes.length := by
  unfold leitura2
  rw [List.length_map]

/-- ★ **A FIBRA É A COCLASSE**: dado um ponto `x₀` com leitura `vals`, os pontos com leitura `vals` são exatamente
as translações por `x₀` (pelo xor) dos pontos do núcleo. -/
theorem a_fibra_e_coclasse {n : ℕ} (axes : List (Fin n → Bool)) (vals : List (ZMod 2))
    (x₀ : Fin n → Bool) (hx₀ : leitura2 axes x₀ = vals) :
    (Finset.univ.filter fun x => leitura2 axes x = vals)
      = (oNucleo axes).image (fun k => xorPonto k x₀) := by
  ext x
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image, oNucleo]
  constructor
  · intro hx
    refine ⟨xorPonto x x₀, ?_, ?_⟩
    · have hlen : vals.length = axes.length := by rw [← hx₀]; exact leitura2_length axes x₀
      rw [leitura2_xor, hx, hx₀, zipAdd_self_zero, hlen]
    · exact xorPonto_invol x x₀
  · intro h
    obtain ⟨k, hk, hkx⟩ := h
    have hlen : vals.length = axes.length := by rw [← hx₀]; exact leitura2_length axes x₀
    rw [← hkx, leitura2_xor, hk, hx₀, zipAdd_zero_esq _ _ hlen]

end TGLExt.AFibraECoclasse

