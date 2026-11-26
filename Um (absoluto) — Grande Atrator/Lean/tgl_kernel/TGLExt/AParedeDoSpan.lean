import TGLExt.ARessurreicao

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A parede do span: um eixo pode estar no span dos eixos lidos sem estar na lista deles

## O que é

O span dos eixos lidos, por xor (a soma em GF(2) de uma sub-lista), e um contraexemplo de duas variáveis: um eixo
fora da lista dos eixos lidos, mas no span deles, não divide o resíduo. Logo checar a pertença à lista não basta
para garantir que a leitura de um eixo novo divida o resíduo; a checagem que serve a esse fim é a pertença ao span.

## O que fica provado [REAL]

1. `xorComb` e `eixoNoSpan` [INPUT, definições]: a combinação por xor de uma lista de eixos e a pertença de um eixo
   ao span de uma lista de eixos (existe uma sub-lista cuja combinação por xor é o eixo).
2. ★ `a_parede_do_eixo_no_span`: com o espectro `S2 = [(eixo0, 1), (eixo1, 0)]` em duas variáveis, todo ponto que
   serve `S2` tem paridade 1 no eixo `eixo01 = {x₀, x₁}`: dentro do resíduo a paridade nesse eixo é constante, e a
   leitura desse eixo não divide o resíduo. Prova por `decide`.
3. `o_eixo01_esta_no_span`: `eixo01` está no span de `[eixo0, eixo1]` (é o xor dos dois).
4. ★ `o_eixo_no_span_nao_esta_na_lista`: `eixo01` não está na lista `[eixo0, eixo1]`. Prova por `decide`.
5. `gateG2linha` [INPUT, definição; contrato nomeado, não teorema]: para uma função `eixoDaMorte`, toda fórmula, todo
   espectro e todo ponto que o serve, o eixo `eixoDaMorte φ w` está fora do span dos eixos do espectro.

## O que NÃO se afirma

* O contraexemplo é um espectro e um eixo explícitos em duas variáveis; não é o eixo da morte de uma fórmula.
  Nada aqui diz, para fórmula alguma, que o eixo da morte (`eixoDaMorteDe`) caia no span.
* A hipótese `hdiv` dos teoremas do laço de `OProcessoDaLuz` checa a pertença à lista (`∉ S.map Prod.fst`), não ao
  span. Medido em `TGLExt.AVacuidadeDosContratos`: `hdiv` e `hpoco` não valem juntas para nenhum `n` e nenhuma
  fórmula (`gates_do_processo_insatisfaziveis`; `hpoco_falso`: a descida do laço não recebe o espectro). Um teorema
  que toma as duas como hipótese vale no vazio e não diz nada sobre fórmula alguma. Esta pedra não toma nenhuma das
  duas como hipótese.
* `gateG2linha` é só definição: este módulo não prova que alguma função `eixoDaMorte` o cumpra, nem que um eixo fora
  do span tome os dois valores dentro do resíduo (resíduo não vazio) [OPEN]. A pista desse lema é a dualidade em
  GF(2): o anulador do núcleo é o span das linhas [KNOWN, álgebra linear; não está neste módulo].
* A refutação de `gateG2linha` como está escrito é esperada, e não é medida no kernel [DERIVED, argumento por cálculo
  fora do kernel; não há teorema `gateG2linha_insatisfazivel`]: para todo `w`, o espectro dos `n` eixos unitários,
  com os valores lidos no próprio `w`, é servido por `w`, e os seus eixos geram todo o cubo; logo `eixoDaMorte φ w`
  está no span, para toda função `eixoDaMorte` e todo `n`, desde que `CNF n` tenha habitante (com `n = 0`, até o
  espectro vazio basta). Se a refutação vale, um teorema que tome `gateG2linha` como hipótese vale no vazio e não diz
  nada sobre fórmula alguma. Esta pedra não toma `gateG2linha` como hipótese.
* Nada aqui diz que algum processo decida SAT, em tempo algum. Decidir SAT sem testemunha, em tempo polinomial, na
  máquina de fita segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AparedeDoSpan

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao

variable {n : ℕ}

/-! ## O span dos eixos -/

/-- [INPUT, definição] A combinação por xor de uma lista de eixos (a soma em GF(2) da lista). -/
def xorComb {n : ℕ} (l : List (Fin n → Bool)) : Fin n → Bool := l.foldl xorPonto (fun _ => false)

/-- [INPUT, definição] O eixo no span do espectro: é a combinação por xor de uma sub-lista dos eixos lidos. -/
def eixoNoSpan {n : ℕ} (axes : List (Fin n → Bool)) (a : Fin n → Bool) : Prop :=
  ∃ sub : List (Fin n → Bool), sub ⊆ axes ∧ xorComb sub = a

/-! ## O contraexemplo -/

/-- O eixo da variável 0. -/
def eixo0 : Fin 2 → Bool := fun i => if i = 0 then true else false

/-- O eixo da variável 1. -/
def eixo1 : Fin 2 → Bool := fun i => if i = 0 then false else true

/-- O eixo das duas variáveis: o xor dos dois. -/
def eixo01 : Fin 2 → Bool := fun _ => true

/-- O espectro do contraexemplo: `x₀ ↦ 1`, `x₁ ↦ 0`. O resíduo é o ponto único (verdadeiro, falso) [DERIVED, por
cálculo; não é teorema deste módulo, que prova só a paridade constante em `eixo01`]. -/
def S2 : Espectro2 2 := [(eixo0, 1), (eixo1, 0)]

/-- ★ **A PAREDE DO EIXO NO SPAN** [REAL]: dentro do resíduo de `S2`, a paridade no eixo `{x₀, x₁}` é constante
`1`; a leitura desse eixo não divide o resíduo. Vale para este espectro e este eixo explícitos, não para o eixo da
morte de uma fórmula. Prova por `decide`. -/
theorem a_parede_do_eixo_no_span :
    ∀ x : Fin 2 → Bool, serveEspectro2 x S2 = true → paridade2 x eixo01 = 1 := by
  decide

/-- [REAL] O eixo das duas variáveis está no span dos dois eixos unitários: `eixo01 = eixo0 ⊞ eixo1`. -/
theorem o_eixo01_esta_no_span : eixoNoSpan [eixo0, eixo1] eixo01 := by
  refine ⟨[eixo0, eixo1], ?_, ?_⟩
  · intro s hs; exact hs
  · ext i
    fin_cases i <;> decide

/-- ★ **A CHECAGEM PELA LISTA NÃO BASTA** [REAL]: o eixo no span não está na lista dos eixos lidos; uma checagem por
pertença à lista o deixaria passar como eixo novo, embora a sua leitura não divida o resíduo. Prova por `decide`. -/
theorem o_eixo_no_span_nao_esta_na_lista : eixo01 ∉ [eixo0, eixo1] := by
  decide

/-! ## A forma do gate pelo span -/

/-- [INPUT, definição; contrato nomeado] **G2′**: o eixo lido pela morte é transversal, fora do span dos eixos do
espectro, para toda fórmula, todo espectro e todo ponto que o serve. É hipótese, não teorema: este módulo não prova
que alguma função o cumpra, nem que um eixo fora do span tome os dois valores dentro do resíduo [OPEN]. A refutação
como está escrito é esperada (os eixos unitários lidos no próprio ponto geram todo o cubo), mas não é medida no kernel
[DERIVED, fora do kernel]. -/
def gateG2linha {n : ℕ} (eixoDaMorte : CNF n → (Fin n → Bool) → (Fin n → Bool)) : Prop :=
  ∀ (φ : CNF n) (S : Espectro2 n) (w : Fin n → Bool),
    serveEspectro2 w S = true → ¬ eixoNoSpan (S.map Prod.fst) (eixoDaMorte φ w)

end TGLExt.AparedeDoSpan

