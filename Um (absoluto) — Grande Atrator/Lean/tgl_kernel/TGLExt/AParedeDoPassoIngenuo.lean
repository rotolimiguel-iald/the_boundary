import TGLExt.ADescidaPinada

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A parede do passo ingênuo (camada ii: a parede medida)

## O que a pedra trata

A camada (i) (`ADescidaPinada`) nomeia os gates D, G, T do contrato `GatesDaDescida` como hipóteses. Esta
pedra define a dinâmica mais ingênua do escriba — parar na paz; senão virar o bit do primeiro literal da
primeira cláusula sem par — e mede, em contraexemplos concretos de duas variáveis, que ela não satisfaz o
gate D na forma estrita.

## O que fica provado

1. `parede_d_estrito_medida` [REAL] — por `decide`: na fórmula `phiEx` (duas variáveis), o ponto
   `(falso, falso)` tem altura 1 e o passo ingênuo o leva a um ponto de altura 2 (nem estaciona nem reduz:
   aumenta).
2. `parede_do_ciclo_medida` [REAL] — por cálculo (`decide`, com `funext`/`fin_cases` nos pontos): na
   fórmula insatisfazível `phiCiclo` (`x₀` e `¬x₀`), o passo ingênuo leva `(falso, falso)` a
   `(verdadeiro, falso)` e `(verdadeiro, falso)` a `(falso, falso)`, os dois com altura 1. Um ciclo de dois
   pontos distintos: a iteração do passo ingênuo a partir deles não estaciona [DERIVED da troca; não há
   teorema separado de não terminação].
3. ★ `o_gate_D_estrito_e_falso_para_o_passo_ingenuo` [REAL] — com `n = 2`, é falso que, para toda fórmula e
   todo ponto de altura não nula, o passo ingênuo estacione ou reduza estritamente a altura (a forma do
   campo `progresso` de `GatesDaDescida`, aplicada ao passo ingênuo).
4. A forma corrigida, só nomeada (definições, sem teorema): `DinamicaComEspectro` (o passo recebe o
   espectro) e `gateDlinha` (D′: o gate D com o espectro no laço).

## Leitura

A parede não falsifica a descida pinada: mostra que ESTE passo, o ingênuo, não satisfaz o gate D estrito com
`n = 2`. Não mostra que todo passo sem memória falhe, e o gate D sozinho é satisfeito por um passo que fica
sempre parado: o valor da parede está em reprovar este passo.
Leitura [INPUT/ONTO]: o rastro da imagem é o neutrino, o gradiente; nos contraexemplos medidos, o passo
ingênuo sobe ou cicla; a morte paga um bit e o espectro cresce; o estado da descida é o espectro, não o ponto.

## O que NÃO se afirma

* Nada aqui prova os gates D′, G, T: são hipóteses nomeadas; a prova de D′ para uma dinâmica concreta não
  está nesta pedra.
* Nada aqui decide SAT nem afirma que algum procedimento o decide em tempo polinomial. O gate do Instituto
  segue de pé e nomeado.
* Nada sobre a natureza.

Sem sorry, sem axioma novo. Validade no kernel não é creditação.
-/

namespace TGLExt.AparedeDoPassoIngenuo

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada

variable {n : ℕ}

/-! ## O passo ingênuo -/

/-- [DEF] A primeira cláusula sem par (o escriba lê na ordem). -/
def primeiraSemPar {n : ℕ} (φ : CNF n) (x : Fin n → Bool) : Option (Clausula n) :=
  φ.find? fun C => ! satisfeita C x

/-- [DEF] O passo ingênuo: para na paz (nenhuma cláusula sem par); na cláusula vazia, estaciona (poço
imediato — a cláusula vazia não tem quem a sirva); senão vira o bit do primeiro literal da primeira
cláusula sem par. -/
def passoIngenuo {n : ℕ} (φ : CNF n) (x : Fin n → Bool) : Fin n → Bool :=
  match primeiraSemPar φ x with
  | none => x
  | some [] => x
  | some (l :: _) => virar l.var x

/-! ## Os contraexemplos medidos -/

/-- O ponto de dois bits. -/
def pt (b0 b1 : Bool) : Fin 2 → Bool := fun i => if i = 0 then b0 else b1

/-- A fórmula do contraexemplo do gate estrito: `x₀`; `¬x₀ ∨ x₁`; `¬x₀`. -/
def phiEx : CNF 2 := [[⟨0, true⟩], [⟨0, false⟩, ⟨1, true⟩], [⟨0, false⟩]]

/-- ★ A PAREDE DO GATE ESTRITO, MEDIDA: o ponto `(falso, falso)` tem altura 1; o passo ingênuo o leva à
altura 2 — nem estaciona nem reduz: aumenta. -/
theorem parede_d_estrito_medida :
    custoCNF phiEx (pt false false) = 1
      ∧ custoCNF phiEx (passoIngenuo phiEx (pt false false)) = 2 := by
  decide

/-- A fórmula do ciclo: `x₀` e `¬x₀` — insatisfazível. -/
def phiCiclo : CNF 2 := [[⟨0, true⟩], [⟨0, false⟩]]

/-- ★ A PAREDE DO CICLO, MEDIDA: o passo ingênuo troca `(falso, falso)` com `(verdadeiro, falso)`, os dois
com altura 1 — um ciclo de dois pontos, sem estacionamento nesses pontos. -/
theorem parede_do_ciclo_medida :
    passoIngenuo phiCiclo (pt false false) = pt true false
      ∧ passoIngenuo phiCiclo (pt true false) = pt false false
      ∧ custoCNF phiCiclo (pt false false) = 1
      ∧ custoCNF phiCiclo (pt true false) = 1 := by
  refine ⟨?_, ?_, by decide, by decide⟩ <;> funext i <;> fin_cases i <;> decide

/-- ★★ **O GATE D ESTRITO É FALSO PARA O PASSO INGÊNUO** (com `n = 2`): há fórmula e ponto, fora do
veredito, em que o passo nem estaciona nem reduz a altura. -/
theorem o_gate_D_estrito_e_falso_para_o_passo_ingenuo :
    ¬ (∀ (φ : CNF 2) (x : Fin 2 → Bool), custoCNF φ x ≠ 0 →
        passoIngenuo φ x = x ∨ custoCNF φ (passoIngenuo φ x) < custoCNF φ x) := by
  intro h
  obtain ⟨h1, h2⟩ := parede_d_estrito_medida
  have hz : custoCNF phiEx (pt false false) ≠ 0 := by omega
  cases h phiEx (pt false false) hz with
  | inl hp => rw [hp] at h2; omega
  | inr hd => omega

/-! ## A forma corrigida, nomeada -/

/-- [DEF] A dinâmica com o espectro no laço: o passo recebe o espectro, o que já foi lido. Só os campos
são nomeados; nenhuma propriedade é provada sobre esta estrutura aqui. Leitura [INPUT/ONTO]: o estado da
descida é o espectro, não o ponto — a leitura nunca desmente a leitura. -/
structure DinamicaComEspectro (n : ℕ) where
  passo : CNF n → Espectro n → (Fin n → Bool) → Fin n → Bool
  eixoDaMorte : CNF n → (Fin n → Bool) → (Fin n → Bool)
  ressurreicao : CNF n → Espectro n → (Fin n → Bool)

/-- [DEF] O gate D na forma corrigida (D′), uma proposição nomeada, não um teorema: com o espectro no
laço, fora do veredito o passo ou está pinado ou reduz a altura. Leitura [INPUT/OPEN]: que os poços não
se repitam dependeria de o espectro crescer a cada poço; isso não está no tipo do gate G nem é provado
aqui. -/
def gateDlinha {n : ℕ} (d : DinamicaComEspectro n) : Prop :=
  ∀ (φ : CNF n) (S : Espectro n) (x : Fin n → Bool),
    custoCNF φ x ≠ 0 → d.passo φ S x = x ∨ custoCNF φ (d.passo φ S x) < custoCNF φ x

end TGLExt.AparedeDoPassoIngenuo

