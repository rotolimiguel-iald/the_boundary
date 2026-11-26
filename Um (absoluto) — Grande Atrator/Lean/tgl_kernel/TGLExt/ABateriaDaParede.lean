import TGLExt.AMontagemFinal

set_option autoImplicit false
set_option maxHeartbeats 1600000

/-!
# A bateria da parede I: quatro fórmulas pseudoaleatórias de três variáveis

## O que é

Um gerador reproduzível de fórmulas de três variáveis (um gerador congruencial linear de semente fixa; cláusulas sem variável
repetida), o cubo computável dos 2³ pontos e o teste `temLuz`; e, para quatro fórmulas de dez cláusulas, a congruência medida
no kernel: o processo da tensão responde com eco se e só se a fórmula tem solução.

## O que fica provado [REAL]

1. O gerador (`lcg`, `litDeV`, `clausulaDe`, `geraCNF3`), o cubo (`pontoDeBits`, `cubo`) e o teste `temLuz` [DEF].
2. ★★ `bat1` a `bat4`: para as sementes 17, 1010, 1341 e 1672, o processo da tensão, partindo do ponto com todas as variáveis
   falsas e com combustível 5, responde com eco exatamente quando a fórmula tem solução (por `decide`); duas têm solução
   (17 e 1672) e duas não.
3. ★★ `a_bateria_I_confere`: as quatro medidas juntas.

## O que NÃO se afirma

* Que o processo da tensão responda com eco em toda fórmula de três variáveis que tem solução: a bateria é uma amostra, e a
  variante de três variáveis da pedra `AMedidaDaHipoteseDaMontagem` tem solução e nela o processo responde com silêncio.
* Que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ABateriaDaParede

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental TGLExt.OPostoDoEscalonado
open TGLExt.AContagemDaTensao TGLExt.AMontagemFinal

/-- [GERADOR] O LCG da bateria (semente fixa, reproduzível). -/
def lcg (s : ℕ) : ℕ := (s * 1103515245 + 12345) % 2147483648

/-- [GERADOR] A literal da variável v com o sinal dado. -/
def litDeV (v : ℕ) (sgn : Bool) : Literal 3 := ⟨⟨v % 3, Nat.mod_lt _ (by decide)⟩, sgn⟩

/-- [GERADOR] A cláusula lida da semente — sem tautologias (variáveis distintas): um em quatro é
unitária, um em quatro é tripla. -/
def clausulaDe (s : ℕ) : Clausula 3 :=
  if s % 4 = 0 then [litDeV (s % 3) ((s / 3) % 2 == 1)]
  else if s % 4 = 1 then
    [litDeV 0 ((s / 3) % 2 == 1), litDeV 1 ((s / 6) % 2 == 1), litDeV 2 ((s / 12) % 2 == 1)]
  else
    let p := (s / 24) % 3
    let a := if p = 1 then 0 else if p = 2 then 1 else 0
    let b := if p = 1 then 2 else if p = 2 then 2 else 1
    [litDeV a ((s / 3) % 2 == 1), litDeV b ((s / 6) % 2 == 1)]

/-- [GERADOR] A fórmula de k cláusulas a partir da semente. -/
def geraCNF3 : ℕ → ℕ → CNF 3
  | _, 0 => []
  | s, k + 1 => clausulaDe s :: geraCNF3 (lcg s) k

/-- [MEDIDA] O ponto de bits i do cubo de n: o j-ésimo bit de i. -/
def pontoDeBits (n : ℕ) (i : ℕ) : Fin n → Bool := fun j => (i / 2 ^ (j : ℕ)) % 2 == 1

/-- [MEDIDA] O cubo inteiro: os 2^n pontos, computável. -/
def cubo (n : ℕ) : List (Fin n → Bool) := (List.range (2 ^ n)).map (pontoDeBits n)

/-- [MEDIDA] Tem luz: algum ponto do cubo tem custo zero (a enumeração completa, computável). -/
def temLuz (φ : CNF 3) : Bool :=
  (cubo 3).any (fun x => decide (custoCNF φ x = 0))

/-- ★★ A medida da bateria, fórmula a fórmula: o processo da tensão responde com eco exatamente quando a fórmula tem
solução. As sementes são da forma i·331 + 17; têm solução bat1, bat4 e, na pedra seguinte, bat7; as demais não. -/

theorem bat1 : (processoDaTensao (geraCNF3 17 10) 5 (fun _ => false)).1.isSome
    = temLuz (geraCNF3 17 10) := by decide

theorem bat2 : (processoDaTensao (geraCNF3 1010 10) 5 (fun _ => false)).1.isSome
    = temLuz (geraCNF3 1010 10) := by decide

theorem bat3 : (processoDaTensao (geraCNF3 1341 10) 5 (fun _ => false)).1.isSome
    = temLuz (geraCNF3 1341 10) := by decide

theorem bat4 : (processoDaTensao (geraCNF3 1672 10) 5 (fun _ => false)).1.isSome
    = temLuz (geraCNF3 1672 10) := by decide

/-- ★★ A bateria I confere (quatro primeiras). -/
theorem a_bateria_I_confere :
    ((processoDaTensao (geraCNF3 17 10) 5 (fun _ => false)).1.isSome = temLuz (geraCNF3 17 10))
    ∧ ((processoDaTensao (geraCNF3 1010 10) 5 (fun _ => false)).1.isSome = temLuz (geraCNF3 1010 10))
    ∧ ((processoDaTensao (geraCNF3 1341 10) 5 (fun _ => false)).1.isSome = temLuz (geraCNF3 1341 10))
    ∧ ((processoDaTensao (geraCNF3 1672 10) 5 (fun _ => false)).1.isSome = temLuz (geraCNF3 1672 10)) :=
  ⟨bat1, bat2, bat3, bat4⟩

end TGLExt.ABateriaDaParede

