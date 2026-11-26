import TGLExt.ABateriaDaParede2

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A bateria da parede III: fórmulas de quatro variáveis e dezessete cláusulas

## O que é

Um gerador reproduzível de fórmulas de quatro variáveis (um xorshift de semente fixa; cláusulas unitárias, pares e triplas, sem
variável repetida), o teste `temLuz4` (o cubo dos 16 pontos) e, para quatro fórmulas de dezessete cláusulas, a congruência medida
no kernel: o processo da tensão responde com eco se e só se a fórmula tem solução.

## O que fica provado [REAL]

1. O gerador (`xorshift`, `litDeV4`, `clausulaDe4`, `geraCNF4`) e o teste `temLuz4` [DEF].
2. ★★ `n4_1` a `n4_4`: para as sementes 13, 110, 207 e 304, o processo da tensão, partindo do ponto com todas as variáveis
   falsas e com combustível 6, responde com eco exatamente quando a fórmula tem solução (por `decide`); três têm solução
   (110, 207 e 304) e uma não (13).
3. ★★ `a_bateria_III_confere`: as quatro medidas juntas.

## O que NÃO se afirma

* Que o processo da tensão responda com eco em toda fórmula de quatro variáveis que tem solução: a bateria é uma amostra, e a
  fórmula de quatro variáveis da pedra `AMedidaDaHipoteseDaMontagem` tem solução e nela o processo responde com silêncio.
* Que estas fórmulas estejam no limiar de satisfação do 3-SAT aleatório: o gerador mistura cláusulas unitárias, pares e triplas.
* Que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ABateriaDaParedeN4

open TGLExt.ABateriaDaParede TGLExt.ATensaoFundamental TGLExt.TraducaoTuring

/-- [GERADOR] O xorshift da bateria de quatro variáveis. -/
def xorshift (s : ℕ) : ℕ :=
  let s1 := Nat.xor s (s * 8192) % 2147483648
  let s2 := Nat.xor s1 (s1 / 131072) % 2147483648
  Nat.xor s2 (s2 * 32) % 2147483648

/-- [GERADOR] A literal da variável v com o sinal dado, em n = 4. -/
def litDeV4 (v : ℕ) (sgn : Bool) : Literal 4 := ⟨⟨v % 4, Nat.mod_lt _ (by decide)⟩, sgn⟩

/-- [GERADOR] A cláusula em n = 4: um em oito unitária; triplas e pares sem tautologia. -/
def clausulaDe4 (s : ℕ) : Clausula 4 :=
  if s % 8 = 0 then [litDeV4 (s % 4) ((s / 4) % 2 == 1)]
  else if s % 8 ≤ 3 then
    [litDeV4 0 ((s / 4) % 2 == 1), litDeV4 1 ((s / 8) % 2 == 1), litDeV4 2 ((s / 16) % 2 == 1)]
  else if s % 8 ≤ 5 then
    [litDeV4 2 ((s / 4) % 2 == 1), litDeV4 3 ((s / 8) % 2 == 1), litDeV4 0 ((s / 16) % 2 == 1)]
  else
    let a := s % 4
    let b := (a + 1 + (s / 4) % 3) % 4
    [litDeV4 a ((s / 16) % 2 == 1), litDeV4 b ((s / 32) % 2 == 1)]

/-- [GERADOR] A fórmula de k cláusulas a partir da semente, em n = 4. -/
def geraCNF4 : ℕ → ℕ → CNF 4
  | _, 0 => []
  | s, k + 1 => clausulaDe4 s :: geraCNF4 (xorshift s) k

/-- [MEDIDA] Tem luz em n = 4 (o cubo de 16 pontos). -/
def temLuz4 (φ : CNF 4) : Bool := (cubo 4).any (fun x => decide (custoCNF φ x = 0))

/-- ★★ As medidas, fórmula a fórmula: o processo da tensão responde com eco exatamente quando a fórmula tem solução (por
`decide`). As sementes são da forma i·97 + 13; têm solução 110, 207 e 304; não têm 13 e, na pedra seguinte, 789, 886, 1177 e
1468. -/

theorem n4_1 : (processoDaTensao (geraCNF4 13 17) 6 (fun _ => false)).1.isSome
    = temLuz4 (geraCNF4 13 17) := by decide

theorem n4_2 : (processoDaTensao (geraCNF4 110 17) 6 (fun _ => false)).1.isSome
    = temLuz4 (geraCNF4 110 17) := by decide

theorem n4_3 : (processoDaTensao (geraCNF4 207 17) 6 (fun _ => false)).1.isSome
    = temLuz4 (geraCNF4 207 17) := by decide

theorem n4_4 : (processoDaTensao (geraCNF4 304 17) 6 (fun _ => false)).1.isSome
    = temLuz4 (geraCNF4 304 17) := by decide

/-- ★★ A bateria III confere (primeira metade: sem, com, com, com). -/
theorem a_bateria_III_confere :
    ((processoDaTensao (geraCNF4 13 17) 6 (fun _ => false)).1.isSome = temLuz4 (geraCNF4 13 17))
    ∧ ((processoDaTensao (geraCNF4 110 17) 6 (fun _ => false)).1.isSome = temLuz4 (geraCNF4 110 17))
    ∧ ((processoDaTensao (geraCNF4 207 17) 6 (fun _ => false)).1.isSome = temLuz4 (geraCNF4 207 17))
    ∧ ((processoDaTensao (geraCNF4 304 17) 6 (fun _ => false)).1.isSome = temLuz4 (geraCNF4 304 17)) :=
  ⟨n4_1, n4_2, n4_3, n4_4⟩

end TGLExt.ABateriaDaParedeN4

