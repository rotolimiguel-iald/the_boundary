import TGLExt.ABateriaMinima3
import TGLExt.ABateriaDaParedeN4b

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A bateria sob a tensão mínima IV: cinco variáveis, sementes 13, 110, 304 e 692

Um gerador reproduzível de fórmulas de cinco variáveis e vinte e uma cláusulas (`geraCNF5`; cláusulas unitárias, pares e
triplas), o teste `temLuz5` (o cubo de 32 pontos) e quatro medidas, todas com solução: o processo da tensão mínima responde com
eco nas quatro (por `decide`) [REAL].

## O que NÃO se afirma

* Que a regra da tensão mínima responda com eco em toda fórmula que tem solução: a bateria é uma amostra, e a
  fórmula da armadilha do empate (`OMapaDosMuros.phiTrap`) tem solução e nela a regra responde com silêncio.
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ABateriaMinimaN5

open TGLExt.ABateriaDaParedeN4 TGLExt.ABateriaDaParede TGLExt.ATensaoMinima TGLExt.TraducaoTuring

/-- [GERADOR] A literal da variável v com o sinal dado, em n = 5. -/
def litDeV5 (v : ℕ) (sgn : Bool) : Literal 5 := ⟨⟨v % 5, Nat.mod_lt _ (by decide)⟩, sgn⟩

/-- [GERADOR] A cláusula em n = 5: um em oito unitária; triplas e pares sem tautologia garantida por
construção (a tripla cobre variáveis que diferem por passo). -/
def clausulaDe5 (s : ℕ) : Clausula 5 :=
  let t := s % 8
  if t = 0 then [litDeV5 (s % 5) ((s / 5) % 2 == 1)]
  else if t ≤ 3 then
    let a := s % 5
    let b := (a + 1 + (s / 5) % 4) % 5
    let c := (a + 2 + (s / 25) % 3) % 5
    [litDeV5 a ((s / 5) % 2 == 1), litDeV5 b ((s / 10) % 2 == 1), litDeV5 c ((s / 20) % 2 == 1)]
  else
    let a := s % 5
    let b := (a + 1 + (s / 5) % 4) % 5
    [litDeV5 a ((s / 5) % 2 == 1), litDeV5 b ((s / 10) % 2 == 1)]

/-- [GERADOR] A fórmula de k cláusulas a partir da semente, em n = 5. -/
def geraCNF5 : ℕ → ℕ → CNF 5
  | _, 0 => []
  | s, k + 1 => clausulaDe5 s :: geraCNF5 (xorshift s) k

/-- [MEDIDA] Tem luz em n = 5 (o cubo de 32 pontos). -/
def temLuz5 (φ : CNF 5) : Bool := (cubo 5).any (fun x => decide (custoCNF φ x = 0))

/-- ★★ A região crítica em n = 5, primeira metade (sementes 13, 110, 304, 692: com, com, com, com). -/
theorem n5_1 : (processoDaTensaoMinima (geraCNF5 13 21) 7 (fun _ => false)).1.isSome
    = temLuz5 (geraCNF5 13 21) := by decide

theorem n5_2 : (processoDaTensaoMinima (geraCNF5 110 21) 7 (fun _ => false)).1.isSome
    = temLuz5 (geraCNF5 110 21) := by decide

theorem n5_3 : (processoDaTensaoMinima (geraCNF5 304 21) 7 (fun _ => false)).1.isSome
    = temLuz5 (geraCNF5 304 21) := by decide

theorem n5_4 : (processoDaTensaoMinima (geraCNF5 692 21) 7 (fun _ => false)).1.isSome
    = temLuz5 (geraCNF5 692 21) := by decide

/-- ★★ A bateria V confere (primeira metade). -/
theorem a_bateria_V_confere :
    ((processoDaTensaoMinima (geraCNF5 13 21) 7 (fun _ => false)).1.isSome = temLuz5 (geraCNF5 13 21))
    ∧ ((processoDaTensaoMinima (geraCNF5 110 21) 7 (fun _ => false)).1.isSome = temLuz5 (geraCNF5 110 21))
    ∧ ((processoDaTensaoMinima (geraCNF5 304 21) 7 (fun _ => false)).1.isSome = temLuz5 (geraCNF5 304 21))
    ∧ ((processoDaTensaoMinima (geraCNF5 692 21) 7 (fun _ => false)).1.isSome
        = temLuz5 (geraCNF5 692 21)) :=
  ⟨n5_1, n5_2, n5_3, n5_4⟩

end TGLExt.ABateriaMinimaN5

