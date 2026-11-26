import TGLExt.OMapaDosMuros

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A busca com a poda

## O que é

A busca com volta atrás sobre as portas limpas. Na cláusula sem par, o laço registra a melhor porta limpa (a de tensão mínima)
e guarda o ponto de escolha (o espectro de antes e as portas restantes). Quando nenhuma porta limpa resta, volta ao ponto mais
recente com portas e tenta a próxima; o ponto esgotado sai da pilha; a pilha vazia é o silêncio.

## O que fica provado [REAL]

1. `oMelhorDe`, `limpasDe`, `mesmaLiteral`, `lacoComPoda`, `processoComPoda` [DEF]: a busca.
2. ★★ `o_eco_da_poda_e_fiel`: a busca só responde com ponto de custo zero.
3. ★★ `a_medida_a_poda_atravessa_os_tres_muros`: a busca responde com eco em `phiTrap`, `phiCE` e `phiFofa2` (por `decide`).
4. ★★ `a_medida_a_poda_e_honesta`: a busca responde com silêncio numa fórmula sem solução e com eco nas fórmulas pequenas das
   pedras anteriores (por `decide`).

## O que NÃO se afirma

* Que a busca responda com eco em toda fórmula que tem solução (que a pilha esgotada só venha quando não há solução): segue
  [OPEN].
* Uma cota polinomial para o número de passos: o número de revisões pode crescer com a fórmula, e a cota segue [OPEN].
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.APodaFechaOContorno

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental TGLExt.OPostoDoEscalonado
open TGLExt.AContagemDaTensao TGLExt.AMontagemFinal TGLExt.OCustoDoProcesso TGLExt.ATensaoMinima
open TGLExt.AContagemDaTensaoMinima TGLExt.OMapaDosMuros

/-- [DEF] O melhor de uma lista de literais pelo custo do retorno (o primeiro no empate). -/
def oMelhorDe {n : ℕ} (φ : CNF n) (S : Espectro2 n) : List (Literal n) → Option (Literal n) :=
  fun ls => ls.foldl (fun acc l => match acc with
    | none => some l
    | some m => if custoDoRetorno φ S l < custoDoRetorno φ S m then some l else some m) none

/-- [DEF] As portas limpas de uma cláusula morta: retornáveis e não-condenadas pelo campo. -/
def limpasDe {n : ℕ} (φ : CNF n) (S : Espectro2 n) (C : Clausula n) : List (Literal n) :=
  C.filter (fun l => retornoUtil S l && !condenadaNoCampo φ S l)

/-- [DEF] A mesma literal (por campos: a estrutura não tem BEq). -/
def mesmaLiteral {n : ℕ} (q r : Literal n) : Bool := (q.var == r.var) && (q.pos == r.pos)

/-- [DEF] **O LAÇO COM A PODA**: a morte registra a melhor porta limpa e empilha o ponto de escolha (o
espectro de antes, as portas restantes); quando nenhuma porta limpa resta — o detector do
retorno-silêncio, provado correto —, o processo revisa: volta ao ponto mais recente com portas e tenta a
próxima; o ponto esgotado sobe; a pilha vazia é o silêncio honesto. -/
def lacoComPoda {n : ℕ} (φ : CNF n) :
    ℕ → Espectro2 n → (Fin n → Bool) → List (Espectro2 n × List (Literal n)) →
      (Option (Fin n → Bool)) × Espectro2 n
  | 0, S, x, _ => (if custoCNF φ x = 0 then some x else none, S)
  | fuel + 1, S, x, pilha =>
    let w := (descerGenerico passoFiel φ S φ.length ([], x)).2
    if custoCNF φ w = 0 then (some w, S)
    else
      match primeiraSemPar φ w with
      | none => (none, S)
      | some C =>
        match limpasDe φ S C with
        | [] =>
          match pilha with
          | [] => (none, S)
          | (S', restantes) :: pilha' =>
            match oMelhorDe φ S' restantes with
            | some r =>
              lacoComPoda φ fuel (S' ++ [(mascaraDeLiteral r, valorDeLiteral r)])
                (ressurreicaoConstrutiva (S' ++ [(mascaraDeLiteral r, valorDeLiteral r)]))
                ((S', restantes.filter (fun q => !mesmaLiteral q r)) :: pilha')
            | none =>
              lacoComPoda φ fuel S' (ressurreicaoConstrutiva S') pilha'
        | limpas =>
          match oMelhorDe φ S limpas with
          | some l =>
            lacoComPoda φ fuel (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])
              (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]))
              ((S, limpas.filter (fun q => !mesmaLiteral q l)) :: pilha)
          | none => (none, S)

/-- [DEF] O processo com a poda: o campo fala o fecho; o laço com a poda corre da pilha vazia. -/
def processoComPoda {n : ℕ} (φ : CNF n) (fuel : ℕ) (x : Fin n → Bool) :=
  let S₀ := fechoDoCampo φ (n + 2) []
  lacoComPoda φ fuel S₀ (ressurreicaoConstrutiva S₀) []

/-- ★★ **O ECO DA PODA É FIEL**: o laço com a poda só responde com ponto de custo zero — nunca inventa
luz, revise o que revisar. -/
theorem o_eco_da_poda_e_fiel {n : ℕ} (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x w : Fin n → Bool) (pilha : List (Espectro2 n × List (Literal n)))
      (S' : Espectro2 n),
      lacoComPoda φ fuel S x pilha = (some w, S') → custoCNF φ w = 0 := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x w pilha S' h
    unfold lacoComPoda at h
    rw [Prod.mk.injEq] at h
    obtain ⟨h1, -⟩ := h
    split at h1
    · rename_i hcond
      have hw := Option.some.inj h1
      subst hw
      exact hcond
    · simp at h1
  | succ f ih =>
    intro S x w pilha S' h
    unfold lacoComPoda at h
    dsimp only [] at h
    split at h
    · rename_i hcond
      rw [Prod.mk.injEq] at h
      obtain ⟨h1, -⟩ := h
      have hw := Option.some.inj h1
      subst hw
      exact hcond
    · split at h
      · simp at h
      · split at h
        · -- o detector: a poda
          split at h
          · simp at h
          · split at h
            · exact ih _ _ _ _ _ h
            · exact ih _ _ _ _ _ h
        · split at h
          · exact ih _ _ _ _ _ h
          · simp at h

/-- ★★ O laço com a poda responde com eco na armadilha do empate, em `phiCE` e na fórmula de paridade par; em cada uma
delas uma das regras locais responde com silêncio. -/
theorem a_medida_a_poda_atravessa_os_tres_muros :
    (processoComPoda phiTrap 10 (fun _ => false)).1.isSome = true
    ∧ (processoComPoda phiCE 10 (fun _ => false)).1.isSome = true
    ∧ (processoComPoda phiFofa2 10 (fun _ => false)).1.isSome = true := by
  decide

/-- ★★ **A PODA É HONESTA E SÃ**: a insatisfazível cala (a pilha esgota — o campo inteiro varrido), e a
sanidade segue (a fofa, a cadeia, a phiMedida). -/
theorem a_medida_a_poda_e_honesta :
    (processoComPoda ([[⟨0, true⟩], [⟨0, false⟩]] : CNF 1) 10 (fun _ => false)).1 = none
    ∧ (processoComPoda phiFofa 10 (fun _ => false)).1.isSome = true
    ∧ (processoComPoda phiCadeia 10 (fun _ => false)).1.isSome = true := by
  decide

end TGLExt.APodaFechaOContorno

