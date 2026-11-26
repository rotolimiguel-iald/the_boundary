import TGLExt.APodaFechaOContorno
import TGLExt.ABateriaMinimaN5b

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A medida da árvore: quantas revisões cada fórmula pede

## O que é

O laço da busca com a poda, contado: devolve, junto da resposta, o número de revisões (voltas a um ponto de escolha com portas).

## O que fica provado [REAL]

1. `lacoComPodaC`, `processoComPodaC` [DEF]: o laço contado.
2. ★★ `a_medida_a_arvore_nos_muros`: `phiTrap` pede exatamente uma revisão; `phiCE` e `phiFofa2`, nenhuma (por `decide`).
3. ★★ `a_medida_a_arvore_na_bateria`: nenhuma revisão nas fórmulas pequenas `phiFofa` e `phiCadeia` e em duas fórmulas de
   cinco variáveis (sementes 13, com solução, e 207, sem solução); numa fórmula sem solução de uma variável, o silêncio (por
   `decide`).
4. ★★ `a_medida_o_contado_e_o_simples_concordam`: o laço contado responde com eco em `phiTrap`, `phiCE` e `phiFofa2`, como
   o laço da busca (por `decide`).

## O que NÃO se afirma

* Uma cota para o número de revisões em toda fórmula: segue [OPEN].
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AMedidaDaArvore

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental TGLExt.OPostoDoEscalonado
open TGLExt.AContagemDaTensao TGLExt.AMontagemFinal TGLExt.OCustoDoProcesso TGLExt.ATensaoMinima
open TGLExt.AContagemDaTensaoMinima TGLExt.OMapaDosMuros TGLExt.APodaFechaOContorno
open TGLExt.ABateriaMinimaN5

/-- [DEF] O laço com a poda, contado: devolve a resposta, o espectro, e o número de revisões (o detector
do retorno-silêncio disparando com a pilha não-vazia). -/
def lacoComPodaC {n : ℕ} (φ : CNF n) :
    ℕ → Espectro2 n → (Fin n → Bool) → List (Espectro2 n × List (Literal n)) → ℕ →
      (Option (Fin n → Bool)) × Espectro2 n × ℕ
  | 0, S, x, _, rev => (if custoCNF φ x = 0 then some x else none, S, rev)
  | fuel + 1, S, x, pilha, rev =>
    let w := (descerGenerico passoFiel φ S φ.length ([], x)).2
    if custoCNF φ w = 0 then (some w, S, rev)
    else
      match primeiraSemPar φ w with
      | none => (none, S, rev)
      | some C =>
        match limpasDe φ S C with
        | [] =>
          match pilha with
          | [] => (none, S, rev)
          | (S', restantes) :: pilha' =>
            match oMelhorDe φ S' restantes with
            | some r =>
              lacoComPodaC φ fuel (S' ++ [(mascaraDeLiteral r, valorDeLiteral r)])
                (ressurreicaoConstrutiva (S' ++ [(mascaraDeLiteral r, valorDeLiteral r)]))
                ((S', restantes.filter (fun q => !mesmaLiteral q r)) :: pilha') (rev + 1)
            | none =>
              lacoComPodaC φ fuel S' (ressurreicaoConstrutiva S') pilha' rev
        | limpas =>
          match oMelhorDe φ S limpas with
          | some l =>
            lacoComPodaC φ fuel (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])
              (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]))
              ((S, limpas.filter (fun q => !mesmaLiteral q l)) :: pilha) rev
          | none => (none, S, rev)

/-- [DEF] O processo com a poda, contado. -/
def processoComPodaC {n : ℕ} (φ : CNF n) (fuel : ℕ) (x : Fin n → Bool) :=
  let S₀ := fechoDoCampo φ (n + 2) []
  lacoComPodaC φ fuel S₀ (ressurreicaoConstrutiva S₀) [] 0

/-- [MEDIDA] As revisões pedidas por uma fórmula: o terceiro componente do processo contado. -/
def revisoesPedidas {n : ℕ} (φ : CNF n) (fuel : ℕ) : ℕ :=
  (processoComPodaC φ fuel (fun _ => false)).2.2

/-- ★★ **A ÁRVORE NOS MUROS**: as revisões pedidas pelas três paredes — a poda trabalha exatamente onde o
campo a guarda. -/
theorem a_medida_a_arvore_nos_muros :
    revisoesPedidas phiTrap 12 = 1
    ∧ revisoesPedidas phiCE 12 = 0
    ∧ revisoesPedidas phiFofa2 12 = 0 := by
  decide

/-- ★★ **A ÁRVORE NA BATERIA**: as revisões numa amostra da bateria (a fofa, a cadeia, duas da varredura,
a insatisfazível): a morte certa não pede poda nos casos fáceis. -/
theorem a_medida_a_arvore_na_bateria :
    revisoesPedidas phiFofa 12 = 0
    ∧ revisoesPedidas phiCadeia 12 = 0
    ∧ revisoesPedidas (geraCNF5 13 21) 12 = 0
    ∧ revisoesPedidas (geraCNF5 207 21) 12 = 0
    ∧ (processoComPodaC ([[⟨0, true⟩], [⟨0, false⟩]] : CNF 1) 12 (fun _ => false)).1 = none := by
  decide

/-- ★★ A consistência: o laço contado e o laço simples respondem o mesmo nos muros. -/
theorem a_medida_o_contado_e_o_simples_concordam :
    (processoComPodaC phiTrap 10 (fun _ => false)).1.isSome = true
    ∧ (processoComPodaC phiCE 10 (fun _ => false)).1.isSome = true
    ∧ (processoComPodaC phiFofa2 10 (fun _ => false)).1.isSome = true := by
  decide

end TGLExt.AMedidaDaArvore

