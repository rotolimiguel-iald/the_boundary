import TGLExt.AContagemDaTensaoMinima

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# O mapa dos muros: três fórmulas, duas regras

## O que é

A fórmula da armadilha do empate (`phiTrap`, quatro variáveis): as cinco cláusulas de `phiCE` mais (x₀ ∨ x₂ ∨ x₃) e
(x₀ ∨ x₂ ∨ ¬x₃). Ela tem solução, e x₀ é falso em toda solução. Os custos de retorno das duas literais de (x₀ ∨ x₁) empatam, e
a ordem da cláusula entrega a escolha a x₀. A regra da máscara com retorno (a morte transversal pela máscara da cláusula, com o
lado trocado, e no span a literal limpa) é medida nas mesmas fórmulas.

## O que fica provado [REAL] (por `decide`)

1. ★★ `a_medida_a_tensao_minima_e_refutada_pela_armadilha`: a regra da tensão mínima responde com silêncio em `phiTrap`, que
   tem solução.
2. ★★ `a_medida_a_mascara_atravessa_as_duas_armadilhas`: a regra da máscara com retorno responde com eco em `phiCE` e em
   `phiTrap`.
3. ★★ `a_medida_a_mascara_mora_na_paridade_par`: a regra da máscara responde com silêncio na fórmula de paridade par
   (`phiFofa2`), que tem solução.
4. ★★ `o_mapa_dos_muros`: a tabela inteira.

## O que NÃO se afirma

* Que alguma das regras locais medidas responda com eco nas três fórmulas: nenhuma responde.
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.OMapaDosMuros

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental TGLExt.OPostoDoEscalonado
open TGLExt.AContagemDaTensao TGLExt.AMontagemFinal TGLExt.OCustoDoProcesso TGLExt.ATensaoMinima
open TGLExt.AContagemDaTensaoMinima

/-- [MEDIDA] A armadilha do empate: o núcleo escondido guardado por ¬x₀ MAIS o pino por resolução que
força x₂ na luz e empata os custos. A luz: (falso, verdadeiro, verdadeiro, falso). -/
def phiTrap : CNF 4 :=
  [[⟨0, true⟩, ⟨1, true⟩],
   [⟨0, false⟩, ⟨1, true⟩, ⟨2, true⟩],
   [⟨0, false⟩, ⟨1, true⟩, ⟨2, false⟩],
   [⟨0, false⟩, ⟨1, false⟩, ⟨2, true⟩],
   [⟨0, false⟩, ⟨1, false⟩, ⟨2, false⟩],
   [⟨0, true⟩, ⟨2, true⟩, ⟨3, true⟩],
   [⟨0, true⟩, ⟨2, true⟩, ⟨3, false⟩]]

/-- [DEF] O laço da máscara com retorno limpo (a morte pela máscara com o filtro do campo): a morte
transversal pela máscara com o lado flipado; no span, a literal limpa. -/
def lacoMascaraERetorno {n : ℕ} (φ : CNF n) :
    ℕ → Espectro2 n → (Fin n → Bool) → (Option (Fin n → Bool)) × Espectro2 n
  | 0, S, x => (if custoCNF φ x = 0 then some x else none, S)
  | fuel + 1, S, x =>
    let w := (descerGenerico passoFiel φ S φ.length ([], x)).2
    if custoCNF φ w = 0 then (some w, S)
    else
      let a := eixoDaMorteDe φ w
      if eixoNoSpanB (S.map Prod.fst) a then
        match primeiraSemPar φ w with
        | none => (none, S)
        | some C =>
          match literalDoRetornoLimpo φ S C with
          | none => (none, S)
          | some l =>
            let S' := S ++ [(mascaraDeLiteral l, valorDeLiteral l)]
            lacoMascaraERetorno φ fuel S' (ressurreicaoConstrutiva S')
      else
        let S' := S ++ [(a, paridade2 w a + 1)]
        lacoMascaraERetorno φ fuel S' (ressurreicaoConstrutiva S')

/-- [DEF] O processo da máscara: o campo fala o fecho primeiro; o laço da máscara corre a partir daí. -/
def processoMascara {n : ℕ} (φ : CNF n) (fuel : ℕ) (x : Fin n → Bool) :=
  lacoMascaraERetorno φ fuel (fechoDoCampo φ (n + 2) [])
    (ressurreicaoConstrutiva (fechoDoCampo φ (n + 2) []))

/-- ★★ A regra da tensão mínima responde com silêncio na armadilha do empate, que tem solução: os custos empatam e a
ordem da cláusula entrega a escolha à literal que nenhuma solução aceita. -/
theorem a_medida_a_tensao_minima_e_refutada_pela_armadilha :
    (processoDaTensaoMinima phiTrap 6 (fun _ => false)).1 = none
    ∧ custoCNF phiTrap (fun i => decide (i = 1) || decide (i = 2)) = 0 := by
  decide

/-- ★★ **A MÁSCARA ATRAVESSA AS DUAS ARMADILHAS**: a regra da máscara com retorno ecoa no phiCE e no
phiTrap. -/
theorem a_medida_a_mascara_atravessa_as_duas_armadilhas :
    (processoMascara phiCE 5 (fun _ => false)).1.isSome = true
    ∧ (processoMascara phiTrap 6 (fun _ => false)).1.isSome = true := by
  decide

/-- ★★ **A MÁSCARA MORA NA PARIDADE PAR**: a mesma regra cala na fofa2 com a luz existente — o muro da
paridade par. -/
theorem a_medida_a_mascara_mora_na_paridade_par :
    (processoMascara phiFofa2 6 (fun _ => false)).1 = none
    ∧ custoCNF phiFofa2 luzFofa2 = 0 := by
  decide

/-- ★★ **O MAPA DOS MUROS**: a tabela medida inteira — cada regra tem o seu muro, e os muros são
complementares: a máscara morre na paridade par (e atravessa as armadilhas); a tensão mínima morre no
empate (e atravessa a paridade par e o núcleo escondido). -/
theorem o_mapa_dos_muros :
    ((processoMascara phiFofa2 6 (fun _ => false)).1 = none
      ∧ (processoMascara phiCE 5 (fun _ => false)).1.isSome = true
      ∧ (processoMascara phiTrap 6 (fun _ => false)).1.isSome = true)
    ∧ ((processoDaTensaoMinima phiFofa2 6 (fun _ => false)).1.isSome = true
      ∧ (processoDaTensaoMinima phiCE 5 (fun _ => false)).1.isSome = true
      ∧ (processoDaTensaoMinima phiTrap 6 (fun _ => false)).1 = none) := by
  decide

end TGLExt.OMapaDosMuros

