import TGLExt.AMedidaDaArvore

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A família de armadilhas profundas: duas medidas

## O que é

Armadilhas encadeadas: a k-ésima porta errada ativa o seu núcleo, que o fecho do campo não condena, a porta certa vem por último
e os custos empatam para que a ordem escolha errado k vezes. As fórmulas `phiTrap2` (seis variáveis, profundidade dois) e
`phiTrap3` (oito variáveis, profundidade três), com a solução conferida.

## O que fica provado [REAL] (por `decide`)

1. ★★ `a_medida_a_profundidade_dois`: a busca com a poda responde com eco em `phiTrap2` com exatamente duas revisões.
2. ★★ `a_medida_a_profundidade_tres`: a busca responde com eco em `phiTrap3` com exatamente três revisões.

## O que NÃO se afirma

* Que o número de revisões cresça linearmente com a profundidade em geral: são duas medidas, de uma família em que as portas de
  cada ponto se esgotam.
* Uma cota para o número de revisões em toda fórmula: segue [OPEN].
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AFamiliaProfunda

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental TGLExt.OPostoDoEscalonado
open TGLExt.AContagemDaTensao TGLExt.AMontagemFinal TGLExt.OCustoDoProcesso TGLExt.ATensaoMinima
open TGLExt.AContagemDaTensaoMinima TGLExt.OMapaDosMuros TGLExt.APodaFechaOContorno TGLExt.AMedidaDaArvore

/-- [MEDIDA] A armadilha de profundidade 2: três portas na cláusula morta — x₀ ativa o núcleo A (em
x₃,x₄), x₁ ativa o núcleo B (em x₄,x₅), x₂ é a certa — mais o pino que força x₃ na luz. A luz: x₂ e x₃
acesas. -/
def phiTrap2 : CNF 6 :=
  [[⟨0, true⟩, ⟨1, true⟩, ⟨2, true⟩],
   [⟨0, false⟩, ⟨3, true⟩, ⟨4, true⟩],
   [⟨0, false⟩, ⟨3, true⟩, ⟨4, false⟩],
   [⟨0, false⟩, ⟨3, false⟩, ⟨4, true⟩],
   [⟨0, false⟩, ⟨3, false⟩, ⟨4, false⟩],
   [⟨1, false⟩, ⟨4, true⟩, ⟨5, true⟩],
   [⟨1, false⟩, ⟨4, true⟩, ⟨5, false⟩],
   [⟨1, false⟩, ⟨4, false⟩, ⟨5, true⟩],
   [⟨1, false⟩, ⟨4, false⟩, ⟨5, false⟩],
   [⟨0, true⟩, ⟨1, true⟩, ⟨3, true⟩, ⟨5, true⟩],
   [⟨0, true⟩, ⟨1, true⟩, ⟨3, true⟩, ⟨5, false⟩]]

/-- [MEDIDA] A armadilha de profundidade 3: quatro portas — x₀, x₁, x₂ ativam os seus núcleos, x₃ é a
certa — mais o pino que força x₄ na luz. A luz: x₃ e x₄ acesas. -/
def phiTrap3 : CNF 8 :=
  [[⟨0, true⟩, ⟨1, true⟩, ⟨2, true⟩, ⟨3, true⟩],
   [⟨0, false⟩, ⟨4, true⟩, ⟨5, true⟩], [⟨0, false⟩, ⟨4, true⟩, ⟨5, false⟩],
   [⟨0, false⟩, ⟨4, false⟩, ⟨5, true⟩], [⟨0, false⟩, ⟨4, false⟩, ⟨5, false⟩],
   [⟨1, false⟩, ⟨5, true⟩, ⟨6, true⟩], [⟨1, false⟩, ⟨5, true⟩, ⟨6, false⟩],
   [⟨1, false⟩, ⟨5, false⟩, ⟨6, true⟩], [⟨1, false⟩, ⟨5, false⟩, ⟨6, false⟩],
   [⟨2, false⟩, ⟨6, true⟩, ⟨7, true⟩], [⟨2, false⟩, ⟨6, true⟩, ⟨7, false⟩],
   [⟨2, false⟩, ⟨6, false⟩, ⟨7, true⟩], [⟨2, false⟩, ⟨6, false⟩, ⟨7, false⟩],
   [⟨0, true⟩, ⟨1, true⟩, ⟨2, true⟩, ⟨4, true⟩, ⟨7, true⟩],
   [⟨0, true⟩, ⟨1, true⟩, ⟨2, true⟩, ⟨4, true⟩, ⟨7, false⟩]]

/-- ★★ **A PROFUNDIDADE 2**: a luz existe; a poda ecoa com exatamente 2 revisões — uma por porta
errada. -/
theorem a_medida_a_profundidade_dois :
    custoCNF phiTrap2 (fun i => decide (i = 2) || decide (i = 3)) = 0
    ∧ (processoComPodaC phiTrap2 20 (fun _ => false)).1.isSome = true
    ∧ (processoComPodaC phiTrap2 20 (fun _ => false)).2.2 = 2 := by
  decide

/-- ★★ **A PROFUNDIDADE 3**: a luz existe; a poda ecoa com exatamente 3 revisões — a escalada é linear
nesta família. -/
theorem a_medida_a_profundidade_tres :
    custoCNF phiTrap3 (fun i => decide (i = 3) || decide (i = 4)) = 0
    ∧ (processoComPodaC phiTrap3 30 (fun _ => false)).1.isSome = true
    ∧ (processoComPodaC phiTrap3 30 (fun _ => false)).2.2 = 3 := by
  decide

end TGLExt.AFamiliaProfunda

