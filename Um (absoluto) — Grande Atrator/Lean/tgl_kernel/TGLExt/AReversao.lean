import TGLExt.ABateriaSobAMemoria1

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A reversão: o registro da busca com o livro não perde o que registra

## O que é

Na busca com a poda e o livro (`ADescidaPinadaEmMemoria`), a morte empilha o ponto de escolha (o espectro e as portas restantes)
e consigna a porta no livro; a revisão volta ao ponto e registra a porta seguinte. As duas operações acrescentam um elemento à
frente de uma lista. Esta pedra prova que esse acréscimo não perde informação: do estado de depois se lê o de antes.

## O que fica provado [REAL]

1. ★★ `a_morte_e_revertivel`: se dois registros da morte dão a mesma pilha e o mesmo livro, o espectro, as portas restantes, a
   pilha de antes, o livro de antes e a porta coincidem (a injetividade do acréscimo à frente da lista).
2. ★★ `a_revisao_e_revertivel`: o mesmo para o registro da revisão: a pilha e o livro de depois determinam o espectro ao qual se
   voltou, a pilha de antes e o livro de antes.
3. ★★ `a_maquina_nunca_perde_registro`: o enunciado da 1, com o nome da máquina.
4. ★ `a_superposicao_nao_aloja_maquina`: o espectro vazio e o livro vazio, postos à frente, não acrescentam nem retiram nada
   (`[] ++ S = S`, `[] ++ mem = mem`): o estado sem registro não tem o que desfazer.

Os três primeiros teoremas usam nenhum axioma.

## O que NÃO se afirma

* Os teoremas falam do acréscimo à pilha e ao livro, não da corrida inteira de `processoComPodaM`: que a corrida inteira seja
  revertível passo a passo (a descida e o fecho do campo incluídos) não está provado aqui.
* Que a reversibilidade limite o tempo da busca: não decorre destes teoremas, e não decorre da reversibilidade em geral, porque
  toda computação pode ser refeita de modo revertível com custo de tempo da mesma ordem [KNOWN: Bennett 1973].
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AReversao

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental TGLExt.OPostoDoEscalonado
open TGLExt.AContagemDaTensao TGLExt.AMontagemFinal TGLExt.OCustoDoProcesso TGLExt.ATensaoMinima
open TGLExt.AContagemDaTensaoMinima TGLExt.OMapaDosMuros TGLExt.APodaFechaOContorno TGLExt.AMedidaDaArvore
open TGLExt.AFamiliaProfunda TGLExt.ADescidaPinadaEmMemoria

/-- ★★ **A MORTE É REVERTÍVEL**: o passo do registro (a morte empilha o espectro de antes e consigna a
porta no livro) é injetivo: dois registros com o mesmo pós-estado tinham o mesmo pré-estado. A pilha
lembra o espectro de antes; o livro lembra a porta. -/
theorem a_morte_e_revertivel {n : ℕ}
    (S S' : Espectro2 n) (rest rest' : List (Literal n))
    (pilha pilha' : List (Espectro2 n × List (Literal n)))
    (l l' : Literal n) (mem mem' : List (Literal n))
    (hpilha : (S, rest) :: pilha = (S', rest') :: pilha')
    (hmem : l :: mem = l' :: mem') :
    S = S' ∧ pilha = pilha' ∧ mem = mem' ∧ rest = rest' ∧ l = l' := by
  have hp := List.cons.inj hpilha
  have hm := List.cons.inj hmem
  obtain ⟨h1, h2⟩ := hp
  obtain ⟨h3, h4⟩ := hm
  -- (S, rest) = (S', rest') → S = S' e rest = rest'
  have hS : S = S' := congrArg Prod.fst h1
  have hrest : rest = rest' := congrArg Prod.snd h1
  exact ⟨hS, h2, h4, hrest, h3⟩

/-- ★★ **A REVISÃO É REVERTÍVEL**: o passo da poda (volta ao ponto e registra a porta seguinte) é
injetivo: o pós determina o espectro ao que se voltou, a pilha de antes e o livro de antes. -/
theorem a_revisao_e_revertivel {n : ℕ}
    (S1 S1' : Espectro2 n) (rest rest' : List (Literal n))
    (pilha pilha' : List (Espectro2 n × List (Literal n)))
    (r r' : Literal n) (mem mem' : List (Literal n))
    (hpilha : (S1, rest) :: pilha = (S1', rest') :: pilha')
    (hmem : r :: mem = r' :: mem') :
    S1 = S1' ∧ pilha = pilha' ∧ mem = mem' := by
  have hp := List.cons.inj hpilha
  have hm := List.cons.inj hmem
  exact ⟨congrArg Prod.fst hp.1, hp.2, hm.2⟩

/-- ★★ **A MÁQUINA NUNCA PERDE O REGISTRADO**: o conteúdo registrado de cada passo é invertível — de todo
pós-estado da pilha e do livro se lê o pré-estado. (O enunciado de `a_morte_e_revertivel`; a corrida inteira não
está neste teorema.) -/
theorem a_maquina_nunca_perde_registro {n : ℕ}
    (S S' : Espectro2 n) (rest rest' : List (Literal n))
    (pilha pilha' : List (Espectro2 n × List (Literal n)))
    (l l' : Literal n) (mem mem' : List (Literal n))
    (hpilha : (S, rest) :: pilha = (S', rest') :: pilha')
    (hmem : l :: mem = l' :: mem') :
    S = S' ∧ pilha = pilha' ∧ mem = mem' ∧ rest = rest' ∧ l = l' :=
  a_morte_e_revertivel S S' rest rest' pilha pilha' l l' mem mem' hpilha hmem

/-- ★ **A SUPERPOSIÇÃO NÃO ALOJA MÁQUINA** (a forma registrada): o estado de máquina sem pilha e sem
livro não tem o que reverter: o espectro vazio e o livro vazio, postos à frente, não acrescentam nem
retiram nada. -/
theorem a_superposicao_nao_aloja_maquina {n : ℕ} (S : Espectro2 n) (mem : List (Literal n)) :
    (([] : Espectro2 n) ++ S = S)
    ∧ (([] : List (Literal n)) ++ mem = mem) := by
  exact ⟨List.nil_append S, List.nil_append mem⟩

end TGLExt.AReversao

