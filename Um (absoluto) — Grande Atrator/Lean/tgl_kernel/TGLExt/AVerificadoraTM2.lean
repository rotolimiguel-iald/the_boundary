import TGLExt.AOQueSouEAContagem
import TGLExt.APonteAoTM2
import TGLExt.ATensaoMinima

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A verificadora TM2: a máquina da casa na língua padrão

## O que é

`aVerificadora` é uma máquina de Turing de pilhas da Mathlib (`Turing.FinTM2`) com seis pilhas (entrada, saída, fórmula, testemunha,
contagem, restauração) que recebe na entrada a fórmula e a testemunha (a leitura) codificadas e escreve na saída o veredito: verdadeiro
se a testemunha satisfaz a fórmula, falso senão. O estado interno é finito (a fase, a falha da cláusula, a polaridade do marcador, o
último símbolo lido, o bit lido). O programa tem quinze fases: copia a entrada invertendo, colhe a testemunha, percorre a fórmula
invertida literal a literal (conta os ticks, lê o marcador, procura o bit, restaura a testemunha, compara, salta o resto da cláusula
satisfeita), escreve o veredito e, antes de parar, drena a fórmula e a testemunha e zera o estado, de modo que a configuração final é a
que a Mathlib exige (`haltList`). A máquina tem duas guardas: quando a testemunha se esgota antes dos ticks do índice, a fase
`esgotaContagem` drena os ticks sem deslocar nem fabricar bit; e o casamento de um literal exige o bit presente antes de comparar o
valor. Um literal cujo índice passa do comprimento da testemunha nunca casa, como em `verificaNaFita`. `correr` itera o passo;
`oVeredito` lê a saída da configuração parada, com combustível.

## O que fica provado [REAL]

1. `PV`, `Fase`, `EstadoV`, `aVerificadora`, `correr`, `oVeredito` [DEF].
2. ★★ `aAuxDaVerificadora`: as pilhas de entrada e de saída têm os alfabetos certos (`TM2ComputableAux`).
3. ★★ `a_medida_a_verificadora_decide`: por `decide`, em cinco casos, o veredito da máquina bate com a verificação da casa.

## O que NÃO se afirma

* A correção geral e o tempo da verificadora, para toda fórmula e toda testemunha: estão em `AMaquinaVerificaCompleta`
  (`a_maquina_verifica`, com combustível de no máximo L² + 20L + 10, sendo L o comprimento da entrada codificada) e em
  `AMaquinaVerificaHabitante` (`a_verificadora_em_tempo_polinomial`).
* Esta máquina recebe a leitura, não a procura: não decide SAT sem testemunha.
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AVerificadoraTM2

open TGLExt TGLExt.TraducaoTuring TGLExt.APonteAoTM2 TGLExt.ATensaoMinima Turing.TM2.Stmt Turing

/-- As pilhas da verificadora. -/
inductive PV where
  | entrada | saida | formula | testemunha | contagem | restaura
  deriving DecidableEq, Fintype

/-- As fases da verificadora. -/
inductive Fase where
  | copiaEntrada | colheTestemunha | inicioClausula | leMarcador
  | contaTicks | procuraBit | esgotaContagem | restauraTestemunha | saltaLiterais
  | escreveVerdade | escreveMentira | parada | limpaFormula | limpaTestemunha | paradaFinal
  deriving DecidableEq, Fintype

/-- O estado interno: a fase, a falha da cláusula, a polaridade do marcador, o último símbolo lido, o
bit lido na procura. -/
structure EstadoV where
  fase : Fase
  falha : Bool
  marcador : Bool
  lido : Option Simbolo
  bitLido : Option Simbolo
  deriving DecidableEq, Fintype

/-- [DEF] **A VERIFICADORA** (TM2): a máquina da casa na língua padrão. -/
def aVerificadora : Turing.FinTM2 where
  K := PV
  k₀ := PV.entrada
  k₁ := PV.saida
  Γ := fun k => match k with
    | PV.saida => Bool
    | _ => Simbolo
  Λ := Fase
  main := Fase.copiaEntrada
  σ := EstadoV
  initialState := ⟨Fase.copiaEntrada, false, false, none, none⟩
  m := fun l =>
    match l with
    | Fase.copiaEntrada =>
      pop PV.entrada (fun s b => { s with lido := b })
        (branch (fun s => s.lido.isSome)
          (push PV.formula (fun s => s.lido.getD Simbolo.zero) (goto fun _ => Fase.copiaEntrada))
          (goto fun _ => Fase.colheTestemunha))
    | Fase.colheTestemunha =>
      peek PV.formula (fun s b => { s with lido := b })
        (branch (fun s => s.lido == some Simbolo.sep)
          (pop PV.formula (fun s _ => s) (goto fun _ => Fase.inicioClausula))
          (branch (fun s => s.lido.isSome)
            (pop PV.formula (fun s _ => s)
              (push PV.testemunha (fun s => s.lido.getD Simbolo.zero)
                (goto fun _ => Fase.colheTestemunha)))
            (goto fun _ => Fase.inicioClausula)))
    | Fase.inicioClausula =>
      peek PV.formula (fun s b => { s with lido := b })
        (branch (fun s => s.lido == some Simbolo.fimF)
          (pop PV.formula (fun s _ => s) (goto fun _ => Fase.inicioClausula))
          (branch (fun s => s.lido == some Simbolo.fimC)
            (pop PV.formula (fun s _ => { s with falha := true })
              (goto fun _ => Fase.leMarcador))
            (branch (fun s => s.lido.isSome)
              (goto fun _ => Fase.leMarcador)
              (goto fun _ => Fase.escreveVerdade))))
    | Fase.leMarcador =>
      peek PV.formula (fun s b => { s with lido := b })
        (branch (fun s => s.lido == some Simbolo.tick)
          (pop PV.formula (fun s _ => s)
            (push PV.contagem (fun _ => Simbolo.tick) (goto fun _ => Fase.leMarcador)))
          (branch (fun s => s.lido == some Simbolo.litT)
            (pop PV.formula (fun s _ => ⟨Fase.procuraBit, s.falha, true, s.lido, s.bitLido⟩)
              (goto fun _ => Fase.procuraBit))
            (branch (fun s => s.lido == some Simbolo.litF)
              (pop PV.formula (fun s _ => ⟨Fase.procuraBit, s.falha, false, s.lido, s.bitLido⟩)
                (goto fun _ => Fase.procuraBit))
              (branch (fun s => s.lido == some Simbolo.fimC)
                (branch (fun s => s.falha)
                  (goto fun _ => Fase.escreveMentira)
                  (pop PV.formula (fun s _ => s) (goto fun _ => Fase.inicioClausula)))
                (branch (fun s => s.falha)
                  (goto fun _ => Fase.escreveMentira)
                  (goto fun _ => Fase.escreveVerdade))))))
    | Fase.contaTicks =>
      goto fun _ => Fase.leMarcador
    | Fase.procuraBit =>
      peek PV.contagem (fun s b => { s with lido := b })
        (branch (fun s => s.lido.isSome)
          (peek PV.testemunha (fun s b => { s with bitLido := b })
            (branch (fun s => s.bitLido.isSome)
              (pop PV.contagem (fun s _ => s)
                (pop PV.testemunha (fun s b => { s with lido := b })
                  (push PV.restaura (fun s => s.lido.getD Simbolo.zero)
                    (goto fun _ => Fase.procuraBit))))
              (goto fun _ => Fase.esgotaContagem)))
          (peek PV.testemunha (fun s b => { s with bitLido := b })
            (goto fun _ => Fase.restauraTestemunha)))
    | Fase.esgotaContagem =>
      pop PV.contagem (fun s b => { s with lido := b })
        (branch (fun s => s.lido.isSome)
          (goto fun _ => Fase.esgotaContagem)
          (goto fun _ => Fase.restauraTestemunha))
    | Fase.restauraTestemunha =>
      peek PV.restaura (fun s b => { s with lido := b })
        (branch (fun s => s.lido.isSome)
          (pop PV.restaura (fun s _ => s)
            (push PV.testemunha (fun s => s.lido.getD Simbolo.zero)
              (goto fun _ => Fase.restauraTestemunha)))
          (branch (fun s => s.bitLido.isSome && ((s.bitLido == some Simbolo.um) == s.marcador))
            (load (fun s => { s with falha := false }) (goto fun _ => Fase.saltaLiterais))
            (goto fun _ => Fase.leMarcador)))
    | Fase.saltaLiterais =>
      peek PV.formula (fun s b => { s with lido := b })
        (branch (fun s => s.lido == some Simbolo.fimC)
          (goto fun _ => Fase.inicioClausula)
          (branch (fun s => s.lido.isSome)
            (pop PV.formula (fun s _ => s) (goto fun _ => Fase.saltaLiterais))
            (goto fun _ => Fase.escreveVerdade)))
    | Fase.escreveVerdade =>
      push PV.saida (fun _ => true) (goto fun _ => Fase.parada)
    | Fase.escreveMentira =>
      push PV.saida (fun _ => false) (goto fun _ => Fase.parada)
    | Fase.parada =>
      goto fun _ => Fase.limpaFormula
    | Fase.limpaFormula =>
      pop PV.formula (fun s b => { s with lido := b })
        (branch (fun s => s.lido.isSome)
          (goto fun _ => Fase.limpaFormula)
          (goto fun _ => Fase.limpaTestemunha))
    | Fase.limpaTestemunha =>
      pop PV.testemunha (fun s b => { s with lido := b })
        (branch (fun s => s.lido.isSome)
          (goto fun _ => Fase.limpaTestemunha)
          (load (fun _ => ⟨Fase.copiaEntrada, false, false, none, none⟩)
            (goto fun _ => Fase.paradaFinal)))
    | Fase.paradaFinal => halt

/-- [DEF] O corredor: itera o passo até parar, devolvendo a configuração parada. -/
def correr (tm : Turing.FinTM2) : tm.Cfg → ℕ → Option tm.Cfg
  | cfg, 0 => some cfg
  | cfg, k + 1 => match Turing.FinTM2.step tm cfg with
    | none => some cfg
    | some c' => correr tm c' k

/-- [DEF] O veredito da verificadora com combustível: a pilha da saída da configuração parada. -/
def oVeredito (entrada : List Simbolo) (fuel : ℕ) : Option (List Bool) :=
  (correr aVerificadora (Turing.initList aVerificadora entrada) fuel).map (fun c => c.stk PV.saida)

/-- ★★ **O AUXILIAR DE ALFABETOS MONTADO**: as pilhas de entrada e saída têm os alfabetos certos —
`TM2ComputableAux` da verificadora. -/
def aAuxDaVerificadora : Turing.TM2ComputableAux Simbolo Bool where
  tm := aVerificadora
  inputAlphabet := Equiv.refl _
  outputAlphabet := Equiv.refl _

/-- ★★ **A MEDIDA: A VERIFICADORA DECIDE** — os cinco casos por `decide` na execução: o veredito da
máquina bate com a verificação da casa, nos muros e nos triviais. (Os combustíveis cobrem a limpeza
final das pilhas antes da parada.) -/
theorem a_medida_a_verificadora_decide :
    oVeredito (codificaEntrada ([[(0, true)]], [true])) 500 = some [true]
    ∧ oVeredito (codificaEntrada ([[(0, true)]], [false])) 500 = some [false]
    ∧ oVeredito (codificaEntrada ([[(0, true), (1, true)], [(0, false), (1, true)]],
        [true, true])) 500 = some [true]
    ∧ oVeredito (codificaEntrada (aFormulaNaFita phiCE,
        aAtribuicaoNaFita (fun i => decide (i = (1 : Fin 3))))) 500 = some [true]
    ∧ oVeredito (codificaEntrada (aFormulaNaFita phiCE,
        aAtribuicaoNaFita (fun _ => true : Fin 3 → Bool))) 500 = some [false] := by
  decide

end TGLExt.AVerificadoraTM2

