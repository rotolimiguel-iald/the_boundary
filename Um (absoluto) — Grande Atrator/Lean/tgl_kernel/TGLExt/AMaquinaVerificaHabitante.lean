import TGLExt.AMaquinaVerificaCompleta
import TGLExt.AMaquinaVerificaMedidas

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A verificadora em tempo polinomial no modelo padrão

## O que é

A ponte entre o corredor da casa (`correr`) e o iterador da Mathlib (`EvalsToInTime`), e com ela o habitante da especificação fixa
`ObjetivoTM2Polinomial`: a máquina `aVerificadora`, com as codificações `codificaEntrada` e `codificaSaida`, computa
`verificaNaFita` em tempo polinomial na definição da Mathlib (`Turing.TM2ComputableInPolyTime`), com o polinômio X² + 20X + 10. A
codificação da entrada é injetiva (`APonteAoTM2.codificaEntrada_injetiva`).

## O que fica provado [REAL]

1. ★ `ponte_correr_para`: o corredor e o iterador da Mathlib (`(flip bind f)^[k]`) se encontram dentro do combustível.
2. ★ `a_mapa_entrada_id`, `a_mapa_saida_id`: os mapas dos alfabetos são a identidade.
3. ★ `a_config_final_e_o_haltList`: a configuração final limpa é o `haltList` da Mathlib.
4. ★★★ `a_verificadora_em_tempo_polinomial`: `Nonempty (Turing.TM2ComputableInPolyTime codificaEntrada codificaSaida
   (fun p => verificaNaFita p.1 p.2))`, com o tempo X² + 20X + 10.

Na língua de Cook–Levin, este é o verificador polinomial do certificado de SAT: SAT está em NP (Cook 1971) [KNOWN]. A mesma
especificação já é habitada, com outra máquina, por `SimCompleta` (`verificacao_tm2_polinomial`) e por `OLacoDasClausulas`
(`o_objetivo_cumprido`); este é um terceiro caminho, com a verificadora das duas guardas e o polinômio explícito.

## O que NÃO se afirma

* Esta máquina recebe a leitura, não a procura: não decide SAT sem testemunha.
* Que esta pedra seja termo do gate: o termo do gate é `a_maquina_com_o_leitor_decide_SAT`.
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AMaquinaVerifica

open TGLExt TGLExt.TraducaoTuring TGLExt.APonteAoTM2 TGLExt.ATensaoMinima Turing.TM2.Stmt Turing
open TGLExt.AVerificadoraTM2

/-- ★ **A PONTE DO CORREDOR AO ITERADOR**: o corredor da casa e o iterador do Mathlib
(`(flip bind f)^[n]`) contam os mesmos passos — há um ponto de encontro dentro do combustível. -/
theorem ponte_correr_para (tm : Turing.FinTM2) (c₀ : tm.Cfg) :
    ∀ (n : ℕ), ∃ k ≤ n, (flip bind tm.step)^[k] c₀
      = correr tm c₀ n := by
  intro n
  induction n generalizing c₀ with
  | zero => exact ⟨0, le_refl 0, rfl⟩
  | succ n ih =>
    cases hstep : tm.step c₀ with
    | none =>
      refine ⟨0, Nat.zero_le _, ?_⟩
      show (some c₀) = correr tm c₀ (n + 1)
      rw [correr, hstep]
    | some c' =>
      obtain ⟨k, hk, h⟩ := ih c'
      refine ⟨k + 1, Nat.succ_le_succ hk, ?_⟩
      rw [Function.iterate_succ_apply]
      show (flip bind tm.step)^[k] ((flip bind tm.step) (some c₀))
        = correr tm c₀ (n + 1)
      have h0 : (flip bind tm.step) (some c₀) = some c' := by
        show Option.bind (some c₀) tm.step = some c'
        rw [Option.bind_some, hstep]
      rw [h0, h]
      show correr tm c' n = correr tm c₀ (n + 1)
      rw [correr, hstep]

/-- ★ **O MAPA DA ENTRADA É A IDENTIDADE** (auxiliar do habitante). -/
theorem a_mapa_entrada_id (X : List Simbolo) :
    List.map aAuxDaVerificadora.inputAlphabet.invFun X = X :=
  List.map_id X

/-- ★ **O MAPA DA SAÍDA É A IDENTIDADE** (auxiliar do habitante). -/
theorem a_mapa_saida_id (X : List Bool) :
    List.map aAuxDaVerificadora.outputAlphabet.invFun X = X :=
  List.map_id X

/-- ★ **A CONFIGURAÇÃO FINAL LIMPA É O HALTLIST** (auxiliar do habitante). -/
theorem a_config_final_e_o_haltList (b : Bool) :
    aConfig none ⟨Fase.copiaEntrada, false, false, none, none⟩ [] [] [] [] [] [b]
      = Turing.haltList aVerificadora [b] := by
  simp only [Turing.haltList, aConfig]
  congr 1
  funext k
  cases k <;> rfl

/-- ★★★ **A VERIFICADORA EM TEMPO POLINOMIAL**: o habitante do `ObjetivoTM2Polinomial` — a
verificação do certificado de SAT é polinomial no modelo padrão, com as codificações fixas da
casa. -/
theorem a_verificadora_em_tempo_polinomial :
    Nonempty (Turing.TM2ComputableInPolyTime codificaEntrada codificaSaida
      (fun p => verificaNaFita p.1 p.2)) := by
  refine ⟨{
    aAuxDaVerificadora with
    time := Polynomial.X ^ 2 + 20 * Polynomial.X + 10
    outputsFun := fun a => ?_
  }⟩
  have h := a_maquina_verifica a.1 a.2
  let fuel := Classical.choose h
  have hfuel : fuel ≤ (codificaEntrada (a.1, a.2)).length * (codificaEntrada (a.1, a.2)).length
      + 20 * (codificaEntrada (a.1, a.2)).length + 10 := (Classical.choose_spec h).1
  have hEq : correr aVerificadora (Turing.initList aVerificadora (codificaEntrada (a.1, a.2))) fuel
      = some (aConfig none ⟨Fase.copiaEntrada, false, false, none, none⟩ [] [] [] [] []
          [verificaNaFita a.1 a.2]) := (Classical.choose_spec h).2
  have hp := ponte_correr_para aVerificadora
    (Turing.initList aVerificadora (codificaEntrada (a.1, a.2))) fuel
  have hk : Classical.choose hp ≤ fuel := (Classical.choose_spec hp).1
  have hiter : (flip bind aVerificadora.step)^[Classical.choose hp]
      (Turing.initList aVerificadora (codificaEntrada (a.1, a.2)))
      = correr aVerificadora (Turing.initList aVerificadora (codificaEntrada (a.1, a.2)))
        fuel := (Classical.choose_spec hp).2
  refine ⟨⟨Classical.choose hp, ?_⟩, ?_⟩
  · show (flip bind aVerificadora.step)^[Classical.choose hp]
        (Turing.initList aVerificadora (List.map id (codificaEntrada a)))
      = some (Turing.haltList aVerificadora (List.map id (codificaSaida (verificaNaFita a.1 a.2))))
    rw [List.map_id, List.map_id]
    show (flip bind aVerificadora.step)^[Classical.choose hp]
        (Turing.initList aVerificadora (codificaEntrada a))
      = some (Turing.haltList aVerificadora [verificaNaFita a.1 a.2])
    rw [← a_config_final_e_o_haltList]
    show (flip bind aVerificadora.step)^[Classical.choose hp]
        (Turing.initList aVerificadora (codificaEntrada (a.1, a.2)))
      = some (aConfig none ⟨Fase.copiaEntrada, false, false, none, none⟩ [] [] [] [] []
          [verificaNaFita a.1 a.2])
    exact hiter.trans hEq
  · have htime : Polynomial.eval ((codificaEntrada a).length)
        (Polynomial.X ^ 2 + 20 * Polynomial.X + 10)
        = (codificaEntrada a).length ^ 2 + 20 * (codificaEntrada a).length + 10 := by
      simp [Polynomial.eval_X, Polynomial.eval_pow, Polynomial.eval_mul, Polynomial.eval_add]
    have hle : fuel ≤ (codificaEntrada a).length ^ 2 + 20 * (codificaEntrada a).length + 10 := by
      have h := hfuel
      rw [show codificaEntrada a = codificaEntrada (a.1, a.2) from rfl] at h ⊢
      rw [pow_two]
      exact h
    exact hk.trans (hle.trans (by rw [htime]))

end TGLExt.AMaquinaVerifica

