import TGLExt.AMontagemFinal

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# O custo do processo: o fecho é finito e o processo registra no máximo 2n equações

## O que é

A contagem das equações que o processo da tensão registra. O laço registra no máximo n equações (cada registro gasta um pivô
novo); o fecho do campo também só registra literais livres, isto é, de eixo novo, e por isso também não passa de n. O custo da
leitura da fórmula não passa do tamanho dela.

## O que fica provado [REAL]

1. `custoCNF_le_length`: o custo da leitura da fórmula não passa do número de cláusulas.
2. ★ `passoDePropagacao_nova`: a literal propagada é sempre pivô novo (nem proibida nem implicada pelo espectro).
3. ★ `o_fecho_conta`: o fecho do campo registra no máximo n leituras forçadas.
4. ★★ `o_processo_registra_no_maximo_2n`: o processo inteiro registra no máximo 2n equações, o fecho mais o laço, cada um
   limitado pelo posto.

## O que NÃO se afirma

* O custo de cada passo na fita da máquina padrão: segue [OPEN].
* Que o processo responda com eco em toda fórmula que tem solução.
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.OCustoDoProcesso

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental TGLExt.OPostoDoEscalonado
open TGLExt.AContagemDaTensao TGLExt.AMontagemFinal

variable {n : ℕ}

/-- O custo da leitura da fórmula não passa do tamanho dela. -/
theorem custoCNF_le_length {n : ℕ} (φ : CNF n) (x : Fin n → Bool) :
    custoCNF φ x ≤ φ.length := by
  unfold custoCNF
  exact List.countP_le_length

/-- ★ **A LITERAL PROPAGADA É PIVÔ NOVO**: o passo de propagação só devolve literais livres — nem
proibidas nem implicadas — logo a sua equação cresce o posto de um. -/
theorem passoDePropagacao_nova {n : ℕ} (S : Espectro2 n) (C : Clausula n) (l : Literal n)
    (h : passoDePropagacao S C = some l) : retornoUtil S l = true := by
  unfold passoDePropagacao at h
  split at h
  · simp at h
  · rename_i hnimpl
    split at h
    · rename_i l' hfiltro
      have hll : l' = l := Option.some.inj h
      rw [hll] at hfiltro
      -- l é livre (não proibida): do filtro
      have hnl : (!literalProibida S l) = true := by
        have hm : l ∈ C.filter (fun l => !literalProibida S l) := by
          rw [hfiltro]; exact List.mem_cons_self
        exact (List.mem_filter.mp hm).2
      -- e não implicada: o campo não tinha nenhuma
      have hanyf : C.any (fun l => literalImplicada S l) = false :=
        Bool.eq_false_iff.mpr hnimpl
      have hnimpl : literalImplicada S l = false := by
        rw [List.any_eq_false] at hanyf
        have hnlC : l ∈ C := by
          have hm : l ∈ C.filter (fun l => !literalProibida S l) := by
            rw [hfiltro]; exact List.mem_cons_self
          exact (List.mem_filter.mp hm).1
        have h2 := hanyf l hnlC
        exact Bool.eq_false_iff.mpr h2
      -- a lógica dos três estados: nem proibida nem implicada ⟹ pivô existe
      unfold retornoUtil
      cases hb : bitPivo (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).1 with
      | some i => rfl
      | none =>
        -- a proibida daria v = 1; a implicada daria v = 0
        have hpro : literalProibida S l = false := by
          cases hp : literalProibida S l with
          | false => rfl
          | true => simp [hp] at hnl
        have himp : literalImplicada S l = false := hnimpl
        unfold literalProibida at hpro
        unfold literalImplicada at himp
        unfold reducaoLiteral at hpro himp
        dsimp only [] at hpro himp
        rw [hb] at hpro himp
        simp only [Option.isNone_none, Bool.true_and] at hpro himp
        have hv1 : (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).2 ≠ 1 := by
          intro h1
          rw [h1] at hpro
          simp at hpro
        have hv0 : (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).2 ≠ 0 := by
          intro h0
          rw [h0] at himp
          simp at himp
        have hdois : ∀ v : ZMod 2, v = 0 ∨ v = 1 := by decide
        obtain hv | hv := hdois (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).2
        · exact (hv0 hv).elim
        · exact (hv1 hv).elim
    · simp at h

/-- ★ **O FECHO DO CAMPO É FINITO**: a propagação registra no máximo n leituras forçadas — cada uma é um
pivô novo. -/
theorem o_fecho_conta {n : ℕ} (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n),
      (fechoDoCampo φ fuel S).length + (oEscalonado S).length ≤ S.length + n := by
  intro fuel
  induction fuel with
  | zero =>
    intro S
    have h := oEscalonado_length_le S
    simp [fechoDoCampo]
    omega
  | succ f ih =>
    intro S
    unfold fechoDoCampo
    split
    · have h := oEscalonado_length_le S
      omega
    · rename_i l hpasso
      obtain ⟨C, hCφ, hC⟩ := passoNoCampo_mem S φ l hpasso
      have hnova := passoDePropagacao_nova S C l hC
      have hcresce := oEscalonado_cresce_um_com_retorno S l hnova
      have hih := ih (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])
      have hlen : (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]).length = S.length + 1 := by
        rw [List.length_append]; simp
      omega

/-- ★★ **O PROCESSO REGISTRA NO MÁXIMO 2n EQUAÇÕES**: o fecho do campo não passa de n, e o laço da tensão
adiciona no máximo n — a máquina inteira é finita no número de variáveis, duas vezes. -/
theorem o_processo_registra_no_maximo_2n {n : ℕ} (φ : CNF n) (fuel : ℕ) (x : Fin n → Bool) :
    ((processoDaTensao φ fuel x).2).length ≤ 2 * n := by
  unfold processoDaTensao
  dsimp only []
  have hfecho := o_fecho_conta φ (n + 2) []
  have hzero : (oEscalonado ([] : Espectro2 n)).length = 0 := rfl
  rw [hzero] at hfecho
  have hlaço := a_tensao_faz_no_maximo_n_mortes φ fuel (fechoDoCampo φ (n + 2) [])
    (ressurreicaoConstrutiva (fechoDoCampo φ (n + 2) []))
  simp at hfecho
  omega

end TGLExt.OCustoDoProcesso

