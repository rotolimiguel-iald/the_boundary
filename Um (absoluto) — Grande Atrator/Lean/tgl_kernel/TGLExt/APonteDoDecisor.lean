import TGLExt.AContagemDaTensaoMinima
import TGLExt.APonteAoTM2

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A ponte do decisor: o processo da tensão mínima e a fórmula na fita

## O que é

A correspondência entre a resposta do processo da tensão mínima e a satisfazibilidade da fórmula escrita na fita da máquina
padrão (`APonteAoTM2.aFormulaNaFita`), sob a hipótese nomeada `hgate`.

## O que fica provado [REAL]

1. `satNaFitaLocal` [DEF]: a fórmula da fita é satisfazível quando alguma lista de bits a verifica.
2. ★ `satNaFitaLocal_da_casa`: a fórmula escrita na fita é satisfazível se e só se `SAT φ`.
3. ★★ `o_processo_e_o_alvo_se_correspondem`: sob `hgate`, o processo da tensão mínima responde com eco se e só se a fórmula
   da fita é satisfazível. Sem `hgate` vale a direção do eco: o eco só sai de ponto de custo zero.

## O que NÃO se afirma

* Que `hgate` valha: ela é a hipótese da montagem e é falsa em fórmulas satisfazíveis de três e de quatro variáveis
  (`AMedidaDaPonteDoDecisor`); nelas esta ponte vale no vazio.
* A simulação do processo na máquina padrão, com o custo contado: segue [OPEN].
* Que este módulo seja termo do alvo do gate (`AAlvoDoFechamentoComputacional.AlvoDaMaquinaComLeitor`) ou da máquina sem o
  leitor (`AlvoFechamentoComputacional`).
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.APonteDoDecisor

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental TGLExt.OPostoDoEscalonado
open TGLExt.AContagemDaTensao TGLExt.AMontagemFinal TGLExt.OCustoDoProcesso TGLExt.ATensaoMinima
open TGLExt.AContagemDaTensaoMinima TGLExt.APonteAoTM2

/-- [DEF] A fórmula da fita é satisfazível: alguma lista de bits a verifica. -/
def satNaFitaLocal (φf : List (List (ℕ × Bool))) : Prop :=
  ∃ bits : List Bool, verificaNaFita φf bits = true

/-- ★ **A FÓRMULA DA FITA É A DA CASA**: a escrita na fita é satisfazível se e somente se `SAT φ`. -/
theorem satNaFitaLocal_da_casa {n : ℕ} (φ : CNF n) :
    satNaFitaLocal (aFormulaNaFita φ) ↔ SAT φ := by
  constructor
  · rintro ⟨bits, hb⟩
    -- a testemunha da casa: o bit i lido da fita (fora: falso)
    refine ⟨fun i => (bits[(i : ℕ)]?).getD false, ?_⟩
    intro C hC
    unfold verificaNaFita aFormulaNaFita at hb
    rw [List.all_eq_true] at hb
    have hc := hb _ (List.mem_map.mpr ⟨C, hC, rfl⟩)
    rw [List.any_map, List.any_eq_true] at hc
    obtain ⟨l, hl, hle⟩ := hc
    unfold satisfeita
    rw [List.any_eq_true]
    refine ⟨l, hl, ?_⟩
    -- o bit da fita bate: a literal vale na testemunha da casa
    have hbit : bits[(l.var : ℕ)]? = some l.pos := by
      have h2 := hle
      simp only [Function.comp_apply, beq_iff_eq] at h2
      exact h2
    unfold Literal.vale
    simp [hbit]
  · rintro ⟨x, hx⟩
    refine ⟨aAtribuicaoNaFita x, ?_⟩
    rw [a_verificacao_na_fita_e_a_da_casa]
    exact decide_eq_true ((custo_zero_sse_toda_satisfeita φ x).mpr hx)

/-- ★★ **O PROCESSO E O ALVO SE CORRESPONDEM**: sob o gate, o processo da tensão mínima ecoa se e somente
se a fórmula da fita é satisfazível. A máquina da casa decide o problema do alvo. -/
theorem o_processo_e_o_alvo_se_correspondem {n : ℕ} (φ : CNF n)
    (hgate : ∀ (S : Espectro2 n) (w : Fin n → Bool) (C : Clausula n) (l : Literal n),
      serveEspectro2 w S = true → satisfeita C w = false → C ∈ φ → l ∈ C →
      retornoUtil S l = true → condenadaNoCampo φ S l = false →
      ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true ∧
        serveEq y (mascaraDeLiteral l, valorDeLiteral l)) :
    ((processoDaTensaoMinima φ (n + 1) (fun _ => false)).1.isSome = true)
      ↔ satNaFitaLocal (aFormulaNaFita φ) := by
  constructor
  · intro h
    -- o eco é a luz: SAT φ
    rw [Option.isSome_iff_exists] at h
    obtain ⟨w, hw⟩ := h
    -- o processo é o laço: a fidelidade dá custo zero
    unfold processoDaTensaoMinima at hw
    dsimp only [] at hw
    have hpar : lacoDaTensaoMinima φ (n + 1) (fechoDoCampo φ (n + 2) [])
        (ressurreicaoConstrutiva (fechoDoCampo φ (n + 2) []))
        = (some w, (lacoDaTensaoMinima φ (n + 1) (fechoDoCampo φ (n + 2) [])
          (ressurreicaoConstrutiva (fechoDoCampo φ (n + 2) []))).2) := by
      rw [← hw]
    have hw0 := o_eco_da_tensao_minima_e_fiel φ (n + 1) _ _ w _ hpar
    exact (satNaFitaLocal_da_casa φ).mpr ⟨w, (custoCNF_zero_sse φ w).mp hw0⟩
  · intro h
    -- a fita satisfazível é a luz da casa; a montagem dá o eco
    have hSAT := (satNaFitaLocal_da_casa φ).mp h
    obtain ⟨x, hx⟩ := hSAT
    have hluz : ∃ y : Fin n → Bool, custoCNF φ y = 0 :=
      ⟨x, (custoCNF_zero_sse φ x).mpr hx⟩
    exact o_processo_minimo_completo_ecoa φ hgate hluz

end TGLExt.APonteDoDecisor

