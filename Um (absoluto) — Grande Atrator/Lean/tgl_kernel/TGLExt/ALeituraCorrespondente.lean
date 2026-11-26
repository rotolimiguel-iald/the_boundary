import TGLExt.ABateriaSobAMemoria1

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A leitura correspondente: a busca com o livro só responde com um ponto que satisfaz a fórmula

## O que é

A resposta «há solução» da busca com a poda e o livro vem com um ponto. Ela é correspondente quando o ponto satisfaz a fórmula
(custo zero). Esta pedra prova que toda resposta com ponto é correspondente.

## O que fica provado [REAL]

1. ★★ `o_eco_da_podaM_e_fiel`: o laço com o livro só responde com ponto de custo zero.
2. ★★ `o_processo_final_so_le_com_correspondencia`: se o processo responde com um ponto, o ponto satisfaz a fórmula.
3. ★★ `a_leitura_falsa_nunca_acontece`: não há resposta com ponto de custo não nulo.

## O que NÃO se afirma

* Que, havendo solução, a busca responda com ela (a completude), nem em quanto tempo: segue [OPEN].
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ALeituraCorrespondente

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

/-- ★★ **O ECO DO LAÇO COM O LIVRO É FIEL**: só responde com ponto de custo zero. -/
theorem o_eco_da_podaM_e_fiel {n : ℕ} (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x w : Fin n → Bool)
      (pilha : List (Espectro2 n × List (Literal n))) (mem : List (Literal n))
      (S' : Espectro2 n) (mem' : List (Literal n)),
      lacoComPodaM φ fuel S x pilha mem = (some w, S', mem') → custoCNF φ w = 0 := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x w pilha mem S' mem' h
    unfold lacoComPodaM at h
    rw [Prod.mk.injEq] at h
    obtain ⟨h1, -⟩ := h
    split at h1
    · rename_i hcond
      have hw := Option.some.inj h1
      subst hw
      exact hcond
    · simp at h1
  | succ f ih =>
    intro S x w pilha mem S' mem' h
    unfold lacoComPodaM at h
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
        · split at h
          · simp at h
          · split at h
            · exact ih _ _ _ _ _ _ _ h
            · exact ih _ _ _ _ _ _ _ h
        · split at h
          · exact ih _ _ _ _ _ _ _ h
          · simp at h

/-- ★★ **A MÁQUINA FINAL SÓ LÊ COM CORRESPONDÊNCIA**: se o processo responde com o ponto, o ponto serve o
campo — a leitura é correspondente, sempre. -/
theorem o_processo_final_so_le_com_correspondencia {n : ℕ} (φ : CNF n) (fuel : ℕ) (x : Fin n → Bool)
    (w : Fin n → Bool)
    (h : (processoComPodaM φ fuel x).1 = some w) :
    custoCNF φ w = 0 := by
  unfold processoComPodaM at h
  dsimp only [] at h
  exact o_eco_da_podaM_e_fiel φ _ _ _ _ _ _ _ _ (Prod.ext_iff.mpr ⟨h, rfl⟩)

/-- ★★ **A LEITURA FALSA NUNCA ACONTECE**: não há ponto devolvido com custo não-zero — a máquina nunca
apresenta `1 = 0 = VERDADEIRO`. -/
theorem a_leitura_falsa_nunca_acontece {n : ℕ} (φ : CNF n) (fuel : ℕ) (x : Fin n → Bool) :
    ¬ ∃ w : Fin n → Bool, (processoComPodaM φ fuel x).1 = some w ∧ custoCNF φ w ≠ 0 := by
  intro ⟨w, h1, h2⟩
  exact h2 (o_processo_final_so_le_com_correspondencia φ fuel x w h1)

end TGLExt.ALeituraCorrespondente

