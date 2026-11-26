import TGLExt.ODentroOuMorto

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# O que sou é a contagem normalizada: a densidade do Nome das soluções

## O que é

No referencial absoluto, o estado é a densidade maximamente misturada (`absoluteRho`), cuja expectativa é o traço normalizado. Lido
sobre o Nome das soluções de uma fórmula, ele dá o número de soluções dividido por 2ⁿ: «o que sou» da fórmula é a sua densidade de
soluções.

## O que fica provado [REAL]

1. ★★ `o_card_do_cubo`: as atribuições de n variáveis são 2ⁿ.
2. ★★ `o_que_sou_da_formula_e_a_contagem`: «o que sou» do Nome das soluções de uma CNF é o número de soluções dividido por 2ⁿ.
3. ★★ `a_expectativa_do_absoluto_e_a_contagem`: a leitura do estado absoluto sobre o Nome das soluções é essa contagem normalizada.
4. ★★★ `a_densidade_decide`: a fórmula tem solução se e só se a contagem normalizada é diferente de zero.
5. ★★★ `a_maquina_le_a_densidade`: com combustível de pelo menos `2·aCotaDoLivro n + 1`, a busca com o livro responde com ponto se e
   só se a contagem normalizada é diferente de zero.

## O que NÃO se afirma

* Que a contagem se calcule em tempo polinomial: contar soluções é problema da classe #P [KNOWN]; nada aqui conta passos.
* Que combustível polinomial baste para a busca do item 5: a cota é exponencial; a polinomial segue [OPEN].
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AOQueSouEAContagem

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo
open TGLExt.OSinalDeRetorno TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental TGLExt.ATensaoMinima
open TGLExt.AContagemDaTensaoMinima TGLExt.OMapaDosMuros TGLExt.APodaFechaOContorno TGLExt.AMedidaDaArvore
open TGLExt.AFamiliaProfunda TGLExt.ADescidaPinadaEmMemoria TGLExt.ALeituraCorrespondente
open TGLExt.OAlgoritmoPenP TGLExt.OLivroConsultado TGLExt.ODentroOuMorto

/-- ★★ **O CARDINAL DO CUBO**: as atribuições de n variáveis são 2ⁿ. -/
theorem o_card_do_cubo (n : ℕ) : Fintype.card (Fin n → Bool) = 2 ^ n := by
  rw [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin]

/-- ★★ **O QUE SOU DA FÓRMULA É A CONTAGEM NORMALIZADA**: no referencial absoluto, o peso do Nome das
soluções de uma CNF é o número de soluções dividido por 2ⁿ. -/
theorem o_que_sou_da_formula_e_a_contagem {n : ℕ} (φ : CNF n) :
    TGLExt.Identidade.oQueSou (nomeDasSolucoes (custoCNF φ))
      = ((((Finset.univ.filter (fun y => custoCNF φ y = 0)).card : ℝ) / ((2 ^ n : ℕ) : ℝ)) : ℂ) := by
  rw [o_que_sou_das_solucoes, o_card_do_cubo]
  push_cast
  rfl

/-- ★★★ **A DENSIDADE DECIDE**: a fórmula tem solução ⟺ a contagem normalizada é diferente de zero. -/
theorem a_densidade_decide {n : ℕ} (φ : CNF n) :
    SAT φ ↔
      ((((Finset.univ.filter (fun y => custoCNF φ y = 0)).card : ℝ) / ((2 ^ n : ℕ) : ℝ)) : ℂ) ≠ 0 := by
  rw [← o_que_sou_da_formula_e_a_contagem]
  constructor
  · intro h
    obtain ⟨y, hy⟩ := h
    exact (o_tempo_cria_a_medida (custoCNF φ)).mpr ⟨y, (custoCNF_zero_sse φ y).mpr hy⟩
  · intro h
    obtain ⟨y, hy⟩ := (o_tempo_cria_a_medida (custoCNF φ)).mp h
    exact ⟨y, (custoCNF_zero_sse φ y).mp hy⟩

/-- ★★ **A EXPECTATIVA DO ABSOLUTO É A CONTAGEM**: a leitura do estado absoluto (ρ_Ψ = I/n) sobre o
Nome das soluções é o número de soluções dividido por 2ⁿ — o traço normalizado em ação. -/
theorem a_expectativa_do_absoluto_e_a_contagem {n : ℕ} (φ : CNF n) :
    TGLExt.gibbs (absoluteRho (Fin n → Bool)) (nomeDasSolucoes (custoCNF φ)).P
      = ((((Finset.univ.filter (fun y => custoCNF φ y = 0)).card : ℝ) / ((2 ^ n : ℕ) : ℝ)) : ℂ) :=
  o_que_sou_da_formula_e_a_contagem φ

/-- ★★★ **A MÁQUINA LÊ A DENSIDADE**: com o combustível da luz, a máquina do livro ecoa exatamente
quando a contagem normalizada é diferente de zero — o algoritmo lê o «o que sou» da fórmula. (O fecho
do circuito: o estado absoluto mede, a máquina decide.) -/
theorem a_maquina_le_a_densidade {n : ℕ} (φ : CNF n) (fuel : ℕ)
    (hcomb : 2 * aCotaDoLivro n + 1 ≤ fuel) :
    (processoComLivro φ fuel (fun _ => false)).1.isSome = true ↔
      ((((Finset.univ.filter (fun y => custoCNF φ y = 0)).card : ℝ) / ((2 ^ n : ℕ) : ℝ)) : ℂ) ≠ 0 := by
  constructor
  · intro h
    obtain ⟨w, hw⟩ := Option.isSome_iff_exists.mp h
    have h0 := o_veredito_e_honesto φ fuel _ w hw
    exact (a_densidade_decide φ).mp ⟨w, (custoCNF_zero_sse φ w).mp h0⟩
  · intro h
    exact o_veredito_decide_com_a_cota φ ((a_densidade_decide φ).mpr h) fuel hcomb

end TGLExt.AOQueSouEAContagem

