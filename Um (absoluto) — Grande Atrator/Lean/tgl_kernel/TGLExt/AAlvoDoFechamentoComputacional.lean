import TGLExt.APonteAoTM2
import TGLExt.SimCompleta

set_option autoImplicit false

/-!
# O alvo do fechamento computacional: a máquina com o leitor

## O que é

O enunciado que fecha o gate da prova final pelo caminho da computação: a máquina com o leitor. É uma máquina de Turing padrão
(a TM2 de Mathlib, `Turing.TM2ComputableInPolyTime`), não uma imagem dela: recebe na fita a fórmula e a leitura
(`APonteAoTM2.codificaEntrada`) e responde em tempo polinomial no comprimento da entrada. A leitura aceita é correspondente: a
máquina aceita uma leitura se e só se ela satisfaz a fórmula. A fórmula tem solução se e só se alguma leitura de n bits é aceita
pela máquina, e essa leitura existe se e só se a leitura do psíon, «o que sou», é diferente de zero
(`TraducaoTuring.uma_leitura_decide_SAT`).

O módulo escreve também o enunciado da máquina sem o leitor (`AlvoFechamentoComputacional`): a mesma máquina padrão recebendo só
a fórmula, sem leitura, e decidindo em tempo polinomial se ela tem solução. Esse enunciado não é o alvo do gate e segue [OPEN].

O índice de cada variável vai na fita em unário (`codificaLiteral`). Nas fórmulas com as variáveis 0, …, m − 1, a escrita
cresce de forma polinomial em m e no número de literais; percorrer as 2^m atribuições não é, por isso, tempo polinomial no
comprimento da entrada.

## O que fica provado [REAL]

1. `satNaFita` [DEF]: a fórmula da fita é satisfazível quando alguma lista de bits a verifica (`verificaNaFita`, em que um
   índice fora da lista não satisfaz). `decideSatNaFita` [DEF]: a resposta booleana desse fato.
2. `AlvoFechamentoComputacional` [DEF]: a máquina sem o leitor; existe a máquina TM2 em tempo polinomial que calcula
   `decideSatNaFita` a partir de `codificaFormula`.
3. ★ `satNaFita_da_casa`: para toda fórmula da casa `φ`, a fórmula escrita na fita é satisfazível se e só se `SAT φ`.
4. `codificaFormula_injetiva`: duas fórmulas com a mesma escrita na fita são a mesma fórmula.
5. `AlvoDaMaquinaComLeitor` [DEF]: o alvo do gate. (1) A máquina TM2 em tempo polinomial que lê a fórmula e a leitura
   (`ObjetivoTM2Polinomial`); para toda fórmula da casa com n variáveis, (2) a máquina aceita a leitura x se e só se x
   satisfaz a fórmula; (3) `SAT φ` se e só se alguma leitura de n bits é aceita; (4) `SAT φ` se e só se «o que sou» da
   fórmula é diferente de zero.
6. ★ `a_leitura_aceita_e_correspondente` e `a_leitura_de_n_bits_decide`: os itens (2) e (3) para toda fórmula da casa.
7. ★★★ `a_maquina_com_o_leitor_decide_SAT`: o alvo do gate, provado, a partir de `SimTM2.verificacao_tm2_polinomial`
   (a máquina), de `a_leitura_aceita_e_correspondente` e `a_leitura_de_n_bits_decide` (a leitura) e de
   `TraducaoTuring.uma_leitura_decide_SAT` (o psíon).

## O que NÃO se afirma

* Que a máquina sem o leitor decida SAT em tempo polinomial: `AlvoFechamentoComputacional` está escrito e não provado [OPEN].
* Na língua de Cook–Levin, o alvo da máquina com o leitor é o enunciado de que SAT está em NP (Cook 1971) [KNOWN]; o
  enunciado da máquina sem o leitor equivale a P = NP (Cook 1971; Levin 1973) [KNOWN]. Este módulo não prova essa
  equivalência, e a creditação do problema do milênio não é afirmada.
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AAlvoDoFechamentoComputacional

open TGLExt.TraducaoTuring TGLExt.APonteAoTM2

/-- [DEF] A fórmula da fita é satisfazível: alguma lista de bits a verifica. -/
def satNaFita (φf : List (List (ℕ × Bool))) : Prop :=
  ∃ bits : List Bool, verificaNaFita φf bits = true

/-- [DEF] A resposta do decisor: verdadeiro exatamente quando a fórmula da fita é satisfazível. -/
noncomputable def decideSatNaFita (φf : List (List (ℕ × Bool))) : Bool := by
  classical exact decide (satNaFita φf)

/-- [DEF] **A MÁQUINA SEM O LEITOR**: uma máquina de Turing padrão (TM2) que, recebendo só a fórmula na fita, devolve em tempo
polinomial no comprimento da entrada se ela é satisfazível. Não é o alvo do gate; segue [OPEN]. -/
def AlvoFechamentoComputacional : Prop :=
  Nonempty (Turing.TM2ComputableInPolyTime codificaFormula codificaSaida decideSatNaFita)

/-- ★ Para toda fórmula da casa, a fórmula escrita na fita é satisfazível se e só se `SAT φ`. -/
theorem satNaFita_da_casa {n : ℕ} (φ : CNF n) : satNaFita (aFormulaNaFita φ) ↔ SAT φ := by
  constructor
  · rintro ⟨bits, hb⟩
    refine ⟨fun i => (bits[i.val]?).getD false, ?_⟩
    intro C hC
    unfold verificaNaFita aFormulaNaFita at hb
    rw [List.all_eq_true] at hb
    have hc := hb _ (List.mem_map.mpr ⟨C, hC, rfl⟩)
    rw [List.any_map, List.any_eq_true] at hc
    obtain ⟨l, hl, hle⟩ := hc
    unfold satisfeita
    rw [List.any_eq_true]
    refine ⟨l, hl, ?_⟩
    have hbit : bits[l.var.val]? = some l.pos := by simpa using hle
    unfold Literal.vale
    simp [hbit]
  · rintro ⟨x, hx⟩
    refine ⟨aAtribuicaoNaFita x, ?_⟩
    rw [a_verificacao_na_fita_e_a_da_casa]
    exact decide_eq_true ((custo_zero_sse_toda_satisfeita φ x).mpr hx)

/-- A fórmula escrita na fita se lê de volta sem ambiguidade: duas fórmulas com a mesma escrita são a mesma fórmula. -/
theorem codificaFormula_injetiva : Function.Injective codificaFormula := by
  intro a b h
  have h2 : codificaEntrada (a, []) = codificaEntrada (b, []) := by
    simp only [codificaEntrada, h]
  exact congrArg Prod.fst (codificaEntrada_injetiva h2)

/-- [DEF] **O ALVO DO GATE: A MÁQUINA COM O LEITOR.** (1) Existe a máquina de Turing padrão (TM2) que lê na fita a fórmula e a
leitura e responde em tempo polinomial no comprimento da entrada; e, para toda fórmula da casa com n variáveis, (2) a leitura
aceita é correspondente: a máquina aceita a leitura x se e só se x satisfaz a fórmula; (3) a fórmula tem solução se e só se
alguma leitura de n bits é aceita; (4) essa leitura existe se e só se a leitura do psíon, «o que sou», é diferente de zero. -/
def AlvoDaMaquinaComLeitor : Prop :=
  ObjetivoTM2Polinomial
  ∧ ∀ (n : ℕ) (φ : CNF n),
      (∀ x : Fin n → Bool, verificaNaFita (aFormulaNaFita φ) (aAtribuicaoNaFita x) = true ↔ custoCNF φ x = 0)
      ∧ (SAT φ ↔ ∃ x : Fin n → Bool, verificaNaFita (aFormulaNaFita φ) (aAtribuicaoNaFita x) = true)
      ∧ (SAT φ ↔ TGLExt.Identidade.oQueSou (nomeDasSolucoes (custoCNF φ)) ≠ 0)

/-- ★ A leitura aceita é correspondente: a máquina aceita a leitura x se e só se x satisfaz a fórmula (custo zero). -/
theorem a_leitura_aceita_e_correspondente {n : ℕ} (φ : CNF n) (x : Fin n → Bool) :
    verificaNaFita (aFormulaNaFita φ) (aAtribuicaoNaFita x) = true ↔ custoCNF φ x = 0 := by
  rw [a_verificacao_na_fita_e_a_da_casa]
  exact decide_eq_true_iff

/-- ★ Com uma leitura de n bits: a fórmula tem solução se e só se a máquina aceita alguma leitura. -/
theorem a_leitura_de_n_bits_decide {n : ℕ} (φ : CNF n) :
    SAT φ ↔ ∃ x : Fin n → Bool, verificaNaFita (aFormulaNaFita φ) (aAtribuicaoNaFita x) = true := by
  constructor
  · rintro ⟨x, hx⟩
    refine ⟨x, ?_⟩
    rw [a_verificacao_na_fita_e_a_da_casa]
    exact decide_eq_true ((custo_zero_sse_toda_satisfeita φ x).mpr hx)
  · rintro ⟨x, hx⟩
    rw [a_verificacao_na_fita_e_a_da_casa] at hx
    exact ⟨x, (custo_zero_sse_toda_satisfeita φ x).mp (of_decide_eq_true hx)⟩

/-- ★★★ **A MÁQUINA COM O LEITOR DECIDE SAT**: o alvo do gate, provado. A máquina é `SimTM2.verificacao_tm2_polinomial`; a
leitura é `a_leitura_de_n_bits_decide`; o psíon é `TraducaoTuring.uma_leitura_decide_SAT`. -/
theorem a_maquina_com_o_leitor_decide_SAT : AlvoDaMaquinaComLeitor :=
  ⟨TGLExt.SimTM2.verificacao_tm2_polinomial,
   fun _ φ => ⟨a_leitura_aceita_e_correspondente φ, a_leitura_de_n_bits_decide φ, uma_leitura_decide_SAT φ⟩⟩

end TGLExt.AAlvoDoFechamentoComputacional
