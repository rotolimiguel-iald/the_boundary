import TGLExt.AReversao
import TGLExt.ALeituraCorrespondente

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# O algoritmo: a busca com a poda e o livro, e a luz que ela nunca perde

## O que é

`oAlgoritmo φ fuel x` é a busca com a poda e o livro (`processoComPodaM`, de `ADescidaPinadaEmMemoria`), com o nome do
algoritmo: o fecho do campo, a descida fiel até o poço, a morte que registra a porta limpa de tensão mínima e empilha o ponto de
escolha, o detector do retorno que dispara a revisão, o livro das portas. Ela responde com um ponto (o eco) ou com nada (o
silêncio). O combustível `fuel` é o número de passos que ela pode dar.

## O que fica provado [REAL]

1. `serveEq_de_vale`, `serveEspectro2_append_de_vale`: a equação de uma literal vale no ponto em que a literal vale.
2. `a_primeira_sem_par_mem`, `a_primeira_sem_par_existe`: havendo custo, a primeira cláusula falsificada existe e é da fórmula.
3. `o_melhor_de_mem`, `o_melhor_de_nao_vazio`, `o_melhor_de_none`: a escolha da tensão mínima devolve um membro da lista, e só
   falha na lista vazia.
4. `mesmaLiteral_vale`, `a_porta_viva_sobrevive_ao_filtro`: a porta que vale na solução atravessa o filtro.
5. ★ `a_porta_da_luz_nas_limpas`: se uma solução serve o espectro e uma cláusula falha no poço, alguma literal dessa cláusula é
   limpa (retornável e não condenada) e vale na solução.
6. ★★ `a_porta_limpa_mantem_a_casa`: um espectro habitado (algum ponto o serve) que recebe uma porta retornável continua
   habitado.
7. `aLuzDoAlgoritmo` [DEF]: a luz é uma solução que serve o espectro corrente, ou que serve um ponto de escolha da pilha com
   uma porta restante que vale nela. `EstadoDaLuz` [DEF]: a luz, o poço servindo o espectro, o espectro habitado e a pilha
   sadia. `a_luz_exige_limpa`: com a luz no espectro e uma cláusula falhando no poço, a lista das portas limpas não é vazia;
   logo a revisão e a subida, que pedem essa lista vazia, só acontecem sem a luz no espectro corrente.
8. ★★★ `a_luz_nunca_cala_estacionada`: sob `EstadoDaLuz`, se o laço responde com silêncio, então esse silêncio é o do
   combustível zero, alcançado num estado em que a luz ainda está presente. A prova mantém o invariante nos quatro movimentos
   (o eco, a morte, a revisão e a subida).
9. ★★ `o_algoritmo_nunca_mede_a_escuridao`: se a fórmula tem solução, todo silêncio do algoritmo é o do combustível zero, com a
   luz presente: o algoritmo nunca responde silêncio por ter esgotado as portas.
10. ★★ `o_algoritmo_ou_ecoa_ou_guarda_a_luz`: se a fórmula tem solução, o algoritmo ou responde com um ponto que a satisfaz, ou
    cala no combustível zero com a luz presente.
11. ★★ `o_algoritmo_so_le_com_correspondencia`: se o algoritmo responde com um ponto, o ponto satisfaz a fórmula.
12. ★★ `a_medida_o_algoritmo_decide`: por `decide`, o algoritmo responde com um ponto em sete fórmulas com solução (`phiTrap`,
    `phiCE`, `phiFofa2` e `phiCadeia`, com combustível 12; as armadilhas encadeadas de profundidade dois e três, com 20 e 30;
    `phiFofa`, com 12) e cala na fórmula sem solução de uma variável.

## O que NÃO se afirma

* Que algum combustível finito baste para o eco em toda fórmula com solução (a terminação): não está provado. O livro registra
  as portas, mas a busca não o consulta, e uma nova morte no mesmo ponto é possível na dinâmica.
* Que combustível polinomial no tamanho da fórmula baste: segue [OPEN]. É a cota que separa esta busca da máquina de Turing sem
  o leitor.
* Que o silêncio decida a falta de solução: o algoritmo nunca responde «não há solução»; na fórmula sem solução ele cala em todo
  combustível, e calar com combustível finito só decidiria se a cota acima estivesse provada.
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.OAlgoritmoPenP

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental TGLExt.OPostoDoEscalonado
open TGLExt.AContagemDaTensao TGLExt.AMontagemFinal TGLExt.OCustoDoProcesso TGLExt.ATensaoMinima
open TGLExt.AContagemDaTensaoMinima TGLExt.OMapaDosMuros TGLExt.APodaFechaOContorno TGLExt.AMedidaDaArvore
open TGLExt.AFamiliaProfunda TGLExt.ADescidaPinadaEmMemoria TGLExt.ALeituraCorrespondente

/-- [DEF] **O ALGORITMO** — a máquina final da casa: o campo fala o fecho; a descida fiel estaciona; a
morte registra a porta de tensão mínima e empilha o ponto de escolha; o detector dispara a revisão; o
livro guarda as portas. -/
def oAlgoritmo {n : ℕ} (φ : CNF n) (fuel : ℕ) (x : Fin n → Bool) := processoComPodaM φ fuel x

/-- A equação da literal é o valor dela: servir a equação é a literal valer. -/
theorem serveEq_de_vale {n : ℕ} (l : Literal n) (y : Fin n → Bool)
    (hly : l.vale y = true) : serveEq y (mascaraDeLiteral l, valorDeLiteral l) := by
  show paridade2 y (mascaraDeLiteral l) = valorDeLiteral l
  rw [paridade2_mascaraLiteral]
  unfold valorDeLiteral
  unfold Literal.vale at hly
  rw [beq_iff_eq] at hly
  rw [hly]

/-- Servir o espectro e a literal valer: o espectro acrescentado é servido. -/
theorem serveEspectro2_append_de_vale {n : ℕ} (S : Espectro2 n) (y : Fin n → Bool) (l : Literal n)
    (hys : serveEspectro2 y S = true) (hly : l.vale y = true) :
    serveEspectro2 y (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]) = true := by
  rw [serveEspectro2_append]
  have h1 : serveEspectro2 y [(mascaraDeLiteral l, valorDeLiteral l)] = true := by
    unfold serveEspectro2
    rw [List.all_cons, List.all_nil]
    show ((paridade2 y (mascaraDeLiteral l) == valorDeLiteral l) && true) = true
    rw [beq_iff_eq.mpr (serveEq_de_vale l y hly)]
    rfl
  rw [h1, hys]
  rfl

/-- A primeira cláusula sem par é membro da fórmula e falsificada no ponto. -/
theorem a_primeira_sem_par_mem {n : ℕ} (φ : CNF n) (x : Fin n → Bool) (C : Clausula n)
    (h : primeiraSemPar φ x = some C) : C ∈ φ ∧ satisfeita C x = false := by
  unfold primeiraSemPar at h
  obtain ⟨hmem, hpred⟩ := encontra_mem _ _ _ h
  refine ⟨hmem, ?_⟩
  have hpred' : (!satisfeita C x) = true := hpred
  cases hs : satisfeita C x with
  | false => rfl
  | true => rw [hs] at hpred'; simp at hpred'

/-- Havendo custo, a primeira cláusula sem par existe. -/
theorem a_primeira_sem_par_existe {n : ℕ} (φ : CNF n) (x : Fin n → Bool)
    (h : custoCNF φ x ≠ 0) : (primeiraSemPar φ x).isSome = true := by
  have hpos : 0 < custoCNF φ x := Nat.pos_of_ne_zero h
  unfold custoCNF at hpos
  rw [List.countP_pos_iff] at hpos
  obtain ⟨C₀, hC₀φ, hC₀sat⟩ := hpos
  unfold primeiraSemPar
  rw [List.find?_isSome]
  exact ⟨C₀, hC₀φ, hC₀sat⟩

/-- A escolha da tensão mínima é membro da lista. -/
theorem o_melhor_de_mem {n : ℕ} (φ : CNF n) (S : Espectro2 n) (ls : List (Literal n)) (r : Literal n)
    (h : oMelhorDe φ S ls = some r) : r ∈ ls := by
  unfold oMelhorDe at h
  have hm := foldl_some_mem φ S ls none r h
  rcases hm with hnone | hmem
  · simp at hnone
  · exact hmem

/-- A escolha nunca trava em lista viva. -/
theorem o_melhor_de_nao_vazio {n : ℕ} (φ : CNF n) (S : Espectro2 n) (ls : List (Literal n))
    (hne : ls ≠ []) : (oMelhorDe φ S ls).isSome = true := by
  obtain ⟨l₀, resto, hl₀⟩ := List.exists_cons_of_ne_nil hne
  unfold oMelhorDe
  rw [hl₀, List.foldl_cons]
  suffices h : ∀ (resto : Clausula n) (acc : Option (Literal n)), acc.isSome = true →
      ((resto.foldl (fun acc l => match acc with
          | none => some l
          | some m => if custoDoRetorno φ S l < custoDoRetorno φ S m then some l else some m)
        acc).isSome = true) by
    exact h resto (some l₀) rfl
  intro resto
  induction resto with
  | nil => intro acc hacc; simpa using hacc
  | cons b bs ih =>
    intro acc hacc
    rw [List.foldl_cons]
    apply ih
    split
    · rfl
    · split
      · rfl
      · rfl

/-- A escolha dá none só na lista vazia. -/
theorem o_melhor_de_none {n : ℕ} (φ : CNF n) (S : Espectro2 n) (ls : List (Literal n))
    (h : oMelhorDe φ S ls = none) : ls = [] := by
  by_contra hne
  have his := o_melhor_de_nao_vazio φ S ls hne
  rw [h] at his
  simp at his

/-- A mesma literal vale igual. -/
theorem mesmaLiteral_vale {n : ℕ} (a b : Literal n) (y : Fin n → Bool)
    (h : mesmaLiteral a b = true) : a.vale y = b.vale y := by
  unfold mesmaLiteral at h
  rw [Bool.and_eq_true] at h
  obtain ⟨hvar, hpos⟩ := h
  rw [beq_iff_eq] at hvar hpos
  unfold Literal.vale
  rw [hvar, hpos]

/-- A porta da luz atravessa o filtro da porta morta: valores diferentes, literais diferentes. -/
theorem a_porta_viva_sobrevive_ao_filtro {n : ℕ} (a b : Literal n) (y : Fin n → Bool)
    (ha : a.vale y = true) (hb : b.vale y = false) : (!mesmaLiteral a b) = true := by
  cases hm : mesmaLiteral a b with
  | false => rfl
  | true =>
    have h := mesmaLiteral_vale a b y hm
    rw [ha, hb] at h
    simp at h

/-- ★ **A PORTA DA LUZ ESTÁ NAS LIMPAS**: com a luz servida no espectro, o poço servindo e a cláusula
falhando nele, existe literal da cláusula que é limpa (retornável e não-condenada) E vale na luz. A
forma que o invariante da luz consome. -/
theorem a_porta_da_luz_nas_limpas {n : ℕ} (φ : CNF n) (S : Espectro2 n) (w : Fin n → Bool)
    (C : Clausula n) (hCφ : C ∈ φ)
    (hpoco : serveEspectro2 w S = true) (hfalha : satisfeita C w = false)
    (y : Fin n → Bool) (hy0 : custoCNF φ y = 0) (hys : serveEspectro2 y S = true) :
    ∃ l : Literal n, l ∈ limpasDe φ S C ∧ l.vale y = true := by
  obtain ⟨l, hlC, hly, hret⟩ := o_sinal_de_retorno_da_luz φ S w C hCφ hpoco hfalha y hy0 hys
  have hcond : condenadaNoCampo φ S l = false := by
    by_contra hcon
    have hcond2 : condenadaNoCampo φ S l = true := by
      cases hc : condenadaNoCampo φ S l with
      | true => rfl
      | false => exact absurd hc hcon
    have hyserve : serveEspectro2 y (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]) = true :=
      serveEspectro2_append_de_vale S y l hys hly
    exact a_condenada_nao_tem_luz φ S l hcond2 ⟨y, hy0, hyserve⟩
  refine ⟨l, ?_, hly⟩
  unfold limpasDe
  rw [List.mem_filter]
  exact ⟨hlC, by rw [Bool.and_eq_true]; exact ⟨hret, by rw [hcond]; rfl⟩⟩

/-- ★★ **A PORTA LIMPA MANTÉM A CASA** (a consistência da morte da poda): o espectro habitado que recebe
a porta retornável continua habitado. A prova: o ponto construído do escalonado acrescentado serve a
reduzida (que tem pivô, pelo retorno útil) e os pivôs reduzidos; servindo a reduzida e os pivôs, serve a
equação nova e o espectro — sem escolha, só leitura. -/
theorem a_porta_limpa_mantem_a_casa {n : ℕ} (S : Espectro2 n) (l : Literal n)
    (hcons : ∃ z : Fin n → Bool, serveEspectro2 z S = true)
    (hret : retornoUtil S l = true) :
    ∃ z : Fin n → Bool, serveEspectro2 z (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]) = true := by
  obtain ⟨y, hyS⟩ := hcons
  have hyE : serveEspectro2 y (oEscalonado S) = true := (oEscalonado_preserva S y ⟨y, hyS⟩).mpr hyS
  -- a reduzida tem pivô (o retorno útil)
  have hpi : (bitPivo (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).1).isSome
      = true := hret
  obtain ⟨q, hq⟩ := Option.isSome_iff_exists.mp hpi
  -- o ponto construído do escalonado acrescentado serve-o, incondicionalmente
  have hiso := oEscalonado_isolada (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])
  have hserve := pontoConstruido_serve _ hiso
  set p := pontoConstruido (oEscalonado (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])) with hp
  refine ⟨p, ?_⟩
  have hpE' : serveEspectro2 p (oEscalonado (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])) = true := by
    unfold serveEspectro2
    rw [List.all_eq_true]
    intro e he
    exact (beq_iff_eq).mpr (hserve e he)
  rw [oEscalonado_snoc] at hpE'
  unfold inserirEscalonado at hpE'
  dsimp only [] at hpE'
  rw [hq] at hpE'
  dsimp only [] at hpE'
  rw [serveEspectro2_cons'] at hpE'
  obtain ⟨hpr, hpmap⟩ := hpE'
  -- servindo a reduzida e os pivôs reduzidos, p serve os pivôs originais
  have hpE : serveEspectro2 p (oEscalonado S) = true :=
    (pivos_reduzidos_preserva p _ _ hpr q _).mp hpmap
  have hpS : serveEspectro2 p S = true := (oEscalonado_preserva S p ⟨y, hyS⟩).mp hpE
  have hpe : serveEq p (mascaraDeLiteral l, valorDeLiteral l) :=
    (reduzirNova_preserva p _ hpE _).mp hpr
  rw [serveEspectro2_append, hpS]
  unfold serveEspectro2
  rw [List.all_cons, List.all_nil]
  show ((paridade2 p (mascaraDeLiteral l) == valorDeLiteral l) && true) = true
  rw [beq_iff_eq.mpr hpe]
  rfl

/-- [DEF] **A LUZ DO ALGORITMO** (o invariante da luz): ou a luz serve o espectro corrente, ou mora num
ponto de escolha da pilha, com uma porta viva restante nela. A luz nunca se perde: desce pinada. -/
def aLuzDoAlgoritmo {n : ℕ} (φ : CNF n) (S : Espectro2 n)
    (pilha : List (Espectro2 n × List (Literal n))) : Prop :=
  (∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true) ∨
    (∃ e : Espectro2 n × List (Literal n), e ∈ pilha ∧
      ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y e.1 = true ∧
        ∃ l : Literal n, l ∈ e.2 ∧ l.vale y = true)

/-- A porta das limpas é retornável. (Auxiliar da sanidade da pilha.) -/
theorem retornoUtil_das_limpas {n : ℕ} (φ : CNF n) (S : Espectro2 n) (C : Clausula n) (r : Literal n)
    (h : r ∈ limpasDe φ S C) : retornoUtil S r = true := by
  unfold limpasDe at h
  have h2 := (List.mem_filter.mp h).2
  rw [Bool.and_eq_true] at h2
  exact h2.1

/-- Com a luz no espectro e o detector disparado (limpas vazia): contradição — o campo sempre deixa a
porta da luz aberta. (Auxiliar que fecha os ramos escuros: a máquina só revisa na ausência da luz.) -/
theorem a_luz_exige_limpa {n : ℕ} (φ : CNF n) (S : Espectro2 n) (w : Fin n → Bool) (C : Clausula n)
    (hpoco : serveEspectro2 w S = true) (hfalha : satisfeita C w = false) (hCφ : C ∈ φ)
    (hluz : ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true) :
    limpasDe φ S C ≠ [] := by
  intro hv
  obtain ⟨y, hy0, hys⟩ := hluz
  obtain ⟨l, hll, _⟩ := a_porta_da_luz_nas_limpas φ S w C hCφ hpoco hfalha y hy0 hys
  rw [hv] at hll
  exact List.not_mem_nil hll

/-- [DEF] **O ESTADO DA LUZ** — o invariante completo da máquina: a luz (no espectro ou na pilha); o
ponto servo fiel (o G1); a casa habitada; e as portas da pilha retornáveis nos seus pontos. -/
structure EstadoDaLuz {n : ℕ} (φ : CNF n) (S : Espectro2 n) (x : Fin n → Bool)
    (pilha : List (Espectro2 n × List (Literal n))) : Prop where
  luz : aLuzDoAlgoritmo φ S pilha
  pocoServe : serveEspectro2 x S = true
  habitada : ∃ z : Fin n → Bool, serveEspectro2 z S = true
  pilhaSadia : ∀ e : Espectro2 n × List (Literal n), e ∈ pilha →
    (∃ z : Fin n → Bool, serveEspectro2 z e.1 = true) ∧ (∀ r ∈ e.2, retornoUtil e.1 r = true)

/-- ★★★ **A LUZ NUNCA CALA ESTACIONADA** — o invariante do algoritmo: sob o estado da luz, o silêncio só
vem do fundo do combustível — e vem com a luz intacta (o invariante mantido até o fim). O algoritmo nunca
mede a escuridão: ou ecoa com correspondência, ou pede mais combustível. (A manutenção do invariante em
cada um dos quatro movimentos: o eco, a morte, a revisão, a subida.) Que algum combustível finito baste
não está neste teorema. -/
theorem a_luz_nunca_cala_estacionada {n : ℕ} (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool)
      (pilha : List (Espectro2 n × List (Literal n))) (mem : List (Literal n))
      (S' : Espectro2 n) (mem' : List (Literal n)),
      EstadoDaLuz φ S x pilha →
      lacoComPodaM φ fuel S x pilha mem = (none, S', mem') →
      ∃ (S₀ : Espectro2 n) (x₀ : Fin n → Bool) (pilha₀ : List (Espectro2 n × List (Literal n)))
        (mem₀ : List (Literal n)),
        aLuzDoAlgoritmo φ S₀ pilha₀ ∧ lacoComPodaM φ 0 S₀ x₀ pilha₀ mem₀ = (none, S', mem') := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x pilha mem S' mem' hI h
    exact ⟨S, x, pilha, mem, hI.luz, h⟩
  | succ f ih =>
    intro S x pilha mem S' mem' hI h
    unfold lacoComPodaM at h
    dsimp only [] at h
    split at h
    · -- o eco: some = none — impossível
      rw [Prod.mk.injEq] at h
      obtain ⟨h1, -⟩ := h
      simp at h1
    · rename_i hcusto
      split at h
      · -- a primeiraSemPar não cala com custo fora do zero
        rename_i hsem
        exfalso
        have his := a_primeira_sem_par_existe φ _ hcusto
        rw [hsem] at his
        simp at his
      · rename_i C hC
        obtain ⟨hCφ, hCfalha⟩ := a_primeira_sem_par_mem φ _ C hC
        have hpoco : serveEspectro2 (descerGenerico passoFiel φ S φ.length ([], x)).2 S = true :=
          o_poco_fiel_serve φ S x hI.pocoServe
        split at h
        · -- o detector disparou: limpas vazia — a luz não está no espectro corrente
          rename_i hlimpas
          have hescuro : ¬ (∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true) :=
            fun hluz => a_luz_exige_limpa φ S _ C hpoco hCfalha hCφ hluz hlimpas
          split at h
          · -- a pilha vazia: a luz teria de estar no espectro — contradição
            exfalso
            cases hI.luz with
            | inl hluz => exact hescuro hluz
            | inr hna =>
              obtain ⟨e, he, -⟩ := hna
              exact absurd he List.not_mem_nil
          · -- a pilha com ponto de escolha: a revisão ou a subida
            rename_i pilha₀ S₁ rest pilha₁
            have hsadia₁ := hI.pilhaSadia (S₁, rest) List.mem_cons_self
            split at h
            · -- ★ A REVISÃO: a próxima porta do ponto de escolha
              rename_i r hr
              have hrmem : r ∈ rest := o_melhor_de_mem φ S₁ rest r hr
              have hcons' := a_porta_limpa_mantem_a_casa S₁ r hsadia₁.1 (hsadia₁.2 r hrmem)
              have hI' : EstadoDaLuz φ (S₁ ++ [(mascaraDeLiteral r, valorDeLiteral r)])
                  (ressurreicaoConstrutiva (S₁ ++ [(mascaraDeLiteral r, valorDeLiteral r)]))
                  ((S₁, rest.filter (fun q => !mesmaLiteral q r)) :: pilha₁) := by
                refine ⟨?_, ?_, ?_, ?_⟩
                · cases hI.luz with
                  | inl hluz => exact absurd hluz hescuro
                  | inr hna =>
                    obtain ⟨e, he, y, hy0, hye, l', hl'rest, hl'y⟩ := hna
                    rw [List.mem_cons] at he
                    cases he with
                    | inl heq =>
                      subst heq
                      cases hrv : r.vale y with
                      | true =>
                        exact Or.inl ⟨y, hy0, serveEspectro2_append_de_vale _ _ r hye hrv⟩
                      | false =>
                        have hsob := a_porta_viva_sobrevive_ao_filtro l' r y hl'y hrv
                        exact Or.inr ⟨(S₁, rest.filter (fun q => !mesmaLiteral q r)),
                          List.mem_cons_self, y, hy0, hye, l',
                          List.mem_filter.mpr ⟨hl'rest, hsob⟩, hl'y⟩
                    | inr htail =>
                      exact Or.inr ⟨e, List.mem_cons_of_mem _ htail, y, hy0, hye, l', hl'rest, hl'y⟩
                · exact o_solver_serve _ hcons'
                · exact hcons'
                · intro e he
                  rw [List.mem_cons] at he
                  cases he with
                  | inl heq =>
                    subst heq
                    refine ⟨hsadia₁.1, ?_⟩
                    intro r₂ hr₂
                    exact hsadia₁.2 r₂ (List.mem_filter.mp hr₂).1
                  | inr htail =>
                    exact hI.pilhaSadia e (List.mem_cons_of_mem _ htail)
              exact ih _ _ _ _ _ _ hI' h
            · -- ★ A SUBIDA: o ponto esgotado sobe — o espectro da cabeça, sem porta nova
              rename_i hmelhor
              have hrest : rest = [] := o_melhor_de_none φ S₁ rest hmelhor
              subst hrest
              have hI' : EstadoDaLuz φ S₁ (ressurreicaoConstrutiva S₁) pilha₁ := by
                refine ⟨?_, ?_, ?_, ?_⟩
                · cases hI.luz with
                  | inl hluz => exact absurd hluz hescuro
                  | inr hna =>
                    obtain ⟨e, he, y, hy0, hye, l', hl'rest, hl'y⟩ := hna
                    rw [List.mem_cons] at he
                    cases he with
                    | inl heq =>
                      subst heq
                      simp at hl'rest
                    | inr htail =>
                      exact Or.inr ⟨e, htail, y, hy0, hye, l', hl'rest, hl'y⟩
                · exact o_solver_serve _ hsadia₁.1
                · exact hsadia₁.1
                · intro e he
                  exact hI.pilhaSadia e (List.mem_cons_of_mem _ he)
              exact ih _ _ _ _ _ _ hI' h
        · -- ★ A MORTE: a porta limpa de tensão mínima, com o ponto de escolha empilhado
          rename_i hnn
          split at h
          · rename_i l hl
            have hlmem : l ∈ limpasDe φ S C := o_melhor_de_mem φ S _ l hl
            have hcons' := a_porta_limpa_mantem_a_casa S l hI.habitada
              (retornoUtil_das_limpas φ S C l hlmem)
            have hI' : EstadoDaLuz φ (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])
                (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]))
                ((S, (limpasDe φ S C).filter (fun q => !mesmaLiteral q l)) :: pilha) := by
              refine ⟨?_, ?_, ?_, ?_⟩
              · cases hI.luz with
                | inl hluz =>
                  obtain ⟨y, hy0, hys⟩ := hluz
                  obtain ⟨l', hl'limpas, hl'y⟩ :=
                    a_porta_da_luz_nas_limpas φ S _ C hCφ hpoco hCfalha y hy0 hys
                  cases hlv : l.vale y with
                  | true =>
                    exact Or.inl ⟨y, hy0, serveEspectro2_append_de_vale _ _ l hys hlv⟩
                  | false =>
                    have hsob := a_porta_viva_sobrevive_ao_filtro l' l y hl'y hlv
                    exact Or.inr ⟨(S, (limpasDe φ S C).filter (fun q => !mesmaLiteral q l)),
                      List.mem_cons_self, y, hy0, hys, l',
                      List.mem_filter.mpr ⟨hl'limpas, hsob⟩, hl'y⟩
                | inr hna =>
                  obtain ⟨e, he, y, hy0, hye, l', hl'rest, hl'y⟩ := hna
                  exact Or.inr ⟨e, List.mem_cons_of_mem _ he, y, hy0, hye, l', hl'rest, hl'y⟩
              · exact o_solver_serve _ hcons'
              · exact hcons'
              · intro e he
                rw [List.mem_cons] at he
                cases he with
                | inl heq =>
                  subst heq
                  refine ⟨hI.habitada, ?_⟩
                  intro r₂ hr₂
                  exact retornoUtil_das_limpas φ S C r₂ (List.mem_filter.mp hr₂).1
                | inr htail => exact hI.pilhaSadia e htail
            exact ih _ _ _ _ _ _ hI' h
          · -- a escolha não trava em lista viva — impossível
            rename_i hmelhor
            exfalso
            have his := o_melhor_de_nao_vazio φ S _ hnn
            rw [hmelhor] at his
            simp at his

/-- ★★ **O ALGORITMO SÓ LÊ COM CORRESPONDÊNCIA** — a fidelidade no nome final: o eco só sai do custo
zero; a máquina nunca apresenta 1 = 0 = VERDADEIRO. -/
theorem o_algoritmo_so_le_com_correspondencia {n : ℕ} (φ : CNF n) (fuel : ℕ) (x w : Fin n → Bool)
    (h : (oAlgoritmo φ fuel x).1 = some w) : custoCNF φ w = 0 :=
  o_processo_final_so_le_com_correspondencia φ fuel x w h

/-- ★★ **O ALGORITMO NUNCA MEDE A ESCURIDÃO** — o corolário no processo: satisfazível, a máquina jamais
responde «não há luz» por veredito; todo silêncio vem do fundo do combustível, com a luz intacta. (Não
se mede a escuridão: só se mede a luz.) -/
theorem o_algoritmo_nunca_mede_a_escuridao {n : ℕ} (φ : CNF n) (fuel : ℕ) (x : Fin n → Bool)
    (hSAT : SAT φ) (S' : Espectro2 n) (mem' : List (Literal n))
    (h : processoComPodaM φ fuel x = (none, S', mem')) :
    ∃ (S₀ : Espectro2 n) (x₀ : Fin n → Bool) (pilha₀ : List (Espectro2 n × List (Literal n)))
      (mem₀ : List (Literal n)),
      aLuzDoAlgoritmo φ S₀ pilha₀ ∧ lacoComPodaM φ 0 S₀ x₀ pilha₀ mem₀ = (none, S', mem') := by
  obtain ⟨y, hy⟩ := hSAT
  have hy0 : custoCNF φ y = 0 := (custoCNF_zero_sse φ y).mpr hy
  have h0 : serveEspectro2 y ([] : Espectro2 n) = true := by unfold serveEspectro2; simp
  have hyfecho : serveEspectro2 y (fechoDoCampo φ (n + 2) []) = true :=
    o_fecho_e_gratuito φ y hy0 (n + 2) [] h0
  have hI₀ : EstadoDaLuz φ (fechoDoCampo φ (n + 2) [])
      (ressurreicaoConstrutiva (fechoDoCampo φ (n + 2) [])) [] :=
    ⟨Or.inl ⟨y, hy0, hyfecho⟩, o_solver_serve _ ⟨y, hyfecho⟩, ⟨y, hyfecho⟩,
      fun e he => absurd he List.not_mem_nil⟩
  unfold processoComPodaM at h
  exact a_luz_nunca_cala_estacionada φ _ _ _ _ _ _ _ hI₀ h

/-- ★★ **O ALGORITMO: OU O ECO COM CORRESPONDÊNCIA, OU O SILÊNCIO COM A LUZ INTACTA** — a disjunção
final: havendo luz, o algoritmo ou a encontra (e o ponto serve o campo), ou cala só por combustível,
guardando a luz. A leitura é correspondente e o silêncio nunca é
veredito. -/
theorem o_algoritmo_ou_ecoa_ou_guarda_a_luz {n : ℕ} (φ : CNF n) (fuel : ℕ) (x : Fin n → Bool)
    (hSAT : SAT φ) :
    (∃ w : Fin n → Bool, (oAlgoritmo φ fuel x).1 = some w ∧ custoCNF φ w = 0) ∨
      (∃ (S' : Espectro2 n) (mem' : List (Literal n)), oAlgoritmo φ fuel x = (none, S', mem') ∧
        ∃ (S₀ : Espectro2 n) (x₀ : Fin n → Bool) (pilha₀ : List (Espectro2 n × List (Literal n)))
          (mem₀ : List (Literal n)),
          aLuzDoAlgoritmo φ S₀ pilha₀ ∧ lacoComPodaM φ 0 S₀ x₀ pilha₀ mem₀ = (none, S', mem')) := by
  cases hr : (oAlgoritmo φ fuel x).1 with
  | some w => exact Or.inl ⟨w, rfl, o_algoritmo_so_le_com_correspondencia φ fuel x w hr⟩
  | none =>
    right
    obtain ⟨R, hR⟩ : ∃ R, oAlgoritmo φ fuel x = R := ⟨_, rfl⟩
    obtain ⟨r, Sm⟩ := R
    obtain ⟨S₂, mem₂⟩ := Sm
    rw [hR] at hr
    dsimp only [] at hr
    subst hr
    unfold oAlgoritmo at hR
    exact ⟨S₂, mem₂, hR, o_algoritmo_nunca_mede_a_escuridao φ fuel x hSAT S₂ mem₂ hR⟩

/-- ★★ **A MEDIDA FINAL: O ALGORITMO DECIDE** — os três muros, a família profunda, a sanidade e a
honestidade, tudo re-medido no nome final. -/
theorem a_medida_o_algoritmo_decide :
    (oAlgoritmo phiTrap 12 (fun _ => false)).1.isSome = true
    ∧ (oAlgoritmo phiCE 12 (fun _ => false)).1.isSome = true
    ∧ (oAlgoritmo phiFofa2 12 (fun _ => false)).1.isSome = true
    ∧ (oAlgoritmo phiTrap2 20 (fun _ => false)).1.isSome = true
    ∧ (oAlgoritmo phiTrap3 30 (fun _ => false)).1.isSome = true
    ∧ (oAlgoritmo phiCadeia 12 (fun _ => false)).1.isSome = true
    ∧ (oAlgoritmo phiFofa 12 (fun _ => false)).1.isSome = true
    ∧ (oAlgoritmo ([[⟨0, true⟩], [⟨0, false⟩]] : CNF 1) 12 (fun _ => false)).1 = none := by
  decide

end TGLExt.OAlgoritmoPenP

