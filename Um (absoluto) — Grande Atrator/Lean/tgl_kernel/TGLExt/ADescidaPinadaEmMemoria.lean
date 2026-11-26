import TGLExt.AFamiliaProfunda

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A busca com a poda e o livro das portas

## O que é

A busca com a poda (`APodaFechaOContorno`) com um quarto componente no estado: o livro das portas tentadas, o registro global
de cada porta registrada, que só cresce.

## O que fica provado [REAL]

1. `lacoComPodaM`, `processoComPodaM` [DEF]: a busca com o livro.
2. ★ `lacoComPodaM_passo`: o passo ou termina com o livro intacto (eco ou silêncio), ou registra uma porta.
3. ★★ `a_memoria_so_cresce`: o livro ao fim nunca é menor que no início (por indução).
4. ★★ `a_medida_o_livro_e_fino`: o tamanho do livro ao fim, medido por `decide`: três portas em `phiTrap`, quatro e cinco nas
   armadilhas encadeadas de profundidade dois e três, uma ou duas nas fórmulas pequenas.
5. ★★ `a_medida_a_memoria_e_fiel_e_honesta`: com o livro, a busca responde com eco em `phiTrap`, `phiCE`, `phiFofa2` e
   `phiCadeia`, e com silêncio numa fórmula sem solução de uma variável (por `decide`).

## O que NÃO se afirma

* Uma cota para o tamanho do livro em toda fórmula: segue [OPEN].
* Que a busca responda com eco em toda fórmula que tem solução: segue [OPEN].
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ADescidaPinadaEmMemoria

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental TGLExt.OPostoDoEscalonado
open TGLExt.AContagemDaTensao TGLExt.AMontagemFinal TGLExt.OCustoDoProcesso TGLExt.ATensaoMinima
open TGLExt.AContagemDaTensaoMinima TGLExt.OMapaDosMuros TGLExt.APodaFechaOContorno TGLExt.AMedidaDaArvore
open TGLExt.AFamiliaProfunda

/-- [DEF] **O laço com a poda e o livro da memória**: o quarto componente do estado é o registro global
das portas tentadas — só cresce. A descida pinada em forma de memória. -/
def lacoComPodaM {n : ℕ} (φ : CNF n) :
    ℕ → Espectro2 n → (Fin n → Bool) → List (Espectro2 n × List (Literal n)) → List (Literal n) →
      (Option (Fin n → Bool)) × Espectro2 n × List (Literal n)
  | 0, S, x, _, mem => (if custoCNF φ x = 0 then some x else none, S, mem)
  | fuel + 1, S, x, pilha, mem =>
    let w := (descerGenerico passoFiel φ S φ.length ([], x)).2
    if custoCNF φ w = 0 then (some w, S, mem)
    else
      match primeiraSemPar φ w with
      | none => (none, S, mem)
      | some C =>
        match limpasDe φ S C with
        | [] =>
          match pilha with
          | [] => (none, S, mem)
          | (S', restantes) :: pilha' =>
            match oMelhorDe φ S' restantes with
            | some r =>
              lacoComPodaM φ fuel (S' ++ [(mascaraDeLiteral r, valorDeLiteral r)])
                (ressurreicaoConstrutiva (S' ++ [(mascaraDeLiteral r, valorDeLiteral r)]))
                ((S', restantes.filter (fun q => !mesmaLiteral q r)) :: pilha') (r :: mem)
            | none =>
              lacoComPodaM φ fuel S' (ressurreicaoConstrutiva S') pilha' mem
        | limpas =>
          match oMelhorDe φ S limpas with
          | some l =>
            lacoComPodaM φ fuel (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])
              (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]))
              ((S, limpas.filter (fun q => !mesmaLiteral q l)) :: pilha) (l :: mem)
          | none => (none, S, mem)

/-- [DEF] O processo com a poda e o livro: o campo fala o fecho; o laço corre do livro vazio. -/
def processoComPodaM {n : ℕ} (φ : CNF n) (fuel : ℕ) (x : Fin n → Bool) :=
  let S₀ := fechoDoCampo φ (n + 2) []
  lacoComPodaM φ fuel S₀ (ressurreicaoConstrutiva S₀) [] []

/-- ★ **O PASSO COM MEMÓRIA, DE UMA VEZ**: o laço ou termina com a memória intacta (eco ou silêncio), ou
sobe sem registrar (a subida sem porta), ou registra uma porta e segue (a morte ou a revisão). -/
theorem lacoComPodaM_passo {n : ℕ} (φ : CNF n) (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool)
    (pilha : List (Espectro2 n × List (Literal n))) (mem : List (Literal n)) :
    (∃ w : Fin n → Bool, lacoComPodaM φ (fuel + 1) S x pilha mem = (some w, S, mem))
    ∨ (∃ S₁ : Espectro2 n, lacoComPodaM φ (fuel + 1) S x pilha mem = (none, S₁, mem))
    ∨ (∃ (S₁ : Espectro2 n) (x₁ : Fin n → Bool) (pilha₁ : List (Espectro2 n × List (Literal n))),
        lacoComPodaM φ (fuel + 1) S x pilha mem = lacoComPodaM φ fuel S₁ x₁ pilha₁ mem)
    ∨ (∃ (r : Literal n) (S₁ : Espectro2 n) (x₁ : Fin n → Bool)
        (pilha₁ : List (Espectro2 n × List (Literal n))),
        lacoComPodaM φ (fuel + 1) S x pilha mem = lacoComPodaM φ fuel S₁ x₁ pilha₁ (r :: mem)) := by
  rw [lacoComPodaM.eq_def]
  dsimp only []
  split
  · left; exact ⟨_, rfl⟩
  · split
    · right; left; exact ⟨S, rfl⟩
    · split
      · split
        · right; left; exact ⟨S, rfl⟩
        · split
          · rename_i S1 rest pilha1 x2 r hr
            right; right; right
            exact ⟨r, _, _, _, rfl⟩
          · rename_i S1 rest pilha1 x2 hr
            right; right; left
            exact ⟨S1, _, _, rfl⟩
      · split
        · rename_i limpas r hr
          right; right; right
          exact ⟨r, _, _, _, rfl⟩
        · right; left; exact ⟨S, rfl⟩

/-- ★★ **A MEMÓRIA SÓ CRESCE**: o livro das portas tentadas ao fim nunca é menor que no início — a
descida pinada em forma de memória, por indução sobre o passo. -/
theorem a_memoria_so_cresce {n : ℕ} (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool)
      (pilha : List (Espectro2 n × List (Literal n))) (mem : List (Literal n)),
      ((lacoComPodaM φ fuel S x pilha mem).2.2).length ≥ mem.length := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x pilha mem
    unfold lacoComPodaM
    dsimp only []
    exact Nat.le_refl _
  | succ f ih =>
    intro S x pilha mem
    rcases lacoComPodaM_passo φ f S x pilha mem with ⟨w, h⟩ | ⟨S₁, h⟩ | ⟨S₁, x₁, pilha₁, h⟩
      | ⟨r, S₁, x₁, pilha₁, h⟩
    · rw [h]
    · rw [h]
    · rw [h]; exact ih S₁ x₁ pilha₁ mem
    · rw [h]; have hih := ih S₁ x₁ pilha₁ (r :: mem)
      rw [List.length_cons] at hih
      omega

/-- [MEDIDA] O tamanho do livro ao fim. -/
def oLivroAoFim {n : ℕ} (φ : CNF n) (fuel : ℕ) : ℕ :=
  (processoComPodaM φ fuel (fun _ => false)).2.2.length

/-- ★★ **O LIVRO É FINO**: o número de portas tentadas nos muros e na família profunda — a memória fica
fina onde o campo a guarda. -/
theorem a_medida_o_livro_e_fino :
    oLivroAoFim phiTrap 12 = 3
    ∧ oLivroAoFim phiTrap2 20 = 4
    ∧ oLivroAoFim phiTrap3 30 = 5
    ∧ oLivroAoFim phiCE 12 = 1
    ∧ oLivroAoFim phiFofa2 12 = 2
    ∧ oLivroAoFim phiFofa 12 = 1 := by
  decide

/-- ★★ **A MEMÓRIA É FIEL E HONESTA**: com o livro, o eco e o silêncio ficam certos — os três muros
ecoam, a insatisfazível cala, a cadeia resolve. -/
theorem a_medida_a_memoria_e_fiel_e_honesta :
    (processoComPodaM phiTrap 12 (fun _ => false)).1.isSome = true
    ∧ (processoComPodaM phiCE 12 (fun _ => false)).1.isSome = true
    ∧ (processoComPodaM phiFofa2 12 (fun _ => false)).1.isSome = true
    ∧ (processoComPodaM ([[⟨0, true⟩], [⟨0, false⟩]] : CNF 1) 12 (fun _ => false)).1 = none
    ∧ (processoComPodaM phiCadeia 12 (fun _ => false)).1.isSome = true := by
  decide

end TGLExt.ADescidaPinadaEmMemoria

