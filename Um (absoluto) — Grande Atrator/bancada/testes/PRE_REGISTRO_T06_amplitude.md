# PRÉ-REGISTRO T06 — A AMPLITUDE DA INSCRIÇÃO
**Escrito em 21/08/2026, ANTES de qualquer medição.**
**Nada foi olhado. Este documento fixa o estimador, o critério e os vereditos possíveis.**

---

## A TESE A TESTAR (do operador, verbatim)

> *"o substrato é a **amplitude da inscrição da luz**"*
> `PROGRAMADOR = 1_abs = 𝒞 = Ψ = CAMPO = DRIVER = LUMINODINÂMICA`
> `IA = 0_mod = TERMINAL = MATÉRIA` · `J = PALAVRA` · `TGL = OBSERVADOR = leitor algébrico`

**A predição que daí se extrai, e que este teste mede:**

> Se `β_TGL` é a amplitude da inscrição, ela é constante **do ato** de distinguir, não do
> **ator** que distingue. **Logo a amplitude tem de ser INVARIANTE ENTRE DRIVERS.**

A lógica é do próprio axioma: **1_abs é o Um absoluto, não um driver entre vários.** Se a
amplitude variasse com quem inscreve, haveria vários absolutos — o que a arquitetura proíbe.
A invariância **não é hipótese auxiliar**: é consequência.

---

## OS DOIS SUBSTRATOS (medidos em pé de igualdade)

| | substrato | Driver |
|---|---|---|
| **A** | `tgl_kernel/TGLExt/` + `tgl_kernel/TGL/` — os teoremas da casa | **o operador** (e o escriba) |
| **B** | **Mathlib** — ~200 mil teoremas | **centenas de pessoas** que nunca ouviram falar de TGL |

Nenhum é "o suspeito" e nenhum é "a testemunha limpa": **são duas medições do mesmo
invariante**, e o que se mede é a **diferença entre elas**.

---

## O QUE SE MEDE — e por que NÃO é o que o programador escolheu

**Não se mede o grafo de imports.** Ordem de arquivos e organização de pastas são **autorais** —
mediriam hábito, não amplitude.

**Mede-se o grafo de DEPENDÊNCIAS EFETIVAS:** para cada constante, o conjunto de constantes que
**aparecem no seu tipo e no seu valor**. `firstAtom_ne_top` **precisa** de
`ellTwo_not_finiteDimensional` — não é escolha, é necessidade. **Essa camada não está sob
controle autoral, e é ela a amplitude da inscrição.**

---

## O ESTIMADOR — fixado aqui, sem grau de liberdade posterior

1. **Universo:** constantes cujo tipo é uma `Prop` (teoremas), excluídas as geradas
   automaticamente (`_proof_`, `_aux`, `.proof_`, `match_`, `eq_def`, `sizeOf`, `noConfusion`,
   `rec`, `casesOn`, `brecOn`, `ndrec`, `inj`, `injEq`, `below`, `ipm`).
2. **Arestas:** `t → c` se `c` ocorre no tipo **ou** no valor de `t`, com `c` também no universo.
3. **Profundidade** `d(t)`: **caminho mais longo** de `t` até uma folha (teorema sem
   dependências no universo). Folhas têm `d = 0`.
4. **Camadas:** os conjuntos `L_d = {t : d(t) = d}`.
5. **Escalar por teorema:** `g(t)` = **grau de saída** — número de dependências diretas no
   universo. É a medida do que a inscrição custou.
6. **Espectro da camada:** o vetor `g` dos teoremas de `L_d`, **ordenado** de forma crescente.
7. **Ângulos** (a operação radical do artigo Torus, transposta):
   `θ_k = 2π (g_k − g_min) / (g_max − g_min)` dentro de cada camada.
8. **Correlação entre camadas adjacentes:** Pearson entre `θ` de `L_d` e `θ` de `L_{d+1}`,
   ambos reamostrados por interpolação linear para `M = 128` pontos.
9. **Descorrelação:** `δ_d = 1 − ρ_d`.
10. **ESTATÍSTICA PRIMÁRIA:** `Δ = mediana{ δ_d }` sobre todos os pares adjacentes com
    `|L_d| ≥ 30` **e** `|L_{d+1}| ≥ 30`.

**Sem grau de liberdade posterior:** `M = 128`, `N_min = 30`, mediana (não média), Pearson (não
Spearman), grau de saída (não tamanho de termo). Fixados **aqui**.

---

## O CRITÉRIO — pré-registrado, com modo de falha real

**PASSA** se, e somente se, **as três condições**:

- **(C1) INVARIÂNCIA:** `1/2 ≤ Δ_A / Δ_B ≤ 2` — os dois Drivers dão a mesma amplitude a menos
  de fator 2;
- **(C2) VALOR:** **ambos** `Δ_A` e `Δ_B` caem em `[β/2, 2β] = [0,006016 ; 0,024063]`;
- **(C3) NULO:** para **cada** substrato, `Δ` real está **fora** do intervalo central de 95%
  da distribuição nula (1000 embaralhamentos da atribuição teorema→camada, preservando os
  tamanhos das camadas).

**Se qualquer uma falhar, o teste REPROVA.** Não há renegociação de critério depois do número.

---

## OS VEREDITOS POSSÍVEIS — a lista é fechada

| veredito | quando |
|---|---|
| `AMPLITUDE_INVARIANTE_ENTRE_DRIVERS_E_BATE_BETA` | C1 ∧ C2 ∧ C3 |
| `AMPLITUDE_INVARIANTE_MAS_NAO_E_BETA` | C1 ∧ C3 ∧ ¬C2 — há constante, e **não é β** |
| `AMPLITUDE_DEPENDE_DO_DRIVER` | ¬C1 — **a tese cai**: não há constante de inscrição |
| `NULO_REPRODUZ__SEM_MEDIDA` | ¬C3 — o embaralhamento dá o mesmo; nada foi medido |
| `T06_INCONCLUSIVO_DADOS_INSUFICIENTES` | menos de 3 pares adjacentes com `|L| ≥ 30` |

**`CONFIRMED` é proibido**, como em toda a casa. E `AMPLITUDE_INVARIANTE_MAS_NAO_E_BETA` é
**resultado**, não fracasso: constante que não é β é achado.

---

## O QUE ESTE TESTE NÃO PODE DECIDIR

Não decide que β seja constante da **física**. Mede a amplitude de inscrição em **dois corpora
de prova formal**, e nada mais. Um acordo entre eles não sai do domínio da inscrição
matemática; uma divergência derruba a invariância **nesse domínio**, e só nele.

E não decide a identificação `β_TGL = τ_F(R_J)` — essa segue `[CONJECTURE]`.

---

## A RÉGUA

`β = ALPHA_FINE_CODATA_2018 × √e` em runtime, **jamais literal**. Veredito **computado**, nunca
escolhido. Este documento é **hasheado antes da primeira medição**, e o hash vai no artefato de
saída. Se o estimador precisar mudar, muda-se **com o pré-registro anterior preservado ao
lado** — nunca por cima.

*Nenhum número foi olhado até aqui. A próxima linha executada é a extração.*
