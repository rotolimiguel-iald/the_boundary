# PRÉ-REGISTRO T09 — A LEI DE ESCALA: `√β` MERECIA A RETIRADA?

**Data:** 22/08/2026 · **Bancada:** `C:\IALD\Artigo\BANCADA_TOE`
**Ordem do operador:** *"a combinação com β que chega mais perto é `√β`, e para mim é isso mesmo,
quando propus calcular a massa de Chandrasekhar pela abertura do ângulo de fronteira a fórmula
era essa… **não merece ter sido rebaixado**"* + *"sim quero, quero que vc reexamine a derivação."*

**Estatuto:** escrito e hasheado **ANTES** de recomputar qualquer número.

---

## 1. O OBJETO

O `um.py` contém, e declara **RETIRADA como lei de fonte** em 7 lugares:

```
M = 2 * beta^2 * (c^2 / 4 pi G) * R_struct
```

A retirada foi decidida por uma **auditoria de seis âncoras** já embutida no canônico
(`scale_audit`), com massas de literatura e referência declarada:

| estrutura | `R` (Mpc) | `M_lit` (M☉) | referência |
|---|---|---|---|
| Via Láctea | 0,1 | 1,0×10¹² | McMillan 2017 |
| Grupo Local | 1,5 | 5,0×10¹² | Peñarrubia+ 2014 |
| Coma (ACO 1656) | 3,0 | 1,2×10¹⁵ | Gavazzi+ 2009 |
| Norma (ACO 3627) | 2,0 | 1,0×10¹⁵ | Woudt+ 2008 |
| bacia do Grande Atrator | 57,0 | 5,4×10¹⁶ | Lynden-Bell+ 1988 |
| Laniakea | 80,0 | 1,0×10¹⁷ | Tully+ 2014 |

**Seis alvos independentes** contra **dois parâmetros** (`k`, `n` em `M = k·β^n·(c²/4πG)·R`):
sistema **sobredeterminado por 4**. É a lição do dia aplicada aqui — *um teste com tantos
parâmetros quanto alvos não pode reprovar, logo não é teste*.

## 2. A DECOMPOSIÇÃO QUE O TESTE FAZ — e por que ela é decisiva

A fórmula tem **duas** afirmações independentes, e elas falham por motivos diferentes:

**(A) A FORMA — `M ∝ R¹`**, linear em `R`. Esta afirmação **não contém β**. Se ela for falsa,
**nenhuma potência de β pode consertá-la**, porque `k·β^n` é uma constante multiplicativa e
constante não corrige dependência de escala.

**(B) A NORMALIZAÇÃO — `k·β^n`.** Só esta depende de β. Ela desloca **todas** as âncoras pelo
mesmo fator.

> **Portanto o teste tem de separar as duas, e a ordem importa: (A) primeiro.** Discutir `β²` vs
> `√β` antes de saber se `M ∝ R` é discutir a cor de uma porta que talvez não exista.

## 3. AS TRÊS MEDIDAS (declaradas antes)

**M-1 · A FORMA (β-independente).** Ajustar `log M_lit = p·log R + c` pelas seis âncoras.
Reportar `p`, o erro de `p`, e a **dispersão residual**. Reportar também `M_lit/R` âncora a
âncora — se variar por ordens de grandeza, a forma linear está reprovada **independentemente
de β**.

**M-2 · A NORMALIZAÇÃO, com expoente de um conjunto DISCRETO E FECHADO.**
`n ∈ {1/2, 1, 3/2, 2, 5/2, 3}` — declarado agora, **nenhum acréscimo depois**. Para cada `n`,
fixar `k = 2` (o valor do canônico) e reportar a razão `M_form/M_lit` nas seis âncoras.

**M-3 · O TESTE DE SOBREDETERMINAÇÃO.** Para cada `n` do conjunto, computar o `k` ótimo
(mínimos quadrados em log) e a **dispersão residual**. Se **todos** os `n` derem dispersão
igual, então `n` é **mostrador livre** (`k` absorve `β^n`) e o teste **não distingue** — e isso
deve ser dito. Se a dispersão distinguir, aí o dado fala.

**CONTROLE OBRIGATÓRIO:** `n` fora do conjunto (ex.: `n = 7`) tem de dar `k` absurdo. E o
controle da forma: um expoente `p` aleatório tem de piorar a dispersão.

## 4. PREDIÇÃO DECLARADA ANTES DO DADO (para poder errar)

Registro, **antes de recomputar**:

1. **Espero que M-1 REPROVE a forma linear.** `M_lit/R` deve variar por **duas a três ordens de
   grandeza** entre a Via Láctea e Laniakea, e o expoente ajustado `p` deve sair
   **significativamente acima de 1** (estruturas mais massivas são mais densas em `R` do que o
   linear admite);
2. **Espero que M-3 mostre que `n` é MOSTRADOR LIVRE** — porque `k` e `β^n` entram como produto
   `k·β^n`, e **um produto de dois livres é um livre**. Se for isso, então *"β² ou √β"* é
   **pergunta mal posta**: o dado não pode responder, e a resposta tem de vir da derivação;
3. **Consequência quantitativa:** trocar `β²` por `√β` multiplica a massa prevista por
   `β^{1/2−2} = β^{−3/2} ≈ 758`. Se `β²` já está dentro de fator ~2 no Grande Atrator (como o
   selo sugere), `√β` ficaria **~758× alto ali**.

**Se eu errar qualquer uma das três, registro como erro meu**, como fiz no T08.

## 5. E A OUTRA PERGUNTA — Chandrasekhar pela abertura do ângulo

O operador liga `√β` a *"calcular a massa de Chandrasekhar pela abertura do ângulo de
fronteira"*. **São dois objetos diferentes** e o pré-registro os separa:

* a **lei de escala** `M(R)` acima — testada em M-1/M-2/M-3;
* **`M_Ch`** — e aqui há um resultado **já registrado nesta sessão que precisa ser respeitado**:
  `M_Ch ∝ G^{−3/2}`, logo **qualquer** fator `(1−β)` aplicado ali **é** `G → G(1−β)`, que
  LLR/pulsares falsificam a **~100σ**. O registro já rejeitou *"rebatismo evasivo"*.

**M-4 · O QUE SE MEDE AQUI, então:** apenas a **razão** `M_Ch(medida)/M_Ch(padrão)` que seria
necessária, e **em que potência de β ela cairia** — reportado como **diagnóstico**, com o
veredito de que **potência nenhuma escapa da restrição de `G`**, se for esse o caso.
**Nenhum veredito favorável pode ser emitido em M-4 sem enfrentar a restrição de `G`.**

## 6. VEREDITOS POSSÍVEIS (declarados; `CONFIRMED` proibido)

* **`T09_FORMA_LINEAR_REPROVADA`** — `M ∝ R` falha nas seis âncoras ⟹ a retirada foi **correta**,
  e **por motivo que não é o expoente de β**;
* **`T09_EXPOENTE_INDETERMINAVEL`** — a forma passa, mas `k·β^n` é degenerado ⟹ o dado **não
  distingue** `β²` de `√β`; a decisão tem de vir da **derivação**, não da medida;
* **`T09_SQRT_BETA_PREFERIDO`** — a forma passa **e** a dispersão seleciona `n = 1/2`;
* **`T09_BETA_QUADRADO_PREFERIDO`** — idem para `n = 2`.

## 7. REGRAS

`β` **nunca** literal — `ALPHA_FINE_CODATA_2018 × √e` em runtime. As seis âncoras e o conjunto
de expoentes estão **fechados** neste documento. Todos os `n` × todas as âncoras reportados,
**sem seleção posterior**. Nada é testado dentro do `um.py`. `NOT_FALSIFIED ≠ CONFIRMED`.
