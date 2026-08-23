# T08 — VEREDITO FINAL, APÓS VERIFICAÇÃO ADVERSARIAL

**Data:** 22/08/2026 · **Pré-registro:** `5609d2db19cbf467ccd7292d124422b5b09d24384a4f85bfdc5617105cef9192`
**Verificação:** quatro refutadores independentes (álgebra · espaço de `P_F` · espaço de
observáveis · ceticismo adversarial) + síntese. Bruto custodiado em
`verificacao/VERIFICACAO_ADVERSARIAL_T08_bruto.json`.

> **ESTATUTO: `TGL_CORPUS_BETA_REFUTED_ON_THE_FINAL_STEP`**
>
> A **construção** do operador sobrevive **inteira e exata**. A **identificação final**
> `β = τ_F(P_F R_J P_F)` está **refutada — e agora por teorema, não por amostragem**.
>
> `REFUTED_ON_THE_FINAL_STEP` **≠ teoria refutada.** O que morreu foi **a rota de medir β
> num corpus**, não `β = α√e`. O gate matemático **não se move** — pela mesma régua que
> proíbe cosmologia de virar prova, um negativo em corpus não move o gate em nenhuma direção.

---

## PARTE I — O QUE SOBREVIVE INTACTO (a construção do operador)

Verificado por réplica independente, com código escrito do zero:

| Camada | Estatuto | Verificação |
|---|---|---|
| `Ψ = vec(M)`, `M = √p`, `‖M‖²_F = Σp = 1` ⟹ `Ψ` unitário | `[REAL]` | `\|·−1\| = 4,4e−16` |
| `Σσ_k² = 1`, `p_k = σ_k²` é distribuição legítima | `[REAL]` | `1,0000000000000004` |
| `ρ_L = MM†`, `ρ_R = M†M`, `Δ = ρ_L ⊗ ρ_R^{−T}` | `[REAL]` | construído e diagonalizado (M 9×7) |
| `Spec(Δ) = {p_i/p_j}`, `K = −log Δ`, `κ_ij = \|log(p_i/p_j)\|` | `[REAL]` | 49 autovalores, erro rel. máx `8,0e−13` |
| `JΔJ = Δ^{−1}`, `JΨ = Ψ`, `JKJ = −K` ⟹ `J\|K\|J = \|K\|` | `[REAL]` **com a correção do ERRO 5** | `‖J_schmidt Ψ − Ψ‖ = 5,35e−16` |
| `sech(κ/2) = 2√(p_ip_j)/(p_i+p_j) = MG/MA` | `[REAL]` | 2e5 pares, 8 décadas, erro máx `1,96e−15` |
| `√e` sai **exato** da traça normalizada | `[REAL]` | testado com `P_F` que **não** comuta (`‖[P_F,R_J]‖ = 2,03`): diferença **exatamente 0,0** |
| A redução `β_C = β_TGL ⟺ A_C = α` | `[REAL]` | legítima; `κ* = 2·arccosh(1/α) = 11,22676` |
| `𝒥_C(A) = J A† J` é **ℂ-linear** e coincide com `recJ` da pedra `TheRecordOfJ` | `[REAL]` | convergência por caminhos independentes |
| **`Ψ_term` construído e legível** (T08b) | `[REAL]` | átomo exibido em 3 corpora; `S ≈ 3,15 nat` |

**A tese do operador de que os dois abertos eram um só está CONFIRMADA na parte
construtiva:** o mesmo `{p_k}` gera `Ψ_term` **e** `R_J`. Isso não caiu.

---

## PARTE II — A ERRATA DA BANCADA (cinco erros meus, e nenhum salva a proposta)

A régua: o número corrige a frase. Corrijo.

### ERRO 1 (grave, conceitual) — o meu argumento estrutural era **FALSO como enunciado**

Eu escrevi: *"uma média normalizada de `sech` sobre qualquer canto largo é **necessariamente**
O(1)"*. **Falso.** Três dos quatro verificadores derrubaram independentemente, com
contraexemplo explícito: espectro log-linear `λ=1`, `r=861` (741 mil pares — canto largo por
qualquer critério) dá `A_C = 0,007278` contra `α = 0,0072974`, a menos de 0,3%.

E a explicação anexa também estava errada: eu disse que o canto teria de se **concentrar** em
`κ ≈ 11,23`. No contraexemplo o canto **não** se concentra — espalha-se de `κ=0` a `860`, e o
valor pequeno vem de **diluição**, não de concentração.

**A forma correta, e ela refuta melhor:** `A ≈ 2π/Λ`, com `Λ = ln(p_max/p_min)`. O que mata a
proposta é o **número**: `A_C = α` exige `Λ = 2π/α = 861 nats` (374 décadas de faixa dinâmica
no espectro de Schmidt) contra os **16–18 nats medidos**. Fator ~4×10⁴ na direção errada.

### ERRO 2 (sério, metodológico) — corte de posto numérico **não declarado** num protocolo hasheado

Em `V=1600`, `σ_min/σ_max = 3,66e−35`. `A_F1` vale `0,8534 / 0,7102 / 0,5852 / 0,5701 / 0,3104`
conforme o corte (`1e−1 / 1e−2 / 1e−3 / platô / todos`). **Fator 2,75 entre convenções**, e o
**sinal** da tendência em `V` se inverte. Um pré-registro que fecha a família de `P_F` mas deixa
a convenção de posto aberta **tem um grau de liberdade não fechado**. Emenda: declarar o platô.

### ERRO 3 (sério, de número) — **"nunca abaixo de 70×" estava otimista**

Sob a convenção mais deflacionária ainda defensável (vocabulário completo do CPC, `V = 7.465`,
mantendo todos os modos nulos), o valor cai a `0,0797 = 10,9·α`. **O enunciado honesto é ~11×,
não 70×.**

### ERRO 4 (sério, de frase) — **"CEGO ao corpus" é falso como medida**

Com corte declarado, `V=600`, real vs 12 embaralhamentos:

| corpus | real | nulo | z |
|---|---|---|---|
| CPC | 0,5322 | 0,5734 ± 0,0019 | **−21,8** |
| manual DJE | 0,6609 | 0,5899 ± 0,0046 | **+15,3** |
| MNI TJRJ | 0,6776 | 0,5726 ± 0,0028 | **+37,4** |

Há sensibilidade real de 7–18%, com significância enorme e **sinal instável entre corpora**.
**"Cego" vira "fracamente sensível, de segunda ordem, com sinal instável".** A correção não
salva nada (modulação de 1,2× sobre um valor que precisaria encolher 78×), mas a refutação
**não pode carregar uma frase que a própria medida falsifica**.

### ERRO 5 (menor, de enunciado) — `J = conjugate-swap` **só vale na base de Schmidt**

Para bigrama dirigido `p(u,v) ≠ p(v,u)`, `M` não é simétrica, e o conjugate-swap na base de
**tokens** leva `vec(M)` em `vec(Mᵀ)` e **não fixa** `Ψ`: `‖J_token Ψ − Ψ‖ = 0,8786`. Na base
de Schmidt fixa exatamente (`5,35e−16`). Não altera o veredito (`R_J` só depende de `|K|`), mas
o enunciado propagava erro.

### LACUNA 6 (grave enquanto durou, **suprida**) — faltava **controle positivo**

Um negativo só vale se o aparelho puder produzir o positivo. Eu não reportei
injeção-recuperação. **Suprida:** no CPC com `V=800`, a casca `|κ − 11,2268| < 0,05` (146 pares
em 369.664 — 0,04% do canto) devolve **0,007296** contra `α = 0,0072974`. **Quatro dígitos.**
O aparelho **enxerga α quando α está lá**. O negativo é negativo real, não instrumento morto.

---

## PARTE III — OS ARGUMENTOS NOVOS, QUE SÃO MELHORES QUE OS MEUS

### (i) TEOREMA DA COMBINAÇÃO CONVEXA — fecha a família **inteira**, não seis membros

Por ciclicidade e `P_F² = P_F`:

```
tau_F(P_F R_J P_F) = Tr(P_F R_J)/Tr(P_F) = SOMA_a  w_a * sech(kappa_a / 2)
```

com `w_a ∈ [0,1]`, `Σw_a = 1` — para **qualquer** projeção (espectral ou não, no fator, no
comutante, no centralizador) e **qualquer** estado normal. O observável é **sempre** uma média
de probabilidade de `sech ∈ (0,1]`.

Consequência dura: `τ_F ≤ α` força `W(κ≤2) ≤ 1,13%`, `W(κ≤6) ≤ 7,3%`, e **≥46% do peso acima de
`κ = 11,2268`**. Não é *"não testamos o canto certo"* — é **o canto certo não existe**.

### (ii) INVARIÂNCIA DE ESCALA — o observável **descarta `U` e `V†`**

`R_J` é diagonal nos pares de Schmidt e `κ_ij` só depende de `p_i/p_j`: o observável é **função
exclusiva de `S`**. Toda a sintaxe — quem segue quem, o que separa um artigo de física do CPC —
mora em `U` e `V†`, **que são jogados fora**. Por isso o nulo do embaralhamento coincidir **era
teorema, não anomalia**.

Formas fechadas, confirmadas a 1e−15:

```
p_k ~ k^-s   =>   A = 2 INT_0^1 t^(s/2)/(1+t^s) dt   —  EXATAMENTE independente de N
    A(1) = 4 - pi   = 0,8584073464102069
    A(2) = ln 2     = 0,6931471805599453
    A(s) ~ pi/s  (assintoticamente)
```

Para `A = α` seria preciso **`s* = 428,18`** (assintota `π/α = 430,5`). Texto real tem
`s ≈ 1,39–1,49`. *(Divergência entre verificadores resolvida: `s*=428,18`; a assintota
`4/(s+2)` estava errada e produzia `546`.)*

### (iii) PISO CONVEXO — **α está abaixo do chão de precisão dupla**

`sech` tem transformada de Fourier positiva ⟹ é **núcleo positivo-definido** (menor autovalor
`0,1916` numa grade de 300 pontos sobre 744 nats) ⟹ a minimização é **QP convexa** com mínimo
único. Para `Λ = 744,4 nats` (a faixa **inteira** do float64, de 1 ao subnormal `5e−324`), o
mínimo sobre **todos os espectros representáveis** é **`0,008387 = 1,149·α`**.

**Nenhum espectro de Schmidt que caiba em precisão dupla pode fazer um canto de forma-produto
valer α.**

### (iv) A ARMADILHA DO VALOR INTERMEDIÁRIO — *"sem que α entre"* é **estruturalmente impossível**

`F3 = 1 > α` e caudas profundas dão `3e−4 < α`. Por continuidade, **sempre existe** `P_F` que dá
α exatamente (cruzamento medido entre `t = 9,81` e `t = 10,14`). Mas o canto que acerta tem
obrigatoriamente `κ` típico resolvendo `sech(κ/2) = α`, isto é **`κ* = 11,2268`** — ou seja,
**qualquer `P_F` bem-sucedido traz α embutido na própria definição**.

Acerto por essa via **não é descoberta; é ajuste de parâmetro contínuo**.

### (v) TETO DE REPRODUTIBILIDADE — não distingue 1/137 de 1/200

`A ≈ 2e^{−κ/2}` ⟹ `dA/A = −dκ/2`. O `κ*` medido varia de **8,98 a 10,81** (14 medidas,
dispersão 1,83 nats = **fator 2,5 em A**). Afirmar α a 4 dígitos exigiria `κ*` estável a
`2e−4` nats — **quatro ordens de grandeza fora**. Um "acerto" teria barra de erro compatível
com 0,005 e com 0,018 **igualmente**.

### (vi) A IRONIA ESTRUTURAL, que deve ficar registrada

As **duas leituras mais canônicas de Tomita–Takesaki** dão **exatamente 1,0** — o extremo
oposto de α:

* o estado vetorial `ω_Ψ`: `|Ψ⟩` vive **só** nos pares diagonais (`κ=0`), logo
  `⟨Ψ|sech(|K|/2)|Ψ⟩ = Σp_k·1 = 1` **exato**, para qualquer corpus, tokenizador e tamanho;
* o canto de Breuer no zero isolado de `K` — que é **precisamente o objeto da hipótese H1 da
  própria TGL** (Three Locks, Nome = 1).

**O canto que a TGL chama de canônico e o axioma `ω(I)=1` que ela usa para normalizar dizem,
ambos, que a leitura natural de `R_J` vale `√e·1 = 1,6487` — que é 137× `β_TGL`, não `β_TGL`.**

### (vii) BAIXA COLATERAL — este teste **nunca poderia validar a assinatura TGL**

Como `τ_F(1) = 1`, o `√e` **cancela exatamente**. O fator `√e` e a Meia-Nat são **decorativos**
aqui; o teste exige que o corpus produza **α sozinho**. E a proposta **nunca nomeou o canal**:
por que um texto tokenizado conheceria uma condição de renormalização a **momento zero da QED**
(`α` é o valor no limite Thomson, `q²→0`, de uma constante que **corre** com a energia)?
A pergunta não foi feita, quanto mais respondida.

### (viii) E a motivação declarada **não corresponde ao objeto selado**

A memória canônica registra **`(1/4)sech²(κ/2)` como densidade**, com
`∫(1/4)sech²(κ/2)dκ = 1 = ω(I)` (confirmado). O proposto é **`sech` simples em traça
normalizada discreta** sobre um canto. São objetos diferentes em **duas** dimensões (o quadrado
e o fator ¼; Lebesgue vs. traça de canto). Sob a leitura canônica **não existe sequer um lugar
onde α pudesse entrar** — o resultado é 1 por construção.

---

## PARTE IV — O QUE O PROTOCOLO SALVOU (e por que ele é a peça mais valiosa)

**`exp(−S_Schmidt)` daria um FALSO POSITIVO A QUATRO DÍGITOS.**

CPC-PT com `V=3200`: `exp(−S) = 0,01203` contra `β_TGL = 0,0120313`.

E é um **dial**: `exp(−S) ~ D^{−0,637} / D^{−0,684} / D^{−0,746} / D^{−0,721}` conforme o corpus
(nem o expoente é universal), cruza `β` em `D = 1643/360/616/383` e `α` em
`D = 3600/748/1204/765`. **Todo corpus pode ser feito bater α ou β escolhendo um inteiro livre.**

**Teria passado se alguém tivesse rodado um corpus só.** É exatamente o falso-positivo que o
pré-registro existe para matar — **e matou**.

**A peça mais valiosa que sobrevive não é matemática: é o protocolo.**

---

## PARTE V — T09: NÃO HÁ CANDIDATO HONESTO NESTA LINHA

Dito sem suavizar. A dicotomia é fechada: qualquer funcional `Φ` da cadeia
`corpus → M → {p_k} → (Δ, K, J)` **ou** depende do dado — e então varia entre corpora e só
iguala α num conjunto de medida nula, que é a definição operacional de ajuste — **ou** não
depende do dado, e então é identidade algébrica gerada por `{e, 2, log, ½}`, enquanto
`α = 0,0072973525693…` é um acoplamento de QED **medido**, sem forma fechada conhecida.
**Eddington tentou exatamente isto e falhou.**

**T09-a (obrigatório):** selar o **catálogo de falsos-positivos** e instituir a regra
**CONSTÂNCIA ANTES DO VALOR** — ver `catalogos/04_CATALOGO_FALSOS_POSITIVOS.md`.

**T09-b (o único que pode dar física):** **trocar o substrato**. Corpus tokenizado **não tem
canal para α**. A linha do Evento 2 (MCMC α-livre) ou o piso dos vazios com LRG/ELG.
A exigência *"sem que α entre em nenhuma etapa"* foi satisfeita — **e é precisamente por isso
que α não pôde sair**.

**Subproduto legítimo (linguística, não física):** `p₂/p₁` é o único funcional testado que
**enxerga** o corpus — `0,499–0,565` (CPC), `0,685–0,731` (DJE), `0,267–0,350` (Python), e
**colapsa** a `0,042/0,086/0,050` sob embaralhamento — uma ordem de grandeza. Separa código de
prosa. Vale como observável linguístico. **Nada nele aponta para α.**
