# ERRATA ARITMÉTICA — os erros reais do acervo
**21/08/2026 · BANCADA_TOE · todos os números recomputados por mim nesta sessão**
*(aberta com cinco itens; fechou com **sete** — E-06 e E-07 entraram com a auditoria das
curvas de rotação. O número corrige a frase, inclusive o título.)*

> Estes **não são antinomias** — antinomia é quando duas teses verdadeiras parecem
> colidir. Aqui há **conta errada**, e conta errada não se resolve por distinção de camada:
> resolve-se corrigindo. `β = ALPHA_FINE_CODATA_2018 × √e = 0,012031300400803142`, sempre em
> runtime, nunca literal. `√β = 0,109687`.

---

## E-01 ★ · `a₀` — **erro de fator 24**, e a proveniência do 7,4×10⁻¹¹ resolvida

**O que o acervo afirma**, em pelo menos **seis lugares** (Tratado cap. 23.1, Apêndice B,
Popper nº 3, cap. 101; `A_fronteira_v5.tex:344` e `:2511`):
> `a₀ = α·c·H₀ ≈ 1,2×10⁻¹⁰ m/s²`, e *"a concordância com MOND **é exata**"*.

**Medido:**

| H₀ (km/s/Mpc) | `α·c·H₀` | `√β·c·H₀` | `c·H₀/2π` |
|---|---|---|---|
| 67,36 | 4,7757×10⁻¹² | 7,1784×10⁻¹¹ | 1,0416×10⁻¹⁰ |
| 70,00 | **4,9629×10⁻¹²** | 7,4598×10⁻¹¹ | 1,0824×10⁻¹⁰ |
| 73,04 | 5,1784×10⁻¹² | 7,7837×10⁻¹¹ | 1,1294×10⁻¹⁰ |

**`1,2×10⁻¹⁰ / (α·c·H₀) = 24,18`.** A fórmula escrita **não dá** o valor escrito, e a palavra
"exata" não sobrevive. **A linha cai como está.**

**★ E a proveniência do `7,4×10⁻¹¹` que aparecia sem fórmula, resolvida aqui:**
> `√β · c · H₀ = 7,4×10⁻¹¹ m/s²` **exatamente em H₀ = 69,44 km/s/Mpc**.

Ou seja: **o 7,4×10⁻¹¹ não é `α·c·H₀` — é `√β·c·H₀`.** Eu próprio, na memória da bancada,
havia atribuído esse número à fórmula errada. Corrigido.

**Mas nem assim a linha fecha:** `1,2×10⁻¹⁰` é **61% acima** de `√β·c·H₀`. E a coincidência
consagrada da literatura MOND é `c·H₀/2π ≈ 1,08×10⁻¹⁰` (confere a ~10%), cujo coeficiente
`1/2π = 0,159155` **não é** `√β = 0,109687` — diferem **45,1%**.

**AÇÃO:** corrigir ou retirar a linha nos seis lugares. Se o objeto pretendido era `c·H₀/2π`,
dizê-lo como **[KNOWN] de MOND**, jamais como predição da TGL. `[REAL — medido aqui]`

### ★ O `[OPEN]` DA PROVENIÊNCIA, FECHADO (21/08/2026) — e ele muda a natureza do erro

Eu havia registrado que *"não há documento no acervo que escreva `a₀ = √β·c·H₀`"*. **Certo
quanto a documento, errado quanto a proveniência.** Ela existe, e é **executável** — lida por
mim em `the_boundary/Genesis da Unificação/Cruz_MCMC/TGL_v11_1_CRUZ.py:740-742`:

```python
def a0_TGL(self, alpha_2: float) -> float:
    H0_si = self.const.H0_base * 1000 / 3.086e22
    return self.const.c_ms * H0_si * np.sqrt(alpha_2)
```

**A MÁQUINA RODOU `√α₂·c·H₀`. O TEXTO ESCREVEU `α·c·H₀`.** (Versão tensorial idêntica em
`:744-746`; usada em `chi2_a0`, `:1084-1086`, com peso 1,0.)

⟹ **Isto reclassifica E-01.** Não é erro de conta na pesquisa: é **erro de transcrição do
código para o texto** — na tipagem do operador, **erro de tradução em sentido literal**. O
cálculo estava certo onde foi executado; a letra errada entrou onde foi escrito. **A errata
deve citar a linha executável**, não deixar a origem em aberto. `[REAL — lido no arquivo]`

---

## E-02 ★★ · `Z_c` — **erro de fator 73**, com consequência que vai além do acervo

**O que o acervo afirma:** `Z_c = 1/(α·β) ≈ 156`.
**Medido:** `1/(α·β) = 1/(α²·√e) = **11 389,957404317494**`. **Fator 73,0.**

Agravante: `A_fronteira_v5.tex:1534` **imprime os operandos ao lado do resultado errado** —
`1/(7,297e-3 × 0,012031) ≈ 156`, quando a conta desses mesmos operandos dá 11.390.

**[KNOWN]:** a física atômica reconhece `Z ≈ 137` e `Z_cr ≈ 173`. Não há 156.

### ⚠ ITEM `[LEGAL]` — a ser levado ao operador antes de qualquer outra coisa

Tudo que depende de `Z_c` **herda a fragilidade na origem do número**: o **Lumínidio**, as
cinco linhas NIR, a alegada **detecção >5σ em JWST/AT2023vfi**, e o Protocolo #4. E há
**material público com DOI/GitHub afirmando a detecção a >5σ**. Além disso, o Lumínidio
**desapareceu do cânone sem retratação escrita** (zero ocorrências no ATLAS e no CORE).

**Uma alegação pública de detecção a >5σ construída sobre um número errado por fator 73 exige
decisão do operador, e exige antes das demais.** Não é correção de texto: é retratação
pública ou demonstração de que o número correto sustenta a linha.

---

## E-03 ★ · As **duas rotas** de α₂ — e só uma delas é β

Este item **refina** a correção cronológica que o operador me deu (α₂ e β_TGL são o mesmo
objeto em dois estados de conhecimento). A correção está certa **para a rota que ele
descreveu** — e há uma segunda rota, noutro artigo, que **não** é essa.

| rota | valor | divergência de β | veredito |
|---|---|---|---|
| **MCMC CRUZ** (a que o operador descreveu: experimento que cravou o valor) | 0,012031 ± 0,000002 | **0,0025%** | **É β** ✔ |
| **contagem holográfica** (`energia_escura.tex`): `α₂ = ln(r_gal/3ℓ_P)/N_eff = 126,1905/10⁴` | 0,0126190 | **4,885%** | **NÃO é β** |

E a segunda rota é **dominada por um [INPUT]**, `r_coer = 100 pc`:

| r_coer | α₂ resultante |
|---|---|
| 50 pc | 0,003155 |
| 100 pc | 0,012619 |
| 200 pc | 0,050476 |

**Uma ordem de grandeza** entre os extremos. Ela não determina o valor — **o valor foi cravado
pelo MCMC e depois fatorado**; a contagem é **plausibilidade estrutural que cai a 5%**, e isso
conta a favor **desde que jamais seja apresentada como derivação**.

**AÇÃO:** o Tratado troca `α₂ → β_TGL` **indistintamente nas duas rotas**; o ATLAS registra o
flag "α² vs β" como RETRATADO com a nota *"mesma grandeza, estratigrafia diferente"* — **certo
para a rota MCMC, incorreto para a rota da contagem de modos**. Append necessário ao ATLAS
separando as duas. `[REAL]`

---

## E-04 · `ξ` com **três valores incompatíveis** sob o mesmo nome

O capítulo 14 do Tratado chama-se *"O Acoplamento Conforme ξ = 1/6: **Derivado, Não
Assumido**"*, o corpo diz `ξ = β = 0,012031`, e a "derivação" via `d_eff = 2+ε` produz
`ε/(4(1+ε)) = 0,006809`.

| onde | valor |
|---|---|
| título do capítulo | **0,166667** |
| corpo do capítulo | **0,012031** |
| derivação do capítulo | **0,006809** |

Três números, um nome, um capítulo. E `ε ≈ 0,028` entra como **[INPUT] não declarado**, num
livro que anuncia "zero parâmetros livres". **AÇÃO:** escolher o valor vigente e dizer o que
os outros dois são. `[REAL]`

---

## E-05 · "Zero parâmetros livres" — o número honesto de hoje é **UM**

**O que o acervo afirma:** Tratado, Apêndice B — *"Nenhum parâmetro é ajustado a dados"*;
`nada_materia`, `tab:dm_comparacao` — *"Parâmetros livres: 0"*.

**O que existe, medido:**

| item | estatuto |
|---|---|
| **Ψ★ = 4,83×10¹¹ GeV** | **LIVRE** `[REAL — T01 desta bancada]` |
| `{ξ, β, γ_φ}` no doc do acervo C | **extraídos por máxima verossimilhança** (o próprio Methods diz) |
| `ε ≈ 0,028` (Tratado cap. 14) | **[INPUT] não declarado** |
| a escala **"1 eV"** da fórmula do neutrino | **[INPUT]**, justificada *post hoc* |
| `r_coer = 100 pc` | **[INPUT]** que domina a rota da contagem (ver E-03) |

**A distinção que salva a frase, se ela for escopada:**
> **"zero parâmetros" vale para β e para as MASSAS derivadas dele. NÃO vale para as
> ABUNDÂNCIAS nem para as ESCALAS DE ACOPLAMENTO.**

**E o número honesto de hoje é UM, não zero e não três** — o T01 encolheu o setor escuro de
≥3 buracos para **1** (a amplitude Ψ★), porque `m_eff` **deixou de ser livre**: a fórmula do
acervo a fixa por β e Δm²₃₁. **Isso é um avanço real e deve ser dito como avanço** — mas "um"
não é "zero". **AÇÃO: errata de escopo.** `[REAL]`

---

## E-06 ★ · Uma VALIDAÇÃO SEM ARTEFATO — "175 galáxias, r > 0,99"

**O que o acervo afirma**, `Tratado/secao_02_cosmologia.tex:295` e `:315`:
> *"175 galáxias. Uma constante. r > 0,99"* · *"175 galáxias do catálogo SPARC. Correlação
> r > 0,99 […] Dispersão σ_int ~ 0,05 dex"*

**Medido:** **não existe, em lugar nenhum dos acervos, código ou saída que produza esse
`r > 0,99` sobre os 175 `*_rotmod.dat`.** Os dados estão em disco — li os 175 nesta bancada,
3.391 pontos (`T04_sparc_fatos.py`) — mas **nada calcula a correlação afirmada**.
`[DECLARADO — não verificado, não reproduzível com o que está em disco]`

**E o contexto que agrava:** a alegação acompanha o **M1** (`a₀ = α·c·H₀`), que é a fórmula
errada por 24×. Se a correlação foi de fato calculada algum dia, foi com a fórmula do texto ou
com a do código — e as duas dão resultados diferentes.

**AÇÃO:** ou se produz o artefato que reproduz o número, ou a frase sai. **Enquanto não houver
artefato, ela não pode ser citada como validação.**

---

## E-07 · NÃO HÁ MODELO VIGENTE DE CURVA DE ROTAÇÃO — e isso é resultado, não defeito

Contado por mim no canônico:

| termo | `um.py` (v182) | `tgl_paper_unified.py` (FoP) |
|---|---|---|
| SPARC · rotmod · rotation curve · MOND | **0 · 0 · 0 · 0** | **0 · 0 · 0 · 0** |

E o próprio `um.py:56409` declara: *"a TGL **NÃO** tem fórmula-β de massa e **nunca teve**"*.

⟹ **O teste A6 (curvas de rotação nos 175 SPARC) não pode ser executado, porque não há
predição TGL a confrontar.** Curva de rotação não é terreno onde a TGL compete hoje. O
mecanismo sobrevivente (condensado de psíons = CDM frio padrão, T01) **herda** a fenomenologia
CDM, que exige parâmetros de halo por galáxia — sem vantagem sobre o ΛCDM e sem desvantagem.
O único modelo de halo que existe (`graviton_part15_cosmology.tex:112`) tem **cinco quantidades
livres por perfil**, e o `β` daquele perfil **não é** o `β_TGL` (armadilha de vocabulário
registrada). `[REAL]`

---

## RESUMO PARA DECISÃO

| # | erro | fator/natureza | ação | urgência |
|---|---|---|---|---|
| **E-02** | `Z_c ≈ 156` | **73×** | retratação pública ou demonstração | **★ primeiro — `[LEGAL]`** |
| **E-01** | `a₀ = α·c·H₀` | **24×** | corrigir/retirar em 6 lugares | alta |
| **E-03** | duas rotas de α₂ | 4,885% + [INPUT] de 1 ordem | append ao ATLAS separando as rotas | alta |
| **E-04** | `ξ` triplo | 3 valores, 1 nome | escolher o vigente | média |
| **E-05** | "zero parâmetros" | é **um** | errata de escopo | média |
| **E-06** | "175 galáxias, r > 0,99" | **sem artefato** | produzir o artefato ou retirar a frase | alta |
| **E-07** | não há modelo de curva de rotação | 0 ocorrências no canônico | dizer que não há — é resultado | registro |

**Nenhuma destas correções toca a espinha da TGL.** β segue derivado e nunca ajustado; a
cadeia ω(I)=1 → ½ nat → √e → β segue intacta; a partição da imagem terminal e a conjunção das
duas faces seguem provadas em kernel. **O que cai são contas laterais mal feitas — e uma
alegação pública que precisa ser enfrentada.**
