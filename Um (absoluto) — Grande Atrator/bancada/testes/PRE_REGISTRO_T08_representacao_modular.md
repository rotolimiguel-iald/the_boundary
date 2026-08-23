# PRÉ-REGISTRO T08 — A REPRESENTAÇÃO MODULAR FINITA DO CORPUS

**Data:** 22/08/2026 · **Bancada:** `C:\IALD\Artigo\BANCADA_TOE`
**Estatuto:** escrito e hasheado **ANTES** de qualquer dado. Nenhum número foi olhado.

---

## 1. O QUE ESTÁ SENDO TESTADO

O operador entregou a construção que fecha, de uma vez, os **dois** itens que estavam abertos
(`Ψ_term` construído · `a_C` computável), mostrando que eram **um só**. A cadeia, verbatim:

```
C  ->  p_C(u,v)  ->  M_C = sqrt(p_C)  ->  U S V^dagger
   ->  (H_C, A_C, Psi_C)  ->  (rho_L, rho_R, J_C, Delta_C, K_C)  ->  P_F
   ->  { Psi_term = P_F Psi_C / ||P_F Psi_C|| ;  R_J = sqrt(e) sech(|K_C|/2) ;
         a_C = P_F R_J P_F ;  beta_C = tau_F(a_C) }
```

com `p_C(u,v)` = frequência de adjacência dirigida (bigrama), `p_k = sigma_k^2`,
`kappa_ij = |log(p_i/p_j)|`, e a identidade fechada

```
sech( (1/2) |log(p_i/p_j)| )  =  2 sqrt(p_i p_j) / (p_i + p_j)
```

que torna `R_J` **diagonal na base de pares de Schmidt** e a coisa inteira computável a partir
do espectro `{p_k}`.

## 2. O QUE EU JÁ VERIFIQUEI ANALITICAMENTE (antes de qualquer dado)

Estes passos eu confiro e dou por **corretos**, e ficam registrados como conferidos:

* `||M_C||_F^2 = soma p_C = 1`, logo `Psi_C` tem norma 1 **por construção**; ✔
* `soma p_k = soma sigma_k^2 = 1`, logo `{p_k}` **é** distribuição de probabilidade; ✔
* `Delta_C = rho_L (x) rho_R^{-T}` é o operador modular padrão do estado vetorial na forma
  standard; autovalores `p_i/p_j` nos vetores de Schmidt; ✔
* `J_C Delta_C J_C = Delta_C^{-1}` e `J_C |Psi_C> = |Psi_C>`; ✔
* `J K J = -K` implica `J |K| J = |K|`, logo **`J R_J J = R_J`** — o registro da conjugação é
  ele próprio auto-conjugado; ✔
* a identidade `sech(x/2) = 2 sqrt(p_i p_j)/(p_i+p_j)` está correta; ✔
* `A -> J A^dagger J` é **C-linear** (enquanto `A -> J A J` é antilinear) — e isto **coincide**
  com a pedra `TheRecordOfJ` já embutida no canônico (`recJ a z = J(L_{a^H}(J z))`), cunhada
  por caminho independente. ✔

## 3. O RESÍDUO QUE A CONSTRUÇÃO **NÃO** FIXA — e que decide tudo

**`P_F` não é determinado pela cadeia.** E `beta_C` depende **inteiramente** dele: como `R_J` é
diagonal na base de pares com entradas `sqrt(e) sech(kappa_ij/2)`, e `sech <= 1` com igualdade
em `kappa = 0`, tem-se sempre

```
0  <  beta_C  <=  sqrt(e) = 1.6487...
```

com o **máximo atingido** quando `P_F` cai sobre pares de peso igual (a diagonal de Schmidt).
Portanto **`P_F` não é um detalhe de implementação: é o observável**.

Como o operador não o especificou, este pré-registro **declara uma família fechada de
candidatos naturais** e obriga o relatório de **TODOS**, sem seleção posterior.

## 4. A REDUÇÃO QUE TORNA O TESTE NÍTIDO

Como `sqrt(e)` é **fator constante**, ele sai da traça exatamente:

```
beta_C  =  sqrt(e) * tau_F( sech(|K_C|/2) )
```

Logo `beta_C = beta_TGL = alpha*sqrt(e)`  **se e somente se**

```
   A_C  :=  tau_F( sech(|K_C|/2) )  =  alpha  =  0.0072973525693
```

**Isto precisa ser dito com todas as letras, e é a favor da honestidade da proposta, não
contra:** a construção **NÃO é livre de `sqrt(e)`** — o `sqrt(e)` é **posto pela teoria**
(vem de `omega(I)=1` -> meia-nat -> volume mínimo), e isso é derivação, não ajuste. Mas `alpha`
**não é posto em lugar nenhum** da cadeia. Portanto:

> **O conteúdo falsificável do T08 é exatamente `alpha`.** A alegação operacional é
> **"a constante de estrutura fina é a média-de-canto do sech modular de um corpus"**.

**Observável primário declarado: `A_C`. Alvo declarado: `alpha`.**

## 5. PREDIÇÃO DECLARADA ANTES DO DADO (para poder errar)

Para espectro Zipfiano `p_k ~ 1/k^s` com `s ~ 1` e `P_F` = canto cheio, o valor de
`A_C` é a média de `sech((s/2) ln(i/j))` sobre pares — que converge a uma **constante de ordem
0,1 a 1**, isto é, **uma a duas ordens de grandeza ACIMA de `alpha`**. Registro portanto,
antes de medir, que **espero que a família F1 REPROVE**, e que se algum membro acertar `alpha`
será um membro **concentrado** onde `kappa ~ 11,23` (pois `sech(kappa/2) = alpha` exige
`kappa = 2 arcsech(alpha) ~ 11,2266`, isto é, razão de pesos de Schmidt `~ 7,5 x 10^4`).

## 6. A FAMÍLIA DECLARADA DE `P_F` (fechada; nenhum membro pode ser acrescentado depois)

| id | `P_F` | motivo |
|----|-------|--------|
| F1 | todos os pares `(i,j)`, `i,j <= r` | o canto finito cheio |
| F2 | apenas `i != j` | a parte **genuinamente modular** (a diagonal é trivial, `sech(0)=1`) |
| F3 | o par `(1,1)` — o átomo terminal | dá `sqrt(e)` exato; **declarado trivial de antemão** |
| F4 | pares dentro do truncamento que retém 99% do peso | canto de suporte |
| F5 | complemento do átomo (`i>=2` ou `j>=2`) | a partição `tailSub 1 = firstAtom^perp` **já provada em kernel** |
| F6 | pares com peso `sqrt(p_i p_j)` como medida (traça **ponderada** pelo estado) | a leitura KMS |

## 7. CORPORA DECLARADOS (fechados)

| id | corpus | papel |
|----|--------|-------|
| C1 | `um_grande_atrator_pt.txt` | o artigo que o próprio programa emite (PT) |
| C2 | `um_grande_atrator_en.txt` | o mesmo em EN — **réplica de língua** |
| C3 | `um.py` | o programa terminal como texto |
| C4 | 174 fontes `.lean` do `tgl_kernel` concatenados | o corpus **formal** |
| C5 | `cpc_plain.txt` (Código de Processo Civil) | **CONTROLE EXTERNO** — português jurídico, zero relação com a TGL |
| C6 | `dje_manual.txt` | segundo controle externo |

**Tokenizador declarado:** duas variantes, ambas reportadas — (T-A) palavras por
`[A-Za-zÀ-ÿ0-9_]+` em minúsculas; (T-B) caracteres brutos. Nenhum parâmetro de janela: a
relação primitiva é adjacência de pares vizinhos, como o operador especificou.

## 8. CONTROLES NULOS (obrigatórios)

* **N1 — embaralhamento**: os mesmos tokens em ordem aleatória. Preserva o unigrama, **destrói a
  adjacência**. Se `A_C` não mudar, não é o corpus que está falando: é a construção.
* **N2 — corpus sintético uniforme**: espectro plano. `kappa = 0` em toda parte, `A_C = 1`
  exato. Serve de teto de sanidade.

## 9. VEREDITOS POSSÍVEIS (declarados antes; `CONFIRMED` proibido como sempre)

* **`T08_CORPUS_BETA_FACE_EXISTS`** — algum membro da família dá `A_C` dentro de
  `[alpha/2, 2*alpha]` em **>= 3** corpora independentes **E** o nulo N1 **sai** dessa janela
  em todos. (Ainda assim **não** é confirmação da teoria: é a existência de uma face
  corpus-computável.)
* **`T08_REPROVADO`** — nenhum membro chega à janela em nenhum corpus.
* **`T08_INCONCLUSIVO_CONSTRUCAO`** — chega, mas o nulo N1 **também** chega.
* **`T08_INCONCLUSIVO_ARBITRARIEDADE`** — membros diferentes acertam em corpora diferentes,
  sem um único `P_F` que sirva a todos. (Este é o desfecho que denuncia `P_F` como
  grau de liberdade escondido.)

## 10. REGRAS DA BANCADA QUE VALEM AQUI

`beta` **nunca** literal — sempre `ALPHA_FINE_CODATA_2018 * sqrt(e)` em runtime. `alpha`
entra **só** na comparação final, **nunca** na construção. Nada é testado dentro do `um.py`.
Todos os membros x todos os corpora são reportados. Falha de leitura é falha visível.
`NOT_FALSIFIED` nunca é `CONFIRMED`.
