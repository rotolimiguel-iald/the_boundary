[REAL — gerência c1 (Claude Code, sessão 6da8f00d), 27/10/2026; para a coordenadora da Central IALD no Codex. ORDEM 027 (a 026 teve duas: 026-M, a fila da memória; 026-X, a vitrine).]

# ORDEM 027 — a consulta DIÁRIA das patentes do acervo e da parte IALD dos e-mails (ordem do operador de 26/10/2026)

## A ordem do operador

Recebida na sessão do Escritório em 26/10/2026 e repassada à Central pela fila da memória (depósito `claude-escritorio/20261026_222543_consulta-diaria-patentes-e-emails`, promovido pela gerência em 27/10 00:19; texto integral da ordem com sha256 `35b97e7ac000de911aeb170d2c90c407ba035a22145844edb981cfeadaba7404`). O trecho que é da Central, verbatim:

«também a consulta dos meus e-mails e a consulta de todas as patentes minhas que sao do acervo da central de patentes da IALD, mas isso pode se tornar uma tarefa diária, monte o comando e transmita para o chatgpt, por favor»

A parte judicial da mesma ordem (carteira forense, certificado digital, tribunais) é do Escritório e já foi montada lá (instrução 234 do livro do escritório, conversa própria em `C:\TUNEL_ESCRITORIO`). **Esta ordem só trata do que é da Central: as patentes e a parte IALD dos e-mails. Pedido de uma casa nunca entra em conversa da outra.**

## O que fazer (modo seguro)

1. **A automação existente vira diária nos dias úteis, de madrugada.** `patentes-leitura-semanal-epct-e-pareceres` (heartbeat `ACTIVE`, hoje `FREQ=WEEKLY;BYDAY=TU;BYHOUR=9;BYMINUTE=0;BYSECOND=0`, conversa de destino `01a0ce09-540f-7351-abf6-e8bf35c814b5`) passa a `FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=3;BYMINUTE=0;BYSECOND=0` (hora de Goiânia), **pela ferramenta oficial**, preservando o destino e o estado, como a coordenadora já fez com o heartbeat científico na ORDEM 026-M. **Não criar outro agendamento** (o próprio prompt o proíbe). Se a ferramenta só aceitar outra forma de rrule, registre no recibo a que ficou e por quê.

2. **Dois modos no mesmo prompt, acrescentados AO LADO do texto atual (nada se apaga):**
   - **(a) Modo diário (segunda a sexta):** só o estado e a data do último documento de cada pedido (os 15 PCTs desta rodada, PCT/BR2026/050568 a 050580, 050582 e 050583; o PCT/BR2025/050558; o raiz), documento novo baixado e preservado nos destinos autorizados como hoje, e a leitura de `C:\IALD\Central de Patentes\andamentos\SINAL.md` (a Sentinela da Central, que lê a RPI do INPI) para repetir no aviso o que ela alarma. **É CURTO: entrar, conferir, sair.** Não reler as ordens inteiras a cada noite.
   - **(b) Modo semanal (terça):** a leitura completa da ORDEM 002 / Parte H da ORDEM 004, exatamente como é hoje (ENTREGA_002_SEMANA nova, comparação com as revisões, REVISAO_012 etc.).
   - Gestão de tokens (`C:\IALD\GESTAO_DE_TOKENS.md`): o Escritório mediu cerca de 1,5 milhão de tokens de entrada por rodada de dois minutos, por causa do AGENTS global de 203 KB; cinco rodadas por semana pagam isso cinco vezes. Por isso o modo diário não deve abrir documento já lido, nem repetir download, nem reler recibos: compara com o estado da noite anterior e para.

3. **A parte IALD dos e-mails.** Na mesma rodada noturna, SE a caixa do operador já estiver acessível na sessão autenticada do app (sem login novo, sem senha, sem 2FA: **parede de login = parar, pedir ao operador UMA vez no aviso e registrar «cobertura incompleta»**), ler SÓ os assuntos da IALD: INPI, WIPO/OMPI/ePCT, Zenodo, GitHub, hospedagem do site, X, fornecedores e contas da IALD Ltda. **Assunto do Escritório ou pessoal: não ler, não relatar, nem o assunto.** Nenhuma resposta, nenhum encaminhamento, nenhum clique em link de e-mail, nenhum download de anexo que não seja documento oficial de patente.

4. **O aviso da manhã.** Ao fim de cada rodada (dias úteis), o operador recebe no app **UM aviso só quando houver** documento, prazo, movimentação ou e-mail da IALD novo, com a lista do que exige decisão dele; **silêncio quando nada mudou** (regra que o prompt já tem). O relatório de cada noite, completo, vai em `C:\TUNEL_PCT\DO_CODEX\DIARIO\<AAAA-MM-DD>.md` (hashes calculados; estado de cada pedido; e-mails da IALD com remetente, data, assunto e o que pedem; seção «O que eu vi e não fiz»; cobertura incompleta dita como tal). Prazo **expressamente fixado em documento**: a entrada permitida em `andamentos\ENTRADA_CODEX.jsonl`, com a fonte, sem duplicar; **nunca inferir prazo do silêncio; falha de busca não é «sem movimentação»**.

5. **Recibo** em `DO_CHATGPT\ORDEM_027\`: a automação como ficou (id, rrule, destino, status, sha256 do prompt novo e do antigo), a conferência de que **nenhum outro agendamento nasceu**, e o relatório da primeira rodada diária. Depósito na fila da memória (ORDEM 026-M), nunca no índice. **Primeira rodada:** já na madrugada de 27 para 28/10 se a ferramenta aceitar hoje; senão, a primeira noite possível, dita no recibo.

## O que NÃO fazer

Não pagar, não emitir GRU, não assinar, não transmitir, não responder comunicação, não alterar controles nem memórias da Central por esta rotina, não logar pelo operador, não enviar mensagem a terceiros, não criar segundo agendamento, não tocar na rotina nem nos arquivos do Escritório (casa separada: `C:\TUNEL_ESCRITORIO`). Login, 2FA e CAPTCHA são pessoais. Sigilo, originais e arquivos existentes preservados. O que o ChatGPT trouxer é `[DECLARADO pelo ChatGPT]` até a Central conferir na fonte.
