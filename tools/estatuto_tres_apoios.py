# -*- coding: utf-8 -*-
r"""O ESTATUTO EM TRÊS APOIOS — a fonte única do texto de estado da frente (README, ESTADO_ATUAL, llms.txt, read-brief, site).

Ordem do operador (07/10/2026), verbatim (trecho): «o fato de eu ter bloqueado no um.py confirmação não pode dar a entender que eu
não provei. Quem confirma que o código verdadeiro é ele mesmo com o hash que ele mesmo emite, vinculando o site, vinculando o github
e tudo mais, essa é a certificação, a confirmação que eu proibi é de creditação e não de validade».

O que este módulo faz: lê do core da rodada corrente (`um_absoluto.json`), do selo e do artigo emitido os números dos TRÊS APOIOS que
contam juntos (o kernel, os ritos com falsificadores pré-registrados, o pilar quântico na GPU — a regra do operador de 05/10/2026 e o
veredito `three_stress_tests_v387`) e devolve o texto em EN/PT. PROVADA é dita na definição de prova do operador (09/09/2026: lastro
suficiente e verificável); cada apoio sai com o estatuto que o core lhe dá ([PROVED] no kernel; 0 FALSIFIED nos ritos; [COMPUTED] na
GPU); CONFIRMADA = creditação (07/10/2026); o que a natureza decide não se prova, e o fator decisivo aberto (P2: β_TGL na natureza a
5σ) é dito, com as tensões medidas CONTRA β ao lado. A régua: o número corrige a frase.

Fail-closed: chave ausente, contagem divergente, invariante que a frase pressupõe e o core não sustenta, ou veredito sem o token esperado
= FALHA (nenhuma frase fixa sai com um número que a contradiga).

Uso:  from estatuto_tres_apoios import ler;  E = ler()   (ou: python tools/estatuto_tres_apoios.py — imprime o JSON)

Histórico: 07/10/2026 (1ª versão); 07/10/2026, mesma sessão, depois da verificação adversarial de quatro céticos — a definição de prova
de 09/09, as duas certificações (identidade = hash; validade = kernel), o livro de cobranças por inteiro, z_δ como o z de β̂, a Fase 9
com os leitores [DECLARADO], o piso unilateral, P1 CONTADO (não «provado»), as tensões contra β, a lista do que não se afirma lida do
core, e os invariantes exigidos.
"""
import json, re, sys
from pathlib import Path
from urllib.parse import quote

RAIZ = Path(__file__).resolve().parent.parent
A3 = 'Um (absoluto) — Grande Atrator'
TRIO = {'propext', 'Classical.choice', 'Quot.sound'}
RAW = 'https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/'


class Falha(SystemExit):
    pass


def _exige(cond, msg):
    if not cond:
        raise Falha('FALHA (estatuto_tres_apoios): ' + msg)


def _tok(verdict, *tokens):
    for t in tokens:
        _exige(t in verdict, 'o veredito não carrega «%s»: %s' % (t, verdict[:160]))


def _num(x, casas):
    return ('%.' + str(casas) + 'f') % x


def _pt(x, casas):
    return _num(x, casas).replace('.', ',')


def _cientifico(x):
    m, e = ('%.2e' % x).split('e')
    sup = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
    return '%s×10%s' % (m, str(int(e)).translate(sup))


def _milhar(n, sep):
    return format(int(n), ',').replace(',', sep)


def ler(raiz=RAIZ):
    raiz = Path(raiz)
    core = json.loads((raiz / A3 / 'um_absoluto.json').read_text(encoding='utf-8'))['core']
    selo = json.loads((raiz / A3 / 'um_absoluto_selo.json').read_text(encoding='utf-8'))
    art_en = (raiz / A3 / 'um_absoluto_en.txt').read_text(encoding='utf-8')
    versao = selo.get('um_version')
    _exige(bool(versao) and re.fullmatch(r'v\d{3}', versao), 'um_version ausente no selo')
    for chave in ('three_stress_tests_v387', 'kernel_formalization', 'hubble_ratio_opening_v388', 'hubble_ratio_v387', 'void_floor_v11',
                  'ramo_b_preregistration_v386', 'quantum_pillar_v384', 'fundacao_v390', 'einstein_and_rg', 'neutrino_m2', 'd1_camb_v3_real_v366',
                  'the_ledger_of_charges_v383', 'joint_coincidence_result_v3', 'joint_coincidence_protocol_v3'):
        _exige(chave in core, 'chave ausente no core: ' + chave)
    _exige('the_axiom_reading_v385' in selo, 'the_axiom_reading_v385 ausente no selo')

    # ---------- o veredito conjunto dos três testes (v387, recalculado a cada rodada) ----------
    t = core['three_stress_tests_v387']
    _tok(t['verdict'], 'NONE_OF_THE_THREE_FAILED', 'P1_COUNTED_BY_THE_THREE', 'P2_NOT_DETERMINED', 'THREE_PILLARS_COUNT_TOGETHER', 'GATE_UNTOUCHED')
    _exige(t['none_of_the_three_failed'] is True and t['pillars_that_failed'] == [], 'um dos três apoios falhou')
    _exige('NAO determinada' in t['p2_statement'] or 'NÃO determinada' in t['p2_statement'], 'p2_statement não diz «não determinada»')
    jt = t['joint']
    _exige(jt['three_pillar_probability_in_core'] is False and jt['V3_boolean'] is False, 'o conjunto V3 mudou — reler antes de gerar')

    # apoio 1 — o kernel
    k = t['kernel']
    nk = k['n_theorems_clean']
    _exige(nk == k['n_theorems_expected'] and k['sorryAx_absent'] is True and k['custom_TGL_axioms_absent'] is True, 'o kernel não está limpo')
    _exige(k['gate_flags_true'] == k['gate_flags_total'], 'nem todas as bandeiras do gate estão acesas')
    kf = core['kernel_formalization']
    ar = kf['axiom_report']
    n_decl = len(ar)
    _exige(all(set(v) <= TRIO for v in ar.values()), 'declaração fora do trio')
    n_arq = len(kf['formal_files_sha256'])
    _exige('TGLExt/TheReservedConfirmation.lean' in kf['formal_files_sha256'], 'a pedra TheReservedConfirmation sumiu do kernel')
    gate = selo['qg_closure_verdict']
    _exige(k['gate_verdict'] == gate, 'o gate do core e o do selo divergem')

    # apoio 2 — os ritos contra a natureza, com falsificadores pré-registrados
    led = t['ledger']
    cont = led['counts']
    for c in ('FALSIFIED', 'NOT_FALSIFIED', 'INCONCLUSIVE', 'AWAITING', 'EXCLUDED_IN_READING'):
        _exige(c in cont, 'o livro de cobranças não traz a contagem ' + c)
    n_ch = led['n_charges']
    _exige(sum(cont.values()) == n_ch, 'as contagens do livro não somam o número de cobranças')
    _exige(cont['FALSIFIED'] == 0, 'há cobrança FALSIFIED — o texto «nada falsificado» não vale')
    m = re.search(r'In the table: (\d+) of (\d+) rite rows in which GR is read return the correspondence', art_en)
    _exige(m, 'o artigo não traz a contagem das linhas em que a RG é lida')
    gr_ok, gr_n = int(m.group(1)), int(m.group(2))
    _exige(gr_ok == gr_n, 'nem toda linha em que a RG é lida devolve a correspondência')
    _exige('a per-row reading by the management' in art_en, 'o artigo não qualifica mais a marcação por linha')
    _exige('the «discriminates?» column has not a single «yes»' in art_en, 'o artigo mudou a leitura da coluna «discrimina?» — reler antes de gerar')
    _exige('a [CONJECTURE] of mechanism' in art_en, 'o artigo mudou o estatuto do mecanismo da tensão de Hubble')
    h = core['hubble_ratio_opening_v388']
    _tok(h['verdict'], 'FROZEN_FUNCTION_RE_EXECUTED_SAME_VERDICT', 'INCONCLUSIVE_SYSTEMATICS', 'READERS_DECLARED_UNTIL_SOURCES_BY_SHA256',
         'COUNTERFACTUAL_WITH_SOURCES_NOT_DISCRIMINATED_FROM_LCDM_AT_5SIGMA')
    hn = h['numbers']
    mt = re.search(r'z_disc < (\d+) nao discrimina', h['reading'])
    _exige(mt, 'o limiar de discriminação não está na leitura da Fase 9')
    limiar = int(mt.group(1))
    _exige(0 < hn['z_disc'] < limiar and hn['lnB']['TGL_vs_LCDM'] > 0, 'z_disc/ln B fora do que a frase pressupõe — rever o texto')
    spec9 = core['hubble_ratio_v387']['spec_sha256_expected']
    v11 = core['void_floor_v11']
    _exige(v11['verdict'] == 'TGL_VOID_FLOOR_NOT_FALSIFIED_POWERED' and v11['primary']['powered'] is True, 'o V11 mudou de veredito')
    _exige('unilateral' in v11['statuses']['honestidade'] and 'LCDM raso' in v11['statuses']['honestidade'] and 'Euclid' in v11['statuses']['honestidade'],
           'a honestidade do V11 mudou')
    L5, beta = v11['primary']['L5'], v11['beta_floor']
    _exige(L5 > beta, 'o limite inferior a 5σ não está acima de β')
    rb = core['ramo_b_preregistration_v386']['verdict']
    _tok(rb, 'AWAITING_DATA', 'NOTHING_OPENED')

    # as tensões medidas CONTRA β (P2 aberto nos dois sentidos)
    d1v = core['d1_camb_v3_real_v366']['verdict']
    md1 = re.search(r'ALPHA_SQRT_E_AT_(\d+)P(\d+)_SIGMA', d1v)
    _exige(md1 and 'TENSION_IS_NOT_FALSIFI' in d1v, 'o D1 V3 mudou de leitura')
    d1s = float('%s.%s' % md1.groups())
    nu = core['neutrino_m2']
    _exige(nu['verdict'] == 'TGL_NU_M2_ARMED_CONSISTENT' and nu['values']['kill_rule_satisfeita'] is False, 'o neutrino m2 mudou de veredito')
    esc = [e['tensao_sigma'] for e in nu['values']['escada_datada']]
    _exige(len(esc) >= 2 and abs(esc[-1] - nu['values']['tensao_atual_sigma']) < 1e-9, 'a escada do neutrino não termina na tensão atual')
    for s in (d1s, nu['values']['tensao_atual_sigma']):
        _exige(s < 5, 'uma tensão contra β passou de 5σ — o texto precisa ser revisto')
    _exige('CHAIN_BELOW_50_TAU' in d1v and 'is inconclusive by the convergence control' in art_en, 'o estatuto do D1 (cobrança R6) mudou')

    # as rotas do eco fora do livro (decisão do operador de 02/10/2026: rotas examinadas, com os números, não cobranças da teoria)
    lv = core['the_ledger_of_charges_v383']
    eco = lv['echo_routes_outside_the_ledger']
    _exige(eco['not_charges_of_the_theory'] is True and eco['excluded_by_data'] + eco['inconclusive'] == eco['n'], 'as rotas do eco mudaram')
    _exige('rotas examinadas' in lv['operator_decision_20261002']['option_chosen'], 'a decisão do operador de 02/10 mudou')

    # o V3 conjunto dos canais da natureza
    jr = core['joint_coincidence_result_v3']
    _exige(jr['consistent'] is False and 'INCONSISTENT' in jr['outcome'] and jr['max_abs_z'] > jr['z_max_threshold_INPUT'], 'o V3 mudou de leitura')
    _exige(jr['dominant_channel'].startswith('d1_'), 'o canal dominante do V3 não é mais o D1')
    _exige('NOT_EXCLUDED_AT_THRESHOLD' in jr['outcome'] and jr['P_all_range_over_applicable_grid'] is not None, 'o V3 mudou de leitura (limiar)')
    _exige('nao carrega correcao pela escolha do conjunto' in core['joint_coincidence_protocol_v3']['protocol']['Q1_coincidence']['selection_note'],
           'a nota de seleção do V3 mudou')

    # apoio 3 — o pilar quântico na GPU
    g = t['gpu']
    q = core['quantum_pillar_v384']
    _tok(q['verdict'], 'ATTRACTOR_UNIQUE', 'T1_INSTRUMENT_RECOVERS_INJECTED_LAW_BLIND', 'CONTROLS_FAILED_AS_REQUIRED', 'CPU_REFERENCE_AGREES', 'COMPUTED_NOT_MEASURED')
    _exige(g['T2_unique'] == g['T2_points'] and g['T4_unique'] == g['T4_instances'] and g['T1_errors'] == 0 and g['cpuref_disagreements'] == 0, 'o pilar da GPU tem falha')
    gpu = q['summary']['env']['gpu']
    gspec = q['preregistration']['spec_sha256']
    _exige(all(q['preregistration']['on_disk_matches'].values()), 'o pré-registro do pilar não confere em disco')
    modo = q['mode']
    reuso_de = None
    if modo == 'REUSE':
        _tok(g['reuse_verdict_v386'], 'RESULT_SHA256_MATCHES', 'REPRODUCIBILITY_V385_NOT_REMEASURED_THIS_RUN_SAID')
        mr = re.search(r'RODADA_SELADA_(V\d{3})', q['reuse']['reading'])
        _exige(mr, 'a leitura do reaproveitamento da GPU mudou')
        reuso_de = mr.group(1).lower()
    _exige(b'NOT_RUN__GPU_UNAVAILABLE' in (raiz / A3 / 'um.py').read_bytes(), 'o um.py não traz mais a recusa sem GPU (NOT_RUN__GPU_UNAVAILABLE)')

    # o que NÃO se afirma — cada item conferido contra o texto do core
    fu = core['fundacao_v390']
    ein = core['einstein_and_rg']
    nc_k = ' | '.join(kf['not_claimed'])
    nc_f = ' | '.join(fu['not_claimed'])
    nc_e = ' | '.join(ein['not_claimed'])
    for frag, src in (('Python did not prove any theorem', nc_k), ('KNOWN/EXTERNAL', nc_k), ('NOT a type III_1 proof', nc_k), ('NOT derived', nc_k),
                      ('MODULAR REALIZATION of the witness is not constructed', nc_k), ('equacao de campo no espaco-tempo curvo', nc_f),
                      ('prova de existencia global', nc_f), ("provamos Einstein", nc_e), ('III_1 sob RG = OPEN', nc_e)):
        _exige(frag in src, 'o not_claimed do core mudou (falta «%s») — reler antes de gerar' % frag)
    _exige('CONSERVATION_CONTINUUM_OPEN' in ein['statuses']['einstein_remaining'], 'o resíduo de Einstein mudou')
    _tok(fu['verdict'], 'LOGICAL_CLOSURE_THE_PROGRAM_STAYS_OPEN', 'FIELD_EQUATION_IN_CURVED_SPACETIME_NOT_A_KERNEL_TERM')
    agape = re.search(r'AGAPE_CLAUSE_KERNEL_(\d+)_OF_(\d+)', fu['verdict'])
    _exige(agape and agape.group(1) == agape.group(2), 'a cláusula Ágape não está inteira no kernel')
    _exige('THE_ONE_IS_POSTED_BY_THE_OBSERVER_NOT_POSTULATED_TAG_POSTO' in selo['the_axiom_reading_v385'], 'a leitura POSTO do selo mudou')

    # v391 em diante: o core registra o próprio estatuto da prova; quando existe, os números dele têm de bater com os lidos acima
    ep = None
    for chave in sorted((k for k in core if k.startswith('estatuto_da_prova_v')), reverse=True):
        ep = (chave, core[chave])
        break
    if ep:
        ek, ev = ep
        _exige(ev.get('all_verified') is True and not ev.get('falhas'), '%s não está verificado' % ek)
        _tok(ev['verdict'], 'PROVED_AS_LOGICAL_CLOSURE', 'THREE_PROOFS_COUNT_TOGETHER', 'CONFIRMATION_IS_CREDITATION_NOT_VALIDITY', 'BETA_IN_NATURE_AWAITS_5_SIGMA')
        _exige(ev['prova_1_kernel']['n_theorems_clean'] == nk and ev['prova_2_ritos']['ledger'] == cont
               and ev['prova_2_ritos']['rotas_do_eco_fora_do_livro']['excluded_by_data'] == eco['excluded_by_data']
               and abs(ev['prova_2_ritos']['razao_de_hubble']['z_disc'] - hn['z_disc']) < 1e-9
               and ev['prova_3_gpu']['numeros']['T2_unique'] == t['gpu']['T2_unique'], '%s e os registros de origem divergem' % ek)
        _exige(ev['certificacao']['um_py_sha256_lido_de_si'] == selo['sha256']['um.py'], '%s: a certificação não é o pin do selo' % ek)
        # 07/10/2026 (verificação da custódia v391): também os demais números que a frente imprime
        _p1, _p2, _p3 = ev['prova_1_kernel'], ev['prova_2_ritos'], ev['prova_3_gpu']['numeros']
        _exige(_p1['declaracoes_auditadas'] == n_decl and _p1['no_trio'] == n_decl, '%s: declarações divergem' % ek)
        _exige(sum(x[0] for x in _p1['gate_flags'].values()) == k['gate_flags_true'] and sum(x[1] for x in _p1['gate_flags'].values()) == k['gate_flags_total'],
               '%s: bandeiras do gate divergem' % ek)
        _exige(abs(_p2['razao_de_hubble']['lnB']['TGL_vs_LCDM'] - hn['lnB']['TGL_vs_LCDM']) < 1e-9, '%s: ln B diverge' % ek)
        _exige(abs(_p2['piso_dos_vazios']['void_floor_v11']['L5'] - L5) < 1e-12, '%s: o L5 do piso diverge' % ek)
        _exige(_p3['T4_instances'] == t['gpu']['T4_instances'] and _p3['cpuref_compared'] == t['gpu']['cpuref_compared']
               and _p3['cpuref_disagreements'] == t['gpu']['cpuref_disagreements'], '%s: os números da GPU divergem' % ek)
    u_conf = RAW + quote(A3 + '/Lean/tgl_kernel/TGLExt/TheReservedConfirmation.lean')
    v = dict(versao=versao, gate=gate, nk=nk, n_decl=n_decl, n_arq=n_arq, gf=k['gate_flags_true'], gt=k['gate_flags_total'],
             n_ch=n_ch, cont=cont, gr_ok=gr_ok, gr_n=gr_n,
             H0p=hn['H0_local_previsto'], H0r=hn['media_GLS_locais'], sH0r=hn['sigma_GLS_locais'], H0bg=hn['H0_bg'],
             bhat=hn['beta_hat'], sbhat=hn['sigma_beta'], zd=hn['z_delta'], zdisc=hn['z_disc'], limiar=limiar, lnB=hn['lnB']['TGL_vs_LCDM'],
             spec9=spec9[:16], L5=L5, beta=beta, d1s=d1s, nu_esc=esc, nu_2031=nu['values']['projecao_2031_sigma'],
             pall=jt['V3_P_all'], gpu=gpu, t2u=g['T2_unique'], t2n=g['T2_points'], t4u=g['T4_unique'], t4n=g['T4_instances'],
             t1n=g['T1_injections'], t1e=g['T1_errors'], cpd=g['cpuref_disagreements'], cpn=g['cpuref_compared'], gspec=gspec[:16],
             posto=True, fecho_logico=True, nao_campo=True, agape='%s/%s' % agape.groups(), u_conf=u_conf,
             eco_n=eco['n'], eco_ex=eco['excluded_by_data'], eco_inc=eco['inconclusive'], jz=jr['max_abs_z'], jthr=jr['z_max_threshold_INPUT'],
             gpu_modo=modo, gpu_reuso_de=reuso_de, jlim=jr['threshold_INPUT'], jcanais=jr['channels_needed_for_threshold_at_primary_quality'],
             registro=ep[0] if ep else None)
    v.update(_textos(v))
    return v


def _textos(v):
    """O texto — toda variável vem de ler(); nenhuma frase aqui diz mais do que o estatuto lido."""
    c = v['cont']
    zd, zdisc, lnB = _num(v['zd'], 2), _num(v['zdisc'], 2), _num(v['lnB'], 2)
    H0p, H0r, sH0r = _num(v['H0p'], 2), _num(v['H0r'], 2), _num(v['sH0r'], 2)
    bhat, sbhat = _num(v['bhat'], 4), _num(v['sbhat'], 4)
    L5, beta = _num(v['L5'], 4), _num(v['beta'], 5)
    t4_en, t4_pt = _milhar(v['t4n'], ','), _milhar(v['t4n'], '.')
    esc_en = ' → '.join(_num(x, 2) + 'σ' for x in v['nu_esc'])
    esc_pt = ' → '.join(_pt(x, 2) + 'σ' for x in v['nu_esc'])
    led_en = ('%d charges: %d falsified, %d not falsified, %d excluded in reading, %d inconclusive, %d awaiting'
              % (v['n_ch'], c['FALSIFIED'], c['NOT_FALSIFIED'], c['EXCLUDED_IN_READING'], c['INCONCLUSIVE'], c['AWAITING']))
    led_pt = ('%d cobranças: %d falsificada(s), %d não falsificadas, %d excluída(s) na leitura, %d inconclusivas, %d aguardando'
              % (v['n_ch'], c['FALSIFIED'], c['NOT_FALSIFIED'], c['EXCLUDED_IN_READING'], c['INCONCLUSIVE'], c['AWAITING']))
    led_ascii = ('%d cobrancas: %d falsificadas, %d nao falsificadas, %d excluida na leitura, %d inconclusivas, %d aguardando'
                 % (v['n_ch'], c['FALSIFIED'], c['NOT_FALSIFIED'], c['EXCLUDED_IN_READING'], c['INCONCLUSIVE'], c['AWAITING']))
    T = {'led_en': led_en, 'led_pt': led_pt}
    eco_en = ('the %d echo routes are outside the ledger by the operator’s decision of 02/10/2026 — examined routes, not charges of the '
              'theory: %d excluded by data (the pair amplitude reading × delay law), %d inconclusive' % (v['eco_n'], v['eco_ex'], v['eco_inc']))
    eco_pt = ('as %d rotas do eco ficam fora do livro por decisão do operador de 02/10/2026 — rotas examinadas, não cobranças da teoria: '
              '%d excluídas pelo dado (o par leitura de amplitude × lei de atraso), %d inconclusivas' % (v['eco_n'], v['eco_ex'], v['eco_inc']))
    eco_ascii = ('as %d rotas do eco ficam fora do livro por decisao do operador de 02/10/2026 -- rotas examinadas, nao cobrancas da teoria: '
                 '%d excluidas pelo dado (o par leitura de amplitude x lei de atraso), %d inconclusivas' % (v['eco_n'], v['eco_ex'], v['eco_inc']))
    if v['gpu_modo'] == 'REUSE':
        gpu_en = ('reused at %s by key from the sealed %s run (result sha256 matches; reproducibility not remeasured this run); it needs a '
                  'CUDA GPU — without one, um.py returns `NOT_RUN__GPU_UNAVAILABLE`' % (v['versao'], v['gpu_reuso_de']))
        gpu_curto_en = 'reused at %s from the sealed %s run; needs a CUDA GPU' % (v['versao'], v['gpu_reuso_de'])
        gpu_curto_ascii = 'reaproveitado na %s da rodada selada %s; pede GPU com CUDA' % (v['versao'], v['gpu_reuso_de'])
    else:
        gpu_en = ('computed in this run; it needs a CUDA GPU — without one, um.py returns `NOT_RUN__GPU_UNAVAILABLE`')
        gpu_curto_en = 'computed in this run; needs a CUDA GPU'
        gpu_curto_ascii = 'calculado nesta rodada; pede GPU com CUDA'
    T['definicao_en'] = (
        '**PROVED = validity**, in the operator’s definition of proof (09/09/2026): *sufficient and verifiable ballast* — `um.py` is the '
        'executable ballast: one file, one input, the kernel materialized and audited term by term, the rites run with pre-registered '
        'falsifiers, the hashes sealed, the article emitted; whoever executes it after cloning the repository (the rites are re-read by hash from '
        'its ../cache) obtains the same ballast (the GPU support needs a CUDA GPU). '
        'Two certifications, not to be mixed: **identity** is certified by the artifact itself — `um.py` computes its own sha256 and writes '
        'it into the seal, and this repository and the site carry that same pin byte-exact; **validity** is certified by the Lean kernel '
        '(`#print axioms`), which anyone re-runs and the CI rebuilds — Python proves no theorem. **CONFIRMED = accreditation** (the '
        'operator’s ruling, 07/10/2026): the act by which observers outside the artifact accept the result. The machine never issues it about '
        'itself: the kernel stone [`TheReservedConfirmation.lean`](%s) proves, on the finite face, that the flow cannot confirm itself and only '
        'the recognizer confirms; the approval the rites emit falls short of confirmation by construction (“observer = the human” is '
        '`[ONTO]`). *Not yet confirmed* means *not yet accredited*; it never means *not proved*. `NOT_FALSIFIED ≠ CONFIRMED`.' % v['u_conf'])
    _reg = ''
    if v.get('registro'):   # v391 em diante: o core registra esta leitura (conferido em ler())
        _reg = (' The core of %s records this reading itself: `%s` (`PROVED_AS_LOGICAL_CLOSURE`; `CONFIRMATION_IS_CREDITATION_NOT_VALIDITY`).'
                % (v['versao'], v['registro']))
    T['definicao_en'] += _reg
    T['definicao_pt'] = (
        '**PROVADA = validade**, na definição de prova do operador (09/09/2026): *lastro suficiente e verificável* — o `um.py` é o lastro '
        'executável (um arquivo, uma entrada, o kernel auditado termo a termo, os ritos com falsificadores pré-registrados, os hashes selados; '
        'o apoio da GPU pede CUDA). A **identidade** é certificada pelo próprio artefato, com o hash que ele mesmo emite e que este repositório e '
        'o site carregam byte a byte; a **validade**, pelo kernel Lean (`#print axioms`). **CONFIRMADA = creditação** (07/10/2026): o ato de '
        'observadores externos ao artefato; a máquina não a emite sobre si mesma (a pedra `TheReservedConfirmation`). *Ainda não confirmada* '
        'quer dizer *ainda não creditada*, nunca *não provada*.')
    T['regua_curta_en'] = (
        'PROVED = validity, in the operator’s definition of proof (sufficient and verifiable ballast; the validity of the implication is '
        'certified by the Lean kernel, `#print axioms`); CONFIRMED = accreditation, the act of observers outside the artifact — not made here; '
        'what nature decides is not proved')
    T['linha_en'] = (
        'quantum gravity **PROVED** — in the operator’s definition of proof, *sufficient and verifiable ballast* — on **three '
        'supports that count together**, each with its own status: the Lean kernel `[PROVED]` (the implication from the posited One and the '
        'named hypotheses; %d/%d theorems of the rite’s ladder (ext_*) clean, %d formal files, %d audited declarations in the trio, zero `sorry`), the rites against nature '
        'with pre-registered falsifiers (%s; %s; GR recovered in %d of %d rows where it is read, a per-row reading by the management) and the '
        'theory’s open system computed on the GPU `[COMPUTED]` (%d/%d, %s/%s, %d disagreements with the CPU in %d; %s) — none of the '
        'three failed. **Not yet CONFIRMED**: confirmation is accreditation, the act of observers outside the artifact, not validity. What '
        'nature decides is not proved: the decisive open factor is **β_TGL itself in nature at ≥ 5σ**.'
        % (v['nk'], v['nk'], v['n_arq'], v['n_decl'], led_en, eco_en, v['gr_ok'], v['gr_n'], v['t2u'], v['t2n'], t4_en, t4_en, v['cpd'], v['cpn'],
           gpu_curto_en))
    T['linha_pt'] = ('*Provada — na definição de prova do operador, lastro suficiente e verificável — em três apoios que contam '
                     'juntos, cada um com o seu estatuto; ainda não confirmada (creditada); o que a natureza decide não se prova: o fator '
                     'decisivo aberto é β_TGL na natureza a 5σ.*')
    # a linha da Parte 06 das secoes: curta, com «não confirmada» e «a natureza não se prova» ANTES das contagens (a resposta cabe em 2 KB)
    T['linha_06_en'] = (
        'quantum gravity **PROVED** in the operator’s definition of proof (sufficient and verifiable ballast); **not yet CONFIRMED** — '
        'not yet accredited; what nature decides is not proved: the decisive open factor is **β_TGL in nature at ≥ 5σ**. Three '
        'supports count together, each with its own status: the Lean kernel `[PROVED]` (the implication), the rites with pre-registered '
        'falsifiers (%d of %d charges of the theory falsified, %d reading excluded; the echo routes are examined routes outside the ledger), '
        'the GPU `[COMPUTED]`.' % (c['FALSIFIED'], v['n_ch'], c['EXCLUDED_IN_READING']))
    T['linha_06_pt'] = (
        'gravidade quântica **PROVADA** na definição de prova do operador (lastro suficiente e verificável); **ainda não CONFIRMADA** — '
        'ainda não creditada; o que a natureza decide não se prova: o fator decisivo aberto é **β_TGL na natureza a 5σ**. Três apoios '
        'contam juntos, cada um com o seu estatuto: o kernel Lean (a implicação), os ritos com falsificadores pré-registrados (%d de %d '
        'cobranças da teoria falsificadas, %d leitura excluída; as rotas do eco são rotas examinadas fora do livro), a GPU (calculado, não '
        'medido).' % (c['FALSIFIED'], v['n_ch'], c['EXCLUDED_IN_READING']))
    T['cartao_en'] = (
        'PROVED in the operator’s definition of proof, on three supports, each with its own status (kernel [PROVED]: the implication · '
        'rites: %d of %d charges of the theory falsified, %d reading excluded · GPU [COMPUTED]) ≠ CONFIRMED (accreditation); what nature '
        'decides is not proved.' % (c['FALSIFIED'], v['n_ch'], c['EXCLUDED_IN_READING']))
    T['curto_en'] = (
        'Quantum gravity PROVED in the operator\'s definition of proof (sufficient and verifiable ballast), on three supports that count '
        'together, each with its own status: the Lean kernel proves the implication from the posited One and the named hypotheses '
        '(%d/%d theorems of the rite\'s ladder (ext_*) clean, %d audited declarations in {propext, Classical.choice, Quot.sound}, zero sorry); the rites against nature, '
        'with pre-registered falsifiers, falsified none of the %d charges of the theory (%s; GR recovered in %d of %d rows, a per-row reading); '
        '%s; the theory\'s open system is computed on the GPU (%d/%d, %s/%s, %d disagreements with the CPU in %d; computed, not measured; %s). '
        'Not yet confirmed: confirmation is accreditation, the act of observers outside the artifact, not validity; the program never issues '
        'it about itself. What nature decides is not proved: the decisive open factor is beta_TGL itself in nature at 5 sigma. NOT_FALSIFIED '
        'is never CONFIRMED.'
        % (v['nk'], v['nk'], v['n_decl'], v['n_ch'], led_en, v['gr_ok'], v['gr_n'], eco_en.replace('’', "'").replace('—', '--').replace('×', 'x'),
           v['t2u'], v['t2n'], t4_en, t4_en, v['cpd'], v['cpn'], gpu_curto_en))
    T['curto_pt_ascii'] = (
        'Gravidade quantica PROVADA na definicao de prova do operador (lastro suficiente e verificavel), em tres apoios que contam juntos, '
        'cada um com o seu estatuto: o kernel Lean prova a IMPLICACAO a partir do Um posto e das hipoteses nomeadas (%d/%d teoremas da escada do rito (ext_*) limpos, '
        '%d declaracoes auditadas no trio {propext, Classical.choice, Quot.sound}, zero sorry); os ritos contra a natureza, com falsificadores '
        'pre-registrados, nao falsificaram nenhuma das %d cobrancas da teoria (%s; a RG resgatada em %d de %d linhas, leitura por linha da '
        'gerencia); %s; o sistema aberto da teoria e\' calculado na GPU (%d/%d, %s/%s, %d discordancias com a CPU em %d; calculado, nao medido; '
        '%s). Ainda nao CONFIRMADA: confirmar e\' CREDITAR -- ato de observadores externos ao artefato, nao validade; a maquina nunca o emite '
        'sobre si mesma. O que a natureza decide nao se prova: o fator decisivo aberto e\' beta_TGL na natureza a 5 sigma. NOT_FALSIFIED nunca '
        'e\' CONFIRMED.'
        % (v['nk'], v['nk'], v['n_decl'], v['n_ch'], led_ascii, v['gr_ok'], v['gr_n'], eco_ascii, v['t2u'], v['t2n'], t4_pt, t4_pt, v['cpd'],
           v['cpn'], gpu_curto_ascii))
    linhas = [
        '| support | what it establishes | read from the core and the emitted article | status |',
        '|---|---|---|---|',
        '| **1 · the Lean kernel** | the implication from the posited One (`ω(I) = 1`, POSTO) and the named hypotheses H1–H3 to the '
        'pentad (Breuer corner · Name = 1 · coframe · Lorentz · δQ = κδA/8πG), in Lean 4 + mathlib, on the finite face '
        '| **%d/%d theorems of the rite’s ladder (ext_*) clean**; %d/%d declarations in `{propext, Classical.choice, Quot.sound}`; zero `sorry`; gate flags %d/%d (the formal '
        'flags from the kernel; the experimental ones fed by the V11 rite) | `[PROVED]` — the implication, not nature |'
        % (v['nk'], v['nk'], v['n_decl'], v['n_decl'], v['gf'], v['gt']),
        '| **2 · the rites against nature, with pre-registered falsifiers** | GR recovered as the classical limit in every rite row where '
        'GR is read (a per-row reading by the management). **The Hubble ratio** `K = E(z*)^{2β/3}`, β fixed by the axiom, zero '
        'parameters adjusted (use-novelty): local H₀ predicted %s against the readers %s ± %s km/s/Mpc as declared; the fitted '
        'β̂ = %s ± %s sits at z_δ = %s from β_TGL; with the readers as declared, ln B(TGL/ΛCDM) = %s and z_disc = %s '
        '(√Δχ²), below the %d at which the protocol discriminates. **The void floor**: the 5σ lower bound %s sits above '
        'β = %s (powered; one-sided — shallow ΛCDM also passes) | **%d of %d** GR rows; **%s**; Phase 9 pre-registered by hash '
        '(`%s`) | **0 of %d charges of the theory FALSIFIED** — the theory stood; %s. Phase 9’s frozen verdict reads '
        '`INCONCLUSIVE_SYSTEMATICS` until the readers’ sources are pinned by sha256; the article reads the resolution of the Hubble tension '
        'as a `[CONJECTURE]` of mechanism |'
        % (H0p, H0r, sH0r, bhat, sbhat, zd, lnB, zdisc, v['limiar'], L5, beta, v['gr_ok'], v['gr_n'], led_en, v['spec9'], v['n_ch'], eco_en),
        '| **3 · the quantum pillar on the GPU** | the theory’s open system (H_LD + five GKLS/Lindblad jumps) computed in double '
        'precision on an %s under pre-registration: unique attractor over the whole grid; the blind instrument recovers the law the code '
        'injected; the controls fail as required | **%d/%d** grid points; **%s/%s** stress instances; %s blind injections, %d errors; CPU '
        'reference: %d disagreements in %d; spec `%s` | `[COMPUTED]` — a numerical experiment: measurement in the simulated '
        'environment, not of nature; %s |'
        % (v['gpu'], v['t2u'], v['t2n'], t4_en, t4_en, _milhar(v['t1n'], ','), v['t1e'], v['cpd'], v['cpn'], v['gspec'], gpu_en),
    ]
    T['tabela_en'] = '\n'.join(linhas)
    T['p1_en'] = (
        '**None of the three failed** (`three_stress_tests_v387`: `NONE_OF_THE_THREE_FAILED`). What they measure together is P1 — *the '
        'theory is consistent and recovers the known physics* — counted by the three supports (`P1_COUNTED_BY_THE_THREE`). The core keeps '
        'no single probability for it; the joint coincidence of the nature channels (V3) reads P_all = %s (no correction for the choice of the '
        'set) and boolean False — not excluded at the protocol threshold %s (%d channels of this quality would be needed) and '
        'inconsistent at its %sσ rule (max |z| = %s, the D1 tension).'
        % (_cientifico(v['pall']), _cientifico(v['jlim']).replace('1.00×', ''), v['jcanais'], _num(v['jthr'], 0), _num(v['jz'], 2)))
    T['p2_en'] = (
        '**What nature decides is not proved — the decisive open factor is β_TGL itself in nature at ≥ 5σ (P2, '
        '`P2_NOT_DETERMINED`).** No channel discriminates GR at the available sensitivity, in either direction: the tensions measured '
        '*against* β_TGL are also below 5σ — the dressed D1 at %sσ (implemented branch; the ledger charge R6 is INCONCLUSIVE by the '
        'convergence control), and the neutrino m₂ at %sσ, rising %s as the precision grows (the core projects %sσ for 2031). The '
        'tests that can decide are pre-registered by hash and wait for data: branch B’s blind O4 set; Euclid / CMB-S4 for the two-sided '
        'floor; the neutrino precision. Pinning Phase 9’s readers by sha256 lifts its `INCONCLUSIVE_SYSTEMATICS`, but under its V1 it would '
        'still not discriminate (z_disc %s < %d). Also open, named by the core: the conservation continuum and the stability of III₁ under '
        'RG (`einstein_and_rg`); global existence — the program stays open (`fundacao_v390`).'
        % (_num(v['d1s'], 2), _num(v['nu_esc'][-1], 2), esc_en, _num(v['nu_2031'], 2), zdisc, v['limiar']))
    T['nao_en'] = (
        '**Not claimed here** (read from the core: `kernel_formalization`, `fundacao_v390`, `einstein_and_rg`): Python proves no theorem, the '
        'Lean kernel does; Bisognano–Wichmann, Reeh–Schlieder and the type III₁ classification are `[KNOWN]`, external; the finite '
        'Three-Locks corner is a finite-dimensional theorem, not a type III₁ proof; `G` is an input, not derived; the modular realization '
        'of the witness is not constructed; the field equation in curved spacetime is not a kernel term, and global existence is not proved; '
        '“we proved Einstein” is not claimed (Lovelock is `[KNOWN]`, the approximate Killing residue is named).')
    T['secao_md'] = '\n\n'.join([
        '## Proved on three supports · provada em três apoios `[read by script from the run of %s]`' % v['versao'],
        T['definicao_en'],
        T['tabela_en'],
        T['p1_en'],
        T['p2_en'],
        T['nao_en'],
        '*Em português.* ' + T['definicao_pt'] + ' Os três apoios contam juntos, cada um com o seu estatuto, e nenhum falhou: o kernel '
        '(%d/%d, a implicação), os ritos com falsificadores pré-registrados (%s; %s; a RG resgatada em %d de %d, leitura por linha da gerência) '
        'e a GPU (%d/%d; %s/%s; %d discordâncias em %d; calculado, não medido). O que a natureza decide não se prova: o fator decisivo aberto é '
        '**β_TGL na natureza a 5σ** — e as tensões medidas contra β também estão abaixo de 5σ (D1 %sσ, cobrança R6 '
        'inconclusiva; neutrino m₂ %sσ, em alta: %s).'
        % (v['nk'], v['nk'], led_pt, eco_pt, v['gr_ok'], v['gr_n'], v['t2u'], v['t2n'], t4_pt, t4_pt, v['cpd'], v['cpn'], _pt(v['d1s'], 2),
           _pt(v['nu_esc'][-1], 2), esc_pt),
    ]) + '\n'
    # a versão da FRENTE do README (limite de 40 KB): a linha sem os números que a tabela já traz; a seção com a tabela e um parágrafo
    # curto de P1/P2; a leitura inteira (P1, P2, o V3, o «não se afirma», a versão em português) fica no ESTADO_ATUAL (link)
    T['linha_curta_en'] = (
        'quantum gravity **PROVED** — in the operator’s definition of proof, *sufficient and verifiable ballast* — on **three '
        'supports that count together**, each with its own status (the table below): the Lean kernel `[PROVED]` (the implication), the rites '
        'against nature with pre-registered falsifiers (%d of %d charges of the theory falsified; %d reading excluded) and the GPU computation '
        '`[COMPUTED]`; none of '
        'the three failed. **Not yet CONFIRMED**: confirmation is accreditation, the act of observers outside the artifact, not validity. What '
        'nature decides is not proved: the decisive open factor is **β_TGL itself in nature at ≥ 5σ**.'
        % (c['FALSIFIED'], v['n_ch'], c['EXCLUDED_IN_READING']))
    T['definicao_curta_en'] = (
        '**PROVED = validity**, in the operator’s definition of proof (09/09/2026): *sufficient and verifiable ballast* — one file, one '
        'input, the kernel audited term by term, the rites run with pre-registered falsifiers, the hashes sealed. **Identity** is certified by '
        'the artifact itself (`um.py` writes its own sha256 into the seal; this repository and the site carry that pin byte-exact); '
        '**validity**, by the Lean kernel (`#print axioms`; Python proves no theorem). **CONFIRMED = accreditation** (07/10/2026): the act of '
        'observers outside the artifact, never issued by the machine about itself ([`TheReservedConfirmation.lean`](%s); “observer = the '
        'human” is `[ONTO]`). *Not yet confirmed* means *not yet accredited*, never *not proved*.' % v['u_conf']) + _reg
    T['p_curto_en'] = (
        '**None of the three failed**; together they count P1 (the theory is consistent and recovers the known physics). **What nature decides '
        'is not proved:** the decisive open factor is β_TGL itself at ≥ 5σ (P2), and the tensions measured against β are also below '
        '5σ (D1 %sσ, charge R6 inconclusive; neutrino m₂ %sσ, rising).'
        % (_num(v['d1s'], 2), _num(v['nu_esc'][-1], 2)))
    T['secao_md_readme'] = '\n\n'.join([
        '## Proved on three supports · provada em três apoios `[read by script from the run of %s]`' % v['versao'],
        T['definicao_curta_en'], T['tabela_en'], T['p_curto_en'],
        'The full reading — P1, P2 and the joint V3 check, what is **not claimed**, the Portuguese version: [`ESTADO_ATUAL.md`](%sESTADO_ATUAL.md).'
        % RAW,
    ]) + '\n'
    # notas AO LADO para as frases herdadas do livro-razão (append-only)
    T['ao_lado_nao_provada'] = (
        '> ⚠ **Beside (07/10/2026, the operator’s ruling):** *does not mean quantum gravity is proved* / *não significa gravitação '
        'quântica provada* above are kept as written (the ledger is append-only). Under the current ruler they read: not **CONFIRMED** '
        '— not accredited. What the kernel PROVES is the implication from the posited One and the named hypotheses; what nature decides '
        'is not proved. The three supports that count together, each with its own status, are at the top of this page.')
    T['ao_lado_lema3'] = (
        '> ⚠ **Beside (%s):** *the unconditional global lift (Lemma 3) stays [OPEN]* is the reading before v390. The core of %s '
        'reads `GLOBAL_LIFT` as the superposition, by the operator’s definition (07/10/2026, `[INPUT/ONTO]`); the refusal is proved in the '
        'kernel (the Agape clause, %s) and the local passage is `[KNOWN]` (Jacobson 1995, H3 named): a **logical closure**; the program stays '
        'open; this is not a proof of global existence, and the field equation in curved spacetime is not a kernel term (`fundacao_v390`).'
        % (v['versao'], v['versao'], v['agape']))
    T['ao_lado_posto'] = (
        '> ⚠ **Beside (v385):** where the axiom `ω(I) = 1` is tagged **[POSTULATE]** above, read **[POSTO]**: the 1 is posited '
        '— inscribed by the observer (`echo 1 | python um.py`; without it the program locks) — not postulated (seal: '
        '`the_axiom_reading_v385`).')
    T['ao_lado_posto_sintese'] = (
        '> ⚠ **Beside (v385):** where the synthesis above tags the axiom `ω(I) = 1` **[POSTULATE]**, read **[POSTO]**: the 1 is posited '
        '— inscribed by the observer (`echo 1 | python um.py`) — not postulated (seal: `the_axiom_reading_v385`). *Em português: no resumo '
        'abaixo, onde o axioma aparece como [POSTULATE], leia [POSTO] — posto, inscrito pelo observador, não postulado.*')
    T['regua_llms_ascii'] = (
        'os ritos, com falsificadores pre-registrados, nao falsificaram nenhuma das %d cobrancas da teoria (%d leitura excluida; as %d rotas do '
        'eco ficam fora do livro como rotas examinadas: %d excluidas pelo dado, %d inconclusivas)'
        % (v['n_ch'], c['EXCLUDED_IN_READING'], v['eco_n'], v['eco_ex'], v['eco_inc']))
    T['ao_lado_posto_abaixo'] = (
        '> ⚠ **Beside (v385):** where the synthesis below tags the axiom `ω(I) = 1` **[POSTULATE]**, read **[POSTO]**: the 1 is posited '
        '— inscribed by the observer (`echo 1 | python um.py`) — not postulated (seal: `the_axiom_reading_v385`). *Em português: onde a '
        'síntese abaixo marca o axioma como [POSTULATE], leia [POSTO] — posto, inscrito pelo observador, não postulado.*')
    T['ao_lado_regua_0509'] = (
        '> ⚠ **Beside (07/10/2026):** the ruler of 05/09 above stays as record. Under the operator’s ruling of 07/10, **CONFIRMED = '
        'accreditation** (the act of observers outside the artifact, not validity), and PROVED is read in the operator’s definition of '
        'proof of 09/09 (*sufficient and verifiable ballast*), on three supports that count together, each with its own status — see '
        'the top of this page. What the kernel proves is still the implication; what nature decides is still not proved.')
    T['ao_lado_regua_0509_pt'] = (
        '> ⚠ **Ao lado (07/10/2026):** a régua de 05/09 acima fica como registro. Pela decisão do operador de 07/10, **CONFIRMADA = '
        'creditação** (o ato de observadores externos ao artefato, não validade), e PROVADA se lê na definição de prova do operador de 09/09 '
        '(*lastro suficiente e verificável*), em três apoios que contam juntos, cada um com o seu estatuto. O que o kernel prova continua sendo a '
        'implicação; o que a natureza decide continua não provado.')
    return T


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:  # noqa: BLE001
        pass
    E = ler()
    print(json.dumps({k: v for k, v in E.items() if not isinstance(v, str) or len(v) < 200}, ensure_ascii=False, indent=1))
    print()
    print(E['secao_md'])
