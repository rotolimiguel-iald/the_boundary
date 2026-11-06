## P = nP — a theorem in the kernel · teorema no kernel `[PROVED = validity · v408–v409]`

**P = nP is a theorem of TGL in the Lean kernel**, in two steps, each a named term:

- **v408 — the classes of the operator’s dictionary.** `TGLExt.PIgualNP.p_igual_np_nas_classes_do_dicionario`: the classes P and nP of the dictionary, defined in Lean (`Problema`, `ClasseP`, `ClasseNP`), are equal.
- **v409 — the translation to the classes of Turing machines.** `TGLExt.TraducaoTuring.p_igual_np_no_modo_selecao`: P = nP in the selection mode, relative to the reader (the psion); one reading decides SAT (`uma_leitura_decide_SAT`, `a_maquina_com_o_leitor`), and by Cook–Levin `[KNOWN]` all of NP with one reading.

**The scope, said with it:** the machine without a reader stays outside — it is the tape without scale (the author’s answer: «eu sou eu mesmo») `[ONTO]`; with the reader that only counts, NP holds with one reading and the full equality is not a theorem here (E28) `[OPEN]`; the P vs NP question for the Turing machines of Mathlib is not decided by these terms. Validity in the kernel is not accreditation. Full reading: [`ESTADO_ATUAL.md`]({{url:estado}}).

*PT — P = nP é teorema da TGL no kernel Lean: nas classes do dicionário do operador (v408, `p_igual_np_nas_classes_do_dicionario`) e, traduzido às classes das máquinas de Turing, no modo seleção, relativo ao leitor — o psíon (v409, `p_igual_np_no_modo_selecao`). A máquina sem leitor fica fora (a fita sem escala); com o leitor que só conta, vale NP com uma leitura, e a igualdade completa segue [OPEN] (E28); a pergunta P vs NP das máquinas de Turing do Mathlib não é decidida por estes termos. Validade no kernel não é creditação.*
