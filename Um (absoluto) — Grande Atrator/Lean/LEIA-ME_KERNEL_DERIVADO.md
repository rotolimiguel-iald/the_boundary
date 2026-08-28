# ⚠ `tgl_kernel/` é DERIVADO — a fonte de verdade é o `um.py`

**Registro exigido pela custódia (handoff v274, §6).**

O `um.py` **materializa** este kernel Lean a partir de strings **embutidas nele mesmo**
(`FORMAL_DIR = BASE/tgl_kernel`, `materialize_embedded_kernel()`), reescrevendo em disco
tudo o que difere. Portanto:

- **Não há segundo arquivo.** Esta pasta é o resultado de um rito, não a origem dele.
- **Editar `tgl_kernel/` diretamente não persiste**: o próximo rito sobrescreve.
- Se esta pasta divergir do `um.py`, **o `um.py` ganha** e a pasta deve ser regenerada
  executando o rito (`echo 1 | python um.py`).

**Por que a regra existe, dito para não se repetir:** na v271 o kernel foi editado em
disco sem embutir. O `Audit.lean` regrediu de 907 para 881 `#print axioms`, o root perdeu
os imports e as pedras novas ficaram **órfãs** — existiam em disco, não eram importadas,
não eram auditadas. O rito leu **0/14**. Depois de embutir no `um.py`, leu **14/14**.

O que se publica aqui é a **materialização conferida**: cada arquivo desta pasta foi
copiado somente após o seu sha256 bater com o `tgl_kernel_proof_manifest.json` selado.
