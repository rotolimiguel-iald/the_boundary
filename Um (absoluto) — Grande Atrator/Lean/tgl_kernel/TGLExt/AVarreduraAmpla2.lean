import TGLExt.AVarreduraAmpla1

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

namespace TGLExt.AVarreduraAmpla2

open TGLExt.ABateriaDaParede TGLExt.ABateriaMinimaN5 TGLExt.ATensaoMinima TGLExt.AVarreduraAmpla1
open TGLExt.TraducaoTuring

/-- ★★ A varredura II: oito sementes a mais, todas conferem. -/
theorem a_varredura_II_confere :
    confere 2665
      ∧ confere 2996
      ∧ confere 3327
      ∧ confere 3658
      ∧ confere 3989
      ∧ confere 4320
      ∧ confere 4651
      ∧ confere 4982
  := by decide

end TGLExt.AVarreduraAmpla2

