import TGLExt.AVarreduraAmpla1

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

namespace TGLExt.AVarreduraAmpla3

open TGLExt.ABateriaDaParede TGLExt.ABateriaMinimaN5 TGLExt.ATensaoMinima TGLExt.AVarreduraAmpla1
open TGLExt.TraducaoTuring

/-- ★★ A varredura III: oito sementes a mais, todas conferem. -/
theorem a_varredura_III_confere :
    confere 5313
      ∧ confere 5644
      ∧ confere 5975
      ∧ confere 6306
      ∧ confere 6637
      ∧ confere 6968
      ∧ confere 7299
      ∧ confere 7630
  := by decide

end TGLExt.AVarreduraAmpla3

