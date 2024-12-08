from _hpresidual_core import __doc__, set_precision, Matrix, Vector
import _hpresidual_core as _core

class HPResidual:
    def __init__(self, A):
        self._cpp = _core.HPResidual( A.data, A.indices, A.indptrs, A.shape[0], A.shape[1], len(A.data) )
