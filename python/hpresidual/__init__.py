from _hpresidual_core import __doc__, set_precision, Matrix, Vector
import _hpresidual_core as _core

def create(A):
    return _core.HPResidual(A.data, A.indices, A.indptr, A.shape[0], A.shape[1])
