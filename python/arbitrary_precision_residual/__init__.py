from _arbitrary_precision_residual_core import __doc__, set_precision, CSRMatrix, Vector
import _arbitrary_precision_residual_core as _core

def create(A):
    return _core.HPResidual(A.data, A.indices, A.indptr, A.shape[0], A.shape[1])
