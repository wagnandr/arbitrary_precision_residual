#include <iostream>
#include <unsupported/Eigen/MPRealSupport>
#include <Eigen/Sparse>

void test();

using CSRMatrix = Eigen::SparseMatrix<mpfr::mpreal>;

CSRMatrix create_csr_matrix(
    const std::vector<double> &data,
    const std::vector<int> &indices,
    const std::vector<int> &indptr,
    const size_t rows,
    const size_t cols);
