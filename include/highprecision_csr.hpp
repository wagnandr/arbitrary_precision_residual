#include <iostream>
#include <unsupported/Eigen/MPRealSupport>
#include <Eigen/Sparse>

void test();

using CSRMatrix = Eigen::SparseMatrix<mpfr::mpreal>;
using Vector = Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1>;

CSRMatrix create_csr_matrix(
    const std::vector<double> &data,
    const std::vector<int> &indices,
    const std::vector<int> &indptr,
    const size_t rows,
    const size_t cols);


Vector create_vector(const std::vector<std::string> & data);

void set_precision(size_t prec);