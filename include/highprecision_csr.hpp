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

CSRMatrix create_csr_matrix(
    double const *const data,
    int const *const indices,
    int const *const indptr,
    const size_t rows,
    const size_t cols,
    const size_t data_size);

Vector create_vector(const std::vector<std::string> &data);

void set_precision(size_t prec);

class HPResidual
{
public:
    HPResidual(
        double const *const data,
        int const *const indices,
        int const *const indptr,
        const size_t rows,
        const size_t cols,
        const size_t data_size);

    void set_x(const std::vector<std::string> &data);
    void set_x(double const *const src);
    void set_b(double const *const src);
    void get_x(double *const dst) const;
    void add_to_x(double const *const to_add);
    void evaluate(double *const dst) const;

private:
    CSRMatrix A;
    Vector b;
    Vector x;
};