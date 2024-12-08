#include "highprecision_csr.hpp"
#include <gmp.h>
#include <unsupported/Eigen/MPRealSupport>

void test()
{
  unsigned int i;
  mpfr_t s, t, u;

  mpfr_init2(t, 200);
  mpfr_set_d(t, 1.0, MPFR_RNDD);
  mpfr_init2(s, 200);
  mpfr_set_d(s, 1.0, MPFR_RNDD);
  mpfr_init2(u, 200);
  for (i = 1; i <= 100; i++)
  {
    mpfr_mul_ui(t, t, i, MPFR_RNDU);
    mpfr_set_d(u, 1.0, MPFR_RNDD);
    mpfr_div(u, u, t, MPFR_RNDD);
    mpfr_add(s, s, u, MPFR_RNDD);
  }
  printf("Sum is ");
  mpfr_out_str(stdout, 10, 0, s, MPFR_RNDD);
  putchar('\n');
  mpfr_clear(s);
  mpfr_clear(t);
  mpfr_clear(u);
  mpfr_free_cache();

  // test
  Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1> vec(2);

  vec(0) = "3.14159265358979323846264338327950288419716939937510";
  vec(1) = "2.71828182845904523536028747135266249775724709369995";
}

CSRMatrix create_csr_matrix(
    const std::vector<double> &data,
    const std::vector<int> &indices,
    const std::vector<int> &indptr,
    const size_t rows,
    const size_t cols)
{
  // Create an Eigen sparse matrix
  CSRMatrix mat(rows, cols);

  // Reserve space for non-zero elements
  mat.reserve(data.size());

  // Fill the matrix
  for (int row = 0; row < rows; ++row)
  {
    for (int idx = indptr[row]; idx < indptr[row + 1]; ++idx)
    {
      mat.insert(row, indices[idx]) = data[idx];
    }
  }

  // Finalize the matrix
  mat.makeCompressed();

  return mat;
}