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

  std::vector<std::string> nums = {"3.14159265358979323846264338327950288419716939937510", "2.71828182845904523536028747135266249775724709369995"};
  Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1> vec2(nums.size());
  for (size_t i = 0; i < nums.size(); i += 1)
    vec2[i] = nums[i];
}

CSRMatrix create_csr_matrix(
    const std::vector<double> &data,
    const std::vector<int> &indices,
    const std::vector<int> &indptr,
    const size_t rows,
    const size_t cols)
{
  return create_csr_matrix(data.data(), indices.data(), indptr.data(), rows, cols, data.size());
}

CSRMatrix create_csr_matrix(
    double const *const data,
    int const *const indices,
    int const *const indptr,
    const size_t rows,
    const size_t cols,
    const size_t data_size)
{
  // Create an Eigen sparse matrix
  CSRMatrix mat2(rows, cols);

  typedef Eigen::Triplet<mpfr::mpreal> T;
  std::vector<T> tripletList;
  tripletList.reserve(data_size);
  for (int row = 0; row < rows; ++row)
  {
    for (int idx = indptr[row]; idx < indptr[row + 1]; ++idx)
    {
      tripletList.emplace_back(row, indices[idx], data[idx]);
    }
  }
  mat2.setFromTriplets(tripletList.begin(), tripletList.end());

  return mat2;
}

Vector create_vector(const std::vector<std::string> &data)
{
  Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1> vec(data.size());
  for (size_t i = 0; i < data.size(); i += 1)
    vec[i] = data[i];
  return vec;
}

void set_precision(size_t prec)
{
  mpfr::mpreal::set_default_prec(prec);
}


HPResidual::HPResidual(
    double const* const data,
    int const* const indices,
    int const* const indptr,
    const size_t rows,
    const size_t cols,
    const size_t data_size
) : A(create_csr_matrix(data, indices, indptr, rows, cols, data_size))
, b(Vector(rows))
, x(Vector(cols))
{}

void HPResidual::set_x(const std::vector<std::string> &data)
{
  for(size_t i = 0; i < x.size(); i += 1)
    x[i] = data[i];
}

void HPResidual::set_x(double const* const src)
{
  for(size_t i = 0; i < x.size(); i += 1)
    x[i] = src[i];
}

void HPResidual::set_b(double const* const src)
{
  for(size_t i = 0; i < b.size(); i += 1)
    b[i] = src[i];
}

void HPResidual::get_x(double * const dst) const
{
  for(size_t i = 0; i < x.size(); i += 1)
    dst[i] = x[i].toDouble();
}

void HPResidual::add_to_x(double const *const to_add)
{
  for(size_t i = 0; i < x.size(); i += 1)
    x[i] += to_add[i];

}

void HPResidual::evaluate(double *const dst) const
{
  Vector d = b - A * x;
  for(size_t i = 0; i < d.size(); i += 1)
    dst[i] = d[i].toDouble();
}