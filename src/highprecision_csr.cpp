#include "highprecision_csr.hpp"
#include <gmp.h>
#include <unsupported/Eigen/MPRealSupport>

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