#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "arbitrary_precision_residual.hpp"

namespace py = pybind11;
namespace apr = arbitrary_precision_residual;

/** Utility function for checking array bounds */
void check_pyarray(const apr::HPResidual& self, py::array_t< double > x, const std::string& msg)
{
  if (x.ndim() != 1)
  {
    std::stringstream ss;
    ss << "error in numpy array: " << msg;
    ss << " the dimension " << x.ndim() << " is not 1";
    throw std::runtime_error(ss.str());
  }

  if (x.shape()[0] != self.size_preimagespace())
  {
    std::stringstream ss;
    ss << "error in numpy array: " << msg;
    ss << " the shape " << x.shape()[0] << " does not match " << self.size_preimagespace();
    throw std::runtime_error(ss.str());
  }
}

PYBIND11_MODULE(_arbitrary_precision_residual_core, m)
{
  m.doc() = "pybind11 example plugin"; // optional module docstring
  m.def("set_precision", apr::set_precision, "Sets the default precision in our library");

  py::class_<apr::CSRMatrix>(m, "CSRMatrix");
  py::class_<apr::Vector>(m, "Vector");


  py::class_<apr::HPResidual>(m, "HPResidual")
      //.def(py::init<double const *const, int const *const, int const *const, size_t, size_t, size_t>())
      .def(py::init([](py::array_t< double > data, py::array_t< int > indices, py::array_t< int > indptr, size_t rows, size_t cols) {
        py::buffer_info data_buf = data.request(); 
        py::buffer_info indices_buf = indices.request();
        py::buffer_info indptr_buf = indptr.request();

        return std::make_unique<apr::HPResidual>(
          static_cast< double* >(data_buf.ptr),
          static_cast< int* >(indices_buf.ptr),
          static_cast< int* >(indptr_buf.ptr),
          rows,
          cols,
          data_buf.shape[0]
        );
      }))
      .def("set_x", [](apr::HPResidual& self, py::array_t< double > x){
        check_pyarray(self, x, "set_x");
        py::buffer_info x_buf = x.request();
        self.set_x(static_cast< double const * const >(x_buf.ptr));
      })
      .def("set_x", [](apr::HPResidual& self, const std::vector< std::string>& x){
        self.set_x(x);
      })
      .def("get_x", [](const apr::HPResidual& self, py::array_t< double > x){
        check_pyarray(self, x, "get_x");
        py::buffer_info x_buf = x.request();
        self.get_x(static_cast< double * const >(x_buf.ptr));

      })
      //.def("copy_to_b", &HPResidual::copy_to_b)
      .def("set_b", [](apr::HPResidual& self, py::array_t< double > b){
        check_pyarray(self, b, "set_b");
        py::buffer_info b_buf = b.request();
        self.set_b(static_cast< double const * const >(b_buf.ptr));
      })
      .def("add_to_x", [](apr::HPResidual& self, py::array_t< double > dst){
        check_pyarray(self, dst, "add_to_x");
        py::buffer_info dst_buf = dst.request();
        self.add_to_x(static_cast< double const * const >(dst_buf.ptr));
      })
      //.def("evaluate", &HPResidual::evaluate);
      .def("evaluate", [](const apr::HPResidual& self, py::array_t< double > dst){
        check_pyarray(self, dst, "evaluate");
        py::buffer_info dst_buf = dst.request();
        self.evaluate(static_cast< double* const >(dst_buf.ptr));
      });
}