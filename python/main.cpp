#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "highprecision_csr.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_hpresidual_core, m)
{
  m.doc() = "pybind11 example plugin"; // optional module docstring
  m.def("set_precision", set_precision, "Sets the default precision in our library");

  py::class_<CSRMatrix>(m, "Matrix");
  py::class_<Vector>(m, "Vector");

  py::class_<HPResidual>(m, "HPResidual")
      //.def(py::init<double const *const, int const *const, int const *const, size_t, size_t, size_t>())
      .def(py::init([](py::array_t< double > data, py::array_t< int > indices, py::array_t< int > indptr, size_t rows, size_t cols) {
        py::buffer_info data_buf = data.request(); 
        py::buffer_info indices_buf = indices.request();
        py::buffer_info indptr_buf = indptr.request();

        return std::make_unique<HPResidual>(
          static_cast< double* >(data_buf.ptr),
          static_cast< int* >(indices_buf.ptr),
          static_cast< int* >(indptr_buf.ptr),
          rows,
          cols,
          data_buf.shape[0]
        );
      }))
      .def("set_x", [](HPResidual& self, py::array_t< double > x){
        py::buffer_info x_buf = x.request();
        self.set_x(static_cast< double const * const >(x_buf.ptr));
      })
      .def("set_x", [](HPResidual& self, const std::vector< std::string>& x){
        self.set_x(x);
      })
      .def("get_x", [](const HPResidual& self, py::array_t< double > x){
        py::buffer_info x_buf = x.request();
        self.get_x(static_cast< double * const >(x_buf.ptr));

      })
      //.def("copy_to_b", &HPResidual::copy_to_b)
      .def("set_b", [](HPResidual& self, py::array_t< double > b){
        py::buffer_info b_buf = b.request();
        self.set_b(static_cast< double const * const >(b_buf.ptr));
      })
      .def("add_to_x", [](HPResidual& self, py::array_t< double > dst){
        py::buffer_info dst_buf = dst.request();
        self.add_to_x(static_cast< double const * const >(dst_buf.ptr));
      })
      //.def("evaluate", &HPResidual::evaluate);
      .def("evaluate", [](const HPResidual& self, py::array_t< double > dst){
        py::buffer_info dst_buf = dst.request();
        self.evaluate(static_cast< double* const >(dst_buf.ptr));
      });
}