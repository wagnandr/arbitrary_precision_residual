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
      .def("set_x_from_string", &HPResidual::set_x_from_string)
      .def("copy_to_x", &HPResidual::copy_to_x)
      .def("copy_from_x", &HPResidual::copy_from_x)
      //.def("copy_to_b", &HPResidual::copy_to_b)
      .def("copy_to_b", [](HPResidual& self, py::array_t< double > b){
        py::buffer_info b_buf = b.request();
        self.copy_to_b(static_cast< double const * const >(b_buf.ptr));
      })
      .def("add_to_x", &HPResidual::add_to_x)
      //.def("evaluate", &HPResidual::evaluate);
      .def("evaluate", [](const HPResidual& self, py::array_t< double > dst){
        py::buffer_info dst_buf = dst.request();
        self.evaluate(static_cast< double* const >(dst_buf.ptr));
      });
}