#include <pybind11/pybind11.h>
#include "highprecision_csr.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_hpresidual_core, m)
{
  m.doc() = "pybind11 example plugin"; // optional module docstring
  m.def("set_precision", set_precision, "Sets the default precision in our library");

  py::class_<CSRMatrix>(m, "Matrix");
  py::class_<Vector>(m, "Vector");

  py::class_<HPResidual>(m, "HPResidual")
      .def(py::init<double const *const, int const *const, int const *const, size_t, size_t, size_t>())
      .def("set_x_from_string", &HPResidual::set_x_from_string)
      .def("copy_to_x", &HPResidual::copy_to_x)
      .def("copy_from_x", &HPResidual::copy_from_x)
      .def("add_to_x", &HPResidual::add_to_x)
      .def("evaluate", &HPResidual::evaluate);
}