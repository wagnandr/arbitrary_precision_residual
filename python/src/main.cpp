#include <pybind11/pybind11.h>
#include "highprecision_csr.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring
  m.def("set_precision", set_precision , "Sets the default precision in our library");
  m.def("test", [](){}, "Sets the default precision in our library");

  py::class_<CSRMatrix>(m, "CSRMatrix")
    ;

}