#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "highprecision_csr.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
namespace mc = macrocirculation;

static int len = 0;
static char * program_name = "";
static char ** argv = &program_name;

PYBIND11_MODULE(_core, m) {
  m.doc() = R"pbdoc(
        HPResidual library
        -----------------------
        .. currentmodule:: hpresidual 
        .. autosummary::
           :toctree: _generate
    )pbdoc";

  m.def("set_precision", set_precision , "Sets the default precision in our library");

  py::class_<CSRMatrix>(m, "CSRMatrix")
    ;



#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}