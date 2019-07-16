#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

#include "KernelWrapper.h"

#include <iostream>

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(gpu_louvain, m)
{
    // m.def("runLouvain", &runLouvain, "Runs Louvain algorithm");

    py::class_<KernelWrapper>(m, "KernelWrapper")
        .def(py::init())
        .def("get_community_idx", &KernelWrapper::get_community_idx)
        .def("get_community_sizes", &KernelWrapper::get_community_sizes)
        .def("get_community_inter", &KernelWrapper::get_community_inter)
        .def("get_hit_classes", &KernelWrapper::get_hit_classes)
        .def("run", &KernelWrapper::run);
}