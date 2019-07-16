#ifndef KERNEL_WRAPPER_H
#define KERNEL_WRAPPER_H

// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <pybind11/iostream.h>
#include <pybind11/numpy.h>

// #include <iostream>

using namespace std;
namespace py = pybind11;

struct KernelWrapper
{

    KernelWrapper() {}

    py::array_t<int> get_community_idx()
    {
        return py::array_t<int>(n, community_idx);
    }

    py::array_t<int> get_community_sizes()
    {
        return py::array_t<int>(n, community_sizes);
    }

    py::array_t<int> get_community_inter()
    {
        return py::array_t<int>(n, community_inter);
    }

    py::array_t<int> get_hit_classes()
    {
        return py::array_t<int>(n, hit_classes);
    }

    void run(int n, py::array_t<int> col_idx, py::array_t<int> prefix_sums, py::array_t<int> degrees, float resolution, float threshold, float class_dens_limit, float class_size_limit);

    int n;
    int *community_idx;
    int *community_sizes;
    int *community_inter;
    int *hit_classes;
};

#endif