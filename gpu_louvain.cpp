#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

#include <iostream>

using namespace std;
namespace py = pybind11;

extern int *kernel_wrapper(int n, int m_edges, int *col_idx, int *prefix_sums, int *degrees, float resolution, float threshold);

py::array runLouvain(int n, py::array col_idx, py::array prefix_sums, py::array degrees, float resolution, float threshold)
{
    cout << "Running Louvain..." << endl;

    py::buffer_info col_idx_info = col_idx.request();
    py::buffer_info prefix_sums_info = prefix_sums.request();
    py::buffer_info degrees_info = degrees.request();

    auto ptr_col_idx = static_cast<int *>(col_idx_info.ptr);
    auto ptr_prefix_sums = static_cast<int *>(prefix_sums_info.ptr);
    auto ptr_degrees = static_cast<int *>(degrees_info.ptr);

    int m_edges = col_idx_info.shape[0];

    printf("n=%d\n", n);
    printf("m_edges=%d\n", m_edges);

    int *result = kernel_wrapper(n, m_edges, ptr_col_idx, ptr_prefix_sums, ptr_degrees, resolution, threshold);

    return py::array(n, result);
}

PYBIND11_MODULE(gpu_louvain, m)
{
    m.def("runLouvain", &runLouvain, "Runs Louvain algorithm");
}