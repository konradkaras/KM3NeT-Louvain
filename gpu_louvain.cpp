#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

#include <iostream>

using namespace std;
namespace py = pybind11;

extern int *kernel_wrapper(int n, int m_edges, int *col_idx, int *prefix_sums, int *degrees, float resolution, float threshold);

py::array_t<int> runLouvain(int n, py::array_t<int> col_idx, py::array_t<int> prefix_sums, py::array_t<int> degrees, float resolution, float threshold)
{
    cout << "Running Louvain..." << endl;

    auto col_idx_info = col_idx.request();
    auto prefix_sums_info = prefix_sums.request();
    auto degrees_info = degrees.request();

    int *ptr_col_idx = (int *) col_idx_info.ptr;
    int *ptr_prefix_sums = (int *) prefix_sums_info.ptr;
    int *ptr_degrees = (int *) degrees_info.ptr;

    int m_edges = col_idx_info.shape[0];

    printf("n=%d\n", n);
    printf("m_edges=%d\n", m_edges);

    int *result = kernel_wrapper(n, m_edges, ptr_col_idx, ptr_prefix_sums, ptr_degrees, resolution, threshold);

    return py::array_t<int>(n, result);
}

PYBIND11_MODULE(gpu_louvain, m)
{
    m.def("runLouvain", &runLouvain, "Runs Louvain algorithm");
}