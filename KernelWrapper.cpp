#include "KernelWrapper.h"
#include "ResponseWrapper.h"

#include <iostream>

extern int *kernel_wrapper(int n, int m_edges, int *col_idx, int *prefix_sums, int *degrees, float resolution, float threshold, ResponseWrapper *wrapper);

void KernelWrapper::run(int n, py::array_t<int> col_idx, py::array_t<int> prefix_sums, py::array_t<int> degrees, float resolution, float threshold)
{
    this->n = n;

    ResponseWrapper *rw = (ResponseWrapper *) malloc(sizeof *rw);

    auto col_idx_info = col_idx.request();
    auto prefix_sums_info = prefix_sums.request();
    auto degrees_info = degrees.request();

    int *ptr_col_idx = (int *)col_idx_info.ptr;
    int *ptr_prefix_sums = (int *)prefix_sums_info.ptr;
    int *ptr_degrees = (int *)degrees_info.ptr;

    int m_edges = col_idx_info.shape[0];

    printf("n=%d\n", n);
    printf("m_edges=%d\n", m_edges);

    cout << "Running Louvain..." << endl;
    kernel_wrapper(n, m_edges, ptr_col_idx, ptr_prefix_sums, ptr_degrees, resolution, threshold, rw);
    this->community_idx = rw->community_idx;
    this->community_sizes = rw->community_sizes;
    this->community_inter = rw->community_inter;
    this->hit_classes = rw->hit_classes;
    cout << "CUDA complete" << endl;

    // return py::array_t<int>(n, result);
}