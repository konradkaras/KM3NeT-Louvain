#include "KernelWrapper.h"
#include "ResponseWrapper.h"

#include <iostream>

extern int *kernel_wrapper(int n, int m_edges, int *col_idx, int *prefix_sums, int *degrees, float resolution, float threshold, ResponseWrapper *wrapper, float class_dens_limit, float class_size_limit);

void KernelWrapper::run(int n, py::array_t<int> col_idx, py::array_t<int> prefix_sums, py::array_t<int> degrees, float resolution, float threshold, float class_dens_limit, float class_size_limit)
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
    printf("class_dens_limit=%f\n", class_dens_limit);
    printf("class_size_limit=%f\n", class_size_limit);

    cout << "Running Louvain..." << endl;
    kernel_wrapper(n, m_edges, ptr_col_idx, ptr_prefix_sums, ptr_degrees, resolution, threshold, rw, class_dens_limit, class_size_limit);
    this->community_idx = rw->community_idx;
    this->community_sizes = rw->community_sizes;
    this->community_inter = rw->community_inter;
    this->hit_classes = rw->hit_classes;
    cout << "CUDA complete" << endl;
    cout << "---------------" << endl;

    // return py::array_t<int>(n, result);
}