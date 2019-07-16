// #define DEBUG_MODE 1
#define DEBUG_NODE 3

#include <stdio.h>
#include <algorithm>
#include <numeric>
#include <cmath>

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc_allocator.h>

#include "ResponseWrapper.h"

__global__ void move_nodes(int n_tot, int *d_col_idx, int *d_prefix_sums, int *d_community_idx, int *d_community_sizes, int *d_tmp_community_idx, int *d_tmp_community_sizes, float resolution) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_tot) {

        //define neighbour range
        int start = 0;
        if (i>0) {
            start = d_prefix_sums[i-1];
        }
        int end = d_prefix_sums[i];

        //modularity
        int current_comm = d_community_idx[i];
        int new_comm = current_comm;
        int n_i = 1;

        bool local_set = false;
        float local_q = 0;
        float max_q = 0;

        #ifdef DEBUG_MODE
        if(i == DEBUG_NODE) {
            printf("....\n");
            printf("========================\n");
            printf("NODE: %d ,COMM: %d \n", i, current_comm);
            printf("START: %d ,END: %d \n", start, end);
            printf("-----------------\n");
        }
        #endif

        //iterate over neighbours of i 
        for(int j = start; j < end; j++) {

            int col = d_col_idx[j];

            //get community of neighbour
            int col_comm = d_community_idx[col];

            int n_comm = d_community_sizes[col_comm];

            // The singlet minimum HEURISTIC
            // if(i == current_comm && d_community_sizes[current_comm] == 1 && col == col_comm && n_comm == 1 && col_comm > current_comm) {
            //     #ifdef DEBUG_MODE
            //     if(i == DEBUG_NODE) {
            //         printf("$$$");
            //         printf("SKIP CHANGE %d to %d \n", i, col);
            //         printf("$$$");
            //     }
            //     #endif

            //     continue;
            // }

            int k_i_comm = 0;   //sum of weights of edges joining i with community
            //search for other neighbors from this community
            for(int n = start; n < end; n++) {
                int col_n = d_col_idx[n];
                //check if its from the same community
                if(d_community_idx[col_n] != col_comm) {
                    continue;
                }

                k_i_comm++;
            }

            local_q = - ( 2*k_i_comm - (2 * n_i * resolution * n_comm) );

            #ifdef DEBUG_MODE
            if(i == DEBUG_NODE) {
                printf("test %d to %d \n", i, col_comm);
                printf("n_i = %d \n", n_i);
                printf("k_i_comm = %d \n", k_i_comm);
                printf("n_comm = %d \n", n_comm);
                printf("local_q = %f \n", local_q);
                printf("------------------ \n");
            }
            #endif

            if(!local_set || local_q <= max_q) {
                if(local_set && local_q == max_q && new_comm < col_comm) {
                    //do nothing
                } else {
                    #ifdef DEBUG_MODE
                    if(i == DEBUG_NODE) {
                        printf("######################\n");
                        printf("migrated [%d] from %d to %d \n", i, new_comm, col_comm);
                        printf("previous q: %f , current q: %f \n", max_q, local_q);
                        printf("######################\n");
                        printf("------------------ \n");
                    }
                    #endif

                    local_set = true;
                    new_comm = col_comm;
                    max_q = local_q;
                }
            }
        }

        d_tmp_community_idx[i] = new_comm;   
        atomicAdd(&d_tmp_community_sizes[new_comm], 1); 
        atomicSub(&d_tmp_community_sizes[current_comm], 1); 
    }
}

__global__ void calculate_community_internal_edges(int n_tot, int *d_col_idx, int *d_prefix_sums, int *d_tmp_community_idx, int *d_tmp_community_inter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_tot) {
        int inter_count = 0;
       
        //define neighbour range
        int start = 0;
        if (i>0) {
            start = d_prefix_sums[i-1];
        }
        int end = d_prefix_sums[i];
        int current_comm = d_tmp_community_idx[i];

        //iterate over neighbours of i 
        for (int j = start; j < end; j++) {
            int col = d_col_idx[j];
            if (d_tmp_community_idx[col] == current_comm) {
                inter_count++;
            }
        }

        atomicAdd(&d_tmp_community_inter[current_comm], inter_count);
    }
}

__global__ void calculate_part_modularity(int n_tot, int *d_tmp_community_inter, int *d_tmp_community_sizes, float *d_part_mod, float resolution) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_tot) {
        float ec = (float) (d_tmp_community_inter[i] / 2.0);
        float nc = (float) (d_tmp_community_sizes[i]);
        d_part_mod[i] = - ( ec - (resolution * nc * nc) );
    }
}

// LINEAR
#define class_slope -0.015
#define class_const 1.0

//LOG
#define class_log_slope -0.333
#define class_log_const 1.583

// EXPOTENTIAL
#define class_exp_a 1.6
#define class_exp_b 0.92
#define class_exp_c 0.2
#define class_exp_xs 4.6

// #define class_size_limit 10.0
// #define class_dens_limit 0.25

#define class_size_limit 29.0
#define class_dens_limit 0.0


__global__ void classify_communities(int n_tot, int *d_community_inter, int *d_community_sizes, int *d_community_class) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_tot) {
        float ec = (float) (d_community_inter[i] / 2.0);
        float nc = (float) (d_community_sizes[i]);
        float density = ec / ((nc*(nc-1.0)) / 2.0);
        if (isnan(density)) {
            density = 0.0;
        }

        // LINEAR
        // float density_limit = class_slope * nc + class_const;

        // LOG
        // float density_limit = class_log_slope * logf(nc) + class_log_const;
        // if(density_limit < class_dens_limit) {
        //     density_limit = class_dens_limit;
        // }

        // EXPOTENTIAL
        // float density_limit = ( class_exp_a * ( pow(class_exp_b, (nc + class_exp_xs)) ) ) + class_exp_c;

        int comm_class = 0;
        if (nc > class_size_limit && density > class_dens_limit) {
            comm_class = 1;
        }
        d_community_class[i] = comm_class;

        // printf("Community: %d \t ec: %g \t nc: %g \t density: %f \t class: %d \n", i, ec, nc, density, comm_class);
    }
}

__global__ void classify_hits(int n_tot, int *d_community_idx, int *d_community_class, int *d_hit_class) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_tot) {
        int comm_idx = d_community_idx[i];
        d_hit_class[i] = d_community_class[comm_idx];
    }
}


// -----------------------------------------------

int * kernel_wrapper(int n, int m_edges, int *col_idx, int *prefix_sums, int *degrees, float resolution, float threshold, ResponseWrapper *wrapper) {


    int block_size = 1024;                          //thread block size
    int nblocks = int(ceilf(n/(float)block_size));  //problem size divided by thread block size rounded up
    dim3 grid(nblocks, 1);
    dim3 threads(block_size, 1, 1);

    // cudaError_t err;

    int *h_community_idx = (int *) malloc(n*sizeof(int));
    std::iota(h_community_idx, h_community_idx + n, 0);

    int *h_community_inter = (int *) malloc(n*sizeof(int));
    std::iota(h_community_inter, h_community_inter + n, 0);

    int *h_hit_class = (int *) malloc(n*sizeof(int));
    std::iota(h_hit_class, h_hit_class + n, 0);

    // define GPU memory pointers  
    int *d_col_idx;
    int *d_prefix_sums;

    int *h_community_sizes = (int *) malloc(n*sizeof(int));
    std::fill_n(h_community_sizes, n, 1);
    int *d_community_sizes;

    int *d_community_idx;

    int *d_tmp_community_idx;
    int *d_tmp_community_sizes;

    int *d_community_inter;
    int *d_tmp_community_inter;

    float *d_part_mod;
    int *d_community_class;
    int *d_hit_class;

    // allocate GPU memory  
    cudaMalloc((void **)&d_col_idx, m_edges*sizeof(int));
    cudaMalloc((void **)&d_prefix_sums, n*sizeof(int));

    cudaMalloc((void **)&d_community_sizes, n*sizeof(int));

    cudaMalloc((void **)&d_community_idx, n*sizeof(int));
    
    cudaMalloc((void **)&d_tmp_community_idx, n*sizeof(int));
    cudaMalloc((void **)&d_tmp_community_sizes, n*sizeof(int));

    cudaMalloc((void **)&d_community_inter, n*sizeof(int));
    cudaMalloc((void **)&d_tmp_community_inter, n*sizeof(int));

    cudaMalloc((void **)&d_part_mod, n*sizeof(float));
    cudaMalloc((void **)&d_community_class, n*sizeof(int));
    cudaMalloc((void **)&d_hit_class, n*sizeof(int));

    // copy data to GPU
    cudaMemcpy(d_col_idx, col_idx, m_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix_sums, prefix_sums, n*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_community_idx, h_community_idx, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_community_sizes, h_community_sizes, n*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_tmp_community_idx, h_community_idx, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tmp_community_sizes, h_community_sizes, n*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(d_community_inter, 0, n*sizeof(int));
    cudaMemset(d_community_class, 0, n*sizeof(int));
    cudaMemset(d_hit_class, 0, n*sizeof(int));

    
    float phase_modularity = 0;

    int iterations = 0;
    int iteration_limit = 15;

    while(true) {
        // 1. Calculate best moves
        move_nodes<<<grid, threads>>>(n, d_col_idx, d_prefix_sums, d_community_idx, d_community_sizes, d_tmp_community_idx, d_tmp_community_sizes, resolution);

        // 2. Calculate internal edges
        cudaMemset(d_tmp_community_inter, 0, n*sizeof(int));
        cudaMemset(d_part_mod, 0, n*sizeof(int));

        calculate_community_internal_edges<<<grid, threads>>>(n, d_col_idx, d_prefix_sums, d_tmp_community_idx, d_tmp_community_inter);
        calculate_part_modularity<<<grid, threads>>>(n, d_tmp_community_inter, d_tmp_community_sizes, d_part_mod, resolution);

        thrust::device_ptr<float> mod_ptr = thrust::device_pointer_cast(d_part_mod);
        float current_modularity = thrust::reduce(mod_ptr, mod_ptr + n);

        // printf("RESOLUTION: %.2f | MODULARITY: %.3f\n", resolution, current_modularity);

        float iter_diff = fabsf((current_modularity - phase_modularity)/phase_modularity);

        if(iter_diff <= threshold || iterations > iteration_limit) {
            // printf("ITERATIONS: %d | SCORE: %.6f\n", iterations, phase_modularity);
            break;
        } else {
            iterations++;
            phase_modularity = current_modularity;
            cudaMemcpy(d_community_idx, d_tmp_community_idx, n*sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_community_sizes, d_tmp_community_sizes, n*sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_community_inter, d_tmp_community_inter, n*sizeof(int), cudaMemcpyDeviceToDevice);
        }
    }

    classify_communities<<<grid, threads>>>(n, d_community_inter, d_community_sizes, d_community_class);
    classify_hits<<<grid, threads>>>(n, d_community_idx, d_community_class, d_hit_class);

    // copy the result to host
    cudaMemcpy(h_community_idx, d_community_idx, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_community_inter, d_community_inter, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_community_sizes, d_community_sizes, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hit_class, d_hit_class, n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_col_idx);
    cudaFree(d_prefix_sums);

    cudaFree(d_community_idx);
    cudaFree(d_community_sizes);

    cudaFree(d_part_mod);

    cudaFree(d_tmp_community_idx);
    cudaFree(d_tmp_community_sizes);

    cudaFree(d_tmp_community_inter);

    cudaFree(d_community_inter);
    cudaFree(d_community_class);
    cudaFree(d_hit_class);

    wrapper->community_idx = h_community_idx;
    wrapper->community_sizes = h_community_sizes;
    wrapper->community_inter = h_community_inter;
    wrapper->hit_classes = h_hit_class;

    return h_hit_class;
}