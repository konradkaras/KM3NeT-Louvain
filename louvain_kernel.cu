// #define DEBUG_MODE 1
#define DEBUG_NODE 3

#include <stdio.h>
#include <algorithm>
#include <numeric>
#include <cmath>

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc_allocator.h>

__global__ void move_nodes(int n_tot, int *d_col_idx, int *d_prefix_sums, int *d_community_idx, int *d_community_sizes, int *d_tmp_community_idx, int *d_tmp_community_sizes, float *d_part_mod, float resolution) {
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
        int new_comm = d_community_idx[i];
        // int n_i = d_community_sizes[current_comm];
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

            // Previous:
            // local_q = (1.0 / (float)m_tot) * ((float)k_i_comm - (resolution * (float)deg_i * (float)k_comm / (2.0 * (float)m_tot)));

            // Changed:
            local_q = - ( 2*k_i_comm - (2 * n_i * resolution * n_comm) );

            #ifdef DEBUG_MODE
            if(i == DEBUG_NODE) {
                printf("test %d to %d \n", i, col_comm);
                printf("n_i = %d \n", n_i);
                // printf("comm_inter_deg = %d \n", comm_inter_deg);
                printf("k_i_comm = %d \n", k_i_comm);
                // printf("deg_i = %d \n", deg_i);
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
        // d_part_mod[i] = local_q;
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

        d_tmp_community_inter[i] = inter_count;
    }
}

__global__ void calculate_community_internal_sum(int n_tot, int *d_tmp_community_idx, int *d_tmp_community_inter, int *d_tmp_community_inter_sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_tot) {
        int comm_sum = 0;
        for(int j = 0; j < n_tot; j++) {
            if(d_tmp_community_idx[j] == i) {
                comm_sum += d_tmp_community_inter[j];
            }
        }

        // edges are bidirectional
        d_tmp_community_inter_sum[i] = comm_sum / 2;
    }
}

__global__ void calculate_part_modularity(int n_tot, int *d_tmp_community_inter_sum, int *d_tmp_community_sizes, float *d_part_mod, float resolution) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_tot) {
        float ec = (float)d_tmp_community_inter_sum[i];
        float nc = (float)d_tmp_community_sizes[i];
        // d_part_mod[i] = ( ( lc/m ) - (resolution * pow((kc)/(2*m), 2.0f)) );
        // d_part_mod[i] = ( ( lc/m ) - (resolution) );
        d_part_mod[i] = - ( ec - (resolution * nc * nc) );
    }
}


// -----------------------------------------------

int * kernel_wrapper(int n, int m_edges, int *col_idx, int *prefix_sums, int *degrees, float resolution, float threshold) {

    int block_size = 1024;                          //thread block size
    int nblocks = int(ceilf(n/(float)block_size));  //problem size divided by thread block size rounded up
    dim3 grid(nblocks, 1);
    dim3 threads(block_size, 1, 1);

    // cudaError_t err;

    int *h_community_idx = (int *) malloc(n*sizeof(int));
    std::iota(h_community_idx, h_community_idx + n, 0);

    // define GPU memory pointers  
    int *d_col_idx;
    int *d_prefix_sums;

    int *h_community_sizes = (int *) malloc(n*sizeof(int));
    std::fill_n(h_community_sizes, n, 1);
    int *d_community_sizes;

    int *d_community_idx;

    int *d_tmp_community_idx;
    int *d_tmp_community_sizes;

    int *d_tmp_community_inter;
    int *d_tmp_community_inter_sum;

    float *d_part_mod;

    // allocate GPU memory  
    cudaMalloc((void **)&d_col_idx, m_edges*sizeof(int));
    cudaMalloc((void **)&d_prefix_sums, n*sizeof(int));

    cudaMalloc((void **)&d_community_sizes, n*sizeof(int));

    cudaMalloc((void **)&d_community_idx, n*sizeof(int));
    
    cudaMalloc((void **)&d_tmp_community_idx, n*sizeof(int));
    cudaMalloc((void **)&d_tmp_community_sizes, n*sizeof(int));

    cudaMalloc((void **)&d_tmp_community_inter, n*sizeof(int));
    cudaMalloc((void **)&d_tmp_community_inter_sum, n*sizeof(int));

    cudaMalloc((void **)&d_part_mod, n*sizeof(float));

    // copy data to GPU
    cudaMemcpy(d_col_idx, col_idx, m_edges*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix_sums, prefix_sums, n*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_community_idx, h_community_idx, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_community_sizes, h_community_sizes, n*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_tmp_community_idx, h_community_idx, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tmp_community_sizes, h_community_sizes, n*sizeof(int), cudaMemcpyHostToDevice);
    
    float phase_modularity = 0;

    int iterations = 0;
    int iteration_limit = 15;

    while(true) {
        // 1. Calculate best moves
        move_nodes<<<grid, threads>>>(n, d_col_idx, d_prefix_sums, d_community_idx, d_community_sizes, d_tmp_community_idx, d_tmp_community_sizes, d_part_mod, resolution);

        // 2. Calculate internal edges
        cudaMemset(d_tmp_community_inter, 0, n*sizeof(int));
        cudaMemset(d_tmp_community_inter_sum, 0, n*sizeof(int));
        cudaMemset(d_part_mod, 0, n*sizeof(int));

        calculate_community_internal_edges<<<grid, threads>>>(n, d_col_idx, d_prefix_sums, d_tmp_community_idx, d_tmp_community_inter);
        calculate_community_internal_sum<<<grid, threads>>>(n, d_tmp_community_idx, d_tmp_community_inter, d_tmp_community_inter_sum);
        calculate_part_modularity<<<grid, threads>>>(n, d_tmp_community_inter_sum, d_tmp_community_sizes, d_part_mod, resolution);

        thrust::device_ptr<float> mod_ptr = thrust::device_pointer_cast(d_part_mod);
        float current_modularity = thrust::reduce(mod_ptr, mod_ptr + n);

        printf("RESOLUTION: %.2f | MODULARITY: %.3f\n", resolution, current_modularity);

        float iter_diff = fabsf((current_modularity - phase_modularity)/phase_modularity);

        if(iter_diff <= threshold || iterations > iteration_limit) {
            printf("ITERATIONS: %d | SCORE: %.6f\n", iterations, phase_modularity);
            break;
        } else {
            iterations++;
            phase_modularity = current_modularity;
            cudaMemcpy(d_community_idx, d_tmp_community_idx, n*sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_community_sizes, d_tmp_community_sizes, n*sizeof(int), cudaMemcpyDeviceToDevice);
        }
    }


    // copy the result to host
    cudaMemcpy(h_community_idx, d_community_idx, n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_col_idx);
    cudaFree(d_prefix_sums);

    cudaFree(d_community_idx);
    cudaFree(d_community_sizes);

    cudaFree(d_part_mod);

    cudaFree(d_tmp_community_idx);
    cudaFree(d_tmp_community_sizes);

    cudaFree(d_tmp_community_inter);
    cudaFree(d_tmp_community_inter_sum);

    return h_community_idx;
}