// #define DEBUG_MODE 1

#include <stdio.h>
#include <algorithm>
#include <numeric>
#include <cmath>

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc_allocator.h>

__global__ void move_nodes(int n_tot, int m_tot, int *d_col_idx, int *d_weights, int *d_prefix_sums, int *d_degrees, int *d_community_idx, int *d_community_degrees, int *d_tmp_community_idx, int *d_tmp_community_degrees, float resolution) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_tot) {

        //define neighbour range
        int start = 0;
        if (i>0) {
            start = d_prefix_sums[i-1];
        }
        int end = d_prefix_sums[i];

        int deg_i = d_degrees[i];

        //modularity
        int current_comm = d_community_idx[i];
        int new_comm = current_comm;
        float local_q = 0;
        float max_q = 0;

        //iterate over neighbours of i 
        for(int j = start; j < end; j++) {

            int col = d_col_idx[j];

            //get community of neighbour
            int col_comm = d_community_idx[col];

            int k_comm = d_community_degrees[col_comm];     //degree of community

            // The singlet minimum HEURISTIC
            if(i == current_comm && deg_i == d_community_degrees[current_comm] && col == col_comm && d_degrees[col] == k_comm && col_comm > current_comm) {
                #ifdef DEBUG_MODE
                if(i == 0) {
                    printf("$$$");
                    printf("SKIP CHANGE %d to %d \n", i, col);
                    printf("$$$");
                }
                #endif

                continue;
            }

            // int k_i_comm = col_weights[j];   //sum of weights of edges joining i with community
            int k_i_comm = 0;   //sum of weights of edges joining i with community
            //search for other neighbors from this community
            for(int n = start; n < end; n++) {
                int col_n = d_col_idx[n];
                //check if its from the same community
                if(d_community_idx[col_n] != col_comm) {
                    continue;
                }

                // k_i_comm += d_weights[n];
                k_i_comm++;
            }

            local_q = (1.0 / (float)m_tot) * ((float)k_i_comm - (resolution * (float)deg_i * (float)k_comm / (2.0 * (float)m_tot)));

            #ifdef DEBUG_MODE
            if(i == 0) {
                printf("=============== \n");
                printf("migrate %d to %d \n", i, col_comm);
                printf("m_tot = %d \n", m_tot);
                // printf("comm_inter_deg = %d \n", comm_inter_deg);
                printf("k_i_comm = %d \n", k_i_comm);
                printf("deg_i = %d \n", deg_i);
                printf("k_comm = %d \n", k_comm);
                printf("local_q = %f \n", local_q);
            }
            #endif

            if(local_q >= max_q) {
                if(local_q == max_q && new_comm < col_comm) {
                    //do nothing
                } else {
                    #ifdef DEBUG_MODE
                    if(i ==0) {
                        printf("$$$$$ \n");
                        printf("migrated [%d] from %d to %d \n", i, new_comm, col_comm);
                        printf("previous q: %f , current q: %f \n", max_q, local_q);
                        printf("$$$$$ \n");
                    }
                    #endif

                    new_comm = col_comm;
                    max_q = local_q;
                }
            }
        }

        d_tmp_community_idx[i] = new_comm;   
        atomicSub(&d_tmp_community_degrees[current_comm], deg_i); 
        atomicAdd(&d_tmp_community_degrees[new_comm], deg_i); 
    }
}

__global__ void calculate_community_internal_edges(int n_tot, int *d_col_idx, int *d_weights, int *d_prefix_sums, int *d_tmp_community_idx, int *d_tmp_community_inter) {
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
                // inter_count += d_weights[j];
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

__global__ void calculate_part_modularity(int n_tot, int m_tot, int *d_tmp_community_inter_sum, int *d_tmp_community_degrees, float *d_part_mod, float resolution) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_tot) {
        float lc = (float)d_tmp_community_inter_sum[i];
        float kc = (float)d_tmp_community_degrees[i];
        float m = (float)m_tot;
        d_part_mod[i] = ( ( lc/m ) - (resolution * pow((kc)/(2*m), 2.0f)) );
    }
}


// -----------------------------------------------

int * kernel_wrapper(int n, int m_edges, int *col_idx, int *prefix_sums, int *degrees, float resolution, float threshold) {

    int block_size = 1024;                          //thread block size
    int nblocks = int(ceilf(n/(float)block_size));  //problem size divided by thread block size rounded up
    dim3 grid(nblocks, 1);
    dim3 threads(block_size, 1, 1);

    cudaError_t err;

    int m_tot = m_edges/2;
    int *h_community_idx = (int *) malloc(n*sizeof(int));
    std::iota(h_community_idx, h_community_idx + n, 0);

    // define GPU memory pointers  
    int *d_col_idx;
    int *d_prefix_sums;
    int *d_degrees;

    int *h_weights = (int *) malloc(m_edges*sizeof(int));
    std::fill_n(h_weights, m_edges, 1);
    int *d_weights;

    int *d_community_idx;
    int *d_community_degrees;

    int *d_tmp_community_idx;
    int *d_tmp_community_degrees;
    int *d_tmp_community_inter;
    int *d_tmp_community_inter_sum;
    float *d_part_mod;

    // allocate GPU memory  
    err = cudaMalloc((void **)&d_col_idx, m_edges*sizeof(int));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_col_idx: %s\n", cudaGetErrorString( err ));
    err = cudaMalloc((void **)&d_prefix_sums, n*sizeof(int));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_prefix_sums: %s\n", cudaGetErrorString( err ));
    err = cudaMalloc((void **)&d_degrees, n*sizeof(int));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_degrees: %s\n", cudaGetErrorString( err ));

    err = cudaMalloc((void **)&d_weights, m_edges*sizeof(int));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_weights: %s\n", cudaGetErrorString( err ));

    err = cudaMalloc((void **)&d_community_idx, n*sizeof(int));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_community_idx: %s\n", cudaGetErrorString( err ));
    err = cudaMalloc((void **)&d_community_degrees, n*sizeof(int));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_community_degrees: %s\n", cudaGetErrorString( err ));
    
    err = cudaMalloc((void **)&d_tmp_community_idx, n*sizeof(int));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_tmp_community_idx: %s\n", cudaGetErrorString( err ));
    err = cudaMalloc((void **)&d_tmp_community_degrees, n*sizeof(int));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_tmp_community_idx: %s\n", cudaGetErrorString( err ));
    err = cudaMalloc((void **)&d_tmp_community_inter, n*sizeof(int));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_tmp_community_inter: %s\n", cudaGetErrorString( err ));
    err = cudaMalloc((void **)&d_tmp_community_inter_sum, n*sizeof(int));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_tmp_community_inter_sum: %s\n", cudaGetErrorString( err ));
    err = cudaMalloc((void **)&d_part_mod, n*sizeof(float));
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMalloc d_part_mod: %s\n", cudaGetErrorString( err ));

    // copy data to GPU
    err = cudaMemcpy(d_col_idx, col_idx, m_edges*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device col_idx: %s\n", cudaGetErrorString( err ));
    err = cudaMemcpy(d_prefix_sums, prefix_sums, n*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device prefix_sums: %s\n", cudaGetErrorString( err ));
    err = cudaMemcpy(d_degrees, degrees, n*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device d_degrees: %s\n", cudaGetErrorString( err ));

    err = cudaMemcpy(d_weights, h_weights, m_edges*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device d_weights: %s\n", cudaGetErrorString( err ));

    err = cudaMemcpy(d_community_idx, h_community_idx, n*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device h_community_idx: %s\n", cudaGetErrorString( err ));
    err = cudaMemcpy(d_community_degrees, degrees, n*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device d_community_degrees: %s\n", cudaGetErrorString( err ));

    err = cudaMemcpy(d_tmp_community_idx, h_community_idx, n*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device d_tmp_community_idx: %s\n", cudaGetErrorString( err ));
    err = cudaMemcpy(d_tmp_community_degrees, degrees, n*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy host to device d_tmp_community_degrees: %s\n", cudaGetErrorString( err ));
    
    float phase_modularity = 0;
    while(true) {
    
        // 1. Calculate best moves
        move_nodes<<<grid, threads>>>(n, m_tot, d_col_idx, d_weights, d_prefix_sums, d_degrees, d_community_idx, d_community_degrees, d_tmp_community_idx, d_tmp_community_degrees, resolution);

        // 2. Calculate internal edges
        // todo: efficient way of creating 0 array every iteration
        err = cudaMemset(d_tmp_community_inter, 0, n*sizeof(int));
        if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemset d_tmp_community_inter: %s\n", cudaGetErrorString( err ));
        err = cudaMemset(d_tmp_community_inter_sum, 0, n*sizeof(int));
        if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemset d_tmp_community_inter_sum: %s\n", cudaGetErrorString( err ));
        err = cudaMemset(d_part_mod, 0, n*sizeof(float));
        if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemset d_part_mod: %s\n", cudaGetErrorString( err ));

        calculate_community_internal_edges<<<grid, threads>>>(n, d_col_idx, d_weights, d_prefix_sums, d_tmp_community_idx, d_tmp_community_inter);
        calculate_community_internal_sum<<<grid, threads>>>(n, d_tmp_community_idx, d_tmp_community_inter, d_tmp_community_inter_sum);
        calculate_part_modularity<<<grid, threads>>>(n, m_tot, d_tmp_community_inter_sum, d_tmp_community_degrees, d_part_mod, resolution);

        thrust::device_ptr<float> mod_ptr = thrust::device_pointer_cast(d_part_mod);
        float current_modularity = thrust::reduce(mod_ptr, mod_ptr + n);

        printf("RESOLUTION: %.1f | MODULARITY: %.6f\n", resolution, current_modularity);

        if((current_modularity - phase_modularity)/phase_modularity <= threshold) {
            break;
        } else {
            phase_modularity = current_modularity;
            err = cudaMemcpy(d_community_idx, d_tmp_community_idx, n*sizeof(int), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy device to device d_community_idx: %s\n", cudaGetErrorString( err ));
            err = cudaMemcpy(d_community_degrees, d_tmp_community_degrees, n*sizeof(int), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy device to device d_community_degrees: %s\n", cudaGetErrorString( err ));
        }
    }


    // copy the result to host
    err = cudaMemcpy(h_community_idx, d_tmp_community_idx, n*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) fprintf(stderr, "Error in cudaMemcpy device to host h_community_idx: %s\n", cudaGetErrorString( err ));

    cudaFree(d_col_idx);
    cudaFree(d_prefix_sums);
    cudaFree(d_degrees);
    cudaFree(d_community_idx);
    cudaFree(d_community_degrees);
    cudaFree(d_tmp_community_idx);

    return h_community_idx;
}