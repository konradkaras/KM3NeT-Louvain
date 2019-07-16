#ifndef RESPONSE_WRAPPER_H
#define RESPONSE_WRAPPER_H

typedef struct
{
    int *community_idx;
    int *community_sizes;
    int *community_inter;
    int *hit_classes;
} ResponseWrapper;

#endif