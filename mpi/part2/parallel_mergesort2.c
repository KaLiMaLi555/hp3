
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <string.h>

#define nodes 8
#define SEED 1
#define RAND_MAX_2 ((1U << 7) - 1)

int random_custom(int rseed)
{
	return ((rseed * 1103515245 + 12345) & RAND_MAX_2);
}
int create_rand()
{
    int seed = rand();

    return random_custom(seed);
}

void printarray(long long int arr[], long long int size)
{
    for (long long int i=0;i<size;i++)
    {
        printf("%lld. \t ", i);
        printf("%lld \n ", arr[i]);

    }
    printf("\n");
}

void linearmerge(long long int arr[], long long int l, long long int m, long long int r);
void mergeSort(long long int arr[], long long int l, long long int r);
void parallelmerge(long long int half1[], long long int half2[], long long int out[], long long int size1, long long int size2);

int main(int argc, char **argv)
{
    int world_rank, world_size;
    long long int tot_arraysize = 1000000000;

    int subgroupsize = 2;

    MPI_Init(&argc, &argv);

    int row_rank, row_size;
    long long int div_arraysize;
    div_arraysize = (long long int)tot_arraysize/nodes;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm row_comm;
    int group = world_rank % subgroupsize;
    MPI_Comm_split(MPI_COMM_WORLD, group, world_rank, &row_comm);

    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    srand(SEED*world_rank+1);

    long long int *div_arr, *tot_arr = NULL;

    div_arr = (long long int *)malloc(div_arraysize * sizeof(long long int));

    for (long long int i=0;i<div_arraysize;i++)
    {
        div_arr[i] = create_rand();
    }

    mergeSort(div_arr, 0, div_arraysize-1);

    MPI_Finalize();
}

void parallelmerge(long long int half1[], long long int half2[], long long int out[], long long int size1, long long int size2)
{
    long long int i, j, k;
    i = 0;
    j = 0;
    k = 0;

    while (i < size1 && j < size2)
    {
        if (half1[i] <= half2[j])
        {
            out[k] = half1[i];
            i++;
            k++;
        }
        else
        {
            out[k] = half2[j];
            j++;
            k++;
        }
    }

    while (i < size1)
    {
        out[k] = half1[i];
        i++;
        k++;
    }

    while (j < size2)
    {
        out[k] = half2[j];
        j++;
        k++;
    }
}

void linearmerge(long long int arr[],long long int l, long long int m, long long int r)
{
    long long int i, j, k;
    long long int n1 = m - l + 1;
    long long int n2 =  r - m;

    /* create temp arrays */
    long long int L[n1], R[n2];

    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1+ j];

    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = l; // Initial index of merged subarray
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there
       are any */
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there
       are any */
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}

/* l is for left index and r is right index of the
   sub-array of arr to be sorted */
void mergeSort(long long int arr[], long long int l, long long int r)
{
    if (l < r)
    {
        // Same as (l+r)/2, but avoids overflow for
        // large l and h
        long long int m = l+(r-l)/2;

        // Sort first and second halves
        mergeSort(arr, l, m);
        mergeSort(arr, m+1, r);

        linearmerge(arr, l, m, r);
    }
}