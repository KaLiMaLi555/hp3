/*
Assignment 3 - Parallel Merge Sort - MPI
To compile:
	mpicc parallel_mergesort.c
To run using 8 nodes:
	mpirun -np 8 ./a.out
Output: Sorted array of 1,000,000,000 numbers.
[Tested till 1,000,000,000 numbers]

*/


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#define nodes 8
#define SEED 1
#define RAND_MAX_2 ((1U << 15) - 1)

int random_custom(int rseed)
{
	return ((rseed * 1103515245 + 12345) & RAND_MAX_2);
}
int create_rand()
{
    int seed = rand();

    return random_custom(seed);
}

void printarray(int arr[], int size)
{	int i = 0;
    for (i=0;i<size;i++)
    {
        // printf("%d. \t ", i);
        // printf("%d \n ", arr[i]);
        printf("%d ", arr[i]);
    }
    printf("\n");
    printf("Total number of values: %d. \n ", size);
    printf("\n");
}

void linearmerge(int arr[], int l, int m, int r);
void mergeSort(int arr[], int l, int r);
void parallelmerge(int half1[], int half2[], int out[], int size1, int size2);

int main(int argc, char **argv)
{
    int world_rank, world_size;
    int tot_arraysize = pow(10, 9);  // Change 8 to 9 for generating 1 billion numbers!

    int subgroupsize = 2;

    MPI_Init(&argc, &argv);

    int row_rank, row_size;
    int div_arraysize;
    div_arraysize = (int)tot_arraysize/nodes;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm row_comm;
    int group = world_rank % subgroupsize;
    MPI_Comm_split(MPI_COMM_WORLD, group, world_rank, &row_comm);

    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    srand(SEED*world_rank+1);

    int *div_arr, *tot_arr = NULL;

    div_arr = (int *)malloc(div_arraysize * sizeof(int));

    // Generating random numbers in each node
    //DEBUG STATEMENT
	//printf("\n --- Generating Random numbers --- world_rank: %d --- row_rank %d \n ", world_rank, row_rank);
    for (int i=0;i<div_arraysize;i++)
    {

        div_arr[i] = create_rand();
        /* DEBUG/TEST STATEMENT: To check the generated output, comment the above line and uncomment the below line */
        //div_arr[i] = i;

    }

    // Sorting in each node
    //DEBUG STATEMENT
	//printf("\n --- Random numbers generated --- world_rank: %d --- row_rank %d \n ", world_rank, row_rank);
	//printf("\n Merge Sort called --- world_rank: %d --- row_rank %d \n ", world_rank, row_rank);

    mergeSort(div_arr, 0, div_arraysize-1);

    //DEBUG STATEMENT
	//printf("\n Merge Sort completed --- world_rank: %d --- row_rank %d \n ", world_rank, row_rank);]

/* DEBUG STATEMENTS: Uncomment the following to print sorted array for a particular world_rank
	if(world_rank <=1)
	{
		printf("\nworld_rank:%d ", world_rank);
    printarray(div_arr, div_arraysize);

	}
*/
    // DEBUG STATEMENT:
	//printf("\n Merge Sort completed --- world_rank: %d --- row_rank %d \n ", world_rank, row_rank);


    int *out, *final;

    if (row_rank >= 2)
    {
        MPI_Send(div_arr, tot_arraysize/8, MPI_INT, row_rank-2, 0, row_comm);
        free(div_arr);
    }
    else
    {
        int *data = (int *)malloc((tot_arraysize/8) * sizeof(int));
        int *out1 = (int *)malloc((tot_arraysize/4) * sizeof(int));

        MPI_Recv(data, tot_arraysize/8, MPI_INT, row_rank+2, 0, row_comm, MPI_STATUS_IGNORE);
        parallelmerge(div_arr, data, out1, (tot_arraysize/8), (tot_arraysize/8));
        free(div_arr);

        if (row_rank == 1)
        {

            MPI_Send(out1, tot_arraysize/4, MPI_INT, 0, 0, row_comm);
            free(out1);
        }
        else if (row_rank == 0)
        {
            data = (int *)realloc(data, (tot_arraysize/4) * sizeof(int));
            out = (int *)malloc((tot_arraysize/2) * sizeof(int));


            MPI_Recv(data, tot_arraysize/4, MPI_INT, 1, 0, row_comm, MPI_STATUS_IGNORE);

            parallelmerge(out1, data, out, (tot_arraysize/4), (tot_arraysize/4));
            free(out1);
            free(data);

            int *temp;

            temp = (int *)malloc((tot_arraysize/2) * sizeof(int));
            final = (int *)malloc((tot_arraysize) * sizeof(int));

            if (world_rank == 1)
            {
                MPI_Send(out, tot_arraysize/2, MPI_INT, 0, 0, MPI_COMM_WORLD);
                free(out);

            }
            else if (world_rank == 0)
            {
                MPI_Recv(temp, tot_arraysize/2, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                parallelmerge(temp, out, final, tot_arraysize/2, tot_arraysize/2);
                free(temp);
                free(out);
            }
        }
    }


//-----------------------------------------------------------//
/* Uncomment the following to load Input generated array
    if(world_rank == 0){
    	tot_arr = (int *)malloc((tot_arraysize) * sizeof(int));
    }

    MPI_Gather(div_arr, tot_arraysize/8, MPI_INT, tot_arr, tot_arraysize/8, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
         // printf("\n********************\nOriginal array\n\n");
         // printarray(tot_arr, tot_arraysize);
         // printf("********************\n");
    }

*/
//-----------------------------------------------------------//

    // Printing Sorted Array
    if (world_rank == 0)
    {
        printf("\n********************\nSorted array: \n\n");
        printarray(final, tot_arraysize);
        printf("********************\n");
    }

    MPI_Finalize();
    return 0;
}

void parallelmerge(int half1[], int half2[], int out[], int size1, int size2)
{
    int i, j, k;
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

void linearmerge(int arr[], int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 =  r - m;

    /* create temp arrays */
    int *L = (int *)malloc(n1*sizeof(int));
    int *R = (int *)malloc(n2*sizeof(int));

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
    free(L);
    free(R);
}

/* l is for left index and r is right index of the
   sub-array of arr to be sorted */
void mergeSort(int arr[], int l, int r)
{
    if (l < r)
    {
        // Same as (l+r)/2, but avoids overflow for
        // large l and h
        int m = l+(r-l)/2;

        // Sort first and second halves
        mergeSort(arr, l, m);
        mergeSort(arr, m+1, r);
        linearmerge(arr, l, m, r);
    }
}