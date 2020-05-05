#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stddef.h>
#include "mpi.h"


#define RAND_MAX_2 ((1U << 31) - 1)

int random_custom(int rseed)
{
	return ((rseed * 1103515245 + 12345) & RAND_MAX_2);
}


int main(int argc, char *argv[])
{
	int i, j, k, rank, size, tag = 99, blksz, sum = 0;

	int N;


	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(rank == 0){
			printf("Input size of the square matrix: \n");
			scanf("%d", &N);
	}

	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int a[N][N],b[N][N];
	int c[N][N];
	int aa[N],cc[N];

	int seed;

	if(rank == 0){
		   seed = rand();

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
		{
			seed = random_custom(seed);
			a[i][j] = seed;
		}

	printf("The matrix A elements by lehmer random_customly generater numbers are:\n");
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			printf("%d ", (a[i][j]));
		}
	printf("\n");
	}

	seed = rand();

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
		{
			seed = random_custom(seed);
			b[i][j] = seed;
		}

	printf("The matrix B elements by lehmer random_customly generater numbers are:\n");
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			printf("%d ", (b[i][j]));
	}
	printf("\n");
	}
	}

	//scatter rows of first matrix to different processes
	MPI_Scatter(a, N*N/size, MPI_INT, aa, N*N/size, MPI_INT,0,MPI_COMM_WORLD);

	//broadcast second matrix to all processes
	MPI_Bcast(b, N*N, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

		  //perform vector multiplication by all processes
		for (i = 0; i < N; i++)
			{
				for (j = 0; j < N; j++)
				{
						sum = sum + aa[j] * b[j][i];  //MISTAKE_WAS_HERE
				}
				cc[i] = sum;
				sum = 0;
			}

	MPI_Gather(cc, N*N/size, MPI_INT, c, N*N/size, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	if (rank == 0){                         //I_ADDED_THIS
		printf ("\n");
			for (i = 0; i < N; i++) {
				for (j = 0; j < N; j++) {
					printf(" %d", a[i][j]);
			}
			printf ("\n");
			}
		printf ("\n\n");
	}
}