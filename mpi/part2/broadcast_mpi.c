// Author: Rishabh Singh
// Comparison of MPI_Bcast with the my_bcast function

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>

int rank, size;
int n;

void my_bcast(void* data, int count, MPI_Datatype datatype, int root,
              MPI_Comm communicator){

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (n >= 2){
    if (rank == 0)
      {
        MPI_Send(data, count, MPI_INT, 1, 0, MPI_COMM_WORLD);
      }
    else
      {
        MPI_Recv(data, count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
  }

  else if (n >= 4){
    if (rank < 2)
    {
      MPI_Send(data, count, MPI_INT, rank + 2, 0, MPI_COMM_WORLD);
    }
    else
    {
      MPI_Recv(data, count, MPI_INT, rank - 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  
  else if (n >= 8)
  {
    if (rank < 4)
    {
      MPI_Send(data, count, MPI_INT, rank + 4, 0, MPI_COMM_WORLD);
    }
    else
    {
      MPI_Recv(data, count, MPI_INT, rank - 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}


int main(int argc, char** argv) {

  int num_elements;
  // int num_trials = atoi(argv[2]);

  MPI_Init(&argc, &argv);

  int world_rank, node;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &node);

  double total_my_bcast_time = 0.0;
  double total_mpi_bcast_time = 0.0;
  int i, *data;

   num_elements = 30;

   if(world_rank == 0)
   {
     printf("\nTaking number of elements to be 30 in the array and initialising serially\n");
  //   scanf("%d", &num_elements);
     data = (int*)malloc(sizeof(int) * num_elements);
  //   printf("\nEnter the elements of the array:");
     for(i=0;i<num_elements;i++)
      {
        data[i] = i;
  //     scanf("%d", &data[i]);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    total_my_bcast_time -= MPI_Wtime();
    my_bcast(data, num_elements, MPI_INT, 0, MPI_COMM_WORLD);
    // Synchronize again before obtaining final time
    MPI_Barrier(MPI_COMM_WORLD);
    total_my_bcast_time += MPI_Wtime();

    // Time MPI_Bcast
    MPI_Barrier(MPI_COMM_WORLD);
    total_mpi_bcast_time -= MPI_Wtime();
    MPI_Bcast(data, num_elements, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    total_mpi_bcast_time += MPI_Wtime();

  // Print off timing information
  if (world_rank == 0)
  {
    printf("Data size = %d\n", num_elements * (int)sizeof(int));
    printf("Average Custom Broadcast time = %lf\n", total_my_bcast_time );
    printf("Average MPI_Bcast time = %lf\n", total_mpi_bcast_time);
  }

  MPI_Finalize();
}
