#include<stdio.h>
#include<omp.h>

int main()
	{
	int i=0;
	omp_set_num_threads(5);
	#pragma omp parallel num_threads(17) private(i)
	{
		i=omp_get_thread_num();		
		printf("Hello World from thread %d\n", i);
		
	}
	#pragma omp parallel num_threads(7) private(i)
	{
		i=omp_get_thread_num();		
		printf("Hello India from thread %d\n", i);
		
	}
	#pragma omp parallel num_threads(8) private(i)
	{
		i=omp_get_thread_num();		
		printf("Hello IIT Kgp from thread %d\n", i);
		
	}
	return 0;
	}
