#include<stdio.h>
#include<omp.h>

int main()
	{
	int i=0;
	omp_set_num_threads(5);
	#pragma omp parallel num_threads(10) private(i)
	{
		i=omp_get_thread_num();		
		printf("Hello World from thread %d\n", i);
		printf("Hello India from thread %d\n", i);
		printf("Hello IIT Kgp from thread %d\n", i);
	}
	return 0;
	}
