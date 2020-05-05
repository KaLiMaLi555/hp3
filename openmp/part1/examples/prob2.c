#include <stdio.h>
#include <omp.h>

int main()
{
	int ii;

	#pragma omp parallel
	{
//	#pragma omp for
		for(ii=0;ii<10;ii++)
			printf("iteration %d from thread %d\n", ii,omp_get_thread_num());
//		#pragma omp barrier
		printf("%d\n",omp_get_thread_num());
	}
	return 0;
}
