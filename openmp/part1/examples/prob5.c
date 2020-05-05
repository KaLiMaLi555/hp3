#include <stdio.h>
#include <omp.h>

int main()
{
	int i,x=0, a[100000],b[100000];

	double wtime;
	omp_set_num_threads(8);
	wtime = omp_get_wtime();

	for(i=0;i<100000;i++) {
		a[i]=i;
		b[i]=100000-i;
	}

	printf("Elapsed time %lf\n", omp_get_wtime()-wtime);

	wtime = omp_get_wtime();

	#pragma omp parallel
	{
		#pragma omp for reduction (+:x)
			for(i=0;i<100000;i++) {
				x = x + a[i]*b[i];
			}
	}

	printf("Elapsed time %lf\n", omp_get_wtime()-wtime);
	printf("%d\n",x);
	return 0;
}
