#include <stdio.h>
#include <omp.h>

int main()
{
	int i,x=0, a[10],b[10];

	for(i=0;i<10;i++) {
		a[i]=i;
		b[i]=10-i;
	}
	#pragma omp parallel
	{
		#pragma omp for
			for(i=0;i<10;i++) {
				#pragma omp atomic
				x += a[i]*b[i];
			}
	}
	printf("%d\n",x);
	return 0;
}
