/*
Kirtan Mali (18AG10016)
MPI Assignment: Histogram Equalization and Sobel Filters
*/

#include <stdlib.h>
#include <math.h>
#include <png.h>
#include <time.h>
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#define THRESH 15


typedef struct Image
{
    int width;
    int height;
    int data[1];
} Image;


Image* blank_image(int height, int width)
{
	int size = width * height * sizeof(int);
	Image* image = (Image*)malloc(sizeof(int)+size);
	memset( image->data, 0, size );
	image->width = width;
	image->height = height;
	return image;
}

int get_pixel( Image* image, int i, int j )
{
	if ( i < 0 && i >= image->height && j < 0 && j >= image->width)
		return 0;
	return image->data[i*image->width+j];
}


void set_pixel( Image* image, int i, int j, int val )
{
	if ( i < 0 && i >= image->width && j < 0 && j >= image->height )
		return;
	image->data[i*image->width+j] = val;
}


Image* read_png_image( const char *file_name )
{
	png_structp png_ptr;
	png_infop info_ptr;
	png_uint_32 width, height;
	FILE *fp;
	Image* image = NULL;
	int x,y;

	if ((fp = fopen(file_name, "rb")) == NULL)(
	png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (png_ptr == NULL) {
		fclose(fp);
		return 0;
	}

	info_ptr = png_create_info_struct(png_ptr);
	if (info_ptr == NULL) {
		fclose(fp);
		png_destroy_read_struct(&png_ptr, NULL, NULL);
		return 0;
	}

	if (setjmp(png_jmpbuf(png_ptr))) {
		png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
		fclose(fp);
		return 0;
	}

	png_init_io(png_ptr, fp);
	png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_STRIP_16 | PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND | PNG_TRANSFORM_STRIP_ALPHA, NULL);

	png_bytep* row_pointers = png_get_rows(png_ptr, info_ptr);
	height = png_get_image_height(png_ptr, info_ptr);
	width = png_get_image_width(png_ptr, info_ptr);
	image = blank_image( height, width );

	for (int i = 0; i < height; i++ ) {
		for(int j = 0; j < width; j++ ) {
			int c = 0;
			unsigned char* ch = (unsigned char*)&c;
			unsigned char* array = row_pointers[i];

			ch[0] = array[j];
			set_pixel(image, i, j, c);
		}
	}

	png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
	fclose(fp);
	return image;
}


void save_image( Image* image, const char* filename ) {
	int i,j;
	FILE *fp;
	fp = fopen(filename, "w");
	fprintf(fp, "P2\n");
	fprintf(fp, "# Written by pgmwrite\n");
	fprintf(fp, "%d %d\n", image->width, image->height);
	fprintf(fp, "%d\n", 255);
	for(i=0;i<image->height; i++)
	{
		for(j=0; j<image->width; j++)
		{
			fprintf(fp, "%3d ", image->data[i*image->width + j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}


int main( int argc, char* argv[] )
{
	int rank, comm_size, imagesize;

	Image* image = read_png_image("sample1.png");

	imagesize = image->height * image->width;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	Image *histinput = blank_image(image->height/comm_size, image->width);
	Image *histoutput = blank_image(image->height/comm_size, image->width);
	Image *finalhist = blank_image(image->height, image->width);
	Image *finalout = blank_image(image->height, image->width);

	MPI_Scatter(&image->data, imagesize/comm_size, MPI_INT, histinput->data, imagesize/comm_size, MPI_INT, 0, MPI_COMM_WORLD);

	long int *freqdist = (long int*)malloc(256*sizeof(long int));
	long int *finalfreqdist = (long int*)malloc(256*sizeof(long int));

	for ( int i = 0; i < histinput->height; i++ ) {
		for( int j = 0; j < histinput->width; j++ ) {
			freqdist[histinput->data[i*histinput->width+j]] = freqdist[histinput->data[i*histinput->width+j]] +1;
		}
	}

	MPI_Allreduce(freqdist, finalfreqdist, 256, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

	for (int  i = 0; i < histinput->height; i++ ) {
		for( int j = 0; j < histinput->width; j++ ) {
			long int value = histinput->data[i*histinput->width+j];
			long sum = 0;
			for(int k=0;k<=value;k++)
			{
				sum = sum + finalfreqdist[k];
			}
			histoutput->data[i*histoutput->width+j] = abs(255*sum/imagesize);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	float sobelx[3][3] = {{-1,0,1}, {-2,0,2}, {-1,0,1}};
	float sobely[3][3] = {{-1,-2,-1}, {0,0,0}, {1,2,1}};
	for (int i=1;i<histoutput->height-1;i++)
	{
		for (int j=1;j<histoutput->width-1;j++)
		{
			float gx = 0;
			float gy = 0;
			for (int k=-1;k<=1;k++)
			{
				for (int m=-1;m<=1;m++)
				{
					gx += sobelx[k+1][m+1] * get_pixel(histoutput, i+k, j+m);
					gy += sobely[k+1][m+1] * get_pixel(histoutput, i+k, j+m);
				}
			}

			float grad = sqrt(pow(gx, 2)+pow(gy, 2));

			if (grad < THRESH)
			{
				grad = 0;
			}
			else
			{
				grad = 255;
			}

			set_pixel(histinput, i, j, (int)grad);
		}
	}


	MPI_Gather(&histoutput->data, imagesize/comm_size, MPI_INT, finalhist->data, imagesize/comm_size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather(&histinput->data, imagesize/comm_size, MPI_INT, finalout->data, imagesize/comm_size, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		save_image(finalhist, "histeql.pgm");
		save_image(finalout, "final.pgm");
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();
}

