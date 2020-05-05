/*
####################################################
Ass. 1: 3D Rotation

Rishabh Singh, Kirtan Mali
####################################################
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

using namespace std;
#define PI acos(-1)

typedef struct coordinate
{
    float x;
    float y;
    float z;
}coordinate;


// ====================================
// Utility function for printing a matrix of row and col
void printMat(float **input, int rows, int cols)
{
    for (int i=0;i<rows;i++)
    {
        for (int j=0;j<cols;j++)
        {
            printf("%0.3lf ", input[i][j]);
        }
        printf("\n");
    }
}

// Utility function for transpose of a give 4x4 matrix
float ** transpose(float **input)
{
	float **output;
	output = (float **)malloc(4*sizeof(float *));

	for (int i=0; i<4; i++)
	{
		output[i] = (float *)malloc(4*sizeof(float));
	}
	for (int i = 0; i < 4; ++i)
	{
        for (int j = 0; j < 4; ++j)
        {
    	    output[j][i] = input[i][j];
        }
    }

    return output;
}

// Utility function for empty mat of size
float **EmptyMat(int rows,int cols)
{
    float **output;
    output = (float **)malloc(rows*sizeof(float *));

    for (int i=0;i<rows;i++)
    {
        output[i] = (float *)malloc(cols*sizeof(float));
    }

    for (int i=0;i<rows;i++)
    {
        for (int j=0;j<cols;j++)
        {
            output[i][j] = 0.0;
        }
    }

    return output;
}

float ** transMat(coordinate P)
{
    float **output = EmptyMat(4, 4);

    for (int i=0;i<4;i++)
    {
        output[i][i] = 1.0;
    }

    output[3][0] = -P.x;
    output[3][1] = -P.y;
    output[3][2] = -P.z;

    return output;
}

float ** inv_transMat(coordinate P)
{
    float **output = EmptyMat(4, 4);

    for (int i=0;i<4;i++)
    {
        output[i][i] = 1.0;
    }

    output[3][0] = P.x;
    output[3][1] = P.y;
    output[3][2] = P.z;

    return output;
}

float ** rotate_x(coordinate p)
{
    float **output = EmptyMat(4, 4);

    float b = sqrt(p.y*p.y + p.z*p.z);

    output[0][0] = 1;
    output[1][1] = p.z/b;
    output[1][2] = p.y/b;
    output[2][1] = (-1*p.y)/b;
    output[2][2] = p.z/b;
    output[3][3] = 1;

    return output;
}

float ** rotate_y(coordinate p)
{
    float **output = EmptyMat(4, 4);

    float b = sqrt(p.y*p.y + p.z*p.z);

    output[0][0] = b;
	output[0][2] = p.x;
	output[1][1] = 1;
	output[2][0] = -1*p.x;
	output[2][2] = b;
	output[3][3] = 1;

    return output;
}

float ** rotate_z(float angle)
{
	float **output;

	angle = (angle*PI)/180;

	output = EmptyMat(4, 4);

	output[0][0] = cos(angle);
	output[0][1] = sin(angle);
	output[1][0] = -1*sin(angle);
	output[1][1] = cos(angle);
	output[2][2] = 1;
	output[3][3] = 1;

	return output;
}

float** MatMul(float **a,float **b)
{
	float **c = EmptyMat(4, 4);
    int i,j,k;
    #pragma omp parallel shared(a,b,c) private(i,j,k)
    {
        #pragma omp for schedule(static)
        for(i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            c[i][j]=0;
        }

        #pragma omp for schedule(static)
        for(i=0;i<4;i++)
        {
            for(j=0;j<4;j++)
            {
                for(k=0;k<4;k++)
                c[i][j]+=a[i][k]*b[k][j];
            }
        }
    }
    return c;
}

// ====================================

int main(int argc, char ** argv)
{
    if (argc < 5)
    {
        printf("Missing a parameter\n");
        exit(1);
    }

    int NUM_THREADS = atoi(argv[1]);
    float angle_of_rotation = strtof(argv[4], NULL);
    omp_set_num_threads(NUM_THREADS);

    FILE *filename4arbitraryAxis = fopen(argv[2], "r");
    FILE *filename4objectFile = fopen(argv[3], "r");

    if (filename4objectFile == NULL)
    {
        printf("File for object not present");
        exit(1);
    }

    if (filename4arbitraryAxis == NULL)
    {
        printf("File for axis not present");
        exit(1);
    }

    char line[100];

    float axis[6];
    int index = -1;
    int prev = 0;

    // Reading axis from the file
    while (fgets(line, 100, filename4arbitraryAxis))
    {
        for (int i=0;i<strlen(line);i++)
        {
            if (line[i] == '(' || line[i] == ',')
            {
                axis[++index] = strtof(line+i+1, NULL);
            }
        }
        break;
    }

    coordinate p, q, val;
    p.x = axis[0];
    p.y = axis[1];
    p.z = axis[2];
    q.x = axis[3];
    q.y = axis[4];
    q.z = axis[5];

    float d = sqrt((q.x-p.x)*(q.x-p.x) + (q.y-p.y)*(q.y-p.y) + (q.z-p.z)*(q.z-p.z));
    // q.x -= p.x;
    // q.y -= p.y;
    // q.z -= p.z;
    // p.x = 0;
    // p.y = 0;
    // p.z = 0;

    val.x = (q.x-p.x)/d;
    val.y = (q.y-p.y)/d;
    val.z = (q.z-p.z)/d;
    int size = 0;
    float **object;

    // Reading object from the file
    while (fgets(line, 100, filename4objectFile))
    {
        if (line[0] == '\n')
        {
            break;
        }

        int index = 0;
        size++;
        object = (float **)realloc(object, size * sizeof(*object));

        object[size-1] = (float *)malloc(3*sizeof(float));

        while (line[index] == ' ')
        {
            index++;
        }

        object[size-1][0] = strtof(line+index, NULL);

        while (line[index] != ' ')
        {
            index++;
        }

        object[size-1][1] = strtof(line+index, NULL);

        while (line[index] == ' ')
        {
            index++;
        }

        while (line[index] != ' ')
        {
            index++;
        }

        object[size-1][2] = strtof(line+index, NULL);
    }

    float **translation = transMat(p);
    float **inv_translation = inv_transMat(p);
    float **R_X = rotate_x(val);
    float **R_Y = rotate_y(val);
    float **R_Z = rotate_z(angle_of_rotation);
    float **inv_R_X = transpose(R_X);
    float **inv_R_Y = transpose(R_Y);
    float **inv_R_Z = transpose(R_Z);

    float **mat1, **mat2, **mat3, **mat4, **mat5, **final_transform, **rotated_object;

    int i, j, k;

        mat1 = MatMul(R_X, translation);
        mat2 = MatMul(R_Z, R_Y);
        mat3 = MatMul(inv_R_X, inv_R_Y);



        mat4 = MatMul(mat2, mat1);
        mat5 = MatMul(inv_translation, mat3);


        final_transform = MatMul(mat5, mat4);


        rotated_object = EmptyMat(size, 3);

        #pragma omp for schedule (dynamic) private(i)
            for(i=0;i<size;i++)
            {
                rotated_object[i][0] += final_transform[0][0]*object[i][0]+final_transform[0][1]*object[i][1]+final_transform[0][2]*object[i][2]+final_transform[0][3];
                rotated_object[i][1] += final_transform[1][0]*object[i][0]+final_transform[1][1]*object[i][1]+final_transform[1][2]*object[i][2]+final_transform[1][3];
                rotated_object[i][2] += final_transform[2][0]*object[i][0]+final_transform[2][1]*object[i][1]+final_transform[2][2]*object[i][2]+final_transform[2][3];
            }


    printMat(rotated_object, size, 3);

    // printf("Elapsed time %lf\n", omp_get_wtime()-new);

    // for (int i=0;i<size;i++)
    // {
    //     for (int j=0;j<4;j++)
    //     {
    //         printf("%lf ", output[i][j]);
    //     }

    //     printf("\n");
    //     break;
    // }

    return 0;
}
