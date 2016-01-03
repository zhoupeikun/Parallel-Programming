#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<sys/time.h>
#include<sys/time.h>
#define H  1000
#define W  1000

int main(int argc, char *argv[])
{
	int matrice[H][W];
	int ligne[H];
	int colonne[W];
	int i = 0, j = 0;
	
	srand(time(NULL));

	for(i = 0; i < H; i++)
	{
	  for(j = 0; j < W; j++)
	   {
	     matrice[H][W] = rand() % 1000;		
	    }	
	}
	
	struct timeval tim;
	gettimeofday(&tim, NULL);
	double t1 = tim.tv_sec + (tim.tv_usec/1000000.0);

	//un vectrue de taille H, la somme de chaque ligne
	#pragma omp parallel for
	for(i = 0; i < H; i++)
	{
	  for(j = 0; j < W; j++)
	   {
	     ligne[H]= matrice[H][W] + ligne[H]	;
	   }   
	}

	//un vecteur de taille W, la somme de chaque colonne
	#pragma omp parallel for
	for(i = 0; i < W; i++)
	{
	  for(j = 0; j < H; j++)
	   {
	     colonne[W]= matrice[H][W] + colonne[W];
	   }	
	}

	gettimeofday(&tim, NULL);
	double t2 = tim.tv_sec + (tim.tv_usec/100000.0);

	printf("Temps total = %f sec\n", t2 - t1);

	return 0;
}







