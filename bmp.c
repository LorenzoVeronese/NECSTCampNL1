#include <stdio.h>
#include <math.h> // compile with flag -lm to use this library
#include "bmp.h"

#define KERNEL_DIM 3
#define PI 3,14159


int loadBMP(char *filename, BMP_Image *image){
	/** Funzione per caricare un'immagine BMP da file
	Parametri:
		nome del file in lettura, puntatore alla struttura in cui memorizzare i dati dell'immagine
	Valore di ritorno:
		0 se il caricamento è avvenuto con successo, un numero diverso da 0 altrimenti
	*/
	FILE *fp;

	// "rb" because it's not a text file
	fp = fopen(filename, "rb");
	// error
	if(fp == NULL){
		printf("Error: impossibile aprire il file in lettura\n");
		return 1;
	}

	// read the first 2 chars and put in attribute magic of image (BMP_Image)
	fread(image->magic, sizeof(image->magic), 1, fp);
	// error
	if(image->magic[0]!='B' || image->magic[1]!='M'){
		printf("Error: tipo di immagine non corretto\n");
		return 2;
	}

	// read header and info
	fread(&image->header, sizeof(image->header), 1, fp);
	fread(&image->info, sizeof(image->info), 1, fp);
	// error
	if(image->info.bits!=8){
		printf("Error: numero di bits/pixel diverso da 8\n");
		return 3;
	}
	if(image->info.width!=DATA_DIM || image->info.height!=DATA_DIM){
		printf("--- Attenzione, dimensioni non corrette ---");
	}

	// read colors and data
	fread(&image->color_table, sizeof(image->color_table), 1, fp);
	fread(image->data, sizeof(image->data), 1, fp);

	fclose(fp);

	return 0;
}


int saveBMP(BMP_Image image, char * filename){
	/** Funzione per salvare una struttura BMP_Image su file
	Parametri:
		puntatore alla struttura da cui leggere i dati dell'immagine, nome del file su cui scrivere
	Valore di ritorno:
		0 se il salvataggio è avvenuto con successo, 1 altrimenti
	*/
	FILE *fp2;

	fp2 = fopen(filename, "wb");
	// error
	if(fp2==NULL){
		printf("Impossibile aprire il file in scrittura\n");
		return 1;
	}

	fwrite(&image.magic, sizeof(image.magic), 1, fp2);
	fwrite(&image.header, sizeof(image.header), 1, fp2);
	fwrite(&image.info, sizeof(image.info), 1, fp2);
	fwrite(&image.color_table, sizeof(image.color_table), 1, fp2);
	fwrite(image.data, sizeof(image.data), 1, fp2);

	fclose(fp2);

	return 0;
}


float gaussian_funct(int x, int y, float sigma){
	/**It gives the value of the gaussian function in a certain point
	PAR:
		int, x: position on x
		int, y: position on y
		unsigned int, sigma: standard deviation. It must be in the interval [0, 10]
	RETURN:
		value of the gaussian function in a certain point
	*/
	return (exp(-1*(x*x + y*y) / (2 * sigma*sigma)) / (M_PI * 2 * sigma*sigma));
	//return ((1 / 2*M_PI*sigma*sigma) * exp(-1*(x*x+y*y)/2*sigma*sigma));
}


void mk_gaussian_kernel(float gaussian_kernel[][KERNEL_DIM], float sigma){
	/**It builds gaussian filter kernel of dim = KERNEL_DIM
	PAR:
		gaussian_kernel: pointer to the matrix containing the kernel
		sigma: standard deviation. It must be in the interval [0, 10]
	RETURN:
		void
	*/
	int i, j, up, down;
	float normalizer;

	// Building kernel
	up = KERNEL_DIM/2;
	down = -1 * up;

	normalizer = 0;
	for(i = down; i <= up; i++){
		for(j = down; j <= up; j++){
			gaussian_kernel[i + up][j + up] = gaussian_funct(i, j, sigma);
			normalizer += gaussian_kernel[i + up][j + up];
		}
	}

	// Normalizing kernel
	for(i = 0; i < KERNEL_DIM; i++){
		for(j = 0; j < KERNEL_DIM; j++){
			gaussian_kernel[i + up][j + up] /= normalizer;
		}
	}

}

int main(){
	// kernel
	float gaussian_kernel[KERNEL_DIM][KERNEL_DIM];
	float sigma;
	// image
	BMP_Image picture;
	// utility
	int check;
	int i, j;


	loadBMP("Immagini/abdomen.bmp", &picture);

	sigma = 1;
	mk_gaussian_kernel(gaussian_kernel, sigma);

	for(i = 0; i <= 2; i++){
		for(j = 0; j <= 2; j++){
			printf("%f ", gaussian_kernel[i][j]);
		}
		printf("\n");
	}

	saveBMP(picture, "prova.bmp");


	return 0;
}
