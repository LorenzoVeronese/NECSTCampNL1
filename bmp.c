#include <stdio.h>
#include <math.h> // compile with flag -lm to use this library
#include <stdlib.h>
#include "bmp.h"

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



float** mk_gaussian_kernel(float sigma, unsigned int kernel_dim){
	/**It builds gaussian filter kernel of dim = KERNEL_DIM
	PAR:
		gaussian_kernel: pointer to the matrix containing the kernel
		sigma: standard deviation. It must be in the interval [0, 10]
	RETURN:
		void
	*/
	float** gaussian_kernel;
	int i, j, up, down;
	float normalizer;

	// Creating kernel
    gaussian_kernel = (float**)malloc(kernel_dim * sizeof(float*));
    for(i = 0; i < kernel_dim; i++){
        gaussian_kernel[i] = (float*)malloc((kernel_dim) * sizeof(float));
    }

	// Building kernel
	up = kernel_dim/2;
	down = -1 * up;

	normalizer = 0;
	for(i = down; i <= up; i++){
		for(j = down; j <= up; j++){
			gaussian_kernel[i + up][j + up] = gaussian_funct(i, j, sigma);
			normalizer += gaussian_kernel[i + up][j + up];
		}
	}

	// Normalizing kernel
	for(i = 0; i < kernel_dim; i++){
		for(j = 0; j < kernel_dim; j++){
			gaussian_kernel[i][j] /= normalizer;
		}
	}

	return gaussian_kernel;
}



void apply_gaussian_filter(BMP_Image* image, float** gaussian_kernel, unsigned int kernel_dim){
	// This will be used to add a black border to the picture
	Pixel** bordered;
	int i, j, h, k;
	unsigned char prova;
	unsigned int border_dim;


	// Creating the black bordered image
	border_dim = kernel_dim/2;
	bordered = (Pixel**)malloc((DATA_DIM + (border_dim*2)) * sizeof(Pixel*));
    for(i = 0; i < (DATA_DIM + ((kernel_dim/2)*2)); i++){
        bordered[i] = (Pixel*)calloc(((DATA_DIM + (border_dim*2))), sizeof(Pixel));
    }

	// i and j slide bordered, h and k slide image
	for(i = border_dim, h = 0; h < DATA_DIM; i++, h++){
		for(j = border_dim, k = 0; k < DATA_DIM; j++, k++){
			bordered[i][j].grey = image -> data[h][k].grey;
		}
	}


	// Apply filter
	// i, j is the up-left angle to start
	for(i = 0; i < DATA_DIM; i++){
		for(j = 0; j < DATA_DIM; j++){
			for(h = 0; h < kernel_dim; h++){
				for(k = 0; k < kernel_dim; k++){
					//D printf("%d, %d\n", i, j);
					bordered[i + h][j + k].grey *= gaussian_kernel[h][k];
					//D bordered[i + h][j + k].grey = 0;
				}
			}
		}
	}

	// i and j slide bordered, h and k slide image
	for(i = border_dim, h = 0; h < DATA_DIM; i++, h++){
		for(j = border_dim, k = 0; k < DATA_DIM; j++, k++){
			printf("%d\n", 'b' - 'a');
			image -> data[h][k].grey = bordered[i][j].grey - 48;
			//printf("%d\n", image -> data[h][k].grey);
		}
	}


	/*D
	for(i = 0; i < DATA_DIM; i++){
		for(j = 0; j < DATA_DIM; j++){
			image.data[i][j].grey = 0;
		}
	}*/

}



int main(){
	// kernel
	float** gaussian_kernel;
	float sigma;
	unsigned int kernel_dim;
	// image
	BMP_Image image;
	// utility
	int check;
	int i, j;


	loadBMP("Immagini/brain.bmp", &image);

	sigma = 1;
	kernel_dim = 3;
	gaussian_kernel = mk_gaussian_kernel(sigma, kernel_dim);


	apply_gaussian_filter(&image, gaussian_kernel, kernel_dim);



	//apply_gaussian_filter(picture, gaussian_kernel);
	for(i = 0; i < kernel_dim; i++){
		for(j = 0; j < kernel_dim; j++){
			printf("%f ", gaussian_kernel[i][j]);
		}
		printf("\n");
	}

	check = saveBMP(image, "prova.bmp");
	printf("%d\n", check);

	return 0;
}
