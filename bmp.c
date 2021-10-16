#include <stdio.h>
#include <math.h> // compile with flag -lm to use this library
#include <stdlib.h>
#include "bmp.h"



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
		x: position on x
		y: position on y
		unsigned int, sigma: standard deviation. It must be in the interval [0, 10]
	RETURN:
		value of the gaussian function in a certain point
	*/
	return (exp(-1*(x*x + y*y) / (2 * sigma*sigma)) / (M_PI * 2 * sigma*sigma));
	//return ((1 / 2*M_PI*sigma*sigma) * exp(-1*(x*x+y*y)/2*sigma*sigma));
}



float** mk_gaussian_kernel(float sigma, unsigned int kernel_dim){
	/**It builds gaussian filter kernel of dim = KERNEL_DIM.
	PAR:
		gaussian_kernel: pointer to the matrix containing the kernel
		sigma: standard deviation. It must be in the interval [0, 10]
	RETURN:
		the pointer to the kernel
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



void extend_borders(BMP_Image* image, Pixel** bordered, int border_dim){
	/**It symmetrically modifies the borders of the image.
	PAR:
		image: image to modify
		bordered: it's borders has to be changed
		border_dim: dimension of the black border
	RETURN:
		void
	*/
	int i, j;


	// Extend borders symmetrically
	// left column
	for(i = border_dim; i < DATA_DIM + border_dim; i++){
		for(j = 0; j < border_dim; j++){
			bordered[i][j].grey = image -> data[i][border_dim + j].grey; //reflect the image
		}
	}
	// down row
	for(i = DATA_DIM + border_dim; i < DATA_DIM + border_dim*2; i++){
		for(j = 0; j < border_dim; j++){
			bordered[i][j].grey = image -> data[i + border_dim - i][j].grey; //reflect the image
		}
	}
	// right column
	for(i = border_dim; i < DATA_DIM + border_dim; i++){
		for(j = DATA_DIM + border_dim; j < DATA_DIM + 2*border_dim; j++){
			bordered[i][j].grey = image -> data[i][j - border_dim].grey; //reflect the image
		}
	}
	// up row
	for(i = 0; i < border_dim; i++){
		for(j = 0; j < DATA_DIM + border_dim; j++){
			bordered[i][j].grey = image -> data[border_dim + i][j].grey; //reflect the image
		}
	}
	// angles
	for(i = 0; i < border_dim; i++){
		bordered[i][i].grey = bordered[i + border_dim][i + border_dim].grey; // up-left
		bordered[DATA_DIM + border_dim*2 - 1 - i][i].grey = bordered[DATA_DIM + border_dim - 1 - i][i + border_dim].grey; // down-left
		bordered[DATA_DIM + border_dim*2 - 1 - i][DATA_DIM + border_dim*2 - 1 - i].grey = bordered[DATA_DIM + border_dim - 1 - i][DATA_DIM + border_dim - 1 - i].grey; // down-right
		bordered[i][DATA_DIM + border_dim*2 - 1 - i].grey = bordered[i + border_dim][DATA_DIM + border_dim - 1 - i].grey; // up-right
	}
}



void apply_gaussian_filter(BMP_Image* image, float** gaussian_kernel, unsigned int kernel_dim){
	/**It applies the gaussian filter to a given image.
	PAR:
		image: image to modify
		gaussian_kernel: the kernel for the convolution
		kernel_dim: dimensions of the kernel
	RETURN:
		void
	*/
	// this will be used to add a black border to the picture
	Pixel** bordered;
	// temporary matrix used to store values of each step
	float** temp_kernel;
	// utility
	int i, j, h, k;
	unsigned int border_dim;
	float total_weight;


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


	extend_borders(image, bordered, border_dim);


	// Create a kernel-dimensioned matrix for the next step
	temp_kernel = (float**)malloc(kernel_dim * sizeof(float*));
    for(i = 0; i < kernel_dim; i++){
        temp_kernel[i] = (float*)malloc((kernel_dim) * sizeof(float));
    }


	// Apply filter
	total_weight = 0;
	// i, j is the up-left angle to start
	for(i = 0; i < DATA_DIM; i++){

		for(j = 0; j < DATA_DIM; j++){
			// in the image, i starts form down
			//D for(h = kernel_dim - 1; h >= 0; h--){
			for(h = 0; h < kernel_dim; h++){
				for(k = 0; k < kernel_dim; k++){
					// compute weighted values
					temp_kernel[kernel_dim - 1 - h][k] = (float)bordered[i + h][j + k].grey * gaussian_kernel[h][k];
					total_weight += temp_kernel[kernel_dim - 1 - h][k];
				}
			}
			// apply weighted values to the center one
			bordered[i + border_dim + 1][j + border_dim + 1].grey = (unsigned char)total_weight;
			total_weight  = 0.0;
		}
	}

	// i and j slide bordered, h and k slide image
	for(i = border_dim, h = 0; h < DATA_DIM; i++, h++){
		for(j = border_dim, k = 0; k < DATA_DIM; j++, k++){
			image -> data[h][k].grey = bordered[i][j].grey;
		}
	}
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

	/**D
	for(i = 0; i < DATA_DIM; i++){
		for(j = 0; j < DATA_DIM; j++){
			image.data[i][j].grey = (unsigned char)((double)image.data[i][j].grey * 0.0);
			printf("%d, %d: %u\n", i, j, image.data[i][j].grey);
		}
		printf("\n");
	}
	*/

	sigma = 1;
	kernel_dim = 7;
	gaussian_kernel = mk_gaussian_kernel(sigma, kernel_dim);

	printf("Kernel scelto: \n");
	for(i = 0; i < kernel_dim; i++){
		for(j = 0; j < kernel_dim; j++){
			printf("%f ", gaussian_kernel[i][j]);
		}
		printf("\n");
	}


	apply_gaussian_filter(&image, gaussian_kernel, kernel_dim);


	check = saveBMP(image, "prova.bmp");
	if(check == 0){
		printf("Immagine modificata correttamente\n");
	}


	return 0;
}
