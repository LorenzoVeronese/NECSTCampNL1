#include <stdio.h>
#include <math.h> // compile with flag -lm to use this library
#include <stdlib.h>
#include "bmp.h"

#define SOBEL_DIM 3
#define DEFAULT_SENSIBILITY 1.4



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



Pixel** mk_bordered(BMP_Image* image, unsigned int kernel_dim){
	/**It creates a copy of image, but surrounded by black borders of dim kernel_dim/2.
	PAR:
		image: image to copy and modify
		kernel_dim: dimensions of the kernel
	RETURN:
		image with black borders
	*/
	Pixel** bordered;
	unsigned int border_dim;
	int i, j, h, k;


	// Creating the black container
	border_dim = kernel_dim/2;
	bordered = (Pixel**)malloc((DATA_DIM + (border_dim*2)) * sizeof(Pixel*));
    for(i = 0; i < (DATA_DIM + ((kernel_dim/2)*2)); i++){
        bordered[i] = (Pixel*)calloc(((DATA_DIM + (border_dim*2))), sizeof(Pixel));
    }

	// Copy the image into the center of the container
	for(i = border_dim, h = 0; h < DATA_DIM; i++, h++){
		for(j = border_dim, k = 0; k < DATA_DIM; j++, k++){
			bordered[i][j].grey = image -> data[h][k].grey;
		}
	}


	return bordered;
}



void convolution(Pixel** bordered, float** kernel, unsigned int kernel_dim){
	/**It makes convolution of a kernel of kernel_dim on a given bordered image
	PAR:
		bordered: image to convolute on
		kernel: kernel for the convolution
		kernel_dim: dimensions of the kernel
	RETURN:
		void. The variable bordered is modified using pointer
	*/
	float** temp_kernel;
	int i, j, h, k;
	float total_weight;
	unsigned int border_dim;

	// Create a kernel-dimensioned matrix for the next step
	temp_kernel = (float**)malloc(kernel_dim * sizeof(float*));
    for(i = 0; i < kernel_dim; i++){
        temp_kernel[i] = (float*)malloc((kernel_dim) * sizeof(float));
    }


	// Apply filter
	border_dim = kernel_dim / 2;
	total_weight = 0.0;
	// i, j is the up-left angle of bordered where I start
	for(i = 0; i < DATA_DIM; i++){
		for(j = 0; j < DATA_DIM; j++){
			//D for(h = kernel_dim - 1; h >= 0; h--){
			for(h = 0; h < kernel_dim; h++){
				for(k = 0; k < kernel_dim; k++){
					// compute weighted values
					if(!(i + h < border_dim || i + h > DATA_DIM + border_dim || j + k < border_dim || j + k > DATA_DIM + border_dim )){
						// in the image, i starts form down
						temp_kernel[kernel_dim - 1 - h][k] = (float)bordered[i + h][j + k].grey * kernel[h][k];
						total_weight += temp_kernel[kernel_dim - 1 - h][k];
					}
				}
			}
			// apply weighted values to the center one
			bordered[i + border_dim][j + border_dim].grey = (unsigned char)total_weight;

			total_weight  = 0.0;
		}
	}
}



void convolution_sobel(float** bordered, float** kernel, unsigned int kernel_dim, float sensibility){
	/**It makes sobel convolution of a kernel of kernel_dim on a given bordered image
	PAR:
		bordered: image to convolute on
		kernel: kernel for the convolution
		kernel_dim: dimensions of the kernel
		sensibility: how to apply thresholding
			It must be in the interval [1, 5]
	RETURN:
		void. The variable bordered is modified using pointer
	*/
	float** temp_kernel;
	int i, j, h, k;
	float total_weight;
	unsigned int border_dim;

	// Create a kernel-dimensioned matrix for the next step
	temp_kernel = (float**)malloc(kernel_dim * sizeof(float*));
    for(i = 0; i < kernel_dim; i++){
        temp_kernel[i] = (float*)malloc((kernel_dim) * sizeof(float));
    }


	// Apply filter
	border_dim = kernel_dim / 2;
	total_weight = 0.0;
	// i, j is the up-left angle of bordered where I start
	for(i = 0; i < DATA_DIM; i++){
		for(j = 0; j < DATA_DIM; j++){
			//D for(h = kernel_dim - 1; h >= 0; h--){
			for(h = 0; h < kernel_dim; h++){
				for(k = 0; k < kernel_dim; k++){
					// compute weighted values
					if(!(i + h < border_dim || i + h > DATA_DIM + border_dim || j + k < border_dim || j + k > DATA_DIM + border_dim )){
						// in the image, i starts form down
						temp_kernel[kernel_dim - 1 - h][k] = bordered[i + h][j + k] * kernel[h][k];
						total_weight += temp_kernel[kernel_dim - 1 - h][k];
					}
				}
			}
			// apply weighted values to the center one
			/*
			if(total_weight > sensibility*100){
				bordered[i + border_dim][j + border_dim].grey = (unsigned char)total_weight; //total_weight
			}
			else{
				bordered[i + border_dim][j + border_dim].grey = (unsigned char)0;
			}*/

			total_weight  = 0.0;
		}
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


	border_dim = kernel_dim/2;
	bordered = mk_bordered(image, kernel_dim);


	convolution(bordered, gaussian_kernel, kernel_dim);


	// Update image according to changes
	for(i = border_dim, h = 0; h < DATA_DIM; i++, h++){
		for(j = border_dim, k = 0; k < DATA_DIM; j++, k++){
			image -> data[h][k].grey = bordered[i][j].grey;
		}
	}
}



float** mk_bordered_float(BMP_Image* image, unsigned int kernel_dim){
	/**It creates a copy of image, but surrounded by black borders of dim kernel_dim/2.
	PAR:
		image: image to copy and modify
		kernel_dim: dimensions of the kernel
	RETURN:
		image with black borders
	*/
	float** bordered;
	unsigned int border_dim;
	int i, j, h, k;


	// Creating the black container
	border_dim = kernel_dim/2;
	bordered = (float**)malloc((DATA_DIM + (border_dim*2)) * sizeof(float*));
    for(i = 0; i < (DATA_DIM + ((kernel_dim/2)*2)); i++){
        bordered[i] = (float*)calloc(((DATA_DIM + (border_dim*2))), sizeof(float));
    }

	// Copy the image into the center of the container
	for(i = border_dim, h = 0; h < DATA_DIM; i++, h++){
		for(j = border_dim, k = 0; k < DATA_DIM; j++, k++){
			bordered[i][j] = (float)((image -> data[h][k].grey));
		}
	}


	return bordered;
}



Pixel** sobel_contours(BMP_Image* image, float sensibility){
	/**It applies the sobel convolution to find contours.
	PAR:
		image: image where to find contours
		sensibility: how much details you want in resulting contours.
			It must be in the interval [1, 5]
		shade: do you want all white (0) or shades (1)?
			It must be 0 or 1
	RETURN:
		Pixel matrix containing contours of the given image
	*/
	// matrix of the sobel kernel
	float** x_sobel_kernel;
	float** y_sobel_kernel;
	float** temp_kernel;
	float** bordered_G1;
	float** bordered_G2;
	float** bordered;
	// utility
	int i, j, h, k;
	int total_weight;
	int border_dim;


	// Creating x_sobel_kernel
	x_sobel_kernel = (float**)malloc(SOBEL_DIM * sizeof(float*));
	for(i = 0; i < SOBEL_DIM; i++){
		x_sobel_kernel[i] = (float*)calloc(SOBEL_DIM, sizeof(float));
	}

	x_sobel_kernel[0][0] = -1;
	x_sobel_kernel[0][2] = 1;
	x_sobel_kernel[1][0] = -2;
	x_sobel_kernel[1][2] = 2;
	x_sobel_kernel[2][0] = -1;
	x_sobel_kernel[2][2] = 1;

	// Creating y_sobel_kernel
	y_sobel_kernel = (float**)malloc(SOBEL_DIM * sizeof(float*));
	for(i = 0; i < SOBEL_DIM; i++){
		y_sobel_kernel[i] = (float*)calloc(SOBEL_DIM, sizeof(float));
	}

	y_sobel_kernel[0][0] = -1;
	y_sobel_kernel[0][1] = -2;
	y_sobel_kernel[0][2] = -1;
	y_sobel_kernel[2][0] = 1;
	y_sobel_kernel[2][1] = 2;
	y_sobel_kernel[2][2] = 1;


	bordered_G1 = mk_bordered_float(image, SOBEL_DIM);
	bordered_G2 = mk_bordered_float(image, SOBEL_DIM);
	bordered = mk_bordered_float(image, SOBEL_DIM);


	convolution_sobel(bordered_G1, x_sobel_kernel, SOBEL_DIM, sensibility);
	convolution_sobel(bordered_G2, y_sobel_kernel, SOBEL_DIM, sensibility);



	for(i = SOBEL_DIM / 2; i < DATA_DIM + SOBEL_DIM/2; i++){
		for(j = SOBEL_DIM / 2; j < DATA_DIM + SOBEL_DIM/2; j++){
			bordered[i][j] = sqrt(pow(bordered_G1[i][j], 2) + pow(bordered_G2[i][j], 2));
			//thresholding
			if(bordered[i][j] < sensibility*100){
				bordered[i][j] = 0.0;
			}
			else{
				bordered[i][j] = 255.0;
			}
		}
	}


	// i and j slide bordered, h and k slide image
	for(i = SOBEL_DIM/2, h = 0; h < DATA_DIM; i++, h++){
		for(j = SOBEL_DIM/2, k = 0; k < DATA_DIM; j++, k++){
			image -> data[h][k].grey = (unsigned char)bordered[i][j];
		}
	}
}



int main(){
	// paramenters
	float** gaussian_kernel;
	float sigma;
	unsigned int kernel_dim;
	float sensibility;
	// image
	BMP_Image image;
	// utility
	int check;
	int i, j;
	char choosen;
	int shade;
	int input;


	loadBMP("Immagini/transaxial.bmp", &image);


	//Setting paramenters
	sigma = 1;
	kernel_dim = 3;
	sensibility = DEFAULT_SENSIBILITY;


	gaussian_kernel = mk_gaussian_kernel(sigma, kernel_dim);


	printf("Kernel scelto: \n");
	for(i = 0; i < kernel_dim; i++){
		for(j = 0; j < kernel_dim; j++){
			printf("%f ", gaussian_kernel[i][j]);
		}
		printf("\n");
	}


	apply_gaussian_filter(&image, gaussian_kernel, kernel_dim);



	sobel_contours(&image, sensibility);


	check = saveBMP(image, "result.bmp");
	if(check == 0){
		printf("Immagine modificata correttamente\n");
	}


	return 0;
}
