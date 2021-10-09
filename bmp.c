#include <stdio.h>
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


int main(){
	BMP_Image prova;
	int check;
	int i, j;

	loadBMP("Immagini/abdomen.bmp", &prova);

	saveBMP(prova, "prova.bmp");


	return 0;
}
