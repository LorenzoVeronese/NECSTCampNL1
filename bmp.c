#include <stdio.h>
#include "bmp.h"

/** Funzione per caricare un'immagine BMP da file
Parametri: 
	nome del file in lettura, puntatore alla struttura in cui memorizzare i dati dell'immagine
Valore di ritorno:
	0 se il caricamento è avvenuto con successo, un numero diverso da 0 altrimenti
*/
int loadBMP(char *filename, BMP_Image *image)
{
	
	FILE *fp;

	fp = fopen(filename, "rb");
	if(fp == NULL)
	{
		printf("Error: impossibile aprire il file in lettura\n");
		return 1;
	}

	fread(image->magic, sizeof(image->magic), 1, fp );

	if(image->magic[0]!='B' || image->magic[1]!='M')
	{
		printf("Error: tipo di immagine non corretto\n");
		return 2;
	}

	fread(&image->header, sizeof(image->header), 1, fp);
	fread(&image->info, sizeof(image->info), 1, fp);

	if(image->info.bits!=8)
	{
		printf("Error: numero di bits/pixel diverso da 8\n");
		return 3;
	}

	if(image->info.width!=DATA_DIM || image->info.height!=DATA_DIM)
	{
		printf("--- Attenzione, dimensioni non corrette ---");
	}

	fread(&image->color_table, sizeof(image->color_table), 1, fp);
	fread(image->data, sizeof(image->data), 1, fp);

	fclose(fp);
	return 0;

}

/** Funzione per salvare una struttura BMP_Image su file
Parametri: 
	puntatore alla struttura da cui leggere i dati dell'immagine, nome del file su cui scrivere
Valore di ritorno:
	0 se il salvataggio è avvenuto con successo, 1 altrimenti
*/
int saveBMP(BMP_Image image, char * filename){
	
	FILE *fp2;
	fp2 = fopen(filename, "wb");

	if(fp2==NULL)
	{
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