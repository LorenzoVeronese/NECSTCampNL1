
// Dimensione immagine in pixel, quadrata
#define DATA_DIM 256             

//Struttura per l'header
typedef struct {
   unsigned int size;
   unsigned short int reserved1;
   unsigned short int reserved2;
   unsigned int offset;
} Header;

//Struttura per l'header info
typedef struct {
   unsigned int size;
   int width,height;
   unsigned short int planes;
   unsigned short int bits;
   unsigned int compression;
   unsigned int imagesize;
   int xresolution,yresolution;
   unsigned int ncolours;
   unsigned int importantcolours;
} Info;

//Struttura per rappresentare un pixel. Nel nostro caso solo 8 bits, un solo campo
typedef struct {
   unsigned char grey;			
} Pixel;

//Struttura immagine
typedef struct{
	unsigned char magic[2];                     //Tipo di immagine
	Header header;                              //Header
	Info info;                                  //Info header
	unsigned char color_table[DATA_DIM*4];      //Tabella colori
	Pixel data[DATA_DIM][DATA_DIM];             //Matrice di pixel   
}BMP_Image;

//Caricamento immagine
int loadBMP(char * filename, BMP_Image *img);

//Salvataggio immagine
int saveBMP(BMP_Image img, char * filename);