#include <cuda_runtime.h>
#include <stdio.h>

// Tenemos que incluir las cabeceras de los archivos donde imlementaremos las fases
#include "fase1.cuh"
#include "fase2.cuh"
#include "fase3.cuh"
#include "fase4.cuh"

#define MAX_VUELOS 1200000
#define RUTA "C:\\dataset_UAH\\Airline_dataset.csv"

// Vamos a crear una estructura donde manejemos todas las variables juntas
struct Dataset {
	int numVuelos;
	float* dep_delay;
	float* arr_delay;
	char** tail_num; //char** es como se define en CUDA un array de strings
	float* weather_delay;
	float* dep_time;
	float* arr_time;
	int* origin_seq_id;
	int* dest_seq_id;
	char** origin_airport;
	char** dest_airport;
};

// Definimos la funcion donde vamos a hacer la lectura del csv
Dataset* leerCSV(const char* ruta) {
	// Abrir el fichero CSV
	FILE* fichero = fopen(ruta, "r");

	// Compruebo que el fichero se ha abierto correctamente
	if (fichero == NULL) {
		printf("ERROR: no se puede abrir el fichero\n");
		return NULL;
	}

	char linea[512];

	// Saltar la cabecera
	fgets(linea, 512, fichero);

	// Leer el fichero
	while (fgets(linea, 512, fichero)) {
		char* campo = strtok(linea, ","); // separamos los campos 
		while (campo != NULL) {
			campo = strtok(NULL, ",");
		}
	}

	// Cerrar el fichero
	fclose(fichero);
}

int main() {
	
	return 0;
}