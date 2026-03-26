#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h> // para que NAN funcione
#include <string.h> // para que strdup funcione
#include <stdlib.h>  // para malloc, atoi, atof
#include <windows.h> // para Sleep

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
	char** tail_num; //char** es como se define en CUDA un array de strings
	int* origin_seq_id;
	char** origin_airport;
	int* dest_seq_id;
	char** dest_airport;
	float* dep_time;
	float* dep_delay;
	float* arr_time;
	float* arr_delay;
	float* weather_delay;
};

// Definimos la funcion donde vamos a hacer la lectura del csv
Dataset* leerCSV(const char* ruta) {
	// Abrir el fichero CSV
	FILE* fichero = fopen(ruta, "r");

	// Compruebo que el fichero se ha abierto correctamente
	if (fichero == NULL) {
		return NULL;
	}

	char linea[512];

	// Saltar la cabecera
	fgets(linea, 512, fichero);

	// Vamos a reservar acceso en memoria
	Dataset* ds = (Dataset*)malloc(sizeof(Dataset));

	ds->tail_num = (char**)malloc(MAX_VUELOS * sizeof(char*));
	ds->origin_seq_id = (int*)malloc(MAX_VUELOS * sizeof(int));
	ds->origin_airport = (char**)malloc(MAX_VUELOS * sizeof(char*));
	ds->dest_seq_id = (int*)malloc(MAX_VUELOS * sizeof(int));
	ds->dest_airport = (char**)malloc(MAX_VUELOS * sizeof(char*)); // char* porque son array de punteros
	ds->dep_time = (float*)malloc(MAX_VUELOS * sizeof(float));
	ds -> dep_delay = (float*)malloc(MAX_VUELOS * sizeof(float));
	ds->arr_time = (float*)malloc(MAX_VUELOS * sizeof(float));
	ds->arr_delay = (float*)malloc(MAX_VUELOS * sizeof(float));
	ds->weather_delay = (float*)malloc(MAX_VUELOS * sizeof(float));

	// Leer el fichero
	int fila = 0;
	while (fgets(linea, 512, fichero) && fila < MAX_VUELOS) {
		int i = 0; // contador de campos que se reinicia en cada linea
		char* campo = strtok(linea, ","); // separamos los campos 
		while (campo != NULL) {
			if (i == 3) {
				ds->tail_num[fila] = (campo == NULL) ? NULL : strdup(campo); // copiamos el string del campo en el array correspondiente
			}
			else if (i == 5) {
				ds->origin_seq_id[fila] = (campo == NULL) ? 0 : atoi(campo); // convertimos el string a entero y lo guardamos
			}
			else if (i == 6) {
				ds->origin_airport[fila] = (campo == NULL) ? NULL : strdup(campo);
			}
			else if (i == 7) {
				ds->dest_seq_id[fila] = (campo == NULL) ? 0 : atoi(campo); // si el campo está vacío guardamos NAN, si no convertimos el string a float
			}
			else if (i == 8) {
				ds->dest_airport[fila] = (campo == NULL) ? NULL : strdup(campo);
			}
			else if (i == 9) {
				ds->dep_time[fila] = (campo == NULL) ? NAN : (float)atoi(campo);
			}
			else if (i == 10) {
				ds->dep_delay[fila] = (campo == NULL) ? NAN : (float)atoi(campo); // si el campo está vacío se añade NAN en float, NULL en string, 0 en int, sino su valor
			}
			else if (i == 11) {
				ds->arr_time[fila] = (campo == NULL) ? NAN :  (float)atoi(campo); // si el campo esta vacio se guarda NAN
			}
			else if (i == 12) { 
				ds->arr_delay[fila] = (campo == NULL) ? NAN : (float)atoi(campo);
			}
			else if (i == 13) {
				ds->weather_delay[fila] = (campo == NULL) ? NAN : atof(campo);
			}
			i++; // incrementamos el contador
			campo = strtok(NULL, ",");
		}
		fila++; // incrementamos el numero de fila
	}

	// Guardamos el numero de vuelos leidos
	ds->numVuelos = fila;

	// Cerrar el fichero
	fclose(fichero);

	return ds; // porque la funcion devuelve Dataset*
}

int main() {
	// Mostramos el título de la practica durante 3 segundos
	printf("======================================================================================================================\n");
	printf("						EL1 PAP 2026						  \n");
	printf("				Candela Sanz Martin y Maria de la Orden Montes					\n");
	printf("======================================================================================================================\n");
	//Sleep(1500);

	// Pedimos la ruta al usuario
	Dataset* ds = NULL;
	while (ds == NULL) {
		printf("\nIntroduzca la ruta base del dataset: ");
		printf("\n(pulse Intro para usar por defecto: C:\\dataset_UAH\\Airline_dataset.csv)\n");
		char ruta[256];
		fgets(ruta, 256, stdin);
		ruta[strcspn(ruta, "\n")] = '\0'; // Eliminar el salto de linea que mete fgets
		// Si pulsa Intro sin escribir nada, usar ruta por defecto
		if (strlen(ruta) == 0) {
			strcpy(ruta, RUTA);
		}

		ds = leerCSV(ruta);
		// Contemplamos la opcion de que el dataset no se cargue bien
		if (ds == NULL) {
			printf("Error al cargar el dataset, intentarlo de nuevo\n");
		}
	}

	printf("Dataset cargado: %d vuelos\n", ds->numVuelos);
	printf("DEBUG arr_delay[20235] = %f\n", ds->arr_delay[20235]);

	// Menú
	int opcion = -1;
	while (opcion != 0) {
		printf("\nMenu:");
		printf("\n1. Despegues");
		printf("\n2. Aterrizajes");
		printf("\n3. Reduccion de Retraso");
		printf("\n4. Histograma de Aeropuertos");
		printf("\n5. Modificar Ruta Base");
		printf("\n6. Salir");
		printf("\nElija una opcion: ");

		// Leemos la opcion seleccionada por el usuario
		scanf("%d", &opcion);

		switch (opcion) {
			case 1: 
				ejecutarFase1(ds -> dep_delay, ds -> numVuelos);
				break;
			case 2: 
				ejecutarFase2(ds->arr_delay, ds->numVuelos, ds->tail_num);
				break;
			case 3: 
				ejecutarFase3(ds->dep_delay, ds->arr_delay, ds->weather_delay, ds->dep_time, ds->arr_time, ds->numVuelos);
				break;
			case 4: printf("Sin implementar"); break;
			case 5: printf("Sin implementar"); break;
			case 6: printf("Sin implementar"); break;
			default: printf("Opcion no valida, introduzca un numero entre 0 y 5\n"); break;
		}
	}
	
	return 0;
}