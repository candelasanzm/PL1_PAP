#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

__constant__ int c_umbral;

__global__ void fase2(float* dev_arrdelay, int numVuelos, int opcion, int* dev_contador, char* dev_tailNum, int* dev_indices, int* dev_tiempo) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numVuelos) return; // corner check
	if (isnan(dev_arrdelay[index])) return;

	int valor = (int)dev_arrdelay[index];
	char* matricula = dev_tailNum + index * 16;

	if ((opcion == 1) && (valor > c_umbral)) {
		printf("- Hilo %d | Matricula: %s | Retraso de %d minutos.\n", index, matricula, valor);
		int pos = atomicAdd(dev_contador, 1);
		dev_tiempo[pos] = valor;
		dev_indices[pos] = index; // guardamos un array de indices que posteriormente se usara para recuperar las matriculas
	}
	else if ((opcion == 2) && (valor < -c_umbral)) {
		printf("- Hilo %d | Matricula: %s | Adelanto de %d minutos.\n", index, matricula, -valor);
		int pos = atomicAdd(dev_contador, 1);
		dev_tiempo[pos] = valor;
		dev_indices[pos] = index;
	}
	else if ((opcion == 3) && (valor == 0)) {
		printf("- Hilo %d | Matricula: %s | Aterriza a tiempo.\n", index, matricula);
		int pos = atomicAdd(dev_contador, 1);
		dev_tiempo[pos] = valor;
		dev_indices[pos] = index;
	}
}

void ejecutarFase2(float* arr_delay, int numVuelos, char** tail_num) {
	// Pedimos la opcion
	int opcion = -1;
	while (opcion != 0) {
		printf("\nElige como filtrar:\n");
		printf("1. Retrasos\n");
		printf("2. Adelantos\n");
		printf("3. Aterrizaje a tiempo\n");
		printf("4. Volver al Menu Principal\n");
		printf("Elija una opcion: ");

		scanf("%d", &opcion);

		int umbral = 0;
		switch (opcion) {
		case 1: // haciendolo de esta manera el case 1 y el case 2 tienen el mismo codigo y usamos operadores ternarios en el kernel para simplificar
		case 2:
			printf("\nEscribe el umbral (en minutos) deseado: ");
			scanf("%d", &umbral);
			if (umbral < 0) {
				printf("El umbral no puede ser negativo\n");
				break;
			}
		case 3: {
			// Copiar umbral en memoria constante
			cudaMemcpyToSymbol(c_umbral, &umbral, sizeof(int));

			// Comprobar caracteristicas hardware
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, 0);
			int hilosPorBloque = prop.maxThreadsPerBlock; // maximo de hilos por bloque
			int numBloques = (numVuelos + hilosPorBloque - 1) / hilosPorBloque;

			// Hay que crear un array plano para poder usar matriculas en la GPU
			char* h_tail = (char*)malloc(numVuelos * 16 * sizeof(char));
			for (int i = 0; i < numVuelos; i++) {
				if (tail_num[i] != NULL) {
					strncpy(h_tail + i * 16, tail_num[i], 15); // copia un string con un limite de caracteres, el 15 porque al final se añade '\0' que indica el final
				}
				else {
					h_tail[i * 16] = '\0';
				}
			}

			// Copiamos arr_delay en la GPU
			int* h_indices = (int*)malloc(numVuelos * sizeof(int));
			int* h_tiempo = (int*)malloc(numVuelos * sizeof(int));
			int h_contador = 0; // hacemos un contador para indicar cuantos vuelos salen

			int* dev_contador, * dev_tiempo, * dev_indices;
			float* dev_arrdelay;
			char* dev_tailNum;

			cudaMalloc((void**)&dev_contador, sizeof(int));
			cudaMalloc((void**)&dev_arrdelay, numVuelos * sizeof(float));
			cudaMalloc((void**)&dev_tailNum, numVuelos * 16 * sizeof(char)); // Reservamos espacio de 16 chars de manera general
			cudaMalloc((void**)&dev_tiempo, numVuelos * sizeof(int));
			cudaMalloc((void**)&dev_indices, numVuelos * sizeof(int));

			cudaMemcpy(dev_contador, &h_contador, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_arrdelay, arr_delay, numVuelos * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_tailNum, h_tail, numVuelos * 16 * sizeof(char), cudaMemcpyHostToDevice);

			dim3 dimGrid(numBloques);
			dim3 dimBlock(hilosPorBloque);
			fase2 << <dimGrid, dimBlock >> > (dev_arrdelay, numVuelos, opcion, dev_contador, dev_tailNum, dev_indices, dev_tiempo);
			cudaDeviceSynchronize();

			cudaMemcpy(&h_contador, dev_contador, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_indices, dev_indices, h_contador * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_tiempo, dev_tiempo, h_contador * sizeof(int), cudaMemcpyDeviceToHost);

			if (h_contador == 0) {
				printf("No hay vuelos con el umbral solicitado\n");
			}
			else {
				printf("Se han encontrado %d vuelos con el umbral deseado\n", h_contador);
				if (opcion == 1) {
					for (int i = 0; i < h_contador; i++) {
						printf("- Matricula %s. Retraso: %d minutos.\n", tail_num[h_indices[i]], h_tiempo[i]);
					}
				}
				else if (opcion == 2) {
					for (int i = 0; i < h_contador; i++) {
						printf("- Matricula %s. Adelanto: %d minutos.\n", tail_num[h_indices[i]], -h_tiempo[i]);
					}
				}
			}

			cudaFree(dev_contador);
			cudaFree(dev_arrdelay);
			cudaFree(dev_tailNum);
			cudaFree(dev_indices);
			cudaFree(dev_tiempo);
			free(h_tail);
			free(h_indices);
			free(h_tiempo);
			break; // para no entrar en el case 3
		}
		case 4: return;
		default: 
			printf("Opcion no valida, introduzca un numero entre 1 y 3\n"); 
			break;
		}
	}
}