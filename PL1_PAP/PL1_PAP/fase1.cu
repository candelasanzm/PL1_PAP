#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fase1(float* dev_depdelay, int numVuelos, int umbral, int opcion, int* dev_contador) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numVuelos) return; // corner check
	if (isnan(dev_depdelay[index])) return;

	int valor = (int)dev_depdelay[index];
	if ((opcion == 1) && (valor > umbral)) {
		printf("Hilo %d. Retraso de %d minutos.\n", index, valor);
		atomicAdd(dev_contador, 1);
	}
	else if ((opcion == 2) && (valor < umbral)) {
		printf("Hilo %d. Adelanto de %d minutos.\n", index, -valor);
		atomicAdd(dev_contador, 1);
	}
	else if ((opcion == 3) && (valor == 0)) {
		printf("Hilo %d. Despegue a tiempo\n", index);
		atomicAdd(dev_contador, 1);
	}
}

void ejecutarFase1(float* dep_delay, int numVuelos) {
	// Pedimos la opcion
	int opcion = -1;
	while (opcion != 0) {
		printf("\nElige como filtrar: \n");
		printf("1. Retrasos\n");
		printf("2. Adelantos\n");
		printf("3. Despegues a tiempo\n");
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
				// Comprobar caracteristicas hardware
				cudaDeviceProp prop;
				cudaGetDeviceProperties(&prop, 0);
				int hilosPorBloque = prop.maxThreadsPerBlock; // maximo de hilos por bloque
				int numBloques = (numVuelos + hilosPorBloque - 1) / hilosPorBloque;

				// Copiamos dep_delay en la GPU
				int h_contador = 0; // hacemos un contador para indicar cuantos vuelos salen

				int* dev_contador;
				float* dev_depdelay;

				cudaMalloc((void**)&dev_contador, sizeof(int));
				cudaMalloc((void**)&dev_depdelay, numVuelos * sizeof(float));

				cudaMemcpy(dev_contador, &h_contador, sizeof(int), cudaMemcpyHostToDevice);
				cudaMemcpy(dev_depdelay, dep_delay, numVuelos * sizeof(float), cudaMemcpyHostToDevice);

				dim3 dimGrid(numBloques);
				dim3 dimBlock(hilosPorBloque);
				fase1 << <dimGrid, dimBlock >> > (dev_depdelay, numVuelos, umbral, opcion, dev_contador);
				cudaDeviceSynchronize();

				cudaMemcpy(&h_contador, dev_contador, sizeof(int), cudaMemcpyDeviceToHost);
				if (h_contador == 0) {
					printf("No hay vuelos con el umbral solicitado\n");
				}
				else {
					printf("Se han encontrado %d vuelos con el umbral deseado\n", h_contador);
				}

				cudaFree(dev_contador);
				cudaFree(dev_depdelay);
				break; // para no entrar en el case 3
			}
			case 4: return;
			default: 
				printf("Opcion no valida, introduzca un numero entre 1 y 3\n"); 
				break;
		}
	}
}