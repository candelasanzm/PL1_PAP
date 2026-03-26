#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <device_launch_parameters.h>

__global__ void fase3_simple(float* dev_datos, int numVuelos, int* dev_resSimple, int esMax) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numVuelos) return;
	if (isnan(dev_datos[i])) return;
	int valor = (int)dev_datos[i];
	if (esMax) {
		atomicMax(dev_resSimple, valor);
	}
	else {
		atomicMin(dev_resSimple, valor);
	}
}

__global__ void fase3_basica(float* dev_datos, int numVuelos, int* dev_resBasica, int esMax) {
	__shared__ float datos_memoria[1024];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numVuelos) return;
	if (isnan(dev_datos[i])) return;
	
	datos_memoria[threadIdx.x] = dev_datos[i];
	__syncthreads();

	int posAnterior;
	if (threadIdx.x > 0) {
		posAnterior = (int)datos_memoria[threadIdx.x - 1];
	}
	else {
		posAnterior = (int)dev_datos[i];
	}

	int posActual = (int)datos_memoria[threadIdx.x];
	int posPosterior;
	if (i + 1 < numVuelos) {
		posPosterior = (int)datos_memoria[threadIdx.x + 1];
	}
	else {
		posPosterior = (int)dev_datos[i];
	}

	int resultado;
	if (esMax) {
		resultado = max(max(posAnterior, posActual), posPosterior);
		atomicMax(dev_resBasica, resultado);
	}
	else {
		resultado = min(min(posAnterior, posActual), posPosterior);
		atomicMin(dev_resBasica, resultado);
	}
}

__global__ void fase3_intermedia(float* dev_datos, int numVuelos, int* dev_resIntermedia, int esMax) {

}

__global__ void fase3_reduccion(float* dev_datos, int numVuelos, int* dev_resReduccion, int esMax) {

}

void ejecutarFase3(float* dep_delay, float* arr_delay, float* weather_delay, float* dep_time, float* arr_time, int numVuelos) {
	// Pedimos la opcion
	int opcion = -1;
	while (opcion != 0) {
		printf("\nElige como filtrar: \n");
		printf("1. Retraso en Despegue\n");
		printf("2. Retraso en Aterrizaje\n");
		printf("3. Retraso por Condiciones Meteorologicas\n");
		printf("4. Hora de Despegue\n");
		printf("5. Hora de Aterrizaje\n");
		printf("6. Volver al Menu Principal\n");
		printf("Elija una opcion: ");

		scanf("%d", &opcion);

		switch (opcion) {
			case 1:
			case 2:
			case 3:
			case 4:
			case 5: {
				int opcion2 = -1;
				while (opcion2 != 0) {
					printf("\nOpciones Disponibles\n");
					printf("1. Maximo\n");
					printf("2. Minimo\n");
					printf("3. Volver Atras\n");
					printf("Seleccione operacion: ");

					scanf("%d", &opcion2);

					switch (opcion2) {
					case 1:
					case 2: {
						// Comprobar caracteristicas hardware
						cudaDeviceProp prop;
						cudaGetDeviceProperties(&prop, 0);
						int hilosPorBloque = prop.maxThreadsPerBlock; // maximo de hilos por bloque
						int numBloques = (numVuelos + hilosPorBloque - 1) / hilosPorBloque;

						int res_simple; // hay que declararlo como int porque atomic no vale para float
						int res_basica;
						int res_intermedia;
						int res_reduccion;

						float* dev_depDelay, * dev_arrDelay, * dev_weatherDelay, * dev_depTime, * dev_arrTime;
						int* dev_resSimple, * dev_resBasica, * dev_resIntermedia, * dev_resReduccion;

						cudaMalloc((void**)&dev_depDelay, numVuelos * sizeof(float));
						cudaMalloc((void**)&dev_arrDelay, numVuelos * sizeof(float));
						cudaMalloc((void**)&dev_weatherDelay, numVuelos * sizeof(float));
						cudaMalloc((void**)&dev_depTime, numVuelos * sizeof(float));
						cudaMalloc((void**)&dev_arrTime, numVuelos * sizeof(float));
						cudaMalloc((void**)&dev_resSimple, sizeof(int));
						cudaMalloc((void**)&dev_resBasica, sizeof(int));
						cudaMalloc((void**)&dev_resIntermedia, sizeof(int));
						cudaMalloc((void**)&dev_resReduccion, sizeof(int));

						cudaMemcpy(dev_depDelay, dep_delay, numVuelos * sizeof(float), cudaMemcpyHostToDevice);
						cudaMemcpy(dev_arrDelay, arr_delay, numVuelos * sizeof(float), cudaMemcpyHostToDevice);
						cudaMemcpy(dev_weatherDelay, weather_delay, numVuelos * sizeof(float), cudaMemcpyHostToDevice);
						cudaMemcpy(dev_depTime, dep_time, numVuelos * sizeof(float), cudaMemcpyHostToDevice);
						cudaMemcpy(dev_arrTime, arr_time, numVuelos * sizeof(float), cudaMemcpyHostToDevice);

						dim3 dimGrid(numBloques);
						dim3 dimBlock(hilosPorBloque);

						if (opcion == 1) {
							float* dev_datos = dev_depDelay;
							if (opcion2 == 1) {
								fase3_simple << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resSimple, 1);
								fase3_basica << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resBasica, 1);
								fase3_intermedia << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resIntermedia, 1);
								fase3_reduccion << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resReduccion, 1);
								cudaDeviceSynchronize();
								cudaMemcpy(&res_simple, dev_resSimple, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_basica, dev_resBasica, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_intermedia, dev_resIntermedia, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_reduccion, dev_resReduccion, sizeof(int), cudaMemcpyDeviceToHost);
								printf("\n[Simple] Max() DEP_DELAY = %d minutos\n", res_simple);
								printf("[Basica] Max() DEP_DELAY = %d minutos\n", res_basica);
								printf("[Intermedia] Max() DEP_DELAY = %d minutos\n", res_intermedia);
								printf("[Reduccion] Max() DEP_DELAY = %d minutos\n", res_reduccion);
							}
							else if (opcion2 == 2) {
								fase3_simple << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resSimple, 0);
								fase3_basica << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resBasica, 0);
								fase3_intermedia << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resIntermedia, 0);
								fase3_reduccion << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resReduccion, 0);
								cudaDeviceSynchronize();
								cudaMemcpy(&res_simple, dev_resSimple, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_basica, dev_resBasica, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_intermedia, dev_resIntermedia, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_reduccion, dev_resReduccion, sizeof(int), cudaMemcpyDeviceToHost);
								printf("[Simple] Min() DEP_DELAY = %d minutos\n", res_simple);
								printf("[Basica] Min() DEP_DELAY = %d minutos\n", res_basica);
								printf("[Intermedia] Min() DEP_DELAY = %d minutos\n", res_intermedia);
								printf("[Reduccion] Min() DEP_DELAY = %d minutos\n", res_reduccion);
							}
						}
						else if (opcion == 2) {
							float* dev_datos = dev_arrDelay;
							if (opcion2 == 1) {
								fase3_simple << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resSimple, 1);
								fase3_basica << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resBasica, 1);
								fase3_intermedia << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resIntermedia, 1);
								fase3_reduccion << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resReduccion, 1);
								cudaDeviceSynchronize();
								printf("\n[Simple] Max() ARR_DELAY = %d minutos\n", res_simple);
								printf("[Basica] Max() ARR_DELAY = %d minutos\n", res_basica);
								printf("[Intermedia] Max() ARR_DELAY = %d minutos\n", res_intermedia);
								printf("[Reduccion] Max() ARR_DELAY = %d minutos\n", res_reduccion);
							}
							else if (opcion2 == 2) {
								fase3_simple << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resSimple, 0);
								fase3_basica << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resBasica, 0);
								fase3_intermedia << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resIntermedia, 0);
								fase3_reduccion << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resReduccion, 0);
								cudaDeviceSynchronize();
								cudaMemcpy(&res_simple, dev_resSimple, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_basica, dev_resBasica, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_intermedia, dev_resIntermedia, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_reduccion, dev_resReduccion, sizeof(int), cudaMemcpyDeviceToHost);
								printf("\n[Simple] Min() ARR_DELAY = %d minutos\n", res_simple);
								printf("[Basica] Min() ARR_DELAY = %d minutos\n", res_basica);
								printf("[Intermedia] Min() ARR_DELAY = %d minutos\n", res_intermedia);
								printf("[Reduccion] Min() ARR_DELAY = %d minutos\n", res_reduccion);
							}
						}
						else if (opcion == 3) {
							float* dev_datos = dev_weatherDelay;
							if (opcion2 == 1) {
								fase3_simple << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resSimple, 1);
								fase3_basica << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resBasica, 1);
								fase3_intermedia << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resIntermedia, 1);
								fase3_reduccion << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resReduccion, 1);
								cudaDeviceSynchronize();
								cudaMemcpy(&res_simple, dev_resSimple, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_basica, dev_resBasica, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_intermedia, dev_resIntermedia, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_reduccion, dev_resReduccion, sizeof(int), cudaMemcpyDeviceToHost);
								printf("\n[Simple] Max() WEATHER_DELAY = %d minutos\n", res_simple);
								printf("[Basica] Max() WEATHER_DELAY = %d minutos\n", res_basica);
								printf("[Intermedia] Max() WEATHER_DELAY = %d minutos\n", res_intermedia);
								printf("[Reduccion] Max() WEATHER_DELAY = %d minutos\n", res_reduccion);
							}
							else if (opcion2 == 2) {
								fase3_simple << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resSimple, 0);
								fase3_basica << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resBasica, 0);
								fase3_intermedia << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resIntermedia, 0);
								fase3_reduccion << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resReduccion, 0);
								cudaDeviceSynchronize();
								cudaMemcpy(&res_simple, dev_resSimple, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_basica, dev_resBasica, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_intermedia, dev_resIntermedia, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_reduccion, dev_resReduccion, sizeof(int), cudaMemcpyDeviceToHost);
								printf("\n[Simple] Min() WEATHER_DELAY = %d minutos\n", res_simple);
								printf("[Basica] Min() WEATHER_DELAY = %d minutos\n", res_basica);
								printf("[Intermedia] Min() WEATHER_DELAY = %d minutos\n", res_intermedia);
								printf("[Reduccion] Min() WEATHER_DELAY = %d minutos\n", res_reduccion);
							}
						}
						else if (opcion == 4) {
							float* dev_datos = dev_depTime;
							if (opcion2 == 1) {
								fase3_simple << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resSimple, 1);
								fase3_basica << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resBasica, 1);
								fase3_intermedia << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resIntermedia, 1);
								fase3_reduccion << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resReduccion, 1);
								cudaDeviceSynchronize();
								cudaMemcpy(&res_simple, dev_resSimple, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_basica, dev_resBasica, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_intermedia, dev_resIntermedia, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_reduccion, dev_resReduccion, sizeof(int), cudaMemcpyDeviceToHost);
								printf("\n[Simple] Max() DEP_TIME = %d minutos\n", res_simple);
								printf("[Basica] Max() DEP_TIME = %d minutos\n", res_basica);
								printf("[Intermedia] Max() DEP_TIME = %d minutos\n", res_intermedia);
								printf("[Reduccion] Max() DEP_TIME = %d minutos\n", res_reduccion);
							}
							else if (opcion2 == 2) {
								fase3_simple << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resSimple, 0);
								fase3_basica << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resBasica, 0);
								fase3_intermedia << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resIntermedia, 0);
								fase3_reduccion << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resReduccion, 0);
								cudaDeviceSynchronize();
								cudaMemcpy(&res_simple, dev_resSimple, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_basica, dev_resBasica, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_intermedia, dev_resIntermedia, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_reduccion, dev_resReduccion, sizeof(int), cudaMemcpyDeviceToHost);
								printf("\n[Simple] Min() DEP_TIME = %d minutos\n", res_simple);
								printf("[Basica] Min() DEP_TIME = %d minutos\n", res_basica);
								printf("[Intermedia] Min() DEP_TIME = %d minutos\n", res_intermedia);
								printf("[Reduccion] Min() DEP_TIME = %d minutos\n", res_reduccion);
							}
						}
						else if (opcion == 5) {
							float* dev_datos = dev_arrTime;
							if (opcion2 == 1) {
								fase3_simple << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resSimple, 1);
								fase3_basica << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resBasica, 1);
								fase3_intermedia << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resIntermedia, 1);
								fase3_reduccion << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resReduccion, 1);
								cudaDeviceSynchronize();
								cudaMemcpy(&res_simple, dev_resSimple, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_basica, dev_resBasica, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_intermedia, dev_resIntermedia, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_reduccion, dev_resReduccion, sizeof(int), cudaMemcpyDeviceToHost);
								printf("\n[Simple] Max() ARR_TIME = %d minutos\n", res_simple);
								printf("[Basica] Max() ARR_TIME = %d minutos\n", res_basica);
								printf("[Intermedia] Max() ARR_TIME = %d minutos\n", res_intermedia);
								printf("[Reduccion] Max() ARR_TIME = %d minutos\n", res_reduccion);
							}
							else if (opcion2 == 2) {
								fase3_simple << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resSimple, 0);
								fase3_basica << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resBasica, 0);
								fase3_intermedia << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resIntermedia, 0);
								fase3_reduccion << <dimGrid, dimBlock >> > (dev_datos, numVuelos, dev_resReduccion, 0);
								cudaDeviceSynchronize();
								cudaMemcpy(&res_simple, dev_resSimple, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_basica, dev_resBasica, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_intermedia, dev_resIntermedia, sizeof(int), cudaMemcpyDeviceToHost);
								cudaMemcpy(&res_reduccion, dev_resReduccion, sizeof(int), cudaMemcpyDeviceToHost);
								printf("\n[Simple] Min() ARR_TIME = %d minutos\n", res_simple);
								printf("[Basica] Min() ARR_TIME = %d minutos\n", res_basica);
								printf("[Intermedia] Min() ARR_TIME = %d minutos\n", res_intermedia);
								printf("[Reduccion] Min() ARR_TIME = %d minutos\n", res_reduccion);
							}
						}

						cudaFree(dev_depDelay);
						cudaFree(dev_arrDelay);
						cudaFree(dev_weatherDelay);
						cudaFree(dev_depTime);
						cudaFree(dev_arrTime);
						cudaFree(dev_resSimple);
						cudaFree(dev_resBasica);
						cudaFree(dev_resIntermedia);
						cudaFree(dev_resReduccion);

						break;
					}

					case 3: return;
					default:
						printf("Opcion no valida, introduzca un numero entre 1 y 3\n");
						break;
					}
				}
			}
			case 6: return;
			default:
				printf("Opcion no valida, introduzca un numero entre 1 y 6\n");
				break;
		}
	}
}