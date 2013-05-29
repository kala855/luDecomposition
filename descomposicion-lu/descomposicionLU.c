//============================================================================
// Name        : LUOpenCL.cpp
// Author      : Lina Maria Perez, John Haiber Osorio,
// Version     :
// Copyright   :
// Description : LU Decomposition using OpenCL, Ansi-style
//============================================================================


#include <CL/cl.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include <math.h>
using namespace std;

//-------------------------------------------------------------------

/////////////////Funcion para imprimir una matriz ////////////////////

int imprimirMatrix(float *matrix, int n, int m, char *mensaje){
	int i,j;
	printf("\n %s: \n",mensaje);
	for(i = 0; i < n; i++){
		for(j = 0;j < m ; j++)
			printf("%f  ",matrix[i*n+j]);
		printf("\n");
	}
	return 0;
}

////////////////////////////////////////////////////////////////////

char* readSource(const char *sourceFilename) {
	FILE *fp;
	int err;
	int size;

	char *source;

	fp = fopen(sourceFilename, "rb");
	if (fp == NULL) {
		printf("Could not open kernel file: %s\n", sourceFilename);
		exit(-1);
	}

	err = fseek(fp, 0, SEEK_END);
	if (err != 0) {
		printf("Error seeking to end of file\n");
		exit(-1);
	}

	size = ftell(fp);
	if (size < 0) {
		printf("Error getting file position\n");
		exit(-1);
	}

	err = fseek(fp, 0, SEEK_SET);
	if (err != 0) {
		printf("Error seeking to start of file\n");
		exit(-1);
	}

	source = (char*) malloc(size + 1);
	if (source == NULL) {
		printf("Error allocating %d bytes for the program source\n", size + 1);
		exit(-1);
	}

	err = fread(source, 1, size, fp);
	if (err != size) {
		printf("only read %d bytes\n", err);
		exit(0);
	}

	source[size] = '\0';

	return source;
}
//////////////////////////////////////////////////////////////////////////////////////////

int prepararOpenCL(float *a, int n, int *pivot)
{
	cl_context context = 0;
	cl_command_queue cmdQueue;
	cl_program program = 0;
	cl_device_id device = 0;
	cl_kernel kernel = 0;
	cl_int errNum;
	cl_uint numDevices = 0;
	cl_int status; // use as return value for most OpenCL functions
	cl_uint numPlatforms = 0;
	cl_platform_id *platforms;
	cl_device_id *devices;
	cl_kernel computeAlphaKernel;
	cl_kernel swapRowsKernel ;
	cl_kernel computeBetaKernel;
	cl_kernel pivotMedioKernel;

	int j=0, rowMax=0, i=0, k=0;
	float big = 0.0;

	char *buildLog;
	size_t buildLogSize;
	char *source;

	const char *sourceFile =
			"FactorizacionLUKernels.cl";

	cl_int buildErr;

///////////// Inicializacion Matriz de permutacion///////////////



	// Query for the number of recognized platforms
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS) {
		printf("clGetPlatformIDs failed\n");
		exit(-1);
	}

	// Make sure some platforms were found
	if (numPlatforms == 0) {
		printf("No platforms detected.\n");
		exit(-1);
	}

	// Allocate enough space for each platform
	platforms = (cl_platform_id*) malloc(numPlatforms * sizeof(cl_platform_id));
	if (platforms == NULL) {
		perror("malloc");
		exit(-1);
	}

	// Fill in platforms
	clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != CL_SUCCESS) {
		printf("clGetPlatformIDs failed\n");
		exit(-1);
	}

	// Print out some basic information about each platform

	/*printf("%u platforms detected\n", numPlatforms);
	for (unsigned int i = 0; i < numPlatforms; i++) {
		char buf[100];
		printf("Platform %u: \n", i);
		status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,
				sizeof(buf), buf, NULL);

		printf("\tVendor: %s\n", buf);

		status |= clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
				sizeof(buf), buf, NULL);
		printf("\tName: %s\n", buf);
		if (status != CL_SUCCESS) {
			printf("clGetPlatformInfo failed\n");
			exit(-1);
		}
	}

	printf("\n");*/

	// Retrieve the number of devices present

	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL,
			&numDevices);

	if (status != CL_SUCCESS) {
		printf("clGetDeviceIDs failed\n");
		exit(-1);
	}

	// Make sure some devices were found
	if (numDevices == 0) {
		printf("No devices detected.\n");
		exit(-1);
	}

	// Allocate enough space for each device
	devices = (cl_device_id*) malloc(numDevices * sizeof(cl_device_id));
	if (devices == NULL) {
		perror("malloc");
		exit(-1);
	}

	// Fill in devices
	status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices,
			devices, NULL);
	if (status != CL_SUCCESS) {
		printf("clGetDeviceIDs failed\n");
		exit(-1);
	}

	// Print out some basic information about each device

	/*printf("%u devices detected\n", numDevices);
	for (unsigned int i = 0; i < numDevices; i++) {
		char buf[100];
		printf("Device %u: \n", i);
		status = clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(buf),
				buf, NULL);
		printf("\tDevice: %s\n", buf);
		status |= clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(buf), buf,
				NULL);
		printf("\tName: %s\n", buf);
		if (status != CL_SUCCESS) {
			printf("clGetDeviceInfo failed\n");
			exit(-1);
		}
	}

	printf("\n");*/

// Create a context and associate it with the devices
	context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);

	if (status != CL_SUCCESS || context == NULL) {
		printf("clCreateContext failed\n");
		exit(-1);
	}


// Create a command queue and associate it with the device you
	// want to execute on
	cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
	if (status != CL_SUCCESS || cmdQueue == NULL) {
		printf("clCreateCommandQueue failed\n");
		exit(-1);
	}


/////////////////////CREATE BUFFERS HERE//////////////////////////////

	cl_mem a_Kernel, rowMax_Kernel, pivot_Kernel; // Input buffers on device

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////Create a buffer for matrix a//////////////////////////////////////////////////////
	a_Kernel = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) *n*n, a, &status);
	if (status != CL_SUCCESS || a_Kernel == NULL) {
		printf("clCreateBuffer failed a_Kernel\n");
		exit(-1);
	}

	//////////////////////////Create a buffer for rowMax//////////////////////////////////////////////////////
	rowMax_Kernel = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * 1, NULL, &status);
	if (status != CL_SUCCESS || rowMax_Kernel == NULL) {
		printf("clCreateBuffer failed rowMax_Kernel\n");
		exit(-1);
	}
	//////////////////////////Create a buffer for pivot_Kernel//////////////////////////////////////////////////////
	pivot_Kernel = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int) * n, pivot, &status);
	if (status != CL_SUCCESS || rowMax_Kernel == NULL) {
		printf("clCreateBuffer failed pivot_Kernel\n");
		exit(-1);
	}

	source = readSource(sourceFile);

	//printf("Program source is:\n%s\n", source);

	// Create a program. The 'source' string is the code from the
	// kernelsFactorizacionLU.cl file.
	program = clCreateProgramWithSource(context, 1, (const char**) &source,
			NULL, &status);
	if (status != CL_SUCCESS) {
		printf("clCreateProgramWithSource failed\n");
		exit(-1);
	}



	// Build (compile & link) the program for the devices.
	// Save the return value in 'buildErr' (the following
	// code will print any compilation errors to the screen)
	buildErr = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

	// If there are build errors, print them to the screen
	if (buildErr != CL_SUCCESS) {
		printf("Program failed to build.\n");
		cl_build_status buildStatus;
		for (unsigned int i = 0; i < numDevices; i++) {
			clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_STATUS,
					sizeof(cl_build_status), &buildStatus, NULL);
			if (buildStatus == CL_SUCCESS) {
				continue;
			}


			clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG, 0,
					NULL, &buildLogSize);
			buildLog = (char*) malloc(buildLogSize);
			if (buildLog == NULL) {
				perror("malloc");
				exit(-1);
			}

			clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
					buildLogSize, buildLog, NULL);

			buildLog[buildLogSize - 1] = '\0';
			printf("Device %u Build Log:\n%s\n", i, buildLog);
			free(buildLog);
		}
		exit(0);
	} else {
		//printf("No build errors\n");
	}

///////////////////////CREATE KERNEL/////////////////////////////////////////

	pivotMedioKernel = clCreateKernel(program,"pivotMedioKernel", &status);
	if (status != CL_SUCCESS) {
		printf("clCreateKernel failed pivotMedioKernel\n");
		exit(-1);
	}

	swapRowsKernel = clCreateKernel(program,"swapRowsKernel", &status);
	if (status != CL_SUCCESS) {
		printf("clCreateKernel failed swapRowsKernel\n");
		exit(-1);
	}

	computeBetaKernel = clCreateKernel(program,"computeBetaKernel", &status);
	if (status != CL_SUCCESS) {
		printf("clCreateKernel failed computeBetaKernel\n");
		exit(-1);
	}

	computeAlphaKernel = clCreateKernel(program,"computeAlphaKernel", &status);
	if (status != CL_SUCCESS) {
		printf("clCreateKernel failed computeAlphaKernel\n");
		exit(-1);
	}

	// Associate the input and output buffers with the kernel///////

	status = clSetKernelArg(pivotMedioKernel, 0,sizeof(cl_mem), &rowMax_Kernel);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg pivotMedioKernel rowMax failed swapRowsKernel\n");
		exit(-1);
	}

	status = clSetKernelArg(pivotMedioKernel, 1, sizeof(cl_mem), &a_Kernel);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg a_Kernel failed pivotMedioKernel\n");
		exit(-1);
	}

	status = clSetKernelArg(pivotMedioKernel, 2, sizeof(cl_mem), &pivot_Kernel);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg pivot_Kernel failed pivotMedioKernel\n");
		exit(-1);
	}


	status = clSetKernelArg(swapRowsKernel, 0,sizeof(cl_mem), &rowMax_Kernel);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg swapRowsKernel rowMax failed swapRowsKernel\n");
		exit(-1);
	}

	status = clSetKernelArg(swapRowsKernel, 2, sizeof(cl_mem), &a_Kernel);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg a_Kernel failed swapRowsKernel\n");
		exit(-1);
	}

	status = clSetKernelArg(computeBetaKernel, 0,sizeof(cl_int), &n);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg n failed computeBetaKernel\n");
		exit(-1);
	}

	status = clSetKernelArg(computeBetaKernel, 1, sizeof(cl_mem), &a_Kernel);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg a_Kernel failed computeBetaKernel\n");
		exit(-1);
	}

	status = clSetKernelArg(computeAlphaKernel, 1, sizeof(cl_mem), &a_Kernel);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg a_Kernel failed computeAlphaKernel\n");
		exit(-1);
	}

	status = clSetKernelArg(computeAlphaKernel, 2,sizeof(cl_int), &n);
	if (status != CL_SUCCESS) {
		printf("clSetKernelArg n failed computeAlphaKernel\n");
		exit(-1);
	}

	size_t global[1];
/////////////////////////Main execution//////////////////////////////////////////
	for(j = 0; j < n ; j++){

/////////////////////////SIZE KERNEL PIVOTE MEDIO////////////////////////////////////////////////////////
		global[0] = 1;
//////////////////////////CALL KERNEL PIVOTE MEDIO/////////////////////////////////////////////
		status = clSetKernelArg(pivotMedioKernel, 3,sizeof(cl_int), &n);
		if (status != CL_SUCCESS) {
			printf("clSetKernelArg pivoteMedioKernel n failed swapRowsKernel\n");
			exit(-1);
		}

		status = clSetKernelArg(pivotMedioKernel, 4,sizeof(cl_int), &j);
		if (status != CL_SUCCESS) {
			printf("clSetKernelArg pivoteMedioKernel n failed swapRowsKernel\n");
			exit(-1);
		}

		status = clEnqueueNDRangeKernel(cmdQueue,pivotMedioKernel, 1, NULL, global, NULL, 0,NULL, NULL);
		if (status != CL_SUCCESS) {
			printf("clEnqueueNDRangeKernel pivotMedioKernel failed\n");
			exit(-1);
		}

/////////////////////////SIZE KERNEL Intercambio Filas////////////////////////////////////////////////////////////////
		global[0] = n;
/////////////////////////CALL KERNEL Intercambio filas//////////////////////////////////////////

		status = clSetKernelArg(swapRowsKernel, 1,sizeof(cl_int), &j);
		if (status != CL_SUCCESS) {
			printf("clSetKernelArg failed swapRowsKernel\n");
			exit(-1);
		}

		status = clEnqueueNDRangeKernel(cmdQueue,swapRowsKernel, 1, NULL, global, NULL, 0,NULL, NULL);
		if (status != CL_SUCCESS) {
			printf("clEnqueueNDRangeKernel swapRowsKernel failed\n");
			exit(-1);
		}

/////////////////////////SIZE KERNEL calculo beta's be////////////////////////////////////////////////////////////////
		global[0] = n - j;
/////////////////////////CALL KERNEL calculo beta's//////////////////////////////////////////
		status = clSetKernelArg(computeBetaKernel, 2,sizeof(cl_int), &j);
		if (status != CL_SUCCESS) {
			printf("clSetKernelArg failed computeBetaKernel\n");
			exit(-1);
		}

		status = clEnqueueNDRangeKernel(cmdQueue,computeBetaKernel, 1, NULL, global, NULL, 0,NULL, NULL);
		if (status != CL_SUCCESS) {
			printf("clEnqueueNDRangeKernel computeBetaKernel failed\n");
			exit(-1);
		}

/////////////////////////SIZE KERNEL calculo alpha's //////////////////////////////////////////
		global[0] =  n - j - 1 ;
/////////////////////////CALL KERNEL calculo alpha's //////////////////////////////////////////
		if(global[0] > 0){
			status = clSetKernelArg(computeAlphaKernel, 0, sizeof(cl_int), &j);
			if (status != CL_SUCCESS) {
				printf("clSetKernelArg failed swapRowsKernel\n");
				exit(-1);
			}
			status = clEnqueueNDRangeKernel(cmdQueue,computeAlphaKernel, 1, NULL, global, NULL, 0,NULL, NULL);
			if (status != CL_SUCCESS) {
				printf("clEnqueueNDRangeKernel computeAlphaKernel failed\n");
				exit(-1);
			}
		}
		//clEnqueueReadBuffer(cmdQueue, a_Kernel, CL_TRUE, 0,sizeof(float)*n*n, a, 0, NULL, NULL);
	}
	clEnqueueReadBuffer(cmdQueue, a_Kernel, CL_TRUE, 0,sizeof(float)*n*n, a, 0, NULL, NULL);
	clEnqueueReadBuffer(cmdQueue, pivot_Kernel, CL_TRUE, 0,sizeof(float)*n, pivot, 0, NULL, NULL);

////////////////////////////////////////////////////////////////////////////
	clReleaseKernel(swapRowsKernel);
	clReleaseKernel(computeBetaKernel);
	clReleaseKernel(computeAlphaKernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(a_Kernel);
	clReleaseContext(context);
	return 0;

}

int imprimirDatosArchivo(int numeroEcuaciones,float deltaX, float tiempoInicial, float tiempoFinal,
						float temperaturaInicial, float temperaturaFinal, float *x){

		int i = 0;
		float tiempo;
		FILE *pFile;
		pFile = fopen("salida", "w");
		fprintf(pFile, "%s,%s \n", "tiempo", "temperatura");
		tiempo = tiempoInicial;
		fprintf(pFile, "%.30f,%.30f \n", tiempoInicial, temperaturaInicial);
		for (i = 0; i < numeroEcuaciones; i++) {
			tiempo = tiempo + deltaX;
			fprintf(pFile, "%.30f,%.30f \n", tiempo, x[i]);
		}
		fprintf(pFile, "%.30f,%.30f \n", tiempoFinal, temperaturaFinal);
		fclose(pFile);
		return EXIT_SUCCESS;
}

int imprimirMatrizLUArchivo(int numeroEcuaciones,float *a){
	int i,j;
	FILE *pFile;
	pFile = fopen("salidaDescomposicionLU", "w");
	for(i = 0; i < numeroEcuaciones; i++){
		for(j = 0;j < numeroEcuaciones ; j++)
			fprintf(pFile, "%f  ", a[i*numeroEcuaciones+j]);
		fprintf(pFile,"\n");
	}
	fclose(pFile);
}

int main(int argc, char *argv[]){

	float temperaturaInicial, temperaturaFinal, temperaturaAmbiente, tiempoFinal,
								tiempoInicial, coeficienteConductividad, deltaX;
	float *a, *b, *x;
	int *pivot;
	int numeroEcuaciones = 0,i,j;
	float divisor = 0.0;
	clock_t start, end;
	double cpu_time_used;

	if(argc != 2){
		printf("Digite el factor que dividirá el delta X. Recuerde que inicia en 0.25.");
		return 1;
	}else{
		divisor = atof(argv[1]);		
	}

	
	/////////////Valores Iniciales////////////////////////////////////////
	temperaturaInicial = 80;
	temperaturaFinal = 28.12011;
	temperaturaAmbiente = 20;
	deltaX = 0.25/divisor;
	tiempoInicial = 0.0;
	tiempoFinal = 2;
	coeficienteConductividad = 1;	
	
	//Reserva de memoria para la matriz y los vectores del programa

	numeroEcuaciones= (int)((tiempoFinal - tiempoInicial)/deltaX) - 1;
	a = (float *)malloc(sizeof(float)*numeroEcuaciones*numeroEcuaciones);
	b = (float *)malloc(sizeof(float)*numeroEcuaciones);
	x = (float *)malloc(sizeof(float)*numeroEcuaciones);
	pivot = (int *)malloc(sizeof(int)*numeroEcuaciones);

	printf("El numero de Ecuaciones es de : %d \n",numeroEcuaciones);

	for (i = 0; i < numeroEcuaciones*numeroEcuaciones; i++){
		a[i] = 0.0;
	}

	// LLenado de los vectores y matrices para la solucion de la ecuacion de enfriamiento de newton

	for (i = 0; i < numeroEcuaciones; i++) {
		b[i] = 2*coeficienteConductividad*deltaX*temperaturaAmbiente;
		pivot[i] = 0;
		x[i] = 0.0;
		for(j = 0; j < numeroEcuaciones; j++){
			if (i == j){
				a[i*numeroEcuaciones+j] = 2*coeficienteConductividad*deltaX;
				if (i < numeroEcuaciones-1){
					a[(i+1)*numeroEcuaciones+j] = -1;
					a[(i*numeroEcuaciones)+(j+1)] = 1;
				}
			}
		}
	}

	b[0] = 2*coeficienteConductividad*deltaX*temperaturaAmbiente + temperaturaInicial;
	b[numeroEcuaciones-1] = 2*coeficienteConductividad*deltaX*temperaturaAmbiente - temperaturaFinal;

	//imprimirMatrix(a,numeroEcuaciones,numeroEcuaciones,"La matriz armada es");

	start = clock();
	prepararOpenCL(a,numeroEcuaciones,pivot);
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Time elapsed: %.30f\n", cpu_time_used);

	//imprimirMatrix(a,numeroEcuaciones,numeroEcuaciones,"La matriz A conteniendo la descomposición LU es");
/*	permutarVector(b,pivot,numeroEcuaciones);
	solveSystemEquationsLU(a,b,pivot,numeroEcuaciones,x);*/

	imprimirMatrizLUArchivo(numeroEcuaciones, a);

	//imprimirDatosArchivo(numeroEcuaciones,deltaX,tiempoInicial,tiempoFinal,temperaturaInicial,temperaturaFinal,x);

	free(a);
	free(pivot);
	free(b);
	free(x);

	return 0;

}
