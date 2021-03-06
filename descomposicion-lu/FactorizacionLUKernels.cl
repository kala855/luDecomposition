__kernel void pivotMedioKernel(__global int *rowMax, __global float *a, __global int *pivot, int n, int j)
{
	int i = 0;
	int k = 0;	
	int big = 0.0;
	rowMax[0] = j;
	for(i = j;i < n; i++){
		k = j;
		if(fabs(a[i*n+k]) > big){
			big = fabs(a[i*n+k]);
			rowMax[0] = i ;
		}
	}
	pivot[j] = rowMax[0];
}


__kernel void swapRowsKernel(__global int *rowMax, int j,__global float *a)
{
	
	if(rowMax[0] != j)
	{
		int i,n;
		local float temp;
		
		temp = 0.0;
		
		i = get_global_id(0);
		n = get_global_size(0);
		
		temp = a[(rowMax[0])*n+i];
		a[(rowMax[0])*n+i] = a[j*n+i];
		a[j*n+i] = temp;
	}
}

__kernel void computeBetaKernel(int n,__global float *a, int j)
{
	int i;
	int k;
 	float sumatoria;
	i = get_global_id(0) + j;
	sumatoria = 0.0;
	for(k = 0;k < j; k++)
		sumatoria = sumatoria + (a[j*n+k]*a[k*n+i]);		
	a[j*n+i] = a[j*n+i] - sumatoria;		
}


__kernel void computeAlphaKernel(int j, __global float *a, int n)
{
	int i;
	float sumatoria;
	int k;		
	i = get_global_id(0) + j;
	sumatoria = 0.0;
	for (k = 0; k <= j-1; k++)
		sumatoria = sumatoria + a[(i+1)*n+k]*a[k*n+j];
	a[(i+1)*n+j]= (1/a[j*n+j])*(a[(i+1)*n+j] - sumatoria);		
}
