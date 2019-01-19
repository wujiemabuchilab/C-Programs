/* Program for computing autocorrelation for a series X of size N */
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>
#include "mkl.h"
#include "mkl_blas.h"
#include "mkl_spblas.h"
#include "mkl_vsl.h"

int main(void) {

	FILE *fpin, *fpout;
	int N, Lag, ii, kk, nthreads, tmp;
	double *rhotmp, *rho, *rhomeanvec, *rhozero, *acf;
	double minusone, rhovar, rhomean, acftmp;
	int intone; /* incremental step of the rows of 'x' in zapxy */
	minusone = -1.0;
	intone = 1;

	printf("\n1. Reading series from files... "); fflush(stdout);
	/* read in series */
	fpin = fopen("vecin.dat","rb");
	if (fpin!=NULL) {
		/* read in the length of the series */
		fread(&N,sizeof(int),1,fpin);
		rho = (double*) mkl_malloc(N*sizeof(double),16);
		rhotmp = (double*) mkl_malloc(N*sizeof(double),16); /* if we don't declare an array of pointers but use 'rhotmp' as a single pointer, we will end up with segmentation fault */
		rhozero = (double*) mkl_malloc(N*sizeof(double),16);
		rhomeanvec = (double*) mkl_malloc(N*sizeof(double),16);
		
		fread(&Lag,sizeof(int),1,fpin);
		if (Lag>=N) {
			printf("Number of lags must be smaller than the total length of the sequence.\n");
			fflush(stdout);
			fclose(fpin);
			exit(1);
		}
		acf = (double*) mkl_malloc((Lag+1)*sizeof(double),16);

		for(ii=0;ii<N;ii++) {
			fread(rhotmp,sizeof(double),1,fpin);
			rhozero[ii] = *rhotmp;
			rho[ii] = *rhotmp;
		}
		fclose(fpin);
	}
	else {
		printf("Error reading series. The file does not exist.\n");
		fflush(stdout);
		fclose(fpin);
		exit(1);
	}
	
	/* check readin */
	fpout = fopen("checkin.dat","wb");
	fwrite(&N,1,sizeof(int),fpout);
	fwrite(&Lag,1,sizeof(int),fpout);
	for(ii=0;ii<N;ii++) {
		acftmp=rho[ii];
		fwrite(&acftmp,1,sizeof(double),fpout);
	}
	fclose(fpout);
	
	/* calculate mean */
	rhomean = 0.0;
	for(ii=0;ii<N;ii++) {
		rhomean +=  rho[ii];
	}
	rhomean = rhomean / N;
	printf("\n2. The mean of the sequence is %f... ",rhomean);

	/* obtain zero-meaned series */
	for(ii=0;ii<N;ii++) {
		rhozero[ii] = rho[ii] - rhomean;
	}
	/* calculate variance */
	rhovar = 0.0;
	for(ii=0;ii<N;ii++) {
		rhovar += rhozero[ii]*rhozero[ii];
	}
	printf("\n3. The variance of the sequence is %f... ",rhovar);
	
	/* initialize output variable */
	for(ii=0;ii<(Lag+1);ii++) {
		acf[ii] = 0.0;
	}
	
	/* change the number of OMP threads from the default value (the number of available cores) to 100 */
	#pragma omp parallel num_threads(46)
	{
		/* The function 'mkl_get_dynamic()' returns a value of 0 or 1: 1 indicates that MKL_DYNAMIC is true, 0 indicates that MKL_DYNAMIC is false. */
		/* This variable indicates whether or not Intel MKL can dynamically change the number of threads, so as to achieve load balancing. */
		/* printf("mkl_get_dynamic=%d\n",mkl_get_dynamic()); */
		#pragma omp master
		{
			nthreads = omp_get_num_threads();
			printf("\n4. Entering OMP Parallel section with %d threads, initializing arrays... ",nthreads);
			fflush(stdout);
		} /* end of omp master */
		/* The omp barrier directive identifies a synchronization point at which threads in a parallel region will wait until all other */
		/* threads in that section reach the same point. Statement execution past the omp barrier point then continues in parallel */
		/* no implicit barrier at the end of the MASTER region, thus we need to add it explicitly */
		#pragma omp barrier
		
		/* loop over 'Lag' = number of lags */
		#pragma omp for
		for(ii=0;ii<(Lag+1);ii++) {
			/* loop to compute autovariance */
			/* refer to NIST Engineering Statistics Handbook for the definition */
			/* URL: http://www.itl.nist.gov/div898/handbook/eda/section3/eda35c.htm */
			for(kk=0;kk<(N - ii);kk++) {
				acf[ii] += rhozero[kk]*rhozero[ii+kk];
			}
			acf[ii] = acf[ii] / rhovar;
		}
	} /* end of omp parallel num_threads */
	
	/* output autocorrelation for various lags */
	fpout = fopen("acfout.dat","wb");
	fwrite(&Lag,1,sizeof(int),fpout);
	for(ii=0;ii<(Lag+1);ii++) {
		acftmp=acf[ii];
		fwrite(&acftmp,1,sizeof(double),fpout);
	}
	fclose(fpout);
	
	printf("\n5. Freeing allocated arrays..."); fflush(stdout);
	if (rho!=NULL) mkl_free(rho);
	if (rhomeanvec!=NULL) mkl_free(rhomeanvec);
	if (rhozero!=NULL) mkl_free(rhozero);
	if (acf!=NULL) mkl_free(acf);
	
	printf("\n6. Exit - success!\n"); fflush(stdout);
} /* end of main() */
