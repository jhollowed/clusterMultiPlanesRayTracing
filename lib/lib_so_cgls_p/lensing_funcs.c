#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include "fft_convolve.h"
//--------------------------------------------------------------------
void roll_a_matrix(double *in, int nx1, int nx2, int roll_nx1, int roll_nx2, double *out) {
	int i,j;
	int index1, index2;

	for(i=0;i<nx1;i++) for(j=0;j<nx2;j++) {
		index1 = (i+roll_nx1)%nx1;
		index2 = (j+roll_nx2)%nx2;
		out[index1*nx2+index2] = in[i*nx2+j];
	}
}
//--------------------------------------------------------------------
void kernel_alphas_iso(int Ncc,double *in1,double *in2,double dsx) {
	int i,j;
	double x,y,r;

	/*for(i=0;i<Ncc;i++) for(j=0;j<Ncc;j++) {*/
		/*if(i <=(Ncc/2)  && j <=(Ncc/2)) {*/
			/*x = (double)(i)*dsx+0.5*dsx;*/
			/*y = (double)(j)*dsx+0.5*dsx;*/
			/*r = sqrt(x*x+y*y);*/

			/*if(r > dsx*(double)Ncc/2.0) {*/
				/*in1[i*Ncc+j] = 0.0;*/
				/*in2[i*Ncc+j] = 0.0;*/
			/*}*/
			/*else {*/
				/*in1[i*Ncc+j] = x/(M_PI*r*r);*/
				/*in2[i*Ncc+j] = y/(M_PI*r*r);*/
			/*}*/

		/*}*/
		/*else {*/
			/*if(i <= Ncc/2 && j > (Ncc/2)) {*/
				/*in1[i*Ncc+j]  =  in1[i*Ncc+Ncc-j];*/
				/*in2[i*Ncc+j]  = -in2[i*Ncc+Ncc-j];*/
			/*}*/
			/*if(i > (Ncc/2) && j <= (Ncc/2)) {*/
				/*in1[i*Ncc+j]  = -in1[(Ncc-i)*Ncc+j];*/
				/*in2[i*Ncc+j]  =  in2[(Ncc-i)*Ncc+j];*/
			/*}*/

			/*if(i > (Ncc/2) && j > (Ncc/2)) {*/
				/*in1[i*Ncc+j]  = -in1[(Ncc-i)*Ncc+Ncc-j];*/
				/*in2[i*Ncc+j]  = -in2[(Ncc-i)*Ncc+Ncc-j];*/
			/*}*/
		/*}*/
	/*}*/

	for(i=0;i<Ncc;i++) for(j=0;j<Ncc;j++) {
		x = ((double)(i-Ncc/2)+0.5)*dsx;
		y = ((double)(j-Ncc/2)+0.5)*dsx;
		r = sqrt(x*x+y*y);

		if(r > dsx*(double)Ncc/2.0) {
			in1[i*Ncc+j] = 0.0;
			in2[i*Ncc+j] = 0.0;
		}
		else {
			in1[i*Ncc+j] = x/(M_PI*r*r);
			in2[i*Ncc+j] = y/(M_PI*r*r);
		}
	}
}
//--------------------------------------------------------------------
void kappa0_to_alphas(double * kappa0, int Nc, double bsz, double * alpha1, double * alpha2) {

	double dsx = bsz/(double)Nc;

	double *alpha1_iso = (double *)calloc(Nc*Nc,sizeof(double));
	double *alpha2_iso = (double *)calloc(Nc*Nc,sizeof(double));

	kernel_alphas_iso(Nc,alpha1_iso,alpha2_iso,dsx);

	/*//---------------*/
	/*convolve_fft(kappa0,alpha1_iso,alpha1,Nc,Nc,dsx,dsx);*/
	/*convolve_fft(kappa0,alpha2_iso,alpha2,Nc,Nc,dsx,dsx);*/

	/*free(alpha1_iso);*/
	/*free(alpha2_iso);*/

	//---------------
	double *alpha1_tmp = (double *)calloc(Nc*Nc,sizeof(double));
	double *alpha2_tmp = (double *)calloc(Nc*Nc,sizeof(double));

	convolve_fft(kappa0,alpha1_iso,alpha1_tmp,Nc,Nc,dsx,dsx);
	convolve_fft(kappa0,alpha2_iso,alpha2_tmp,Nc,Nc,dsx,dsx);

	free(alpha1_iso);
	free(alpha2_iso);

	roll_a_matrix(alpha1_tmp, Nc, Nc, Nc/2, Nc/2, alpha1);
	roll_a_matrix(alpha2_tmp, Nc, Nc, Nc/2, Nc/2, alpha2);

	free(alpha1_tmp);
	free(alpha2_tmp);

	//---------------
}
