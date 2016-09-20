/* 
Note: I only tested it for C compiler on UNIX.  Use "gcc ASAMs.c -lm" or
      "cc ASAMs.c -lm" to compile. The program runs 6 types of Adaptive
      SAM (ASAM) function approximators at once -- Sinc SAM, Gaussian SAM,
      Cauchy SAM, Hyperbolic tangent SAM, Laplace SAM, and Triangle SAM.
      You may need to run fewer types of SAMs at a time if your computer
      cannot allocate enough memory.  Please report any syntax corrections
      for other compilers (C++, Visual C, etc.)
      
   Program ASAMs - learns a function on an interval according to
   the learning equations  "What is the Best Shape for a Fuzzy Set
   in Function  Approximation?" in the Proceedings of the Fifth IEEE
   International Conference on Fuzzy Systems (FUZZ-IEEE 96) vol. 2,
   pp. 1237-1243 by Sanya Mitaim and Bart Kosko.

   ASAMs date 6 March 1997, version 0.9.   Function is a simple
   function of x and this program is self-contained - it has all inputs
   defined in the program.

   The program uses the same parameters described in the paper except that
   it terminates prematurely due to time constraint.  Users may wish to
   modify the parameters in the program (e.g. threshold values, learning
   coefficients, etc) to better fit their requirement.

   Run this program as is. Later you can vary parameters and the function.

   ASAMs will display the mean squared error of function approximation
   as it iterates the learning equations for the known function f(x) for
   x between MINX and MAXX. (See sample[i] in the function "realfunction").
   ASAMs as issued has 12 fuzzy sets on the input side and 12 sets on
   the output side (for each kind of ASAM).  Program has parameters you
   can change.  These include the number of sample data points, the number
   of if-part fuzzy sets on the input side, and the function itself. Note
   that the variable NUMPAT is the number of sets on the input side.  The
   output side of the ASAMs has same number of sets as the input side but
   output side sets have only centroids and volumes (areas in 1-D).  They
   have no definite shape (due the structure of SAM).

   Learning occurs in a SAM when any parameter of any set functions
   changes.  This program first defines sets then tunes them for better
   approximation.  The purpose of this program is then to:

   1. Place if-part sets on the input (x) side of the SAM.
   2. Place output (f(x)) side sets.
   3. change set locations, widths, centroids, and volumes for better
   approximation.

   Notes:  
   (1) Textbooks define sinc(x) as sin(pi*x)/(pi*x)  but this program
       allocates sinc functions over the x axis and the argument is
       actually:

       Sin(in)/in       where in = (x-m)/d.

       Thus the center of each sinc is at "m" on the x axis. d is a "width"
       parameter.

   (2) Textbook defines hyperbolic tangent function tanh(x) as
                     tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))
       which is of S-shape.  We define the hyperbolic tangent set function
       as in eq (9) in the paper so that it is of bell curve.

   (3) Triangle set functions in this program are symmetric.  It has the
       same width to both left and right in eq (2) in the paper.


   INPUT
   Self-contained function in "realfunction" and parameters as below.

   OUTPUT
   (1) Prints out the summed squared error as iterations proceed.
   (2) Leaves data files "XXX.par" for various parameters where XXX
       will be SincSAM, GssSAM, ChySAM, TanhSAM, LaplSAM, and TriSAM.
   (3) Leaves data files "XXX.dat" containing x and F (input-output pairs
       for each type of SAM: FSinc.dat, FGss.dat, etc.)

*************************************************************************
   Sample output values when run as is are:

                  Mean-Squared Errors
Iterations  SincSAM  GaussSAM  CauchySAM  TanhSAM LaplaceSAM TriangleSAM
     0       13.905     9.201     9.134     8.771      8.739       8.972
   100       2.283     7.172     6.973     6.011      5.553       8.486
   200       1.894     5.647     5.850     4.484      4.195       8.048
   300       1.707     4.645     5.317     3.791      3.782       7.651
   400       1.559     4.109     5.014     3.505      3.544       7.292
   500       1.426     3.845     4.782     3.352      3.354       6.965

Parameters of each SAM are in the following files:
       Sinc SAM:     SincSAM.par
       Gaussian SAM: GssSAM.par
       Cauchy SAM:   ChySAM.par
       Hyperbolic Tangent SAM: TanhSAM.par
       Laplace SAM:  LaplSAM.par
       Triangle SAM: TriSAM.par

Format of the storage: for if-part sets which have 2 parameters:
         If-part parameters        Then-part parameters
         [<location> <width>       <centroid> <Volume>]
For other if-part sets such as Tanh and Triangle, the program will
write 3 parameters of if-part sets from each rule first, then the
2 parameters of the then-part sets (centroids and volumes).

File <SincSAM.par> contains (shown here only parameters of 2 rules):
  msinc       dsinc       censinc     Vsinc --these do not appear in the file
-1.3533e-02  1.6469e-02  9.9671e+00  1.0000e+00
9.2573e-02  3.5614e-02  -2.8431e-01  1.0001e+00

File <TanhSAM.par> contains (shown here only parameters of 2 rules):
 mtanh      ltanh       dtanh       centanh     dtanh --not appear in the file
-1.7507e-02  2.9996e-02  1.4122e-02  9.9671e+00 9.9969e-01
8.6182e-02  5.4757e-02  3.3092e-02  -2.8445e-01 1.0003e+00

*************************************************************************
*/

#include <stdio.h>
#include  <math.h>
#include <stdlib.h>
#define NUMPAT  12     /* number of fuzzy sets on input or output side*/
#define NUMSAM  200    /* use NUMSAM number of data pairs for training */
#define NUMDES  400    /* use NUMDES number of data pairs for testing */
float MINX = 0.0;      /* lower limit for x axis  */
float MAXX = 1.0;      /* upper limit for x axis  */
int TotalIters = 500; /* iterations to perform before termination */

float x[NUMSAM];      /* x values for training */
float sample[NUMSAM]; /* f(x) values (little f in paper) for training */
float xtest[NUMDES];  /* x values for testing */
float des[NUMDES];    /* f(x) values (little f in paper) for testing */

/********** Sinc SAM's parameters **********/
float msinc[NUMPAT];    /* "location" of sinc if-part set function */
float dsinc[NUMPAT];    /* "width" of sinc if-part set function */
float censinc[NUMPAT];  /* then-part set centroid in sinc SAM */
float Vsinc[NUMPAT];    /* volume (area) of then-part set in sinc SAM */
float asinc[NUMPAT];    /* sinc set values */
float densinc, xmdsinc[NUMPAT]; /* miscel. parameters */
float SincSAM();     /* Sinc SAM */
float SincSAMapprox();
float Fsinc[NUMDES];    /* Sinc SAM function approximation */
/*******************************************/

/******** Gaussian SAM's parameters ********/
float mgs[NUMPAT];      /* "location" of gaussian if-part set function */
float dgs[NUMPAT];      /* "width" of gaussian if-part set function */
float cengs[NUMPAT];    /* then-part set centroid in Gaussian SAM */
float Vgs[NUMPAT];      /* volume (area) of then-part set in Gaussian SAM */
float ags[NUMPAT];      /* gaussian set values */
float dengs, xmdgs[NUMPAT];  /* miscel. parameters */
float GaussianSAM(); /* Gaussian SAM */
float GaussianSAMapprox();
float Fgss[NUMDES];    /* Gaussian SAM function approximation */
/*******************************************/

/********  Cauchy SAM's parameters  ********/
float mchy[NUMPAT];     /* "location" of cauchy if-part set function */
float dchy[NUMPAT];     /* "width" of cauchy if-part set function */
float cenchy[NUMPAT];   /* then-part set centroid in Cauchy SAM*/
float Vchy[NUMPAT];     /* volume (area) of then-part set in Cauchy SAM */
float achy[NUMPAT];     /* cauchy set values */
float denchy, xmdchy[NUMPAT]; /* miscel. parameters */
float CauchySAM();  /* Cauchy SAM */ 
float CauchySAMapprox();
float Fchy[NUMDES];    /* Cauchy SAM function approximation */
/******************************************/

/********   Tanh SAM's parameters  ********/
float mtanh[NUMPAT];    /* "location" of tanh if-part set function */
float ltanh[NUMPAT];    /* parameters in tanh set function */
float dtanh[NUMPAT];    /* parameters in tanh set function */
float Dtanh[NUMPAT];    /* normalization of a(x) */
float centanh[NUMPAT];  /* then-part set centroid in tanh SAM*/
float Vtanh[NUMPAT];    /* volume (area) of then-part set in tanh SAM*/
float a_tanh[NUMPAT];   /* tanh set values */
float dentanh, xmdtanh[NUMPAT],      /* miscel. parameters */
  xmldtanh[NUMPAT], tanhl_d[NUMPAT], l_d[NUMPAT], 
  tanhxmd[NUMPAT], tanhxmld[NUMPAT];
float TanhSAM();    /* Tanh SAM */
float TanhSAMapprox();
float Ftanh[NUMDES];    /* Hyperbolic tangent SAM function approximation */
/******************************************/

/******** Laplace SAM's parameters ********/
float mlp[NUMPAT];     /* "location" of laplace if-part set function */
float dlp[NUMPAT];     /* "width" of laplace if-part set function */
float cenlp[NUMPAT];   /* then-part set centroid in laplace SAM */
float Vlp[NUMPAT];     /* volume (area) of then-part set in Laplace SAM */
float alp[NUMPAT];    /* laplace set values */
float denlp, fxmdlp[NUMPAT];  /* miscel. parameters */
float LaplaceSAM();  /* Laplace SAM */
float LaplaceSAMapprox();
float Flapl[NUMDES];    /* Laplace SAM function approximation */
/******************************************/

/******** Triangle SAM's parameters ********/
float mtri[NUMPAT];    /* "location" of triangle set function */
float dtri[NUMPAT];    /* "width" of triangle set function */
float centri[NUMPAT];  /* then-part set centroid in Triangle SAM */
float Vtri[NUMPAT];    /* volume (area) of then-parrt set in Triangle SAM */
float atri[NUMPAT];    /* set function */
float dentri; /* miscel. parameters */
float TriangleSAM(); /* Triangle SAM */
float TriangleSAMapprox();
float Ftri[NUMDES];    /* Triangle SAM function approximation */
/******************************************/


void SaveSAMsParameters();
void SaveSAMsApproximation();
float SIGN();
void writedat();

void main()
{
  realfunction();
  ASAMsInitialize();
  printf("\n                  Mean-Squared Errors\n");
  printf("Iterations  SincSAM  GaussSAM  CauchySAM  TanhSAM LaplaceSAM TriangleSAM\n");
  ComputeMSEs(0);
  ASAMsLearning();
  SaveSAMsParameters();
  SaveSAMsApproximation();
  system("PAUSE");
  return EXIT_SUCCESS;
}

int ASAMsInitialize()
{
  SincSAMinit(); 
  GaussianSAMinit(); 
  CauchySAMinit(); 
  TanhSAMinit(); 
  LaplaceSAMinit(); 
  TriangleSAMinit(); 
}

int ASAMsLearning()
{
  int j;
  for (j=1;j<=TotalIters;j++) {
    SincSAMlearn();
    GaussianSAMlearn();
    CauchySAMlearn();
    TanhSAMlearn();
    LaplaceSAMlearn();
    TriangleSAMlearn();
    if (j%100 == 0)   ComputeMSEs(j);  /* Compute MSEs every 100 iterations */
  }
}

ComputeMSEs(iter)
int iter;
{
  float MSEsincSAM, MSEgaussSAM, MSEcauchySAM, MSEtanhSAM, 
        MSElaplaceSAM, MSEtriangleSAM;

  MSEsincSAM = SincSAMapprox();
  MSEgaussSAM = GaussianSAMapprox();
  MSEcauchySAM = CauchySAMapprox();
  MSEtanhSAM = TanhSAMapprox();
  MSElaplaceSAM = LaplaceSAMapprox();
  MSEtriangleSAM = TriangleSAMapprox();
  printf (" %5d       %1.3f     %1.3f     %1.3f     %1.3f      %1.3f       %1.3f\n",
	  iter,MSEsincSAM, MSEgaussSAM, MSEcauchySAM, MSEtanhSAM,
	  MSElaplaceSAM,MSEtriangleSAM);
}

int realfunction()
{
  int i;
  float step, xold;
  step = (MAXX-MINX)/(float)(NUMSAM-1);
  xold = MINX-step;
  for (i=0;i<NUMSAM;i++)   {
    x[i] = xold + step;
    xold = x[i];
  }
  for (i=0;i<NUMSAM;i++)  {
    sample[i] = (1.0+10.0*exp(-(x[i]-0.7)*(x[i]-0.7)*100.0))*
      sin(125.0/(x[i]+1.5))/(x[i]+0.1);
  }

  step = (MAXX-MINX)/(float)(NUMDES-1);
  xold = MINX-step;
  for (i=0;i<NUMDES;i++)   {
    xtest[i] = xold + step;
    xold = xtest[i];
  }
  for (i=0;i<NUMDES;i++)  {
    des[i] = (1.0+10.0*exp(-(xtest[i]-0.7)*(xtest[i]-0.7)*100.0))*
      sin(125.0/(xtest[i]+1.5))/(xtest[i]+0.1);
  }
}

void SaveSAMsParameters()
{
  printf("\nParameters of each SAM are in the following files:\n");
  printf("       Sinc SAM:     SincSAM.par\n");
  printf("       Gaussian SAM: GssSAM.par\n");
  printf("       Cauchy SAM:   ChySAM.par\n");
  printf("       Hyperbolic Tangent SAM: TanhSAM.par\n");
  printf("       Laplace SAM:  LaplSAM.par\n");
  printf("       Triangle SAM: TriSAM.par\n");
  printf("\nFormat of the storage: for if-part sets which have 2 parameters:");
  printf("\n         If-part parameters        Then-part parameters");
  printf("\n       [<location> <width>       <centroid> <Volume>]\n");
  printf("For other if-part sets such as Tanh and Triangle, the program will\n");
  printf("write 3 parameters of if-part sets from each rule first, then the\n");
  printf("2 parameters of the then-part sets (centroids and volumes).\n");
  writepara_sinc("SincSAM.par");
  writepara_gaussian("GssSAM.par");
  writepara_cauchy("ChySAM.par");
  writepara_tanh("TanhSAM.par");
  writepara_laplace("LaplSAM.par");
  writepara_triangle("TriSAM.par");
}

void SaveSAMsApproximation()
{
  printf("\nPoints of each SAM approximation (x F) are in the files:\n");
  printf("       Desired function (f): function.dat\n");
  printf("       Sinc SAM:     FSincSAM.dat\n");
  printf("       Gaussian SAM: FGssSAM.dat\n");
  printf("       Cauchy SAM:   FChySAM.dat\n");
  printf("       Hyperbolic Tangent SAM: FTanhSAM.dat\n");
  printf("       Laplace SAM:  FLaplSAM.dat\n");
  printf("       Triangle SAM: FTriSAM.dat\n");
  writedat("function.dat",xtest,des);
  writedat("FSincSAM.dat",xtest,Fsinc);
  writedat("FGssSAM.dat",xtest,Fgss);
  writedat("FChySAM.dat",xtest,Fchy);
  writedat("FTanhSAM.dat",xtest,Ftanh);
  writedat("FLaplSAM.dat",xtest,Flapl);
  writedat("FTriSAM.dat",xtest,Ftri);
}

void writedat(out,xx,yy)
char out[50];
float xx[NUMDES], yy[NUMDES];
{
  FILE *ofp;
  int i;
  if ((ofp = fopen(out, "w")) == NULL) {
    printf("Could not open output file\n");
    exit(1);
  }
  for (i=0;i<NUMDES;i++)
    fprintf(ofp,"%1.4e  %1.4e\n", xx[i], yy[i]);
  fclose(ofp);  
}

/****************************************************/
/****************************************************/
int SincSAMlearn()
{
  int i,k;
  float mu_m, mu_d, mu_cen, mu_V, fuzoutsinc;
  float cenerr, cenerr_den, cenerrV_den, cosxmd, acosxmdcenerrV_den;
  float dEdd, dEdm, dEdF, dEdFacosblabla;
  
  mu_m   = 1e-8;
  mu_d   = 1e-8;
  mu_cen = 1e-8;
  mu_V   = 1e-8;
  
  for (k=0;k<NUMSAM;k++)  {
    fuzoutsinc = SincSAM(x[k]);
    dEdF = -(sample[k]-fuzoutsinc);
    for (i=0;i<NUMPAT;i++)  {
      if (asinc[i] != 0.0)  {
	cenerr = censinc[i]-fuzoutsinc;
	cenerr_den = cenerr/densinc;
	cenerrV_den = cenerr_den*Vsinc[i];
	cosxmd = cos(xmdsinc[i]);
	acosxmdcenerrV_den = (asinc[i]-cosxmd)*cenerrV_den;
	dEdFacosblabla = dEdF*acosxmdcenerrV_den;
	if (xmdsinc[i] != 0.0)
	  dEdm = dEdFacosblabla/(x[k]-msinc[i]);
	else dEdm = 0.0;
	dEdd = dEdFacosblabla/dsinc[i];
	
	dsinc[i] -= mu_d*dEdd;
	msinc[i] -= mu_m*dEdm;
	censinc[i] -= mu_cen*dEdF*Vsinc[i]*asinc[i]/densinc; 
	Vsinc[i] -= mu_V*dEdF*asinc[i]*cenerr_den;
      }
    }
  }
}

float SincSAM(in)
float in;
{
  int i;
  float av, num=0.0;
  
  densinc = 0.0;
  for (i=0;i<NUMPAT;i++)  {
    xmdsinc[i] = (in - msinc[i])/dsinc[i];
    if (xmdsinc[i] == 0.0)  asinc[i] = 1.0;
    else  asinc[i] = sin(xmdsinc[i])/xmdsinc[i];
    
    av = asinc[i]*Vsinc[i];
    densinc = densinc+av;
    num = num+av*censinc[i];
  }
  if (densinc > 0.0)  return num/densinc;
  else if (densinc < 0.0)  {
    printf("densinc = %f < 0!!\n",densinc);
    return 0.0;
  }
  else                 return 0.0;
}

int SincSAMinit()
{
  int i;
  float j,STEP,base;
  STEP = ((float)(NUMSAM-1))/((float)(NUMPAT-1));
  base = (float)(MAXX-MINX)/(float)(NUMPAT-1);
  i = -1;
  for (j=0.0;j<(float)NUMSAM;j+=STEP)  {
    i += 1;
    censinc[i] = sample[(int)(j+.5)];
    dsinc[i] = base/3.1416;
    msinc[i] = x[(int)(j+.5)];
    Vsinc[i] = 1.0;
  }
}

float SincSAMapprox()
{
  int k;
  float err, sumerr=0.0;

  for (k=0;k<NUMDES;k++) {
    Fsinc[k] = SincSAM(xtest[k]);
    err = des[k]-Fsinc[k];
    sumerr += err*err;
  }
  return sumerr/(float)NUMDES;
}

int writepara_sinc(out)
char out[50];
{
  FILE *ofp;
  int i;
  if ((ofp = fopen(out, "w")) == NULL) {
    printf("Could not open output file\n");
    exit(1);
  }
  for (i=0;i<NUMPAT;i++)
    fprintf(ofp,"%1.4e  %1.4e  %1.4e  %1.4e\n",
	    msinc[i], dsinc[i], censinc[i], Vsinc[i]);
  fclose(ofp);
}

/****************************************************/
/****************************************************/
int GaussianSAMlearn()
{
  int i,k;
  float mu_m, mu_d, mu_cen, mu_V, fuzoutgs;
  float cenerr, a_den, pix, cenerr_d;
  float dEdm, dEdd, dEdF, dEdFa_den;
  
  mu_m   = 2e-8;
  mu_d   = 2e-8;
  mu_cen = 1e-8;
  mu_V   = 1e-8;
  
  for (k=0;k<NUMSAM;k++) {
    fuzoutgs = GaussianSAM(x[k]);
    dEdF = -(sample[k]-fuzoutgs);
    if (dengs != 0.0) {
      for (i=0;i<NUMPAT;i++) {
	a_den = ags[i]/dengs;
	pix = a_den*Vgs[i];
	cenerr = cengs[i]-fuzoutgs;
	cenerr_d = cenerr/dgs[i];
	
	dEdm = dEdF*pix*cenerr_d*xmdgs[i];
	dEdd = dEdm*xmdgs[i];
	
	dEdFa_den = dEdF*a_den;
	
	dgs[i] -= mu_d*dEdd;
	mgs[i] -= mu_m*dEdm;
	cengs[i] -= mu_cen*Vgs[i]*dEdFa_den;
	Vgs[i] -= mu_V*dEdFa_den*cenerr;
      }
    }
  }
}

float GaussianSAM(in)
float in;
{
  int i;
  float av, num=0.0;

  dengs = 0.0;
  for (i=0;i<NUMPAT;i++)  {
    xmdgs[i] = (in-mgs[i])/dgs[i];
    ags[i] = exp(-xmdgs[i]*xmdgs[i]);
    av = ags[i]*Vgs[i];
    dengs = dengs+av;
    num = num+av*cengs[i];
  }
  if (dengs != 0.0)  return num/dengs;
  else               return 0.0;
}

int GaussianSAMinit()
{
  int i;
  float j,STEP,base;
  
  STEP = ((float)(NUMSAM-1))/((float)(NUMPAT-1));
  base = (float)(MAXX-MINX)/(float)(NUMPAT-1);
  i = -1;;
  for (j=0.0;j<(float)NUMSAM;j+=STEP) {
    i += 1;
    cengs[i] = sample[(int)(j+.5)];
    dgs[i] = base/1.52;
    mgs[i] = x[(int)(j+.5)];
    Vgs[i] = 1.0;
  }
}

float GaussianSAMapprox()
{
  int k;
  float err, sumerr=0.0;

  for (k=0;k<NUMDES;k++) {
    Fgss[k] = GaussianSAM(xtest[k]);
    err = des[k]-Fgss[k];
    sumerr += err*err;
  }
  return sumerr/(float)NUMDES;
}

int writepara_gaussian(out)
char out[50];
{
  FILE *ofp;
  int i;
  if ((ofp = fopen(out, "w")) == NULL) {
    printf("Could not open output file\n");
    exit(1);
  }
  for (i=0;i<NUMPAT;i++) {
    fprintf(ofp,"%1.4e  %1.4e  %1.4e  %1.4e\n",
	    mgs[i], dgs[i], cengs[i], Vgs[i]);
  }
  fclose(ofp);
}

/****************************************************/
/****************************************************/
int CauchySAMlearn()
{
  int i,k;
  float mu_m, mu_d, mu_cen, mu_V, fuzoutchy;
  float cenerr, a_den, pix, cenerr_d;
  float dEdm, dEdd, dEdF, dEdFa_den;
  
  mu_d   = 2e-8;  
  mu_m   = 2e-8;
  mu_cen = 1e-8;
  mu_V   = 1e-8;
  
  for (k=0;k<NUMSAM;k++) {
    fuzoutchy = CauchySAM(x[k]);
    dEdF = -(sample[k]-fuzoutchy);
    if (denchy != 0.0) {
      for (i=0;i<NUMPAT;i++) {
	a_den = achy[i]/denchy;
	pix = a_den*Vchy[i];
	cenerr = cenchy[i]-fuzoutchy;
	cenerr_d = cenerr/dchy[i];
	
	dEdm = dEdF*pix*cenerr_d*xmdchy[i]*achy[i];
	dEdd = dEdm*xmdchy[i];
	
	dEdFa_den = dEdF*a_den;
	
	dchy[i] -= mu_d*dEdd;
	mchy[i] -= mu_m*dEdm;
	cenchy[i] -= mu_cen*Vchy[i]*dEdFa_den; 
	Vchy[i] -= mu_V*dEdFa_den*cenerr;
      }
    }
  }
}

float CauchySAM(in)
float in;
{
  int i;
  float av,num;
  
  denchy = 0.0;
  num = 0.0;
  for (i=0;i<NUMPAT;i++)  {
    xmdchy[i] = (in - mchy[i])/dchy[i];
    achy[i] = 1.0/(1.0 + xmdchy[i]*xmdchy[i]);

    av = achy[i]*Vchy[i];
    denchy = denchy+av;
    num = num+av*cenchy[i];
  }
  if (denchy != 0.0)  return num/denchy;
  else                return 0.0;
}

int CauchySAMinit()
{
  int i;
  float j,STEP,base;
  STEP = ((float)(NUMSAM-1))/((float)(NUMPAT-1));
  base = (float)(MAXX-MINX)/(float)(NUMPAT-1);
  i = -1;
  for (j=0.0;j<(float)NUMSAM;j+=STEP) {
    i += 1;
    cenchy[i] = sample[(int)(j+.5)];
    dchy[i] = base/3.0;
    mchy[i] = x[(int)(j+.5)];
    Vchy[i] = 1.0;
  }
}

float CauchySAMapprox()
{
  int k;
  float err, sumerr=0.0;

  for (k=0;k<NUMDES;k++) {
    Fchy[k] = CauchySAM(xtest[k]);
    err = des[k]-Fchy[k];
    sumerr += err*err;
  }
  return sumerr/(float)NUMDES;
}

int writepara_cauchy(out)
char out[50];
{
  FILE *ofp;
  int i;
  if ((ofp = fopen(out, "w")) == NULL) {
    printf("Could not open output file\n");
    exit(1);
  }
  for (i=0;i<NUMPAT;i++) 
    fprintf(ofp,"%1.4e  %1.4e  %1.4e  %1.4e\n",
	    mchy[i], dchy[i], cenchy[i], Vchy[i]);
  fclose(ofp);
}

/****************************************************/
/****************************************************/
int TanhSAMlearn()
{
  int i,k;
  float mu_m, mu_d, mu_l, mu_cen, mu_V, fuzouttanh;
  float dD, tanh2xmd, tanh2xmld, cenerr, ajtanhl_d2;
  float dEdaj, dEdFa_den, dEdd, dEdl, dEdm, dEdF, dFdaj;
  float dajdl, dajdd, dajdm;

  mu_d   = 1e-8;
  mu_l   = 1e-8;
  mu_m   = 1e-8;
  mu_cen = 1e-8;
  mu_V   = 1e-8;

  for (k=0;k<NUMSAM;k++)  {
    fuzouttanh = TanhSAM(x[k]);
    dEdF = -(sample[k]-fuzouttanh);
    if (dentanh != 0.0)  {
      for (i=0;i<NUMPAT;i++)  {
	if (a_tanh[i] > 0.0) {
	  cenerr = centanh[i]-fuzouttanh;
	  dFdaj = Vtanh[i]*cenerr/dentanh;
	  dD = dtanh[i]*Dtanh[i];
	  tanh2xmd = tanhxmd[i]*tanhxmd[i];
	  tanh2xmld = tanhxmld[i]*tanhxmld[i];
	  ajtanhl_d2 = 2.0*a_tanh[i]*(1.0 - tanhl_d[i]*tanhl_d[i]);
	  
	  dajdm = (tanh2xmd - tanh2xmld)/dD;
	  dajdd = (xmdtanh[i]*tanh2xmd - xmldtanh[i]*tanh2xmld - 2.0*l_d[i]
		   + ajtanhl_d2*l_d[i])/dD; 
	  dajdl = (2.0 - tanh2xmld - tanh2xmd - ajtanhl_d2)/dD;
	  
	  dEdaj = dEdF*dFdaj;
	  dEdd = dEdaj*dajdd;
	  dEdl = dEdaj*dajdl;
	  dEdm = dEdaj*dajdm;
	  dEdFa_den = dEdF*a_tanh[i]/dentanh;
	  
	  dtanh[i] -= mu_d*dEdd;
	  ltanh[i] -= mu_l*dEdl;
	  mtanh[i] -= mu_m*dEdm;
	  centanh[i] -= mu_cen*dEdFa_den*Vtanh[i];
	  Vtanh[i] -= mu_V*dEdFa_den*cenerr;
	}
      }
    }
  }
}

float TanhSAM(in)
float in;
{
  int i;
  float xm, xml, av, num;

  dentanh = 0.0;
  num = 0.0;
  for (i=0;i<NUMPAT;i++) {
    xm = in - mtanh[i] + ltanh[i];
    xml = in - mtanh[i] - ltanh[i];
    xmdtanh[i] = xm/dtanh[i];
    xmldtanh[i] = xml/dtanh[i];
    l_d[i] = ltanh[i]/dtanh[i];
    tanhl_d[i] = tanh(l_d[i]);
    Dtanh[i] = 2*tanhl_d[i];
    tanhxmd[i] = tanh(xmdtanh[i]);
    tanhxmld[i] = tanh(xmldtanh[i]);
    a_tanh[i] = (tanhxmd[i]-tanhxmld[i])/Dtanh[i];
    av = a_tanh[i]*Vtanh[i];
    dentanh = dentanh+av;
    num = num+av*centanh[i];
  }
  if (dentanh != 0.0)  return num/dentanh;
  else                 return 0.0;
}

int TanhSAMinit()
{
  int i;
  float j,STEP,base;
  STEP = ((float)(NUMSAM-1))/((float)(NUMPAT-1));
  base = (float)(MAXX-MINX)/(float)(NUMPAT-1);
  i = -1;
  for (j=0.0;j<(float)NUMSAM;j+=STEP)  {
    i += 1;
    dtanh[i] = base/3.0;
    ltanh[i] = base/2.0;
    mtanh[i] = x[(int)(j+.5)];
    centanh[i] = sample[(int)(j+.5)];
    Vtanh[i] = 1.0;
  }
}

float TanhSAMapprox()
{
  int k;
  float err, sumerr=0.0;

  for (k=0;k<NUMDES;k++) {
    Ftanh[k] = TanhSAM(xtest[k]);
    err = des[k]-Ftanh[k];
    sumerr += err*err;
  }
  return sumerr/(float)NUMDES;
}

int writepara_tanh(out)
char out[50];
{
  FILE *ofp;
  int i;

  if ((ofp = fopen(out, "w")) == NULL) {
    printf("Could not open output file\n");
    exit(1);
  }
  for (i=0;i<NUMPAT;i++) {
    fprintf(ofp,"%1.4e  %1.4e  %1.4e  %1.4e %1.4e\n", 
	    mtanh[i], ltanh[i], dtanh[i], centanh[i], Vtanh[i]);
  }
  fclose(ofp);
}

/****************************************************/
/****************************************************/
int LaplaceSAMlearn()
{
  int i,k;
  float mu_m, mu_d, mu_cen, mu_V, fuzoutlp;
  float cenerr, err, xm, dEdm, dEdd, dEdFpicenerr_fd;
  float pj;  /* convex coefficient */

  mu_m   = 1e-8;  /* learning rate */
  mu_d   = 1e-8; 
  mu_cen = 1e-8;
  mu_V   = 1e-8;
  
  for (k=0;k<NUMSAM;k++)  {
    fuzoutlp = LaplaceSAM(x[k]);
    err = sample[k]-fuzoutlp;
    if (denlp != 0.0)  {
      for (i=0;i<NUMPAT;i++)  {
	xm = x[k]-mlp[i];
	cenerr = cenlp[i] - fuzoutlp;
	pj = Vlp[i]*alp[i]/denlp;
	dEdFpicenerr_fd = -err*pj*cenerr/fabs(dlp[i]);
	dEdm = dEdFpicenerr_fd*SIGN(xm);
	dEdd = dEdFpicenerr_fd*SIGN(dlp[i])*fxmdlp[i];
	
	mlp[i] -=  mu_m*dEdm;  
	dlp[i] -= mu_d*dEdd;
	Vlp[i] += mu_V*err*alp[i]*cenerr/denlp;
	cenlp[i] += mu_cen*err*pj;
      }
    }
  }
}

float LaplaceSAM(in)
float in;
{
  int i;
  float xm, av, num;
  
  denlp = 0.0;
  num = 0.0;
  for (i=0;i<NUMPAT;i++) {
    xm = in-mlp[i];
    fxmdlp[i] = fabs(xm/dlp[i]);
    alp[i] = exp(-fxmdlp[i]);
    av = alp[i]*Vlp[i];
    denlp = denlp+av;
    num = num+av*cenlp[i];
  }
  if (denlp != 0.0)  return num/denlp;
  else               return 0.0;
}

int LaplaceSAMinit()
{
  int i;
  float j,STEP,base;
  STEP = ((float)(NUMSAM-1))/((float)(NUMPAT-1));
  base = (float)(MAXX-MINX)/(float)(NUMPAT-1);
  i = -1;
  for (j=0.0;j<(float)NUMSAM;j+=STEP) {
    i += 1;
    cenlp[i] = sample[(int)(j+.5)];
    dlp[i] = base/3.0;
    mlp[i] = x[(int)(j+.5)];
    Vlp[i] = 1.0;
  }
}

float LaplaceSAMapprox()
{
  int k;
  float err, sumerr=0.0;

  for (k=0;k<NUMDES;k++) {
    Flapl[k] = LaplaceSAM(xtest[k]);
    err = des[k]-Flapl[k];
    sumerr += err*err;
  }
  return sumerr/(float)NUMDES;
}

int writepara_laplace(out)
char out[50];
{
  FILE *ofp;
  int i;
  
  if ((ofp = fopen(out, "w")) == NULL) {
    printf("Could not open output file\n");
    exit(1);
  }
  for (i=0;i<NUMPAT;i++) {
    fprintf(ofp,"%f  %f  %f  %f", mlp[i], dlp[i], cenlp[i], Vlp[i]);
    fprintf(ofp,"\n");
  }
  fclose(ofp);
}

/****************************************************/
/****************************************************/
int TriangleSAMlearn()  /* Use Guassian learning laws to tune m and d */
{
  int i,k;
  float mu_d, mu_m, mu_cen, mu_V, fuzouttri;
  float cenerr, err, dEdd, dEdm;
  float pj;  /* convex coefficient */

  mu_d   = 1e-8;
  mu_m   = 1e-8;
  mu_cen = 1e-8;
  mu_V   = 1e-8;

  for (k=0;k<NUMSAM;k++)  {
    fuzouttri = TriangleSAM(x[k]);
    err = sample[k]-fuzouttri;
    if (dentri != 0.0)  {
      for (i=0;i<NUMPAT;i++)  {
	cenerr = centri[i] - fuzouttri;
	pj = Vtri[i]*atri[i]/dentri;
	dEdm = - err*pj*cenerr*(x[k]-mtri[i])/(dtri[i]*dtri[i]);
	dEdd = dEdm*(x[k]-mtri[i])/dtri[i];
	dtri[i] -= mu_d*dEdd;
	mtri[i] -= mu_m*dEdm;
	Vtri[i] += mu_V*err*atri[i]*cenerr/dentri;
	centri[i] += mu_cen*err*pj;
      }
    }
  }
}

float TriangleSAM(in)
float in;
{
  int i;
  float av, num, fabs_delta;

  dentri = 0.0;
  num = 0.0;

  for (i=0;i<NUMPAT;i++)  {
    fabs_delta = fabs(in - mtri[i]);
    if (fabs_delta <= dtri[i])
      atri[i] = 1.0 - fabs_delta/dtri[i];
    else atri[i] = 0.0;
    av = atri[i]*Vtri[i];
    dentri = dentri+av;
    num = num+av*centri[i];
  }
  if (dentri != 0.0)  return num/dentri;
  else                return 0.0;
}

int TriangleSAMinit()
{
  int i;
  float j,STEP,base;
  STEP = (float)(NUMSAM-1)/(float)(NUMPAT-1);
  base = (float)(MAXX-MINX)/(float)(NUMPAT-1);
  i = -1;
  for (j=0.0;j<(float)NUMSAM;j+=STEP) {
    i += 1;
    centri[i] = sample[(int)(j+.5)];
    dtri[i] = base;
    mtri[i] = x[(int)(j+.5)];
    Vtri[i] = 1.0;
  }
}

float TriangleSAMapprox()
{
  int k;
  float err, sumerr=0.0;

  for (k=0;k<NUMDES;k++) {
    Ftri[k] = TriangleSAM(xtest[k]);
    err = des[k]-Ftri[k];
    sumerr += err*err;
  }
  return sumerr/(float)NUMDES;
}

int writepara_triangle(out)
char out[50];
{
  FILE *ofp;
  int i;
  
  if ((ofp = fopen(out, "w")) == NULL)  {
    printf("Could not open output file\n");
    exit(1);
  }
  for (i=0;i<NUMPAT;i++) 
    fprintf(ofp,"%f  %f  %f  %f  %f\n",
	    mtri[i], dtri[i], dtri[i], centri[i], Vtri[i]);
  fclose(ofp);
}
/****************************************************/
/****************************************************/


float SIGN(xx)
float xx;
{
  float ss;
  if (xx > 0.0) ss = 1.0;
  else if (xx < 0.0) ss = -1.0;
  else ss = 0.0;
  return ss;
}
