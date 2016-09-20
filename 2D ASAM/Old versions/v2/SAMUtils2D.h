#ifndef SAMUTILS2D_H
#define SAMUTILS2D_H

// Attempting extension to 2D set functions using just gauss set function

typedef Eigen::MatrixXd matrix;

/*******Utility Functions*********************/
enum Fitfxn {gauss, sinc, cauchy};
void Initialize(const vector<double>& xvals, const vector<double>& yvals, const matrix& fxvals, 
	int _numpat, int _numsamx, int _numsamy, int _numdesx, int _numdesy);
vector<double> eigvec2stdvec(const Eigen::VectorXd  & vec);
matrix siftMatrix(const matrix &m, vector<int> idx, vector<int> idy);
void filefail(string out);
void summarizeInput();
void WriteEpoch(string basename, int epoch);
void WriteParams(string);

/********Variables, Arrays Defaults*********************/
extern int NUMPAT;    /* number of fuzzy sets on input or output side*/
extern int NUMSAMX;    /* use NUMSAMX number of points on training x-grid */
extern int NUMSAMY;    /* use NUMSAMY number of points on training y-grid */
extern int NUMDESX;    /* use NUMSAMX number of points on testing x-grid */
extern int NUMDESY;    /* use NUMSAMY number of points on testing y-grid */
/* 76^2 = 5776 ~ 0.5* 100^2 */
extern double MINX;      /* lower limit for x axis  */
extern double MINY;      /* lower limit for y-axis  */
extern double MAXX;      /* upper limit for x axis  */
extern double MAXY;      /* upper limit for y-axis  */
extern vector<double> x;      /* x values for training */
extern vector<double> y;      /* y values for training */
extern vector<double> xtest;  /* x values for testing */
extern vector<double> ytest;  /* y values for testing */
extern matrix sample; /* f(x) values (little f in paper) for training */
extern matrix des;    /* f(x) values (little f in paper) for testing */
extern double base;	/* generic nonzero param to use in dispersion initial assignments */
extern string name[];

/******** Gaussian SAM's parameters ********/
extern vector<double> cengs;    /* then-part set centroid in Gaussian SAM */
extern vector<double> Vgs;      /* volume (area) of then-part set in Gaussian SAM */
extern vector<double> ags;      /* gaussian set values */

extern matrix mgs;		/* "location" of gaussian if-part set function */
extern matrix dgs;		/* "width" of gaussian if-part set function */
extern matrix Fgss;    /* Gaussian SAM function approximation */

extern double dengs;
extern vector<double> xmdgs;  /* miscel. parameters */
extern vector<double> ymdgs;  /* miscel. parameters */

/********** Sinc SAM's parameters **********/
extern vector<double> censinc;  /* then-part set centroid in sinc SAM */
extern vector<double> Vsinc;    /* volume (area) of then-part set in sinc SAM */
extern vector<double> asinc;    /* sinc set values */

extern matrix msinc;    /* "location" of sinc if-part set function */
extern matrix dsinc;    /* "width" of sinc if-part set function */
extern matrix Fsinc;    /* Sinc SAM function approximation */

extern double densinc;
extern vector<double> xmdsinc; /* miscel. parameters */
extern vector<double> ymdsinc; /* miscel. parameters */

/********  Cauchy SAM's parameters  ********/
extern vector<double> cenchy;   /* then-part set centroid in Cauchy SAM*/
extern vector<double> Vchy;     /* volume (area) of then-part set in Cauchy SAM */
extern vector<double> achy;     /* cauchy set values */

extern matrix mchy;     /* "location" of cauchy if-part set function */
extern matrix dchy;     /* "width" of cauchy if-part set function */
extern matrix Fchy;    /* Cauchy SAM function approximation */

extern double denchy;
extern vector<double> xmdchy; /* miscel. parameters */
extern vector<double> ymdchy; /* miscel. parameters */

/*******************************************/
/****Gaussian ASAM Functions****************/
void GaussianSAMlearn();
double GaussianSAM(double xin, double yin);		/* Gaussian SAM */
void GaussianSAMinit();
double GaussianSAMapprox();

/****Sinc ASAM Functions****************/
void SincSAMinit();
double SincSAM(double xin, double yin);
void SincSAMlearn();
double SincSAMapprox();

/****Cauchy ASAM Functions****************/
void CauchySAMinit();
double CauchySAM(double xin, double yin);
void CauchySAMlearn();
double CauchySAMapprox();

/*******************************************/
/******Aggregate ASAM Calls**********************/
void ASAMsInitialize();
void ASAMsLearn();
vector<double> ASAMsApprox();

//matrix strideMatrix(const matrix &m, float dcol, float drow);

#endif