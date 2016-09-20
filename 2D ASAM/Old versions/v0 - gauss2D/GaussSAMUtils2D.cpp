#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>

#include <Eigen/Core>

using namespace std;

#include "GaussSAMUtils2D.h"

/********Variables, Arrays Defaults*********************/
int NUMPAT = 10;    /* number of fuzzy sets on input or output side*/
int NUMSAMX = 76;    /* use NUMSAMX number of points on training x-grid */
int NUMSAMY = 76;    /* use NUMSAMY number of points on training y-grid */
int NUMDESX = 100;    /* use NUMSAMX number of points on testing x-grid */
int NUMDESY = 100;    /* use NUMSAMY number of points on testing y-grid */
/* 76^2 = 5776 ~ 0.5* 100^2 */
static unsigned int adaptCounter = 1;	/*Adaptation Step Counter*/
double MINX;      /* lower limit for x axis  */
double MINY;      /* lower limit for y-axis  */
double MAXX;      /* upper limit for x axis  */
double MAXY;      /* upper limit for y-axis  */
vector<double> x;      /* x values for training */
vector<double> y;      /* y values for training */
vector<double> xtest;  /* x values for testing */
vector<double> ytest;  /* y values for testing */
matrix sample; /* f(x) values (little f in paper) for training */
matrix des;    /* f(x) values (little f in paper) for testing */
string name = "Gauss";
//enum Fitfxn {gauss};

/******** Gaussian SAM's parameters ********/
vector<double> cengs;    /* then-part set centroid in Gaussian SAM */
vector<double> Vgs;      /* volume (area) of then-part set in Gaussian SAM */
vector<double> ags;      /* gaussian set values */

matrix mgs;      /* "location" of gaussian if-part set function */
matrix dgs;      /* "width" of gaussian if-part set function */
matrix Fgss;    /* Gaussian SAM function approximation */

double dengs;
vector<double> xmdgs;  /* miscel. parameters */
vector<double> ymdgs;  /* miscel. parameters */

void Initialize(const vector<double>& xvals, const vector<double>& yvals, const matrix& fxvals, 
	int _numpat, int _numsamx, int _numsamy, int _numdesx, int _numdesy)
{
	if ( (xvals.size() != fxvals.cols()) || (yvals.size() != fxvals.rows()) ){
		std::cerr << "(x,y) <-> f(x,y) Mismatch!!!";
		exit(1);
	}	

	::NUMPAT = _numpat;     /* number of fuzzy sets on input or output side*/
	::NUMSAMX = _numsamx;    /* use NUMSAMX number of points on training x-grid */
	::NUMSAMY = _numsamy;    /* use NUMSAMY number of points on training y-grid */
	::NUMDESX = _numdesx;    /* use NUMDESX number of points on testing x-grid */
	::NUMDESY = _numdesy;    /* use NUMDESY number of points on testing y-grid */

	float stride = min(float(::NUMDESX)/float(::NUMSAMX), float(::NUMDESY)/float(::NUMSAMY));  //float stride => non-uniform sampling
	::xtest = xvals;  /* x values for testing */
	::ytest = yvals;  /* x values for testing */
	::des = fxvals.transpose();    
	/* des is f testing matrix. 
	note tranposition. more transparent indexing (x,y)->(r,c)*/

	/* x values for training: for(unsigned int tx2 = 0; tx2 < xvals.size(); tx2+=2){::x.push_back(xvals.at(tx2));} */
	for(float ix = 0; ix < ::xtest.size(); ix += stride )
		::x.push_back(xvals[int(ix)]);
	/* y values for training: for(vector<double>::iterator iy = ::ytest.begin(); iy < ::ytest.end(); iy+=stride)::y.push_back(*iy); */
	for(float iy = 0; iy < ::ytest.size(); iy+=stride )
		::y.push_back(yvals[int(iy)]);
	/* f(x,y) values for training (little f in paper) */
	sample = strideMatrix(des,stride,stride); //every 2nd element row-wise and column-wise

	::MINX = *(std::min_element(::xtest.begin(), ::xtest.end()));      /* lower limit for x-axis  */
	::MAXX = *(std::max_element(::xtest.begin(), ::xtest.end()));      /* upper limit for x-axis  */
	::MINX = *(std::min_element(::ytest.begin(), ::ytest.end()));      /* lower limit for y-axis  */
	::MAXY = *(std::max_element(::ytest.begin(), ::ytest.end()));      /* upper limit for y-axis  */
}

/****Gaussian SAM ****/
void GaussianSAMinit(){
	double cSTEP = ((double)(::NUMSAMX-1))/((double)(NUMPAT-1));
	double base = max((double)(MAXX-MINX)/(double)(NUMPAT-1), (double)(MAXY-MINY)/(double)(NUMPAT-1));

	adaptCounter = 1;
	int i = 0;

	/******** Gaussian SAM's parameters ********/
	::Vgs.assign(NUMPAT, 1.0);      /* volume (area) of then-part set in Gaussian SAM */
	::cengs.assign(NUMPAT, 0.0);    /* then-part set centroid in Gaussian SAM */
	::ags.assign(NUMPAT, 0.0);		/* gaussian set values */

	::xmdgs.assign(NUMPAT, 0.0);  /* miscel. parameters */
	::ymdgs.assign(NUMPAT, 0.0);  /* miscel. parameters */

	::mgs.resize(NUMPAT,2); //::mgs.fill(0);		/* "location" of gaussian if-part set function */
	mgs.setRandom();	mgs = (mgs.array() + 1)*0.5;
	mgs.col(0) = (mgs.col(0).array())*(::MAXX-::MINX)+::MINX;
	mgs.col(1) = (mgs.col(1).array())*(::MAXY-::MINY)+::MINY;

	::dgs.resize(NUMPAT,2); ::dgs.fill(base/1.52);      /* "width" of gaussian if-part set function */
	::Fgss.resize(NUMDESX, NUMDESY); ::Fgss.fill(0);    /* Gaussian SAM function approximation */
	// (NUMDESX, NUMDESY) being mindful of transposition step in init. #rows = #x, #cols = #y;

	for (double j=0.0; j<(double)::NUMSAMX; j+=cSTEP, i++)
		cengs[i] = sample((int)(j+0.5), (int)(j+0.5));
}

double GaussianSAM(double xin, double yin){
	int i; double av;
	double num=0.0;
	dengs = 0.0;

	for (i=0;i<NUMPAT;i++){			  //foreach fuzzy rule...
		xmdgs[i] = (xin-mgs(i,0))/dgs(i,0);    //calc intermediate centered gaussian xvar
		ymdgs[i] = (yin-mgs(i,1))/dgs(i,1);    //calc intermediate centered gaussian yvar
		ags[i] = exp(-(xmdgs[i]*xmdgs[i]) - (ymdgs[i]*ymdgs[i])); //calc Gaussian fit value
		av = ags[i]*Vgs[i];               //calc fit-scaled volume of then-part
		dengs = dengs+av;				  //calc denominator of SAM
		num = num+av*cengs[i];			  //calc numerator of SAM
	}
	if (dengs != 0.0)  return num/dengs;
	else               return 0.0;
}

void GaussianSAMlearn(){ // New variable notation
	int i,r,c;
	double mu_m, mu_d, mu_cen, mu_V, fuzoutgs;
	double cenerr, a_den, pix;
	double dEdF, dEdFa_den;
	double dEdmx, dEdmy;
	double dEddx, dEddy;

	mu_d   = 1/adaptCounter ;	/* learning rates */
	mu_m   = 1/adaptCounter;	/* modified to an approx harmonic series */
	mu_cen = 1/adaptCounter;	/* sum(mu(t)) ~ 1/t = inf */
	mu_V   = 1/adaptCounter;	/* sum(mu(t)^2) ~ 1/t^2 < inf */

	for (r=0;r<NUMSAMX;r++) {
		for (c=0;c<NUMSAMY;c++) {
			//Work on training set
			fuzoutgs = GaussianSAM(x[r], y[c]);	//calc current F[x] 
			dEdF = -(sample(r,c)-fuzoutgs);	//calc -(f[x]-F[x]); dEdF = -epsilon
			if (dengs != 0.0) {				//dengs: SAM denom. shared from GaussianSAM
				for (i=0;i<NUMPAT;i++) {		//Update params foreach Rule using error info
					a_den = ags[i]/dengs;
					dEdFa_den = dEdF*a_den;		//Not quite dEdc[i]. missing V[i] factor
					pix = a_den*Vgs[i];
					cenerr = cengs[i]-fuzoutgs;

					dEdmx = dEdF*pix*cenerr*xmdgs[i]/dgs(i,0);
					dEdmy = dEdF*pix*cenerr*ymdgs[i]/dgs(i,1);
					dEddx = dEdmx*xmdgs[i];
					dEddy = dEdmy*ymdgs[i];

					/* Done Laws */
					cengs[i] -= mu_cen*dEdFa_den*Vgs[i];	//-=mu_c*dEdc[i]
					Vgs[i] -= mu_V*dEdFa_den*cenerr;
					dgs(i,0) -= mu_d*dEddx; dgs(i,1) -= mu_d*dEddy;
					mgs(i,0) -= mu_m*dEdmx; mgs(i,1) -= mu_m*dEdmy;

					/* Incorrect/Incomplete Laws */
				}
			}
		}
	}
	adaptCounter++;		// Advance adaptation clock
}

double GaussianSAMapprox(){         //returns MSE of current approximation

	for(int r=0; r < Fgss.rows(); r++)
		for (int c=0; c < Fgss.cols(); c++)
			Fgss(r,c) = GaussianSAM(xtest[r], ytest[c]);

	matrix err = des - Fgss;
	double sumerr = err.squaredNorm();
	return sumerr/(double)(NUMDESX*NUMDESY);
}

/**** Utility Functions ****/
vector<double> eigvec2stdvec(const Eigen::VectorXd  & vec){
	vector<double> fin;
	for(int i = 0; i < vec.size(); i++)	fin.push_back(vec(i));
	return fin;
}

matrix strideMatrix(const matrix &m, float drow, float dcol){
	if (drow<=1 && dcol<=1) return m;

	int nr = int(m.rows()/drow), nc = int(m.cols()/dcol);
	matrix fin(nr, nc); fin.fill(0);
	for(float r = 0; r<m.rows(); r+=drow)
		for(float c = 0; c<m.cols(); c+=dcol)
			fin(int(r/drow), int(c/dcol)) = m(int(r),int(c));
	return fin;	
}

void filefail(string out){
	cerr << "Could not open output file: " << out << endl ;
	system("PAUSE");
	exit(1);
}

void summarizeInput(){
	std::cout << endl << "Testing matrix dimensions: ("<< ::des.rows() <<" x " << ::des.cols() << ")" << endl ;
	std::cout << "Training matrix dimensions: ("<< ::sample.rows() <<" x " << ::sample.cols() << ")" << endl ;
	std::cout << "Number of Rules: "<< ::NUMPAT << endl ;
	std::cout << "Range of (x,y): (" 
		<< ::x.front() << ", " << ::x.back() << ") x (" 
		<< ::y.front() << ", " << ::y.back() << ")" << endl;
}

void WriteParams(string out){//enum Fitfxn {gauss};
	ofstream ofp;
	ofp.open(out.data(), ios::out);
	if (ofp.fail()) filefail(out);

	/* m_x  m_y  d_x  d_y  c  V*/
	for(int i=0;i<NUMPAT;i++)
		ofp << mgs(i,0) << "   " << mgs(i,1) << "   " << 
		dgs(i,0) << "   " << dgs(i,1) << "   " << 
		cengs[i]<< "   " << Vgs[i] <<endl;
	ofp.close();
}

void WriteEpoch(string basename, int epoch){
	//enum Fitfxn {gauss};
	ofstream ofp; 
	string out;
	string bnam;
	ostringstream s;
	s << name << "-" << epoch;
	bnam = basename + s.str() + ".dat";

	out = "./" + name + "/" + bnam ;
	WriteParams("./" + name + "/" + "Parameters.par");
	ofp.open(out.data(), ios::out);
	if (ofp.fail()) filefail(out);

	ofp << 0.0 << "\t" ;
	for (int x=0; x<NUMDESX; x++) ofp << ::xtest[x] << "\t";

	for (int y=0; y<NUMDESY; y++){
		ofp << endl << ::ytest[y] << "\t" ;
		for (int x=0; x<NUMDESX; x++)
			ofp << Fgss(x, y) << "\t";
	}//for loop order undoes transposition done in initialization step
	ofp.close();
}