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

#include "SAMUtils2D.h"

/********Variables, Arrays Defaults*********************/
int NUMPAT = 10;    /* number of fuzzy sets on input or output side*/
int NUMSAMX = 71;    /* use NUMSAMX number of points on training x-grid */
int NUMSAMY = 71;    /* use NUMSAMY number of points on training y-grid */
int NUMDESX = 100;    /* use NUMSAMX number of points on testing x-grid */
int NUMDESY = 100;    /* use NUMSAMY number of points on testing y-grid */
/* 71^2 = 5041 ~ 0.5* 100^2 */
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
double base;	/* generic nonzero param to use in dispersion initial assignments */
string name[3] = {"Gauss", "Sinc", "Cauchy"};
//enum Fitfxn {gauss, sinc, cauchy};

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

/********** Sinc SAM's parameters **********/
vector<double> censinc;  /* then-part set centroid in sinc SAM */
vector<double> Vsinc;    /* volume (area) of then-part set in sinc SAM */
vector<double> asinc;    /* sinc set values */

matrix msinc;    /* "location" of sinc if-part set function */
matrix dsinc;    /* "width" of sinc if-part set function */
matrix Fsinc;    /* Sinc SAM function approximation */

double densinc;
vector<double> xmdsinc; /* miscel. parameters */
vector<double> ymdsinc; /* miscel. parameters */

/********  Cauchy SAM's parameters  ********/
vector<double> cenchy;   /* then-part set centroid in Cauchy SAM*/
vector<double> Vchy;     /* volume (area) of then-part set in Cauchy SAM */
vector<double> achy;     /* cauchy set values */

matrix mchy;     /* "location" of cauchy if-part set function */
matrix dchy;     /* "width" of cauchy if-part set function */
matrix Fchy;    /* Cauchy SAM function approximation */

double denchy;
vector<double> xmdchy; /* miscel. parameters */
vector<double> ymdchy; /* miscel. parameters */



/**** Utility Functions ****/

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

void WriteParams(string out, int fxn){//enum Fitfxn {gauss, sinc, cauchy};
	int i;
	ofstream ofp(out.data(), ios::out);  
	if (ofp.fail()) filefail(out);

	/* Output format: m_x  m_y  d_x  d_y  c  V*/

	switch( fxn ) {
	case gauss:
		for (i=0;i<NUMPAT;i++)
			ofp << mgs(i,0) << "   " << mgs(i,1) << "   " << 
			dgs(i,0) << "   " << dgs(i,1) << "   " << 
			cengs[i]<< "   " << Vgs[i] <<endl;
		break; 
	case sinc:
		for (i=0;i<NUMPAT;i++)
			ofp << msinc(i,0) << "   " << msinc(i,1) << "   " << 
			dsinc(i,0) << "   " << dsinc(i,1) << "   " << 
			censinc[i]<< "   " << Vsinc[i] <<endl;
		break;
	case cauchy:
		for (i=0;i<NUMPAT;i++)
			ofp << mchy(i,0) << "   " << mchy(i,1) << "   " << 
			dchy(i,0) << "   " << dchy(i,1) << "   " << 
			cenchy[i]<< "   " << Vchy[i] <<endl;
		break;
	}
	ofp.close();
}

void WriteEpoch(string basename, int epoch){//enum Fitfxn {gauss, sinc, cauchy};
	ofstream ofp[3]; 
	string out[3];
	int j = 0;
	string bnam;

	for(j=gauss; j<=cauchy; j++){
		ostringstream s;
		s << name[j] << "-" << epoch;
		bnam = basename + s.str() + ".dat";

		out[j] = "./" + name[j] + "/" + bnam ;
		WriteParams("./" + name[j] + "/" + "Parameters.par", j);
		ofp[j].open(out[j].data(), ios::out);
		if (ofp[j].fail()) filefail(out[j]);

		switch( j ) {
		case gauss:
			ofp[j] << 0.0 << "\t" ;
			for (int x=0; x<NUMDESX; x++) ofp[j] << ::xtest[x] << "\t";
			for (int y=0; y<NUMDESY; y++){
				ofp[j] << endl << ::ytest[y] << "\t" ;
				for (int x=0; x<NUMDESX; x++) ofp[j] << Fgss(x, y) << "\t";
			}
			break;
		case sinc:
			ofp[j] << 0.0 << "\t" ;
			for (int x=0; x<NUMDESX; x++) ofp[j] << ::xtest[x] << "\t";
			for (int y=0; y<NUMDESY; y++){
				ofp[j] << endl << ::ytest[y] << "\t" ;
				for (int x=0; x<NUMDESX; x++) ofp[j] << Fsinc(x, y) << "\t";
			}
			break;
		case cauchy:
			ofp[j] << 0.0 << "\t" ;
			for (int x=0; x<NUMDESX; x++) ofp[j] << ::xtest[x] << "\t";
			for (int y=0; y<NUMDESY; y++){
				ofp[j] << endl << ::ytest[y] << "\t" ;
				for (int x=0; x<NUMDESX; x++) ofp[j] << Fchy(x, y) << "\t";
			}
			break;
		}
		ofp[j].close(); //for loop order undoes transposition done in initialization steps
	}
}

vector<double> eigvec2stdvec(const Eigen::VectorXd  & vec){
	vector<double> fin;
	for(int i = 0; i < vec.size(); i++)	fin.push_back(vec(i));
	return fin;
}

matrix siftMatrix(const matrix &m, vector<int> idx, vector<int> idy){
	if( ( *max_element(idx.begin(), idx.end()) >= m.rows() ) || 
		( *max_element(idy.begin(), idy.end()) >= m.cols() ) || 
		( *min_element(idx.begin(), idx.end()) <0 )  || 
		( *min_element(idy.begin(), idy.end()) <0 ) 
		){
			cerr << "Error sifting matrix values... ";
			system("pause");
			exit(1);
	}

	int r, c, nr = idx.size(), nc = idy.size();
	matrix fin(nr, nc); fin.fill(0);
	for(r = 0; r < nr; r++)
		for(c = 0; c < nc; c++)
			fin(r, c) = m( idx[r], idy[c] );
	return fin;	

}

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

	/* Selection indices: makes sure float issue don't cause (x,y)->sample mismatches */
	vector<int> idx, idy; unsigned int i;
	for(float ix = 0; ix < ::xtest.size(); ix+=stride) idx.push_back(int(ix));
	for(float iy = 0; iy < ::ytest.size(); iy+=stride) idy.push_back(int(iy));

	for(i=0; i < idx.size(); i++) ::x.push_back(xvals[ idx[i] ]);
	for(i=0; i < idy.size(); i++) ::y.push_back(yvals[ idy[i] ]);

	/* f(x,y) values for training (little f in paper) */
	sample = siftMatrix(des, idx, idy); 

	::MINX = *(std::min_element(::xtest.begin(), ::xtest.end()));      /* lower limit for x-axis  */
	::MAXX = *(std::max_element(::xtest.begin(), ::xtest.end()));      /* upper limit for x-axis  */

	::MINY = *(std::min_element(::ytest.begin(), ::ytest.end()));      /* lower limit for y-axis  */
	::MAXY = *(std::max_element(::ytest.begin(), ::ytest.end()));      /* upper limit for y-axis  */
	::base = max((double)(MAXX-MINX)/(double)(NUMPAT-1), (double)(MAXY-MINY)/(double)(NUMPAT-1));
}

/****Gaussian SAM ****/
void GaussianSAMinit(){
	double cSTEP = ((double)(::NUMSAMX-1))/((double)(NUMPAT-1));

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

	::dgs.resize(NUMPAT,2); ::dgs.fill(0.5 /*base/1.52*/);      /* "width" of gaussian if-part set function */
	::Fgss.resize(NUMDESX, NUMDESY); ::Fgss.fill(0);    /* Gaussian SAM function approximation */
	// (NUMDESX, NUMDESY) being mindful of transposition step in init. #rows = #x, #cols = #y;
	
	Eigen::VectorXd rndcen(::NUMPAT);
	rndcen.setRandom(); rndcen = (rndcen.array()+1)*0.5*sample.maxCoeff();
	cengs = eigvec2stdvec(rndcen); //random centroid initialization. assume min(f)==0

	/*double j; int i;
	for(j=0.0, i=0; j<(double)::NUMSAMX, i<::NUMPAT; j+=cSTEP, i++)
		cengs[i] = sample((int)(j+0.5), (int)(j+0.5));*/
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

	mu_d   = 5e-2/adaptCounter;
	mu_m   = 5e-1/adaptCounter;
	mu_cen = 5e-2/adaptCounter;
	mu_V   = 5e-2/adaptCounter;

	for (r=0;r<NUMSAMX;r++) {
		for (c=0;c<NUMSAMY;c++) {
			//Work on training set
			fuzoutgs = GaussianSAM(x[r], y[c]);	//calc current F[x] 
			dEdF = -(sample(r,c)-fuzoutgs);	//calc -(f[x]-F[x]); dEdF = -epsilon
			if (dengs != 0.0) {				//dengs: SAM denom. shared from GaussianSAM
				for (i=0;i<NUMPAT;i++) {
					a_den = ags[i]/dengs;
					dEdFa_den = dEdF*a_den;
					pix = a_den*Vgs[i];
					cenerr = cengs[i]-fuzoutgs;

					dEdmx = dEdF*pix*cenerr*xmdgs[i]/dgs(i,0);
					dEddx = dEdmx*xmdgs[i];

					dEdmy = dEdF*pix*cenerr*ymdgs[i]/dgs(i,1);
					dEddy = dEdmy*ymdgs[i];

					cengs[i] -= mu_cen*dEdFa_den*Vgs[i];
					Vgs[i] -= mu_V*dEdFa_den*cenerr; Vgs[i] = abs(Vgs[i]);
					dgs(i,0) -= mu_d*dEddx; dgs(i,1) -= mu_d*dEddy;
					mgs(i,0) -= mu_m*dEdmx; mgs(i,1) -= mu_m*dEdmy;
				}
			}
		}
	}
}

double GaussianSAMapprox(){         //returns MSE of current approximation
	int r,c;
	for(r=0; r < Fgss.rows(); r++)
		for (c=0; c < Fgss.cols(); c++)
			Fgss(r,c) = GaussianSAM(xtest[r], ytest[c]);
	matrix err = des - Fgss;
	double sumerr = err.lpNorm<1>();
	return sumerr/(double)(NUMDESX*NUMDESY);
}

/****Sinc SAM ****/
void SincSAMinit(){
	double cSTEP = ((double)(::NUMSAMX-1))/((double)(NUMPAT-1));

	/********** Sinc SAM's parameters *********/
	::Vsinc.assign(NUMPAT, 1.0);    /* volume (area) of then-part set in sinc SAM */
	::censinc.assign(NUMPAT, 0.0);  /* then-part set centroid in sinc SAM */
	::asinc.assign(NUMPAT, 0.0);    /* sinc set values */

	::xmdsinc.assign(NUMPAT, 0.0); /* miscel. parameters */
	::ymdsinc.assign(NUMPAT, 0.0); /* miscel. parameters */
	::densinc=0;

	::msinc.resize(NUMPAT, 2);			/* "location" of sinc if-part set function */
	::msinc.setRandom();	msinc = (msinc.array() + 1)*0.5;
	::msinc.col(0) = (msinc.col(0).array())*(::MAXX-::MINX)+::MINX;
	::msinc.col(1) = (msinc.col(1).array())*(::MAXY-::MINY)+::MINY;

	::dsinc.resize(NUMPAT, 2); ::dsinc.fill(1 /*2*base/3.1416*/);    /* "width" of sinc if-part set function */
	::Fsinc.resize(NUMDESX, NUMDESY); ::Fsinc.fill(0);	/* Sinc SAM function approximation */
	// (NUMDESX, NUMDESY) being mindful of transposition step in init. #rows = #x, #cols = #y;
	Eigen::VectorXd rndcen(::NUMPAT);
	rndcen.setRandom(); rndcen = (rndcen.array()+1)*0.5*sample.maxCoeff();
	censinc = eigvec2stdvec(rndcen); //random centroid initialization. assume min(f)==0
	/*double j; int i;
	for (j=0.0, i=0; j<(double)::NUMSAMX, i<::NUMPAT; j+=cSTEP, i++){
		censinc[i] = sample((int)(j), (int)(j));
	}*/
}

double SincSAM(double xin, double yin){
	int i;
	double av=0, num=0, sx, sy;
	densinc = 0;

	for (i=0;i<NUMPAT;i++){
		xmdsinc[i] = (xin - msinc(i,0))/dsinc(i,0);
		ymdsinc[i] = (yin - msinc(i,1))/dsinc(i,1);

		if (xmdsinc[i] == 0.0) sx = 1.0;
		else sx = sin(xmdsinc[i])/xmdsinc[i]; 
		if(ymdsinc[i] == 0.0) sy = 1.0;
		else sy = sin(ymdsinc[i])/ymdsinc[i];

		asinc[i] = sx*sy;
		av = asinc[i]*Vsinc[i];
		densinc = densinc+av;
		num = num+av*censinc[i];
	}
	if (densinc != 0.0)
		return num/densinc;
	else
		return 0.0;
}

void SincSAMlearn(){
	int i,r,c;
	double mu_m, mu_d, mu_cen, mu_V, fuzoutsinc;
	double cenerr, cenerr_den, cenerrV_den, cosxmd, cosymd, acosxmdcenerrV_den, acosymdcenerrV_den;
	double dEddx, dEddy, dEdmx, dEdmy, dEdF, dEdFacosblablax, dEdFacosblablay;
	double xmd, ymd, sx, sy;
	mu_cen = 5e-2/adaptCounter;
	mu_V   = 1e-7;

	/* MAGIC Numbers!!! */
	mu_d   = 9.25e-2/adaptCounter;
	mu_m   = 9e-1/adaptCounter;

	for (r=0; r < NUMSAMX ;r++){
		for (c=0; c < NUMSAMY ; c++){
			fuzoutsinc = SincSAM(x[r], y[c]);
			dEdF = -(sample(r,c) - fuzoutsinc);
			for (i=0;i<NUMPAT;i++)  {
				if (asinc[i] != 0.0)  {
					cenerr = censinc[i]-fuzoutsinc;
					cenerr_den = cenerr/densinc;
					cenerrV_den = cenerr_den*Vsinc[i];

					xmd = (x[r]-msinc(i,0))/dsinc(i,0); ymd = (y[c]-msinc(i,1))/dsinc(i,1);
					sx = sin(xmd)/xmd; sy = sin(ymd)/ymd;
					cosxmd = cos(xmd); cosymd = cos(ymd); 
					acosxmdcenerrV_den = (asinc[i]-sy*cosxmd)*cenerrV_den; //Note add'l sy
					acosymdcenerrV_den = (asinc[i]-sx*cosymd)*cenerrV_den; //Note add'l sx
					dEdFacosblablax = dEdF*acosxmdcenerrV_den;
					dEdFacosblablay = dEdF*acosymdcenerrV_den;
					if (xmdsinc[i] != 0.0)
						dEdmx = dEdFacosblablax/(x[r]-msinc(i,0));
					else dEdmx = 0.0;
					if (ymdsinc[i] != 0.0)
						dEdmy = dEdFacosblablay/(y[c]-msinc(i,1));
					else dEdmy = 0.0;
					dEddx = dEdFacosblablax/dsinc(i,0);
					dEddy = dEdFacosblablay/dsinc(i,1);


					/* Modified Laws */
					censinc[i] -= mu_cen*dEdF*Vsinc[i]*asinc[i]/densinc; 
					Vsinc[i] -= mu_V*dEdF*asinc[i]*cenerr_den; Vsinc[i] = abs(Vsinc[i]);

					msinc(i,0) -= mu_m*dEdmx; dsinc(i,0) -= mu_d*dEddx;
					msinc(i,1) -= mu_m*dEdmy; dsinc(i,1) -= mu_d*dEddy;
				}
			}
		}
	}
}

double SincSAMapprox(){
	int r,c;
	for(r=0; r < Fsinc.rows(); r++)
		for(c=0; c < Fsinc.cols(); c++)
			Fsinc(r,c) = SincSAM(xtest[r], ytest[c]);
	matrix err = des - Fsinc;
	double sumerr = err.lpNorm<1>();
	return sumerr/(double)(NUMDESX*NUMDESY);
}

/****Cauchy SAM ****/
void CauchySAMinit(){
	double cSTEP = ((double)(::NUMSAMX-1))/((double)(NUMPAT-1));

	/********  Cauchy SAM's parameters  ********/
	::Vchy.assign(NUMPAT, 1.0);     /* volume (area) of then-part set in Cauchy SAM */
	::cenchy.assign(NUMPAT, 0.0);   /* then-part set centroid in Cauchy SAM*/
	::achy.assign(NUMPAT, 0.0);     /* cauchy set values */

	::mchy.resize(NUMPAT,2);			/* "location" of cauchy if-part set function */
	::mchy.setRandom();	mchy = (mchy.array() + 1)*0.5;
	::mchy.col(0) = (mchy.col(0).array())*(::MAXX-::MINX)+::MINX;
	::mchy.col(1) = (mchy.col(1).array())*(::MAXY-::MINY)+::MINY;

	::Fchy.resize(NUMDESX, NUMDESY); ::Fchy.fill(0);	/* Cauchy SAM function approximation */
	::dchy.resize(NUMPAT,2); ::dchy.fill(1 /*4*base/3.0*/);       /* "width" of cauchy if-part set function */	

	::xmdchy.assign(NUMPAT, 0.0); /* miscel. parameters */
	::ymdchy.assign(NUMPAT, 0.0); /* miscel. parameters */
		Eigen::VectorXd rndcen(::NUMPAT);
	rndcen.setRandom(); rndcen = (rndcen.array()+1)*0.5*sample.maxCoeff();
	cenchy = eigvec2stdvec(rndcen); //random centroid initialization. assume min(f)==0
	/*double j; int i;
	for (j=0.0, i=0; j<(double)::NUMSAMX, i<::NUMPAT; j+=cSTEP, i++)
		cenchy[i] = sample((int)(j+0.5), (int)(j+0.5));*/
}

double CauchySAM(double xin, double yin){
	int i;	double av,num, cx, cy;
	denchy = 0.0;
	num = 0.0;

	for (i=0;i<NUMPAT;i++){
		xmdchy[i] = (xin - mchy(i,0))/dchy(i,0);
		ymdchy[i] = (yin - mchy(i,1))/dchy(i,1);
		cx = 1.0/(1.0 + xmdchy[i]*xmdchy[i]);
		cy = 1.0/(1.0 + ymdchy[i]*ymdchy[i]);

		achy[i] = cx*cy;
		av = achy[i]*Vchy[i];
		denchy = denchy+av;
		num = num+av*cenchy[i];
	}
	if (denchy != 0.0)  return num/denchy;
	else                return 0.0;
}

void CauchySAMlearn(){
	int i,r,c;
	double mu_m, mu_d, mu_cen, mu_V, fuzoutchy;
	double cenerr, a_den, pix;
	double dEdF, dEdFa_den;
	double dEdmx, dEddx, dEdmy, dEddy, cenerr_dx, cenerr_dy;
	double xmd, ymd, cx, cy;

	mu_d   = 2e-2/adaptCounter; 
	mu_m   = 2e-2/adaptCounter;
	mu_cen = 1e-5;
	mu_V   = 1e-6;

	for (r=0;r<NUMSAMX;r++) {
		for (c=0;c<NUMSAMY;c++) {
			//Work on training set
			fuzoutchy = CauchySAM(x[r], y[c]);
			dEdF = -(sample(r,c)-fuzoutchy);
			if (denchy != 0.0) {
				for (i=0;i<NUMPAT;i++) {
					a_den = achy[i]/denchy;
					pix = a_den*Vchy[i];
					cenerr = cenchy[i]-fuzoutchy;

					xmd = (x[r]-mchy(i,0))/dchy(i,0); ymd = (y[c]-mchy(i,1))/dchy(i,1);
					cx = 1.0/(1.0 + xmd*xmd);
					cy = 1.0/(1.0 + ymd*ymd);

					cenerr_dx = cenerr/dchy(i,0);
					cenerr_dy = cenerr/dchy(i,1);
					dEdmx = dEdF*pix*cenerr_dx*xmd*cx;
					dEdmy = dEdF*pix*cenerr_dy*ymd*cy;
					dEddx = dEdmx*xmd;
					dEddy = dEdmy*ymd;

					dEdFa_den = dEdF*a_den;

					cenchy[i] -= mu_cen*Vchy[i]*dEdFa_den;
					Vchy[i] -= mu_V*dEdFa_den*cenerr;

					dchy(i,0) -= mu_d*dEddx; dchy(i,1) -= mu_d*dEddy;
					mchy(i,0) -= mu_m*dEdmx; mchy(i,1) -= mu_m*dEdmy;
				}
			}
		}
	}
}

double CauchySAMapprox(){
	int r,c;
	for(r=0; r < Fchy.rows(); r++)
		for(c=0; c < Fchy.cols(); c++)
			Fchy(r,c) = ::CauchySAM(xtest[r], ytest[c]);
	matrix err = des - Fchy;
	double sumerr = err.lpNorm<1>();
	return sumerr/(double)(NUMDESX*NUMDESY);
}

/******Aggregate ASAM Calls*************/
void ASAMsInitialize(){
	adaptCounter = 1;

	GaussianSAMinit();
	SincSAMinit();  
	CauchySAMinit(); 
}

void ASAMsLearn(){
	GaussianSAMlearn();
	SincSAMlearn();  
	CauchySAMlearn(); 

	adaptCounter++;
}

vector<double> ASAMsApprox(){// err vector fill order important!
	vector<double> err;
	//enum Fitfxn {gauss, sinc, cauchy};
	err.push_back( GaussianSAMapprox() );
	err.push_back( SincSAMapprox() );
	err.push_back( CauchySAMapprox() ); 

	return err;
}