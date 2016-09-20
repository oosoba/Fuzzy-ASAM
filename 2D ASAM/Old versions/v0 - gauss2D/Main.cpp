#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
//typedef Eigen::MatrixXd matrix;

using namespace std;
using namespace Eigen;

#include "GaussSAMUtils2D.h"

int main(int argc, char *argv[]){
	int numRules = 15;
	int epochSize = 500; int adaptIters = epochSize*200;
	int temp; const int fnums = 4;
	string mkcmd = "md " +::name;		
	string rmcmd = "rm -r " +::name;
	string fnames[4] = {"FxnGen.dat", "OutFxn.dat", "OutFxn2.dat", "Errors.dat"};
	fstream fxnio[4] = {
		fstream(fnames[0].data(), ios::in),
		fstream(fnames[1].data(), ios::out), 
		fstream(fnames[2].data(), ios::out),
		fstream(fnames[3].data(), ios::out)
	};
	fxnio[1].precision(9);fxnio[2].precision(9);fxnio[3].precision(9);
	MatrixXd fxn;
	VectorXd xin, yin;

	if ( fxnio[0].fail() || fxnio[1].fail() || fxnio[2].fail() || fxnio[3].fail() ){
		cout<<"file i/o error.\n";
		system("PAUSE"); return EXIT_FAILURE; 
	}

	// Determine dynamic matrix size
	int nrows=0, ncols=0; double dtemp=0;
	string line=" ";
	while(getline(fxnio[0], line)) nrows++;
	istringstream buf(line);
	while(buf.good()){ buf >> dtemp; ncols++;} 
	fxn.resize(nrows, ncols);
	fxn.fill(0);
	cout << "\nInput matrix dimensions = " << 
		fxn.rows() << "x" << fxn.cols() << endl;

	// Read in function values
	fxnio[0].clear(); fxnio[0].seekg(0);
	int r=0, c=0; 
	while(getline(fxnio[0], line)){
		buf.clear(); buf.str(line);		
		while(buf.good()){
			buf >> fxn(r,c);
			fxnio[1] << fxn(r,c) << "\t";
			c++;
		}
		fxnio[1] << endl;
		r++; c = 0;
	}	
	
	xin = (fxn.row(0)).tail(nrows-1); vector<double> xv = eigvec2stdvec(xin) ;	
	yin = (fxn.col(0)).tail(nrows-1); vector<double> yv = eigvec2stdvec(yin) ;	
	fxn = fxn.bottomRightCorner(fxn.rows()-1, fxn.cols()-1).eval();
	fxnio[2]<< "Test: \n"  << fxn << endl;

	Initialize(xv, yv, fxn, numRules, (int) (0.76*xin.size()), (int) (0.76*yin.size()), xin.size(), yin.size());
	GaussianSAMinit();		
	system(rmcmd.data());
	system(mkcmd.data());

	summarizeInput();
	std::cout << "Epoch Size: "<< epochSize << endl;

	int k = 0; double error=0;
	string bname ("fuzzyF");
	while( k < adaptIters ) { //Error Criterion or Iteration Limit
		GaussianSAMlearn();
		if (k% epochSize == 0){
			error = GaussianSAMapprox();
			WriteEpoch( bname, k );

			fxnio[3] << k << "\t" << error << endl;
			cout << "iter# " << k << endl;			
		}
		k++;
	}

	for(temp = 0; temp < fnums ; temp++) fxnio[temp].close();	
	std::cout << "\nDone. Final MSE = " << error << endl;
	system("PAUSE");
	return EXIT_SUCCESS;
}