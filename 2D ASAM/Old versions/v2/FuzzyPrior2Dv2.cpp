#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>

using namespace std;
using namespace Eigen;

#include "SAMUtils2D.h"

int main(int argc, char *argv[]){

	int numRules = 6;
	int epochSize = 400; int adaptIters = epochSize*8;
	matrix fxn;
	VectorXd xin, yin;
	int temp; const int fnums = 3;
	
	string mkcmd = "md ";
	string rmcmd = "rm -r ";
	
	string fnames[3] = {"FxnGen.dat", "OutFxn.dat", "Errors.dat"};
	fstream fxnio[3] = {
		fstream(fnames[0].data(), ios::in),
		fstream(fnames[1].data(), ios::out), 
		fstream(fnames[2].data(), ios::out)
	};
	fxnio[1].precision(9);fxnio[2].precision(9);

	if ( fxnio[0].fail() || fxnio[1].fail() || fxnio[2].fail()){
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
	
	xin = (fxn.row(0)).tail(ncols-1); vector<double> xv = eigvec2stdvec(xin) ;	
	yin = (fxn.col(0)).tail(nrows-1); vector<double> yv = eigvec2stdvec(yin) ;	
	fxn = fxn.bottomRightCorner(fxn.rows()-1, fxn.cols()-1).eval();

	Initialize(xv, yv, fxn, numRules, (int) (0.7*xin.size()), (int) (0.7*yin.size()), xin.size(), yin.size());
	ASAMsInitialize();		
	//(Make) Dirs for each fit fxn.
	// Reset Record by removing dirs.
	fxnio[2] << "Iter# ";
	for(int t = 0; t < 3; t++){
		rmcmd = "rm -r " + name[t];
		mkcmd = "md " + name[t];
		system(rmcmd.data());  //Reset record for new runs.
		system(mkcmd.data()); 
		fxnio[2] << "\t    " << name[t];
	}
	fxnio[2] << endl;

	summarizeInput();
	cout << "Epoch Size: "<< epochSize << endl;

	int k = 0; vector<double> errors;
	vector<double>::const_iterator i;
	double minerr; int loc; bool minQ;
	string bname ("fuzzyF");
	while( k <= adaptIters ) {
		ASAMsLearn();
		if (k% epochSize == 0){
			errors = ASAMsApprox();
			WriteEpoch( bname, k );
			minerr = *(std::min_element(errors.begin(), errors.end()));
			fxnio[2] << k ;
			minQ = false; loc = 0;
			for ( i = errors.begin(); i < errors.end() ; i++ ){ //Log MSEs & Locate Min.
				fxnio[2] << "\t" << *i ;
				if ( (!minQ) && (*i != minerr) ) loc++;
				else minQ = true;
			}
			fxnio[2]<< endl;
			cout << "iter# " << k << ": Min. Error = " << minerr 
				<< " using " << name[loc] << " fit function." << endl;
		}
		k++;
	}

	for(temp = 0; temp < fnums ; temp++) fxnio[temp].close();	
	std::cout << "\nDone. Final min MSE = " << errors[loc] << endl;
	system("PAUSE");
	return EXIT_SUCCESS;
}