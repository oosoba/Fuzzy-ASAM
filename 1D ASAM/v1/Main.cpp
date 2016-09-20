#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>


using namespace std;

#include "SAMUtils.h"

/*
Ideas: 
	- Compute MAD instead of MSE ???

*/

vector<double> xin, fx;  //should use a map or valarray container instead.
/* To Work in subdirs:*/
string md = "md ";		//make dir on unices or windows+cygwin
string rm = "rm -r ";	//Remove dir on unices or windows+cygwin
string mkcmd;
string rmcmd;
string bname ("fuzzyF");	//basename for all fuzzy approx files


int main(int argc, char *argv[])
{
	int nRules = 12; //5; // # of Rules. Formerly = 12;
	int epochSize = 2000; // Report min error solution after epoch # of steps
	int adaptIters = epochSize*30;
	int defaultPrec = cout.precision();
	string iname, line;
	string filename = "OutFxn.dat";
	string errlog = "Errors.dat";
	//enum Fitfxn {gauss, cauchy, tanhyp, laplace, triangle, sinc};
	double data1=0, data2=0;
	vector<double> errors(6, 100);
	int fxnpts, k;
	if(argc!=1) // Passing execution arguments
		iname = argv[2];	//=> Can also pass another function file by name
	else
		iname = "FxnGen.dat"; //consider passing 3 of rules too...

	ifstream fxnfile(iname.data(), ios::in);
	ofstream errfile(errlog.data(), ios::out);
	ofstream pdfile(filename.data(), ios::out);
	if ( pdfile.fail() || fxnfile.fail() || errfile.fail() ){
		std::cout<<"file i/o error.\n";
		system("PAUSE");
		return EXIT_SUCCESS; 
	}
	cout<<"File Opening done \n";
	//Need to change precision of I/O pipes here...
	pdfile.precision(9);
	while(std::getline(fxnfile, line)){
		istringstream buf(line);
		buf >> data1 >> data2;
		pdfile << data1 << "\t" << data2 << endl; 
		/*need this file to show  precision of approximand
		in c++ compared to precision of fxn source*/
		xin.push_back(data1); fx.push_back(data2);
	}
	
	fxnpts = fx.size();
	InitializeAll(nRules, (int) (0.5*fxnpts), (int) fxnpts);
	InitializeFxn(xin, fx);
	std::cout << endl << "Number of Testing Points: "<< ::des.size() <<"\t" << ::NUMDES;
	std::cout << endl << "Number of Training Samples: "<< ::sample.size() <<"\t" << ::NUMSAM ;
	std::cout << endl << "Number of Rules: "<< ::NUMPAT ;
	std::cout << endl << "Range of x: " << xin.front() << "<-->" << xin.back() << endl;
	std::cout << endl << "Epoch Size: "<< epochSize << endl;

	//(Make) Dirs for each fit fxn.
	// Reset Record by removing dirs.
	errfile << "Iter# ";
	for(int t = 0; t < 6; t++){ // t<5 omits Sinc SAM
		rmcmd = rm + name[t];
		mkcmd = md + name[t];		
		system(rmcmd.data());  //Reset record for new runs.
		system(mkcmd.data()); 
		errfile << "\t    " << name[t];
	}
	errfile << endl; cout << "\n \n" << endl;

	k = 0; 
	vector<double>::const_iterator i;
	double minerr; int loc; bool minQ;
	ASAMsInitialize();
    while( /*vecmin(errors) > 0.0001*/  k < adaptIters ) { //Error Criterion or Iteration Limit
		ASAMsLearn();
		if (k%epochSize == 0){
			errors = ASAMsApprox();
			minerr = vecmin(errors);

			WriteEpoch( bname, k );

			errfile << k;
			minQ = false; loc = 0;
			for ( i = errors.begin(); i < errors.end() ; i++ ){ //Log MSEs & Locate Min.
				errfile << "\t" << *i ;
				if ( (!minQ) && (*i != minerr) ) loc++;
				else minQ = true;
			}
			errfile << endl;
			cout << "iter# " << k << ": Min. Error = " << minerr 
				<< " using " << name[loc] << " fit function." << endl;			
		}
		k++;
	}

	cout << endl;
	fxnfile.close();
	errfile.close();
	pdfile.close();	
	system("PAUSE");
    return EXIT_SUCCESS;
}