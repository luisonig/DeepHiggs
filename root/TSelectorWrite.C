#include <fstream>
#include <sstream>
#include <string>
#ifndef NDEBUG
  #include <iostream>
#endif

#include "TSelectorReader.h"
#include "TSelectorWrite.h"


extern "C" void heft_OLP_Start(const char *fname, int *ierr);
extern "C" void heft_OLP_EvalSubProcess(int, double*, double, double*, double*);
extern "C" void heft_OLP_Option(const char * assignment, int* success);
extern "C" void heft_OLP_PrintParameter(const char * filename);

extern "C" void full_OLP_Start(const char *fname, int *ierr);
extern "C" void full_OLP_EvalSubProcess(int, double*, double, double*, double*);
extern "C" void full_OLP_Option(const char * assignment, int* success);
extern "C" void full_OLP_PrintParameter(const char * filename);

//extern "C" void full_OLP_Start_qp(const char *fname, int *ierr);
//extern "C" void full_OLP_EvalSubProcess_qp(int, double*, double, double*, double*);
//extern "C" void full_OLP_Option_qp(const char * assignment, int* success);
//extern "C" void full_OLP_PrintParameter_qp(const char * filename);


// --------------------------------------------------------------------------- //
// Selector
// --------------------------------------------------------------------------- //

TSelectorWrite::TSelectorWrite(const int multiplicity)
  : debug(0), multip(multiplicity)
{
  std::cout<<"*************************************"<<std::endl;
  std::cout<<"*** REWEIGHTING SELECTOR SELECTED ***"<<std::endl;
  std::cout<<"*************************************"<<std::endl;
}

TSelectorWrite::~TSelectorWrite()
{
//
}

int TSelectorWrite::Type()
{
  return 2;
}

void TSelectorWrite::SetFileName(string fname, string extension)
{
  //outfilename=fname.substr(0, fname.size()-5)+suffix+".root";
  outfilename=fname+extension;
  std::cout<<"Reweighted events save in file: ";
  std::cout<<outfilename<<std::endl;
  return;
}

void TSelectorWrite::SetParameters(double nf, double mt, double mb, double mbms)
{
  // Input parameters are VEV, mZ and alpha:
  alpha = 1./128.802223294837; // As computed in Sherpa
  vev   = 246.;
  Nf    = nf; 
  mZ    = 91.1876;
  mT    = mt;
  mB    = mb;
  mBMS  = mbms;

  // Derived parameters:
  e2 = alpha*4.*pi;
  mW = sqrt(mZ*mZ/2.+sqrt(mZ*mZ*mZ*mZ/4.-pi*alpha*mZ*mZ*vev*vev));
  sw = sqrt(1-mW*mW/(mZ*mZ));
  
  if (debug) {
    std::cout<<"--- Parameter values ---"<<std::endl;
    std::cout<<"e2 = "<<e2<<std::endl;
    std::cout<<"mW = "<<mW<<std::endl;
    std::cout<<"sw = "<<sw<<std::endl;
  }
  return;
}

void TSelectorWrite::Init(const TSelectorReader* reader)
{
  input_ed_ntuples = &reader->ed_ntuples;
  input_id = &reader->id;
  input_nparticle = &reader->nparticle;
  input_ncount = &reader->ncount;
  input_px = &reader->px[0];
  input_py = &reader->py[0];
  input_pz = &reader->pz[0];
  input_E  = &reader->E[0];
  input_px_f = &reader->px_f[0];
  input_py_f = &reader->py_f[0];
  input_pz_f = &reader->pz_f[0];
  input_E_f  = &reader->E_f[0];
  input_alphas = &reader->alphas;
  input_kf = &reader->kf[0];
  input_ps_wgt = &reader->ps_wgt;
  input_weight = &reader->weight;
  input_weight2 = &reader->weight2;
  input_me_wgt = &reader->me_wgt;
  input_me_wgt2 = &reader->me_wgt2;
  input_x1 = &reader->x1;
  input_x2 = &reader->x2;
  input_x1p = &reader->x1p;
  input_x2p = &reader->x2p;
  input_id1 = &reader->id1;
  input_id2 = &reader->id2;
  input_fac_scale = &reader->fac_scale;
  input_ren_scale = &reader->ren_scale;
  input_nuwgt = &reader->nuwgt;
  input_usr_wgts = &reader->usr_wgts[0];
  input_alphaspower = &reader->alphaspower;
  input_part = &reader->part[0];

  outputfile  = new TFile(outfilename.c_str(),"recreate");

  n_fChain = new TTree("t3","Reweighted Ntuple");
  n_fChain->Branch("id", &n_id, "id/I");
  n_fChain->Branch("nparticle", &n_nparticle, "nparticle/I");
  n_fChain->Branch("ncount", &n_ncount, "ncount/I");
  n_fChain->Branch("px", n_px, "px[nparticle]/D");
  n_fChain->Branch("py", n_py, "py[nparticle]/D");
  n_fChain->Branch("pz", n_pz, "pz[nparticle]/D");
  n_fChain->Branch("E", n_E, "E[nparticle]/D");
  n_fChain->Branch("alphas", &n_alphas, "alphas/D");
  n_fChain->Branch("kf", n_kf, "kf[nparticle]/I");
  n_fChain->Branch("ps_wgt", &n_ps_wgt, "ps_wgt/D");
  n_fChain->Branch("weight", &n_weight, "weight/D");
  n_fChain->Branch("weight2", &n_weight2, "weight2/D");
  n_fChain->Branch("me_wgt", &n_me_wgt, "me_wtg/D");
  n_fChain->Branch("me_wgt2", &n_me_wgt2, "me_wtg2/D");
  n_fChain->Branch("x1", &n_x1, "x1/D");
  n_fChain->Branch("x2", &n_x2, "x2/D");
  n_fChain->Branch("x1p", &n_x1p, "x1p/D");
  n_fChain->Branch("x2p", &n_x2p, "x2p/D");
  n_fChain->Branch("id1", &n_id1, "id1/I");
  n_fChain->Branch("id2", &n_id2, "id2/I");
  n_fChain->Branch("id1p", &n_id1p, "id1p/I");
  n_fChain->Branch("id2p", &n_id2p, "id2p/I");
  n_fChain->Branch("fac_scale", &n_fac_scale, "fac_scale/D");
  n_fChain->Branch("ren_scale", &n_ren_scale, "ren_scale/D");
  n_fChain->Branch("nuwgt", &n_nuwgt, "nuwgt/I");
  n_fChain->Branch("usr_wgts", &n_usr_wgts, "usr_wgts[nuwgt]/D");
  n_fChain->Branch("alphasPower", &n_alphaspower, "alphasPower/B");
  n_fChain->Branch("part", n_part, "part[2]/C");

  InitializeOLP();

}


void TSelectorWrite::Notify()
{
  return;
}

bool TSelectorWrite::Process()
{

  PrepareEvent();
  Reweight();

  return true;
}

void TSelectorWrite::SlaveBegin()
{
  // pass
}

void TSelectorWrite::SlaveTerminate()
{
  // The SlaveTerminate() function is called after all entries or objects
  // have been processed.

  outputfile->Write();
  outputfile->Close();
  
  FinalizeStat();
  //analysis->analysis_finalize(this);
}


void TSelectorWrite::PrepareEvent()
{
  if (event_prev_id != get_event_id()) {
    event_prev_id = get_event_id();
    new_event = 1;
    event_groups++;
  }
  else {
    new_event = 0;
  }
  
  return;
}


int TSelectorWrite::InitializeOLP()
{
  // Setup varibles:
  const string fname1("OLE_order_h1j.olc");
  const string fname2("OLE_order_h2j.olc");
  const string fname3("OLE_order_h3j.olc");
  int ierr(0);

  // Transmit to GoSam and initialize it:
  stringstream ss;
  ss<<fixed<<setprecision(12)<<mW;
  const string mW_str = "mW=" + ss.str();
  const string mW_str_full = "mdlMW=" + ss.str();

  ss.str(std::string());
  ss<<fixed<<setprecision(12)<<mZ;
  const string mZ_str = "mZ=" + ss.str();
  const string mZ_str_full = "mdlMZ=" + ss.str();

  ss.str(std::string());
  ss<<fixed<<setprecision(12)<<sw;
  const string sw_str = "sw=" + ss.str();
  const string sw_str_full = "mdlsw=" + ss.str();
  
  ss.str(std::string());
  ss<<fixed<<setprecision(12)<<mT;
  const string mT_str = "mT=" + ss.str();
  const string mT_str_full = "mdlMT=" + ss.str();    
  
  ss.str(std::string());
  ss<<fixed<<setprecision(12)<<mB;
  const string mB_str = "mB=" + ss.str();
  const string mB_str_full = "mdlMB=" + ss.str();   
  
  ss.str(std::string());
  ss<<fixed<<setprecision(12)<<mBMS;
  const string mBMS_str = "mBMS=" + ss.str();

  ss.str(std::string());
  ss<<fixed<<setprecision(12)<<Nf;
  const string Nf_str = "Nf=" + ss.str();


  //const string mdlmsd3_str_full = "mdlMsd3=10000.0" ;   
  //const string mdlmsd6_str_full = "mdlMsd6=10000.0" ;
  //const string mdlmsu3_str_full = "mdlMsu3=10000.0" ;     
  //const string mdlmsu6_str_full = "mdlMsu6=10000.0" ;
  const string mdlmh01_str_full = "mdlMH=125.0" ;  
  const string mdlmtp_str_full = "mdlMTP=250.0";
  const string mdlwt_str_full= "mdlWT=0.0";
//   const string mdlmsd3_str_full = "mdlMsd3=171.2" ;
//   const string mdlmsu3_str_full = "mdlMsu3=171.2" ;  
//   const string mdlmsd6_str_full = "mdlMsd6=171.2" ;  
//   const string mdlmsu6_str_full = "mdlMsu6=171.2" ;  
//   const string mdlwsu6_str_full = "mdlWsu6=0.0";
//   const string mdlwsd6_str_full = "mdlWsd6=0.0";
//   const string mdlwsu3_str_full = "mdlWsu3=0.0";
//   const string mdlwsd3_str_full = "mdlWsd3=0.0";
  
  
#ifndef DISABLE_OLP
  heft_OLP_Option(mZ_str.c_str(), &ierr);
  heft_OLP_Option(mW_str.c_str(), &ierr);
  heft_OLP_Option(mT_str.c_str(), &ierr);    
  heft_OLP_Option(mBMS_str.c_str(), &ierr);
  
  //std::cout<<"------------->"<<mB_str<<std::endl;
//   full_OLP_Option(Nf_str.c_str(), &ierr);  
//   full_OLP_Option(mZ_str_full.c_str(), &ierr);
//   full_OLP_Option(mW_str_full.c_str(), &ierr);
   full_OLP_Option(mT_str_full.c_str(), &ierr);    
//   full_OLP_Option(mB_str_full.c_str(), &ierr); 
  
  
  //full_OLP_Option(mBMS_str.c_str(), &ierr);
//   full_OLP_Option(mdlmsd3_str_full.c_str(), &ierr); 
//   full_OLP_Option(mdlmsd6_str_full.c_str(), &ierr); 
//   full_OLP_Option(mdlmsu3_str_full.c_str(), &ierr); 
//   full_OLP_Option(mdlmsu6_str_full.c_str(), &ierr);
//   full_OLP_Option(mdlwsu6_str_full.c_str(), &ierr);
//   full_OLP_Option(mdlwsd6_str_full.c_str(), &ierr);   
//   full_OLP_Option(mdlwsu3_str_full.c_str(), &ierr); 
//   full_OLP_Option(mdlwsd3_str_full.c_str(), &ierr);   
  full_OLP_Option(mdlmh01_str_full.c_str(), &ierr);
  full_OLP_Option(mdlmtp_str_full.c_str(), &ierr); 
  full_OLP_Option(mdlwt_str_full.c_str(), &ierr);

  switch(multip){
  case 1:
    heft_OLP_Start(fname1.c_str(),&ierr);
    full_OLP_Start(fname1.c_str(),&ierr);
    FillSubProcessMap_h1j();
    break;
  case 2:
    heft_OLP_Start(fname2.c_str(),&ierr);
    full_OLP_Start(fname2.c_str(),&ierr);
    FillSubProcessMap_h2j();
    break;
  case 3:
    heft_OLP_Start(fname3.c_str(),&ierr);
    full_OLP_Start(fname3.c_str(),&ierr);
    FillSubProcessMap_h3j();
    break;
  }

  if (debug) {
    heft_OLP_PrintParameter("heft_parameters.dat");
    full_OLP_PrintParameter("full_parameters.dat");
  }

#else  // DISABLE_OLP
  std::cout << "Fake calls to OLP_Option and OLP_Start"<<std::endl;
#endif // DISABLE_OLP

  return ierr;
}

void TSelectorWrite::Reweight()
{
  PseudoJetVector particles;
  double Etot = 0.0;
  double momenta[(get_nparticle()+2)*5];

  for (Int_t j=0; j<get_nparticle(); j++) {
    Etot+=get_E(j);
  }

  fastjet::PseudoJet vec1 = fastjet::PseudoJet(0., 0., get_x1()*Etot/(get_x1()+get_x2()), get_x1()*Etot/(get_x1()+get_x2()));
  vec1.set_user_index(get_id1());
  fastjet::PseudoJet vec2 = fastjet::PseudoJet(0., 0.,-get_x2()*Etot/(get_x1()+get_x2()), get_x2()*Etot/(get_x1()+get_x2()));
  vec2.set_user_index(get_id2());
  particles.push_back(vec1);
  particles.push_back(vec2);

  // Create and fill particle kinematic arrays:
  for (Int_t i=0; i<get_nparticle(); i++){

    fastjet::PseudoJet vec = fastjet::PseudoJet(get_px(i), get_py(i), get_pz(i), get_E(i));
    vec.set_user_index(get_kf(i));
    particles.push_back(vec);
  }

  //std::cout<<"--->multip ="<<multip<<std::endl;
  SubProcessto2 flavlist4;
  SubProcessto3 flavlist5;
  SubProcessto4 flavlist6;
  flavlist4.fill(0);
  flavlist5.fill(0);
  flavlist6.fill(0);

  for (unsigned i=0; i<particles.size(); i++){
    momenta[0+i*5] = particles[i].E();
    momenta[1+i*5] = particles[i].px();
    momenta[2+i*5] = particles[i].py();
    momenta[3+i*5] = particles[i].pz();
    momenta[4+i*5] = particles[i].m();

    switch(multip){
    case 1:
      flavlist4[i] = particles[i].user_index();
      break;
    case 2:
      flavlist5[i] = particles[i].user_index();
      break;
    case 3:
      flavlist6[i] = particles[i].user_index();
      break;
    }
  } 

  std::map<SubProcessto2, int>::iterator it2;
  it2 = h1j_SubProcesses.find(flavlist4);
  std::map<SubProcessto3, int>::iterator it3;
  it3 = h2j_SubProcesses.find(flavlist5);
  std::map<SubProcessto4, int>::iterator it4;
  it4 = h3j_SubProcesses.find(flavlist6);

  switch(multip){
  case 1:
    if ( it2 != h1j_SubProcesses.end()){
      if (debug) {
	PrintEvent(particles);
	std::cout<<"subprocess = "<<h1j_SubProcesses[flavlist4]<<std::endl;
      }
    }
    else {
      std::cerr<<"ERROR SUBPROCESS NOT FOUND!\n---> "
	       <<flavlist4[0]<<" "<<flavlist4[1]<<" -> "
	       <<flavlist4[2]<<" "<<flavlist4[3]<<std::endl;
      return;
    }
    break;
  case 2:
    if ( it3 != h2j_SubProcesses.end()){
      if (debug) {
  	PrintEvent(particles);
  	std::cout<<"subprocess = "<<h2j_SubProcesses[flavlist5]<<std::endl;
      }
    }
    else {
      std::cerr<<"ERROR SUBPROCESS NOT FOUND!\n---> "
  	       <<flavlist5[0]<<" "<<flavlist5[1]<<" -> "
  	       <<flavlist5[2]<<" "<<flavlist5[3]<<" "<<flavlist5[4]<<std::endl;
      return;
    }
    break;
  case 3:
    if ( it4 != h3j_SubProcesses.end()){
      if (debug) {
  	PrintEvent(particles);
  	std::cout<<"subprocess = "<<h3j_SubProcesses[flavlist6]<<std::endl;
      }
    }
    else {
      std::cerr<<"ERROR SUBPROCESS NOT FOUND!\n---> "
  	       <<flavlist6[0]<<" "<<flavlist6[1]<<" -> "
  	       <<flavlist6[2]<<" "<<flavlist6[3]<<" "<<flavlist6[4]<<std::endl;
      return;
    }
    break;
  }

  double heft_wgt = 0.;
  double full_wgt = 0.;
  double pdfs = 0.;
  double params[10] = {};
  double res_heft[4] = {};
  double res_full[4] = {};

  // alphas for OLP:
  params[0]=1.0;

#ifndef DISABLE_OLP
  switch(multip){
  case 1:
    heft_OLP_EvalSubProcess(h1j_SubProcesses[flavlist4], momenta, orig_ren_scale(), params, res_heft);
    full_OLP_EvalSubProcess(h1j_SubProcesses[flavlist4], momenta, orig_ren_scale(), params, res_full);
    //std::cout<<res_full[0]<<" "<<res_full[1]<<" "<<res_full[2]<<std::endl;
    break;
  case 2:
    heft_OLP_EvalSubProcess(h2j_SubProcesses[flavlist5], momenta, orig_ren_scale(), params, res_heft);
    full_OLP_EvalSubProcess(h2j_SubProcesses[flavlist5], momenta, orig_ren_scale(), params, res_full);
    break;
  case 3:
    heft_OLP_EvalSubProcess(h3j_SubProcesses[flavlist6], momenta, orig_ren_scale(), params, res_heft);
    full_OLP_EvalSubProcess(h3j_SubProcesses[flavlist6], momenta, orig_ren_scale(), params, res_full);
    break;
  }

  double gs = sqrt(orig_alphas()*4.0*pi);
  
  // res_heft[0] = double pole;
  // res_heft[1] = single pole;
  // res_heft[2] = finite part;
  // res_heft[3] = born;

  // res_full[0] = double pole;
  // res_full[1] = single pole;
  // res_full[2] = finite part;

  switch(multip){
  case 1:
    double gs_ufo,e_ufo;
    gs_ufo=1.219777963704922;
    e_ufo=0.31345100004952897;      
    // for heft take Born
    heft_wgt = orig_ps_wgt()*res_heft[3]*pow(gs,6)*e2;//*pow(gs_ufo,6)*pow(e_ufo,2);//*pow(gs,6)*e2;
    // for full take finite part of loop induced
    //full_wgt = orig_ps_wgt()*res_full[2]*(2.*pi)*pow(gs_ufo,6)*pow(e_ufo,2)/64.0/pow(pi,4);
    full_wgt = orig_ps_wgt()*res_full[2]*(2.*pi)/64.0/pow(pi,4)/pow(gs_ufo,6)/pow(e_ufo,2)*pow(gs,6)*e2;
    //std::cout<<heft_wgt<<" "<<full_wgt<<std::endl;
    break;
  case 2:
    // for heft take Born
    heft_wgt = orig_ps_wgt()*res_heft[3]*pow(gs,8)*e2;
    // for full take finite part of loop induced
    full_wgt = orig_ps_wgt()*res_full[2]*(2.*pi)*pow(gs,8)*e2/64.0/pow(pi,4);
    break;
  case 3:
    // for heft take Born
    heft_wgt = orig_ps_wgt()*res_heft[3]*pow(gs,10)*e2;
    // for full take finite part of loop induced
    full_wgt = orig_ps_wgt()*res_full[2]*(2.*pi)*pow(gs,10)*e2/64.0/pow(pi,4);
    break;
  }
  
  if(debug){
    std::cout<<"------ CHECK -----"<<std::endl;
    std::cout<<"heft = "<<heft_wgt<<std::endl;
    std::cout<<"orig = "<<orig_me_wgt()<<std::endl;
    std::cout<<"full = "<<full_wgt<<std::endl;
    std::cout<<"psw  = "<<orig_ps_wgt()<<std::endl;
    std::cout<<"as(Q)= "<<orig_alphas()<<std::endl;
    std::cout<<"ratio heft/orig = "<<heft_wgt/orig_me_wgt()<<std::endl;
    std::cout<<"ratio heft/full = "<<heft_wgt/full_wgt<<std::endl;
    std::cout<<"ratio full/orig = "<<full_wgt/orig_me_wgt()<<std::endl;
    std::cout<<"full_me ="<<full_wgt/orig_ps_wgt()<<std::endl;
    std::cout<<"orig_me ="<<orig_me_wgt()/orig_ps_wgt()<<std::endl;
    std::cout<<" "<<std::endl;
  }
  

#else  // DISABLE_OLP
  std::cout << "Fake calls to OLP_EvalSubProcess"<<std::endl;
  heft_wgt = orig_me_wgt();
  full_wgt = orig_me_wgt();
#endif // DISABLE_OLP

  int check = CheckPoint(heft_wgt, full_wgt);
  if (check != 0){
    std::cout<<"Unstable point, set to zero."<<std::endl;
    full_wgt=0.0;
  }

  CopyEvent(0);
  
  pdfs      = orig_weight()/orig_me_wgt();
  n_weight  = full_wgt*pdfs;
  n_weight2 = n_weight;
  n_me_wgt  = full_wgt;
  n_me_wgt2 = n_me_wgt;

  n_fChain->Fill();

  return;
}

int TSelectorWrite::CheckPoint(double heft_wgt, double full_wgt)
{
//  if (abs(heft_wgt/orig_me_wgt()-1.0) > 1e-5){
//    full_wgt = 0.0;
//    std::cout<<"WARNING: Born ratio unstable"<<std::endl;
//    return 1;
//  }

  // if (full_wgt != full_wgt) {
  //   full_wgt = 0.0;
  //   std::cout<<"WARNING: NaN detected"<<std::endl;
  //   return 1;
  // }
  
  // if (abs(full_wgt) == std::numeric_limits<double>::infinity()) {
  //   full_wgt = 0.0;
  //   std::cout<<"WARNING: INF detected"<<std::endl;
  //   return 1;
  // }

  if (not isfinite(full_wgt)) {
    full_wgt = 0.0;
    std::cout<<"WARNING: INF/NaN detected"<<std::endl;
    return 1;
  }
  
  if (abs(full_wgt/heft_wgt) > 100.0) {
    full_wgt = 0.0;
    std::cout<<"WARNING: K-factor check failed"<<std::endl;
    return 1;
  }  
  return 0;
}


void TSelectorWrite::CopyEvent(bool fillchain)
{

  n_id        = get_event_id();
  n_nparticle = get_nparticle();
  n_ncount    = get_ncount();
  
  for (int i=0; i< n_nparticle; i++){
  n_px[i] = get_px(i);
  n_py[i] = get_py(i);
  n_pz[i] = get_pz(i);
  n_E[i]  = get_E(i);
  n_kf[i] = get_kf(i);
}

  n_alphas  = orig_alphas();
  n_ps_wgt  = orig_ps_wgt();
  n_weight  = orig_weight();
  n_weight2 = orig_weight2();
  n_me_wgt  = orig_me_wgt();
  n_me_wgt2 = orig_me_wgt2();

  n_x1  = get_x1();
  n_x2  = get_x2();
  n_x1p = get_x1p();
  n_x2p = get_x2p();
  n_id1 = get_id1();
  n_id2 = get_id2();
  
  n_fac_scale = orig_fac_scale();
  n_ren_scale = orig_ren_scale();
  n_nuwgt = get_nuwgt();

  for (int i=0; i<n_nuwgt; i++){
  n_usr_wgts[i] = orig_usr_wgts(i);  //[nuwgt]
  }
  n_alphaspower = get_alphaspower();

  for (int i=0; i<2; i++){
  n_part[i] = get_part(i);
  }

  if (fillchain) n_fChain->Fill();

  return;
}


void TSelectorWrite::FinalizeStat()
{
  std::cout << std::endl;
  std::cout << "--------------------" 
	    << std::endl;
  std::cout << "Finalize: "
	    << event_groups << " event groups"
	    << std::endl;
  return;
}


void TSelectorWrite::PrintEvent(PseudoJetVector particles)
{
  cout.precision(15);
  cout.setf(ios::scientific, ios::floatfield);

  std::cout<<"--------------------\n";
  std::cout<<"proc = "
	   <<particles[0].user_index()<<" "<<particles[1].user_index()<<" -> ";
  for(unsigned i=2; i<particles.size(); i++){
    std::cout<<particles[i].user_index()<<" ";
  }
  std::cout<<std::endl;
  for(unsigned i=0; i<particles.size(); i++){
    std::cout<<particles[i].E() <<"\t"
	     <<particles[i].px()<<"\t"
	     <<particles[i].py()<<"\t"
	     <<particles[i].pz()<<";\t m="
	     <<particles[i].m()<<std::endl;
  }
}

void TSelectorWrite::TestReweighting()
{

  std::cout<<"==================== REWEIGHTING ====================="<<std::endl;
  PseudoJetVector particles;

  Double_t Etot = 0.0;

  for (Int_t j=0; j<get_nparticle(); j++) {
    Etot+=get_E(j);
  }

  fastjet::PseudoJet vec1 = fastjet::PseudoJet(0., 0., get_x1()*Etot/(get_x1()+get_x2()), get_x1()*Etot/(get_x1()+get_x2()));
  vec1.set_user_index(get_id1());
  fastjet::PseudoJet vec2 = fastjet::PseudoJet(0., 0.,-get_x2()*Etot/(get_x1()+get_x2()), get_x2()*Etot/(get_x1()+get_x2()));
  vec2.set_user_index(get_id2());
  particles.push_back(vec1);
  particles.push_back(vec2);

  // Create and fill particle kinematic arrays:
  for (Int_t i=0; i<get_nparticle(); i++){

    fastjet::PseudoJet vec = fastjet::PseudoJet(get_px(i), get_py(i), get_pz(i), get_E(i));
    vec.set_user_index(get_kf(i));
    particles.push_back(vec);
  }

  PrintEvent(particles);
}

void TSelectorWrite::FillSubProcessMap_h1j()
{
  // h1j_SubProcesses[{ 1, -1, 25, 21}] = 0;
  // h1j_SubProcesses[{ 1, 21, 25,  1}] = 1;
  // h1j_SubProcesses[{-1,  1, 25, 21}] = 2;
  // h1j_SubProcesses[{-1, 21, 25, -1}] = 3;
  // h1j_SubProcesses[{21,  1, 25,  1}] = 4;
  // h1j_SubProcesses[{21, -1, 25, -1}] = 5; 
  // h1j_SubProcesses[{ 4, -4, 25, 21}] = 6;
  // h1j_SubProcesses[{ 4, 21, 25,  4}] = 7;
  // h1j_SubProcesses[{-4,  4, 25, 21}] = 8;
  // h1j_SubProcesses[{-4, 21, 25, -4}] = 9;
  // h1j_SubProcesses[{21,  4, 25,  4}] = 10;
  // h1j_SubProcesses[{21, -4, 25, -4}] = 11;
  // h1j_SubProcesses[{ 2, -2, 25, 21}] = 12;
  // h1j_SubProcesses[{ 2, 21, 25,  2}] = 13;
  // h1j_SubProcesses[{-2,  2, 25, 21}] = 14;
  // h1j_SubProcesses[{-2, 21, 25, -2}] = 15;
  // h1j_SubProcesses[{21,  2, 25,  2}] = 16;
  // h1j_SubProcesses[{21, -2, 25, -2}] = 17;
  // h1j_SubProcesses[{21, 21, 25, 21}] = 18;
  // h1j_SubProcesses[{ 3, -3, 25, 21}] = 19;
  // h1j_SubProcesses[{ 3, 21, 25,  3}] = 20;
  // h1j_SubProcesses[{-3,  3, 25, 21}] = 21;
  // h1j_SubProcesses[{-3, 21, 25, -3}] = 22;
  // h1j_SubProcesses[{21,  3, 25,  3}] = 23;
  // h1j_SubProcesses[{21, -3, 25, -3}] = 24;
  // h1j_SubProcesses[{ 5, -5, 25, 21}] = 0;
  // h1j_SubProcesses[{ 5, 21, 25,  5}] = 1;
  // h1j_SubProcesses[{-5,  5, 25, 21}] = 2;
  // h1j_SubProcesses[{-5, 21, 25, -5}] = 3;
  // h1j_SubProcesses[{21,  5, 25,  5}] = 4;
  // h1j_SubProcesses[{21, -5, 25, -5}] = 5;

  h1j_SubProcesses[{ 1, -1, 25, 21}] = 0;
  h1j_SubProcesses[{ 1, 21, 25,  1}] = 1;
  h1j_SubProcesses[{-1,  1, 25, 21}] = 2;
  h1j_SubProcesses[{-1, 21, 25, -1}] = 3;
  h1j_SubProcesses[{21,  1, 25,  1}] = 4;
  h1j_SubProcesses[{21, -1, 25, -1}] = 5; 
  h1j_SubProcesses[{ 4, -4, 25, 21}] = 0;
  h1j_SubProcesses[{ 4, 21, 25,  4}] = 1;
  h1j_SubProcesses[{-4,  4, 25, 21}] = 2;
  h1j_SubProcesses[{-4, 21, 25, -4}] = 3;
  h1j_SubProcesses[{21,  4, 25,  4}] = 4;
  h1j_SubProcesses[{21, -4, 25, -4}] = 5;
  h1j_SubProcesses[{ 2, -2, 25, 21}] = 0;
  h1j_SubProcesses[{ 2, 21, 25,  2}] = 1;
  h1j_SubProcesses[{-2,  2, 25, 21}] = 2;
  h1j_SubProcesses[{-2, 21, 25, -2}] = 3;
  h1j_SubProcesses[{21,  2, 25,  2}] = 4;
  h1j_SubProcesses[{21, -2, 25, -2}] = 5;
  h1j_SubProcesses[{21, 21, 25, 21}] = 6;
  h1j_SubProcesses[{ 3, -3, 25, 21}] = 0;
  h1j_SubProcesses[{ 3, 21, 25,  3}] = 1;
  h1j_SubProcesses[{-3,  3, 25, 21}] = 2;
  h1j_SubProcesses[{-3, 21, 25, -3}] = 3;
  h1j_SubProcesses[{21,  3, 25,  3}] = 4;
  h1j_SubProcesses[{21, -3, 25, -3}] = 5;
  h1j_SubProcesses[{ 5, -5, 25, 21}] = 0;
  h1j_SubProcesses[{ 5, 21, 25,  5}] = 1;
  h1j_SubProcesses[{-5,  5, 25, 21}] = 2;
  h1j_SubProcesses[{-5, 21, 25, -5}] = 3;
  h1j_SubProcesses[{21,  5, 25,  5}] = 4;
  h1j_SubProcesses[{21, -5, 25, -5}] = 5;

  return;
}

void TSelectorWrite::FillSubProcessMap_h2j()
{
  // h2jsubprocesses[{1, 1, 25, 1, 1}] = 1;
  // h2jsubprocesses[{1, -1, 25, 1, -1}] = 2;
  // h2jsubprocesses[{1, -1, 25, 2, -2}] = 12;
  // h2jsubprocesses[{1, -1, 25, 3, -3}] = 12;
  // h2jsubprocesses[{1, -1, 25, 4, -4}] = 12;
  // h2jsubprocesses[{1, -1, 25, 5, -5}] = 12;
  // h2jsubprocesses[{1, -1, 25, 21, 21}] = 5;
  // h2jsubprocesses[{1, 2, 25, 1, 2}] = 13;
  // h2jsubprocesses[{1, -2, 25, 1, -2}] = 14;
  // h2jsubprocesses[{1, 3, 25, 1, 3}] = 13;
  // h2jsubprocesses[{1, -3, 25, 1, -3}] = 14;
  // h2jsubprocesses[{1, 4, 25, 1, 4}] = 13;
  // h2jsubprocesses[{1, -4, 25, 1, -4}] = 14;
  // h2jsubprocesses[{1, 5, 25, 1, 5}] = 13;
  // h2jsubprocesses[{1, -5, 25, 1, -5}] = 14;
  // h2jsubprocesses[{1, 21, 25, 21, 1}] = 6;
  // h2jsubprocesses[{-1, 1, 25, 1, -1}] = 4;
  // h2jsubprocesses[{-1, 1, 25, 2, -2}] = 15;
  // h2jsubprocesses[{-1, 1, 25, 3, -3}] = 15;
  // h2jsubprocesses[{-1, 1, 25, 4, -4}] = 15;
  // h2jsubprocesses[{-1, 1, 25, 5, -5}] = 15;
  // h2jsubprocesses[{-1, 1, 25, 21, 21}] = 7;
  // h2jsubprocesses[{-1, -1, 25, -1, -1}] = 3;
  // h2jsubprocesses[{-1, 2, 25, 2, -1}] = 16;
  // h2jsubprocesses[{-1, -2, 25, -1, -2}] = 17;
  // h2jsubprocesses[{-1, 3, 25, 3, -1}] = 16;
  // h2jsubprocesses[{-1, -3, 25, -1, -3}] = 17;
  // h2jsubprocesses[{-1, 4, 25, 4, -1}] = 16;
  // h2jsubprocesses[{-1, -4, 25, -1, -4}] = 17;
  // h2jsubprocesses[{-1, 5, 25, 5, -1}] = 16;
  // h2jsubprocesses[{-1, -5, 25, -1, -5}] = 17;
  // h2jsubprocesses[{-1, 21, 25, 21, -1}] = 8;
  // h2jsubprocesses[{2,1, 25, 1, 2}] = 18;
  // h2jsubprocesses[{2, -1, 25, 2, -1}] = 14;
  // h2jsubprocesses[{2, 2, 25, 2, 2}] = 1;
  // h2jsubprocesses[{2, -2, 25, 1, -1}] = 12;
  // h2jsubprocesses[{2, -2, 25, 2, -2}] = 2;
  // h2jsubprocesses[{2, -2, 25, 3, -3}] = 12;
  // h2jsubprocesses[{2, -2, 25, 4, -4}] = 12;
  // h2jsubprocesses[{2, -2, 25, 5, -5}] = 12;
  // h2jsubprocesses[{2, -2, 25, 21, 21}] = 5;
  // h2jsubprocesses[{2, 3, 25, 3, 2}] = 18;
  // h2jsubprocesses[{2, -3, 25, 2, -3}] = 14;
  // h2jsubprocesses[{2, 4, 25, 2, 4}] = 13;
  // h2jsubprocesses[{2, -4, 25, 2, -4}] = 14;
  // h2jsubprocesses[{2, 5, 25, 5, 2}] = 18;
  // h2jsubprocesses[{2, -5, 25, 2, -5}] = 14;
  // h2jsubprocesses[{2, 21, 25, 21, 2}] = 6;
  // h2jsubprocesses[{-2, 1, 25, 1, -2}] = 16;
  // h2jsubprocesses[{-2, -1, 25, -1, -2}] = 19;
  // h2jsubprocesses[{-2, 2, 25, 1, -1}] = 15;
  // h2jsubprocesses[{-2, 2, 25, 2, -2}] = 4;
  // h2jsubprocesses[{-2, 2, 25, 3, -3}] = 15;
  // h2jsubprocesses[{-2, 2, 25, 4, -4}] = 15;
  // h2jsubprocesses[{-2, 2, 25, 5, -5}] = 15;
  // h2jsubprocesses[{-2, 2, 25, 21, 21}] = 7;
  // h2jsubprocesses[{-2, -2, 25, -2, -2}] = 3;
  // h2jsubprocesses[{-2, 3, 25, 3, -2}] = 16;
  // h2jsubprocesses[{-2, -3, 25, -3, -2}] = 19;
  // h2jsubprocesses[{-2, 4, 25, 4, -2}] = 16;
  // h2jsubprocesses[{-2, -4, 25, -2, -4}] = 17;
  // h2jsubprocesses[{-2, 5, 25, 5, -2}] = 16;
  // h2jsubprocesses[{-2, -5, 25, -5, -2}] = 19;
  // h2jsubprocesses[{-2, 21, 25, 21, -2}] = 8;
  // h2jsubprocesses[{3, 1, 25, 1, 3}] = 18;
  // h2jsubprocesses[{3, -1, 25, 3, -1}] = 14;
  // h2jsubprocesses[{3, 2, 25, 3, 2}] = 13;
  // h2jsubprocesses[{3, -2, 25, 3, -2}] = 14;
  // h2jsubprocesses[{3, 3, 25, 3, 3}] = 1;
  // h2jsubprocesses[{3, -3, 25, 1, -1}] = 12;
  // h2jsubprocesses[{3, -3, 25, 2, -2}] = 12;
  // h2jsubprocesses[{3, -3, 25, 3, -3}] = 2;
  // h2jsubprocesses[{3, -3, 25, 4, -4}] = 12;
  // h2jsubprocesses[{3, -3, 25, 5, -5}] = 12;
  // h2jsubprocesses[{3, -3, 25, 21, 21}] = 5;
  // h2jsubprocesses[{3, 4, 25, 3, 4}] = 13;
  // h2jsubprocesses[{3, -4, 25, 3, -4}] = 14;
  // h2jsubprocesses[{3, 5, 25, 3, 5}] = 13;
  // h2jsubprocesses[{3, -5, 25, 3, -5}] = 14;
  // h2jsubprocesses[{3, 21, 25, 21, 3}] = 6;
  // h2jsubprocesses[{-3, 1, 25, 1, -3}] = 16;
  // h2jsubprocesses[{-3, -1, 25, -1, -3}] = 19;
  // h2jsubprocesses[{-3, 2, 25, 2, -3}] = 16;
  // h2jsubprocesses[{-3, -2, 25, -3, -2}] = 17;
  // h2jsubprocesses[{-3, 3, 25, 1, -1}] = 15;
  // h2jsubprocesses[{-3, 3, 25, 2, -2}] = 15;
  // h2jsubprocesses[{-3, 3, 25, 3, -3}] = 4;
  // h2jsubprocesses[{-3, 3, 25, 4, -4}] = 15;
  // h2jsubprocesses[{-3, 3, 25, 5, -5}] = 15;
  // h2jsubprocesses[{-3, 3, 25, 21, 21}] = 7;
  // h2jsubprocesses[{-3, -3, 25, -3, -3}] = 3;
  // h2jsubprocesses[{-3, 4, 25, 4, -3}] = 16;
  // h2jsubprocesses[{-3, -4, 25, -3, -4}] = 17;
  // h2jsubprocesses[{-3, 5, 25, 5, -3}] = 16;
  // h2jsubprocesses[{-3, -5, 25, -3, -5}] = 17;
  // h2jsubprocesses[{-3, 21, 25, 21, -3}] = 8;
  // h2jsubprocesses[{4, 1, 25, 1, 4}] = 18;
  // h2jsubprocesses[{4, -1, 25, 4, -1}] = 14;
  // h2jsubprocesses[{4, 2, 25, 2, 4}] = 18;
  // h2jsubprocesses[{4, -2, 25, 4, -2}] = 14;
  // h2jsubprocesses[{4, 3, 25, 3, 4}] = 18;
  // h2jsubprocesses[{4, -3, 25, 4, -3}] = 14;
  // h2jsubprocesses[{4, 4, 25, 4, 4}] = 1;
  // h2jsubprocesses[{4, -4, 25, 1, -1}] = 12;
  // h2jsubprocesses[{4, -4, 25, 2, -2}] = 12;
  // h2jsubprocesses[{4, -4, 25, 3, -3}] = 12;
  // h2jsubprocesses[{4, -4, 25, 4, -4}] = 2;
  // h2jsubprocesses[{4, -4, 25, 5, -5}] = 12;
  // h2jsubprocesses[{4, -4, 25, 21, 21}] = 5;
  // h2jsubprocesses[{4, 5, 25, 5, 4}] = 18;
  // h2jsubprocesses[{4, -5, 25, 4, -5}] = 14;
  // h2jsubprocesses[{4, 21, 25, 21, 4}] = 6;
  // h2jsubprocesses[{-4, 1, 25, 1, -4}] = 16;
  // h2jsubprocesses[{-4, -1, 25, -1, -4}] = 19;
  // h2jsubprocesses[{-4, 2, 25, 2, -4}] = 16;
  // h2jsubprocesses[{-4, -2, 25, -2, -4}] = 19;
  // h2jsubprocesses[{-4, 3, 25, 3, -4}] = 16;
  // h2jsubprocesses[{-4, -3, 25, -3, -4}] = 19;
  // h2jsubprocesses[{-4, 4, 25, 1, -1}] = 15;
  // h2jsubprocesses[{-4, 4, 25, 2, -2}] = 15;
  // h2jsubprocesses[{-4, 4, 25, 3, -3}] = 15;
  // h2jsubprocesses[{-4, 4, 25, 4, -4}] = 4;
  // h2jsubprocesses[{-4, 4, 25, 5, -5}] = 15;
  // h2jsubprocesses[{-4, 4, 25, 21, 21}] = 7;
  // h2jsubprocesses[{-4, -4, 25, -4, -4}] = 3;
  // h2jsubprocesses[{-4, 5, 25, 5, -4}] = 16;
  // h2jsubprocesses[{-4, -5, 25, -5, -4}] = 19;
  // h2jsubprocesses[{-4, 21, 25, 21, -4}] = 8;
  // h2jsubprocesses[{5, 1, 25, 1, 5}] = 18;
  // h2jsubprocesses[{5, -1, 25, 5, -1}] = 14;
  // h2jsubprocesses[{5, 2, 25, 5, 2}] = 13;
  // h2jsubprocesses[{5, -2, 25, 5, -2}] = 14;
  // h2jsubprocesses[{5, 3, 25, 3, 5}] = 18;
  // h2jsubprocesses[{5, -3, 25, 5, -3}] = 14;
  // h2jsubprocesses[{5, 4, 25, 5, 4}] = 13;
  // h2jsubprocesses[{5, -4, 25, 5, -4}] = 14;
  // h2jsubprocesses[{5, 5, 25, 5, 5}] = 1;
  // h2jsubprocesses[{5, -5, 25, 1, -1}] = 12;
  // h2jsubprocesses[{5, -5, 25, 2, -2}] = 12;
  // h2jsubprocesses[{5, -5, 25, 3, -3}] = 12;
  // h2jsubprocesses[{5, -5, 25, 4, -4}] = 12;
  // h2jsubprocesses[{5, -5, 25, 5, -5}] = 2;
  // h2jsubprocesses[{5, -5, 25, 21, 21}] = 5;
  // h2jsubprocesses[{5, 21, 25, 21, 5}] = 6;
  // h2jsubprocesses[{-5, 1, 25, 1, -5}] = 16;
  // h2jsubprocesses[{-5, -1, 25, -1, -5}] = 19;
  // h2jsubprocesses[{-5, 2, 25, 2, -5}] = 16;
  // h2jsubprocesses[{-5, -2, 25, -5, -2}] = 17;
  // h2jsubprocesses[{-5, 3, 25, 3, -5}] = 16;
  // h2jsubprocesses[{-5, -3, 25, -3, -5}] = 19;
  // h2jsubprocesses[{-5, 4, 25, 4, -5}] = 16;
  // h2jsubprocesses[{-5, -4, 25, -5, -4}] = 17;
  // h2jsubprocesses[{-5, 5, 25, 1, -1}] = 15;
  // h2jsubprocesses[{-5, 5, 25, 2, -2}] = 15;
  // h2jsubprocesses[{-5, 5, 25, 3, -3}] = 15;
  // h2jsubprocesses[{-5, 5, 25, 4, -4}] = 15;
  // h2jsubprocesses[{-5, 5, 25, 5, -5}] = 4;
  // h2jsubprocesses[{-5, 5, 25, 21, 21}] = 7;
  // h2jsubprocesses[{-5, -5, 25, -5, -5}] = 3;
  // h2jsubprocesses[{-5, 21, 25, 21, -5}] = 8;
  // h2jsubprocesses[{21, 1, 25, 21, 1}] = 9;
  // h2jsubprocesses[{21, -1, 25, 21, -1}] = 10;
  // h2jsubprocesses[{21, 2, 25, 21, 2}] = 9;
  // h2jsubprocesses[{21, -2, 25, 21, -2}] = 10;
  // h2jsubprocesses[{21, 3, 25, 21, 3}] = 9;
  // h2jsubprocesses[{21, -3, 25, 21, -3}] = 10;
  // h2jsubprocesses[{21, 4, 25, 21, 4}] = 9;
  // h2jsubprocesses[{21, -4, 25, 21, -4}] = 10;
  // h2jsubprocesses[{21, 5, 25, 21, 5}] = 9;
  // h2jsubprocesses[{21, -5, 25, 21, -5}] = 10;
  // h2jsubprocesses[{21, 21, 25, 1, -1}] = 11;
  // h2jsubprocesses[{21, 21, 25, 2, -2}] = 11;
  // h2jsubprocesses[{21, 21, 25, 3, -3}] = 11;
  // h2jsubprocesses[{21, 21, 25, 4, -4}] = 11;
  // h2jsubprocesses[{21, 21, 25, 5, -5}] = 11;
  // h2jsubprocesses[{21, 21, 25, 21, 21}] = 0;

  h2j_SubProcesses[{ 1,  1, 25,  1,  1}] = 1;
  h2j_SubProcesses[{ 1, -1, 25,  1, -1}] = 2;
  h2j_SubProcesses[{ 1, -1, 25,  2, -2}] = 12;
  h2j_SubProcesses[{ 1, -1, 25,  3, -3}] = 12;
  h2j_SubProcesses[{ 1, -1, 25,  4, -4}] = 12;
  h2j_SubProcesses[{ 1, -1, 25,  5, -5}] = 12;
  h2j_SubProcesses[{ 1, -1, 25, 21, 21}] = 5;
  h2j_SubProcesses[{ 1,  2, 25,  1,  2}] = 13;
  h2j_SubProcesses[{ 1, -2, 25,  1, -2}] = 14;
  h2j_SubProcesses[{ 1,  3, 25,  1,  3}] = 13;
  h2j_SubProcesses[{ 1, -3, 25,  1, -3}] = 14;
  h2j_SubProcesses[{ 1,  4, 25,  1,  4}] = 13;
  h2j_SubProcesses[{ 1, -4, 25,  1, -4}] = 14;
  h2j_SubProcesses[{ 1,  5, 25,  1,  5}] = 13;
  h2j_SubProcesses[{ 1, -5, 25,  1, -5}] = 14;
  h2j_SubProcesses[{ 1, 21, 25, 21,  1}] = 6;
  h2j_SubProcesses[{-1,  1, 25,  1, -1}] = 4;
  h2j_SubProcesses[{-1,  1, 25,  2, -2}] = 15;
  h2j_SubProcesses[{-1,  1, 25,  3, -3}] = 15;
  h2j_SubProcesses[{-1,  1, 25,  4, -4}] = 15;
  h2j_SubProcesses[{-1,  1, 25,  5, -5}] = 15;
  h2j_SubProcesses[{-1,  1, 25, 21, 21}] = 7;
  h2j_SubProcesses[{-1, -1, 25, -1, -1}] = 3;
  h2j_SubProcesses[{-1,  2, 25,  2, -1}] = 16;
  h2j_SubProcesses[{-1, -2, 25, -1, -2}] = 17;
  h2j_SubProcesses[{-1,  3, 25,  3, -1}] = 16;
  h2j_SubProcesses[{-1, -3, 25, -1, -3}] = 17;
  h2j_SubProcesses[{-1,  4, 25,  4, -1}] = 16;
  h2j_SubProcesses[{-1, -4, 25, -1, -4}] = 17;
  h2j_SubProcesses[{-1,  5, 25,  5, -1}] = 16;
  h2j_SubProcesses[{-1, -5, 25, -1, -5}] = 17;
  h2j_SubProcesses[{-1, 21, 25, 21, -1}] = 8;
  h2j_SubProcesses[{ 2,  1, 25,  1,  2}] = 18;
  h2j_SubProcesses[{ 2, -1, 25,  2, -1}] = 14;
  h2j_SubProcesses[{ 2,  2, 25,  2,  2}] = 1;
  h2j_SubProcesses[{ 2, -2, 25,  1, -1}] = 12;
  h2j_SubProcesses[{ 2, -2, 25,  2, -2}] = 2;
  h2j_SubProcesses[{ 2, -2, 25,  3, -3}] = 12;
  h2j_SubProcesses[{ 2, -2, 25,  4, -4}] = 12;
  h2j_SubProcesses[{ 2, -2, 25,  5, -5}] = 12;
  h2j_SubProcesses[{ 2, -2, 25, 21, 21}] = 5;
  h2j_SubProcesses[{ 2,  3, 25,  3,  2}] = 18;
  h2j_SubProcesses[{ 2, -3, 25,  2, -3}] = 14;
  h2j_SubProcesses[{ 2,  4, 25,  2,  4}] = 13;
  h2j_SubProcesses[{ 2, -4, 25,  2, -4}] = 14;
  h2j_SubProcesses[{ 2,  5, 25,  5,  2}] = 18;
  h2j_SubProcesses[{ 2, -5, 25,  2, -5}] = 14;
  h2j_SubProcesses[{ 2, 21, 25, 21,  2}] = 6;
  h2j_SubProcesses[{-2,  1, 25,  1, -2}] = 16;
  h2j_SubProcesses[{-2, -1, 25, -1, -2}] = 19;
  h2j_SubProcesses[{-2,  2, 25,  1, -1}] = 15;
  h2j_SubProcesses[{-2,  2, 25,  2, -2}] = 4;
  h2j_SubProcesses[{-2,  2, 25,  3, -3}] = 15;
  h2j_SubProcesses[{-2,  2, 25,  4, -4}] = 15;
  h2j_SubProcesses[{-2,  2, 25,  5, -5}] = 15;
  h2j_SubProcesses[{-2,  2, 25, 21, 21}] = 7;
  h2j_SubProcesses[{-2, -2, 25, -2, -2}] = 3;
  h2j_SubProcesses[{-2,  3, 25,  3, -2}] = 16;
  h2j_SubProcesses[{-2, -3, 25, -3, -2}] = 19;
  h2j_SubProcesses[{-2,  4, 25,  4, -2}] = 16;
  h2j_SubProcesses[{-2, -4, 25, -2, -4}] = 17;
  h2j_SubProcesses[{-2,  5, 25,  5, -2}] = 16;
  h2j_SubProcesses[{-2, -5, 25, -5, -2}] = 19;
  h2j_SubProcesses[{-2, 21, 25, 21, -2}] = 8;
  h2j_SubProcesses[{ 3,  1, 25,  1,  3}] = 18;
  h2j_SubProcesses[{ 3, -1, 25,  3, -1}] = 14;
  h2j_SubProcesses[{ 3,  2, 25,  3,  2}] = 13;
  h2j_SubProcesses[{ 3, -2, 25,  3, -2}] = 14;
  h2j_SubProcesses[{ 3,  3, 25,  3,  3}] = 1;
  h2j_SubProcesses[{ 3, -3, 25,  1, -1}] = 12;
  h2j_SubProcesses[{ 3, -3, 25,  2, -2}] = 12;
  h2j_SubProcesses[{ 3, -3, 25,  3, -3}] = 2;
  h2j_SubProcesses[{ 3, -3, 25,  4, -4}] = 12;
  h2j_SubProcesses[{ 3, -3, 25,  5, -5}] = 12;
  h2j_SubProcesses[{ 3, -3, 25, 21, 21}] = 5;
  h2j_SubProcesses[{ 3,  4, 25,  3,  4}] = 13;
  h2j_SubProcesses[{ 3, -4, 25,  3, -4}] = 14;
  h2j_SubProcesses[{ 3,  5, 25,  3,  5}] = 13;
  h2j_SubProcesses[{ 3, -5, 25,  3, -5}] = 14;
  h2j_SubProcesses[{ 3, 21, 25, 21,  3}] = 6;
  h2j_SubProcesses[{-3,  1, 25,  1, -3}] = 16;
  h2j_SubProcesses[{-3, -1, 25, -1, -3}] = 19;
  h2j_SubProcesses[{-3,  2, 25,  2, -3}] = 16;
  h2j_SubProcesses[{-3, -2, 25, -3, -2}] = 17;
  h2j_SubProcesses[{-3,  3, 25,  1, -1}] = 15;
  h2j_SubProcesses[{-3,  3, 25,  2, -2}] = 15;
  h2j_SubProcesses[{-3,  3, 25,  3, -3}] = 4;
  h2j_SubProcesses[{-3,  3, 25,  4, -4}] = 15;
  h2j_SubProcesses[{-3,  3, 25,  5, -5}] = 15;
  h2j_SubProcesses[{-3,  3, 25, 21, 21}] = 7;
  h2j_SubProcesses[{-3, -3, 25, -3, -3}] = 3;
  h2j_SubProcesses[{-3,  4, 25,  4, -3}] = 16;
  h2j_SubProcesses[{-3, -4, 25, -3, -4}] = 17;
  h2j_SubProcesses[{-3,  5, 25,  5, -3}] = 16;
  h2j_SubProcesses[{-3, -5, 25, -3, -5}] = 17;
  h2j_SubProcesses[{-3, 21, 25, 21, -3}] = 8;
  h2j_SubProcesses[{ 4,  1, 25,  1,  4}] = 18;
  h2j_SubProcesses[{ 4, -1, 25,  4, -1}] = 14;
  h2j_SubProcesses[{ 4,  2, 25,  2,  4}] = 18;
  h2j_SubProcesses[{ 4, -2, 25,  4, -2}] = 14;
  h2j_SubProcesses[{ 4,  3, 25,  3,  4}] = 18;
  h2j_SubProcesses[{ 4, -3, 25,  4, -3}] = 14;
  h2j_SubProcesses[{ 4,  4, 25,  4,  4}] = 1;
  h2j_SubProcesses[{ 4, -4, 25,  1, -1}] = 12;
  h2j_SubProcesses[{ 4, -4, 25,  2, -2}] = 12;
  h2j_SubProcesses[{ 4, -4, 25,  3, -3}] = 12;
  h2j_SubProcesses[{ 4, -4, 25,  4, -4}] = 2;
  h2j_SubProcesses[{ 4, -4, 25,  5, -5}] = 12;
  h2j_SubProcesses[{ 4, -4, 25, 21, 21}] = 5;
  h2j_SubProcesses[{ 4,  5, 25,  5,  4}] = 18;
  h2j_SubProcesses[{ 4, -5, 25,  4, -5}] = 14;
  h2j_SubProcesses[{ 4, 21, 25, 21,  4}] = 6;
  h2j_SubProcesses[{-4,  1, 25,  1, -4}] = 16;
  h2j_SubProcesses[{-4, -1, 25, -1, -4}] = 19;
  h2j_SubProcesses[{-4,  2, 25,  2, -4}] = 16;
  h2j_SubProcesses[{-4, -2, 25, -2, -4}] = 19;
  h2j_SubProcesses[{-4,  3, 25,  3, -4}] = 16;
  h2j_SubProcesses[{-4, -3, 25, -3, -4}] = 19;
  h2j_SubProcesses[{-4,  4, 25,  1, -1}] = 15;
  h2j_SubProcesses[{-4,  4, 25,  2, -2}] = 15;
  h2j_SubProcesses[{-4,  4, 25,  3, -3}] = 15;
  h2j_SubProcesses[{-4,  4, 25,  4, -4}] = 4;
  h2j_SubProcesses[{-4,  4, 25,  5, -5}] = 15;
  h2j_SubProcesses[{-4,  4, 25, 21, 21}] = 7;
  h2j_SubProcesses[{-4, -4, 25, -4, -4}] = 3;
  h2j_SubProcesses[{-4,  5, 25,  5, -4}] = 16;
  h2j_SubProcesses[{-4, -5, 25, -5, -4}] = 19;
  h2j_SubProcesses[{-4, 21, 25, 21, -4}] = 8;
  h2j_SubProcesses[{ 5,  1, 25,  1,  5}] = 18;
  h2j_SubProcesses[{ 5, -1, 25,  5, -1}] = 14;
  h2j_SubProcesses[{ 5,  2, 25,  5,  2}] = 13;
  h2j_SubProcesses[{ 5, -2, 25,  5, -2}] = 14;
  h2j_SubProcesses[{ 5,  3, 25,  3,  5}] = 18;
  h2j_SubProcesses[{ 5, -3, 25,  5, -3}] = 14;
  h2j_SubProcesses[{ 5,  4, 25,  5,  4}] = 13;
  h2j_SubProcesses[{ 5, -4, 25,  5, -4}] = 14;
  h2j_SubProcesses[{ 5,  5, 25,  5,  5}] = 1;
  h2j_SubProcesses[{ 5, -5, 25,  1, -1}] = 12;
  h2j_SubProcesses[{ 5, -5, 25,  2, -2}] = 12;
  h2j_SubProcesses[{ 5, -5, 25,  3, -3}] = 12;
  h2j_SubProcesses[{ 5, -5, 25,  4, -4}] = 12;
  h2j_SubProcesses[{ 5, -5, 25,  5, -5}] = 2;
  h2j_SubProcesses[{ 5, -5, 25, 21, 21}] = 5;
  h2j_SubProcesses[{ 5, 21, 25, 21,  5}] = 6;
  h2j_SubProcesses[{-5,  1, 25,  1, -5}] = 16;
  h2j_SubProcesses[{-5, -1, 25, -1, -5}] = 19;
  h2j_SubProcesses[{-5,  2, 25,  2, -5}] = 16;
  h2j_SubProcesses[{-5, -2, 25, -5, -2}] = 17;
  h2j_SubProcesses[{-5,  3, 25,  3, -5}] = 16;
  h2j_SubProcesses[{-5, -3, 25, -3, -5}] = 19;
  h2j_SubProcesses[{-5,  4, 25,  4, -5}] = 16;
  h2j_SubProcesses[{-5, -4, 25, -5, -4}] = 17;
  h2j_SubProcesses[{-5,  5, 25,  1, -1}] = 15;
  h2j_SubProcesses[{-5,  5, 25,  2, -2}] = 15;
  h2j_SubProcesses[{-5,  5, 25,  3, -3}] = 15;
  h2j_SubProcesses[{-5,  5, 25,  4, -4}] = 15;
  h2j_SubProcesses[{-5,  5, 25,  5, -5}] = 4;
  h2j_SubProcesses[{-5,  5, 25, 21, 21}] = 7;
  h2j_SubProcesses[{-5, -5, 25, -5, -5}] = 3;
  h2j_SubProcesses[{-5, 21, 25, 21, -5}] = 8;
  h2j_SubProcesses[{21,  1, 25, 21,  1}] = 9;
  h2j_SubProcesses[{21, -1, 25, 21, -1}] = 10;
  h2j_SubProcesses[{21,  2, 25, 21,  2}] = 9;
  h2j_SubProcesses[{21, -2, 25, 21, -2}] = 10;
  h2j_SubProcesses[{21,  3, 25, 21,  3}] = 9;
  h2j_SubProcesses[{21, -3, 25, 21, -3}] = 10;
  h2j_SubProcesses[{21,  4, 25, 21,  4}] = 9;
  h2j_SubProcesses[{21, -4, 25, 21, -4}] = 10;
  h2j_SubProcesses[{21,  5, 25, 21,  5}] = 9;
  h2j_SubProcesses[{21, -5, 25, 21, -5}] = 10;
  h2j_SubProcesses[{21, 21, 25,  1, -1}] = 11;
  h2j_SubProcesses[{21, 21, 25,  2, -2}] = 11;
  h2j_SubProcesses[{21, 21, 25,  3, -3}] = 11;
  h2j_SubProcesses[{21, 21, 25,  4, -4}] = 11;
  h2j_SubProcesses[{21, 21, 25,  5, -5}] = 11;
  h2j_SubProcesses[{21, 21, 25, 21, 21}] = 0;
  
  return; 
}

void TSelectorWrite::FillSubProcessMap_h3j()
{
  h3j_SubProcesses[{1, 1,25, 21, 1, 1}] = 20;
  h3j_SubProcesses[{1, -1,25, 21, 1, -1}] = 21;
  h3j_SubProcesses[{1, -1,25, 21, 2, -2}] = 0;
  h3j_SubProcesses[{1, -1,25, 21, 3, -3}] = 0;
  h3j_SubProcesses[{1, -1,25, 21, 4, -4}] = 0;
  h3j_SubProcesses[{1, -1,25, 21, 5, -5}] = 0;
  h3j_SubProcesses[{1, -1,25, 21, 21, 21}] = 13;
  h3j_SubProcesses[{1, 2,25, 21, 1, 2}] = 1;
  h3j_SubProcesses[{1, -2,25, 21, 1, -2}] = 2;
  h3j_SubProcesses[{1, 3,25, 21, 1, 3}] = 1;
  h3j_SubProcesses[{1, -3,25, 21, 1, -3}] = 2;
  h3j_SubProcesses[{1, 4,25, 21, 1, 4}] = 1;
  h3j_SubProcesses[{1, -4,25, 21, 1, -4}] = 2;
  h3j_SubProcesses[{1, 5,25, 21, 1, 5}] = 1;
  h3j_SubProcesses[{1, -5,25, 21, 1, -5}] = 2;
  h3j_SubProcesses[{1, 21,25, 1, 1, -1}] = 22;
  h3j_SubProcesses[{1, 21,25, 1, 2, -2}] = 3;
  h3j_SubProcesses[{1, 21,25, 1, 3, -3}] = 3;
  h3j_SubProcesses[{1, 21,25, 1, 4, -4}] = 3;
  h3j_SubProcesses[{1, 21,25, 1, 5, -5}] = 3;
  h3j_SubProcesses[{1, 21,25, 21, 21, 1}] = 14;
  h3j_SubProcesses[{-1, 1,25, 21, 1, -1}] = 23;
  h3j_SubProcesses[{-1, 1,25, 21, 2, -2}] = 4;
  h3j_SubProcesses[{-1, 1,25, 21, 3, -3}] = 4;
  h3j_SubProcesses[{-1, 1,25, 21, 4, -4}] = 4;
  h3j_SubProcesses[{-1, 1,25, 21, 5, -5}] = 4;
  h3j_SubProcesses[{-1, 1,25, 21, 21, 21}] = 15;
  h3j_SubProcesses[{-1, -1,25, 21, -1, -1}] = 24;
  h3j_SubProcesses[{-1, 2,25, 21, 2, -1}] = 5;
  h3j_SubProcesses[{-1, -2,25, 21, -1, -2}] = 6;
  h3j_SubProcesses[{-1, 3,25, 21, 3, -1}] = 5;
  h3j_SubProcesses[{-1, -3,25, 21, -1, -3}] = 6;
  h3j_SubProcesses[{-1, 4,25, 21, 4, -1}] = 5;
  h3j_SubProcesses[{-1, -4,25, 21, -1, -4}] = 6;
  h3j_SubProcesses[{-1, 5,25, 21, 5, -1}] = 5;
  h3j_SubProcesses[{-1, -5,25, 21, -1, -5}] = 6;
  h3j_SubProcesses[{-1, 21,25, 1, -1, -1}] = 25;
  h3j_SubProcesses[{-1, 21,25, -1, 2, -2}] = 7;
  h3j_SubProcesses[{-1, 21,25, -1, 3, -3}] = 7;
  h3j_SubProcesses[{-1, 21,25, -1, 4, -4}] = 7;
  h3j_SubProcesses[{-1, 21,25, -1, 5, -5}] = 7;
  h3j_SubProcesses[{-1, 21,25, 21, 21, -1}] = 16;
  h3j_SubProcesses[{2, 1,25, 21, 1, 2}] = 8;
  h3j_SubProcesses[{2, -1,25, 21, 2, -1}] = 2;
  h3j_SubProcesses[{2, 2,25, 21, 2, 2}] = 20;
  h3j_SubProcesses[{2, -2,25, 21, 1, -1}] = 0;
  h3j_SubProcesses[{2, -2,25, 21, 2, -2}] = 21;
  h3j_SubProcesses[{2, -2,25, 21, 3, -3}] = 0;
  h3j_SubProcesses[{2, -2,25, 21, 4, -4}] = 0;
  h3j_SubProcesses[{2, -2,25, 21, 5, -5}] = 0;
  h3j_SubProcesses[{2, -2,25, 21, 21, 21}] = 13;
  h3j_SubProcesses[{2, 3,25, 21, 3, 2}] = 8;
  h3j_SubProcesses[{2, -3,25, 21, 2, -3}] = 2;
  h3j_SubProcesses[{2, 4,25, 21, 2, 4}] = 1;
  h3j_SubProcesses[{2, -4,25, 21, 2, -4}] = 2;
  h3j_SubProcesses[{2, 5,25, 21, 5, 2}] = 8;
  h3j_SubProcesses[{2, -5,25, 21, 2, -5}] = 2;
  h3j_SubProcesses[{2, 21,25, 2, 1, -1}] = 3;
  h3j_SubProcesses[{2, 21,25, 2, 2, -2}] = 22;
  h3j_SubProcesses[{2, 21,25, 2, 3, -3}] = 3;
  h3j_SubProcesses[{2, 21,25, 2, 4, -4}] = 3;
  h3j_SubProcesses[{2, 21,25, 2, 5, -5}] = 3;
  h3j_SubProcesses[{2, 21,25, 21, 21, 2}] = 14;
  h3j_SubProcesses[{-2, 1,25, 21, 1, -2}] = 5;
  h3j_SubProcesses[{-2, -1,25, 21, -1, -2}] = 9;
  h3j_SubProcesses[{-2, 2,25, 21, 1, -1}] = 4;
  h3j_SubProcesses[{-2, 2,25, 21, 2, -2}] = 23;
  h3j_SubProcesses[{-2, 2,25, 21, 3, -3}] = 4;
  h3j_SubProcesses[{-2, 2,25, 21, 4, -4}] = 4;
  h3j_SubProcesses[{-2, 2,25, 21, 5, -5}] = 4;
  h3j_SubProcesses[{-2, 2,25, 21, 21, 21}] = 15;
  h3j_SubProcesses[{-2, -2,25, 21, -2, -2}] = 24;
  h3j_SubProcesses[{-2, 3,25, 21, 3, -2}] = 5;
  h3j_SubProcesses[{-2, -3,25, 21, -3, -2}] = 9;
  h3j_SubProcesses[{-2, 4,25, 21, 4, -2}] = 5;
  h3j_SubProcesses[{-2, -4,25, 21, -2, -4}] = 6;
  h3j_SubProcesses[{-2, 5,25, 21, 5, -2}] = 5;
  h3j_SubProcesses[{-2, -5,25, 21, -5, -2}] = 9;
  h3j_SubProcesses[{-2, 21,25, -2, 1, -1}] = 7;
  h3j_SubProcesses[{-2, 21,25, 2, -2, -2}] = 25;
  h3j_SubProcesses[{-2, 21,25, -2, 3, -3}] = 7;
  h3j_SubProcesses[{-2, 21,25, -2, 4, -4}] = 7;
  h3j_SubProcesses[{-2, 21,25, -2, 5, -5}] = 7;
  h3j_SubProcesses[{-2, 21,25, 21, 21, -2}] = 16;
  h3j_SubProcesses[{3, 1,25, 21, 1, 3}] = 8;
  h3j_SubProcesses[{3, -1,25, 21, 3, -1}] = 2;
  h3j_SubProcesses[{3, 2,25, 21, 3, 2}] = 1;
  h3j_SubProcesses[{3, -2,25, 21, 3, -2}] = 2;
  h3j_SubProcesses[{3, 3,25, 21, 3, 3}] = 20;
  h3j_SubProcesses[{3, -3,25, 21, 1, -1}] = 0;
  h3j_SubProcesses[{3, -3,25, 21, 2, -2}] = 0;
  h3j_SubProcesses[{3, -3,25, 21, 3, -3}] = 21;
  h3j_SubProcesses[{3, -3,25, 21, 4, -4}] = 0;
  h3j_SubProcesses[{3, -3,25, 21, 5, -5}] = 0;
  h3j_SubProcesses[{3, -3,25, 21, 21, 21}] = 13;
  h3j_SubProcesses[{3, 4,25, 21, 3, 4}] = 1;
  h3j_SubProcesses[{3, -4,25, 21, 3, -4}] = 2;
  h3j_SubProcesses[{3, 5,25, 21, 3, 5}] = 1;
  h3j_SubProcesses[{3, -5,25, 21, 3, -5}] = 2;
  h3j_SubProcesses[{3, 21,25, 3, 1, -1}] = 3;
  h3j_SubProcesses[{3, 21,25, 3, 2, -2}] = 3;
  h3j_SubProcesses[{3, 21,25, 3, 3, -3}] = 22;
  h3j_SubProcesses[{3, 21,25, 3, 4, -4}] = 3;
  h3j_SubProcesses[{3, 21,25, 3, 5, -5}] = 3;
  h3j_SubProcesses[{3, 21,25, 21, 21, 3}] = 14;
  h3j_SubProcesses[{-3, 1,25, 21, 1, -3}] = 5;
  h3j_SubProcesses[{-3, -1,25, 21, -1, -3}] = 9;
  h3j_SubProcesses[{-3, 2,25, 21, 2, -3}] = 5;
  h3j_SubProcesses[{-3, -2,25, 21, -3, -2}] = 6;
  h3j_SubProcesses[{-3, 3,25, 21, 1, -1}] = 4;
  h3j_SubProcesses[{-3, 3,25, 21, 2, -2}] = 4;
  h3j_SubProcesses[{-3, 3,25, 21, 3, -3}] = 23;
  h3j_SubProcesses[{-3, 3,25, 21, 4, -4}] = 4;
  h3j_SubProcesses[{-3, 3,25, 21, 5, -5}] = 4;
  h3j_SubProcesses[{-3, 3,25, 21, 21, 21}] = 15;
  h3j_SubProcesses[{-3, -3,25, 21, -3, -3}] = 24;
  h3j_SubProcesses[{-3, 4,25, 21, 4, -3}] = 5;
  h3j_SubProcesses[{-3, -4,25, 21, -3, -4}] = 6;
  h3j_SubProcesses[{-3, 5,25, 21, 5, -3}] = 5;
  h3j_SubProcesses[{-3, -5,25, 21, -3, -5}] = 6;
  h3j_SubProcesses[{-3, 21,25, -3, 1, -1}] = 7;
  h3j_SubProcesses[{-3, 21,25, -3, 2, -2}] = 7;
  h3j_SubProcesses[{-3, 21,25, 3, -3, -3}] = 25;
  h3j_SubProcesses[{-3, 21,25, -3, 4, -4}] = 7;
  h3j_SubProcesses[{-3, 21,25, -3, 5, -5}] = 7;
  h3j_SubProcesses[{-3, 21,25, 21, 21, -3}] = 16;
  h3j_SubProcesses[{4, 1,25, 21, 1, 4}] = 8;
  h3j_SubProcesses[{4, -1,25, 21, 4, -1}] = 2;
  h3j_SubProcesses[{4, 2,25, 21, 2, 4}] = 8;
  h3j_SubProcesses[{4, -2,25, 21, 4, -2}] = 2;
  h3j_SubProcesses[{4, 3,25, 21, 3, 4}] = 8;
  h3j_SubProcesses[{4, -3,25, 21, 4, -3}] = 2;
  h3j_SubProcesses[{4, 4,25, 21, 4, 4}] = 20;
  h3j_SubProcesses[{4, -4,25, 21, 1, -1}] = 0;
  h3j_SubProcesses[{4, -4,25, 21, 2, -2}] = 0;
  h3j_SubProcesses[{4, -4,25, 21, 3, -3}] = 0;
  h3j_SubProcesses[{4, -4,25, 21, 4, -4}] = 21;
  h3j_SubProcesses[{4, -4,25, 21, 5, -5}] = 0;
  h3j_SubProcesses[{4, -4,25, 21, 21, 21}] = 13;
  h3j_SubProcesses[{4, 5,25, 21, 5, 4}] = 8;
  h3j_SubProcesses[{4, -5,25, 21, 4, -5}] = 2;
  h3j_SubProcesses[{4, 21,25, 4, 1, -1}] = 3;
  h3j_SubProcesses[{4, 21,25, 4, 2, -2}] = 3;
  h3j_SubProcesses[{4, 21,25, 4, 3, -3}] = 3;
  h3j_SubProcesses[{4, 21,25, 4, 4, -4}] = 22;
  h3j_SubProcesses[{4, 21,25, 4, 5, -5}] = 3;
  h3j_SubProcesses[{4, 21,25, 21, 21, 4}] = 14;
  h3j_SubProcesses[{-4, 1,25, 21, 1, -4}] = 5;
  h3j_SubProcesses[{-4, -1,25, 21, -1, -4}] = 9;
  h3j_SubProcesses[{-4, 2,25, 21, 2, -4}] = 5;
  h3j_SubProcesses[{-4, -2,25, 21, -2, -4}] = 9;
  h3j_SubProcesses[{-4, 3,25, 21, 3, -4}] = 5;
  h3j_SubProcesses[{-4, -3,25, 21, -3, -4}] = 9;
  h3j_SubProcesses[{-4, 4,25, 21, 1, -1}] = 4;
  h3j_SubProcesses[{-4, 4,25, 21, 2, -2}] = 4;
  h3j_SubProcesses[{-4, 4,25, 21, 3, -3}] = 4;
  h3j_SubProcesses[{-4, 4,25, 21, 4, -4}] = 23;
  h3j_SubProcesses[{-4, 4,25, 21, 5, -5}] = 4;
  h3j_SubProcesses[{-4, 4,25, 21, 21, 21}] = 15;
  h3j_SubProcesses[{-4, -4,25, 21, -4, -4}] = 24;
  h3j_SubProcesses[{-4, 5,25, 21, 5, -4}] = 5;
  h3j_SubProcesses[{-4, -5,25, 21, -5, -4}] = 9;
  h3j_SubProcesses[{-4, 21,25, -4, 1, -1}] = 7;
  h3j_SubProcesses[{-4, 21,25, -4, 2, -2}] = 7;
  h3j_SubProcesses[{-4, 21,25, -4, 3, -3}] = 7;
  h3j_SubProcesses[{-4, 21,25, 4, -4, -4}] = 25;
  h3j_SubProcesses[{-4, 21,25, -4, 5, -5}] = 7;
  h3j_SubProcesses[{-4, 21,25, 21, 21, -4}] = 16;
  h3j_SubProcesses[{5, 1,25, 21, 1, 5}] = 8;
  h3j_SubProcesses[{5, -1,25, 21, 5, -1}] = 2;
  h3j_SubProcesses[{5, 2,25, 21, 5, 2}] = 1;
  h3j_SubProcesses[{5, -2,25, 21, 5, -2}] = 2;
  h3j_SubProcesses[{5, 3,25, 21, 3, 5}] = 8;
  h3j_SubProcesses[{5, -3,25, 21, 5, -3}] = 2;
  h3j_SubProcesses[{5, 4,25, 21, 5, 4}] = 1;
  h3j_SubProcesses[{5, -4,25, 21, 5, -4}] = 2;
  h3j_SubProcesses[{5, 5,25, 21, 5, 5}] = 20;
  h3j_SubProcesses[{5, -5,25, 21, 1, -1}] = 0;
  h3j_SubProcesses[{5, -5,25, 21, 2, -2}] = 0;
  h3j_SubProcesses[{5, -5,25, 21, 3, -3}] = 0;
  h3j_SubProcesses[{5, -5,25, 21, 4, -4}] = 0;
  h3j_SubProcesses[{5, -5,25, 21, 5, -5}] = 21;
  h3j_SubProcesses[{5, -5,25, 21, 21, 21}] = 13;
  h3j_SubProcesses[{5, 21,25, 5, 1, -1}] = 3;
  h3j_SubProcesses[{5, 21,25, 5, 2, -2}] = 3;
  h3j_SubProcesses[{5, 21,25, 5, 3, -3}] = 3;
  h3j_SubProcesses[{5, 21,25, 5, 4, -4}] = 3;
  h3j_SubProcesses[{5, 21,25, 5, 5, -5}] = 22;
  h3j_SubProcesses[{5, 21,25, 21, 21, 5}] = 14;
  h3j_SubProcesses[{-5, 1,25, 21, 1, -5}] = 5;
  h3j_SubProcesses[{-5, -1,25, 21, -1, -5}] = 9;
  h3j_SubProcesses[{-5, 2,25, 21, 2, -5}] = 5;
  h3j_SubProcesses[{-5, -2,25, 21, -5, -2}] = 6;
  h3j_SubProcesses[{-5, 3,25, 21, 3, -5}] = 5;
  h3j_SubProcesses[{-5, -3,25, 21, -3, -5}] = 9;
  h3j_SubProcesses[{-5, 4,25, 21, 4, -5}] = 5;
  h3j_SubProcesses[{-5, -4,25, 21, -5, -4}] = 6;
  h3j_SubProcesses[{-5, 5,25, 21, 1, -1}] = 4;
  h3j_SubProcesses[{-5, 5,25, 21, 2, -2}] = 4;
  h3j_SubProcesses[{-5, 5,25, 21, 3, -3}] = 4;
  h3j_SubProcesses[{-5, 5,25, 21, 4, -4}] = 4;
  h3j_SubProcesses[{-5, 5,25, 21, 5, -5}] = 23;
  h3j_SubProcesses[{-5, 5,25, 21, 21, 21}] = 15;
  h3j_SubProcesses[{-5, -5,25, 21, -5, -5}] = 24;
  h3j_SubProcesses[{-5, 21,25, -5, 1, -1}] = 7;
  h3j_SubProcesses[{-5, 21,25, -5, 2, -2}] = 7;
  h3j_SubProcesses[{-5, 21,25, -5, 3, -3}] = 7;
  h3j_SubProcesses[{-5, 21,25, -5, 4, -4}] = 7;
  h3j_SubProcesses[{-5, 21,25, 5, -5, -5}] = 25;
  h3j_SubProcesses[{-5, 21,25, 21, 21, -5}] = 16;
  h3j_SubProcesses[{21, 1,25, 1, 1, -1}] = 26;
  h3j_SubProcesses[{21, 1,25, 1, 2, -2}] = 10;
  h3j_SubProcesses[{21, 1,25, 1, 3, -3}] = 10;
  h3j_SubProcesses[{21, 1,25, 1, 4, -4}] = 10;
  h3j_SubProcesses[{21, 1,25, 1, 5, -5}] = 10;
  h3j_SubProcesses[{21, 1,25, 21, 21, 1}] = 17;
  h3j_SubProcesses[{21, -1,25, 1, -1, -1}] = 27;
  h3j_SubProcesses[{21, -1,25, -1, 2, -2}] = 11;
  h3j_SubProcesses[{21, -1,25, -1, 3, -3}] = 11;
  h3j_SubProcesses[{21, -1,25, -1, 4, -4}] = 11;
  h3j_SubProcesses[{21, -1,25, -1, 5, -5}] = 11;
  h3j_SubProcesses[{21, -1,25, 21, 21, -1}] = 18;
  h3j_SubProcesses[{21, 2,25, 2, 1, -1}] = 10;
  h3j_SubProcesses[{21, 2,25, 2, 2, -2}] = 26;
  h3j_SubProcesses[{21, 2,25, 2, 3, -3}] = 10;
  h3j_SubProcesses[{21, 2,25, 2, 4, -4}] = 10;
  h3j_SubProcesses[{21, 2,25, 2, 5, -5}] = 10;
  h3j_SubProcesses[{21, 2,25, 21, 21, 2}] = 17;
  h3j_SubProcesses[{21, -2,25, -2, 1, -1}] = 11;
  h3j_SubProcesses[{21, -2,25, 2, -2, -2}] = 27;
  h3j_SubProcesses[{21, -2,25, -2, 3, -3}] = 11;
  h3j_SubProcesses[{21, -2,25, -2, 4, -4}] = 11;
  h3j_SubProcesses[{21, -2,25, -2, 5, -5}] = 11;
  h3j_SubProcesses[{21, -2,25, 21, 21, -2}] = 18;
  h3j_SubProcesses[{21, 3,25, 3, 1, -1}] = 10;
  h3j_SubProcesses[{21, 3,25, 3, 2, -2}] = 10;
  h3j_SubProcesses[{21, 3,25, 3, 3, -3}] = 26;
  h3j_SubProcesses[{21, 3,25, 3, 4, -4}] = 10;
  h3j_SubProcesses[{21, 3,25, 3, 5, -5}] = 10;
  h3j_SubProcesses[{21, 3,25, 21, 21, 3}] = 17;
  h3j_SubProcesses[{21, -3,25, -3, 1, -1}] = 11;
  h3j_SubProcesses[{21, -3,25, -3, 2, -2}] = 11;
  h3j_SubProcesses[{21, -3,25, 3, -3, -3}] = 27;
  h3j_SubProcesses[{21, -3,25, -3, 4, -4}] = 11;
  h3j_SubProcesses[{21, -3,25, -3, 5, -5}] = 11;
  h3j_SubProcesses[{21, -3,25, 21, 21, -3}] = 18;
  h3j_SubProcesses[{21, 4,25, 4, 1, -1}] = 10;
  h3j_SubProcesses[{21, 4,25, 4, 2, -2}] = 10;
  h3j_SubProcesses[{21, 4,25, 4, 3, -3}] = 10;
  h3j_SubProcesses[{21, 4,25, 4, 4, -4}] = 26;
  h3j_SubProcesses[{21, 4,25, 4, 5, -5}] = 10;
  h3j_SubProcesses[{21, 4,25, 21, 21, 4}] = 17;
  h3j_SubProcesses[{21, -4,25, -4, 1, -1}] = 11;
  h3j_SubProcesses[{21, -4,25, -4, 2, -2}] = 11;
  h3j_SubProcesses[{21, -4,25, -4, 3, -3}] = 11;
  h3j_SubProcesses[{21, -4,25, 4, -4, -4}] = 27;
  h3j_SubProcesses[{21, -4,25, -4, 5, -5}] = 11;
  h3j_SubProcesses[{21, -4,25, 21, 21, -4}] = 18;
  h3j_SubProcesses[{21, 5,25, 5, 1, -1}] = 10;
  h3j_SubProcesses[{21, 5,25, 5, 2, -2}] = 10;
  h3j_SubProcesses[{21, 5,25, 5, 3, -3}] = 10;
  h3j_SubProcesses[{21, 5,25, 5, 4, -4}] = 10;
  h3j_SubProcesses[{21, 5,25, 5, 5, -5}] = 26;
  h3j_SubProcesses[{21, 5,25, 21, 21, 5}] = 17;
  h3j_SubProcesses[{21, -5,25, -5, 1, -1}] = 11;
  h3j_SubProcesses[{21, -5,25, -5, 2, -2}] = 11;
  h3j_SubProcesses[{21, -5,25, -5, 3, -3}] = 11;
  h3j_SubProcesses[{21, -5,25, -5, 4, -4}] = 11;
  h3j_SubProcesses[{21, -5,25, 5, -5, -5}] = 27;
  h3j_SubProcesses[{21, -5,25, 21, 21, -5}] = 18;
  h3j_SubProcesses[{21, 21,25, 21, 1, -1}] = 19;
  h3j_SubProcesses[{21, 21,25, 21, 2, -2}] = 19;
  h3j_SubProcesses[{21, 21,25, 21, 3, -3}] = 19;
  h3j_SubProcesses[{21, 21,25, 21, 4, -4}] = 19;
  h3j_SubProcesses[{21, 21,25, 21, 5, -5}] = 19;
  h3j_SubProcesses[{21, 21,25, 21, 21, 21}] = 12;
    
  return; 
}
