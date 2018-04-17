#ifndef TSELECTOR_WRITE_H
#define TSELECTOR_WRITE_H

#include <string>
#include <TFile.h>
#include <fastjet/ClusterSequence.hh>
#include "TSelectorMain.h"

class TSelectorReader;

class TSelectorWrite : public TSelectorMain
{
 public :

  TTree          *n_fChain;

  Int_t           n_id;
  Int_t           n_nparticle;
  Int_t           n_ncount;
  Double_t        n_px[MAXNPARTICLE];   //[nparticle]
  Double_t        n_py[MAXNPARTICLE];   //[nparticle]
  Double_t        n_pz[MAXNPARTICLE];   //[nparticle]
  Double_t        n_E[MAXNPARTICLE];    //[nparticle]
  Double_t        n_alphas;
  Int_t           n_kf[MAXNPARTICLE];   //[nparticle]
  Double_t        n_ps_wgt;
  Double_t        n_weight;
  Double_t        n_weight2;
  Double_t        n_me_wgt;
  Double_t        n_me_wgt2;
  Double_t        n_x1;
  Double_t        n_x2;
  Double_t        n_x1p;
  Double_t        n_x2p;
  Int_t           n_id1;
  Int_t           n_id2;
  Int_t           n_id1p;
  Int_t           n_id2p;
  Double_t        n_fac_scale;
  Double_t        n_ren_scale;
  Int_t           n_nuwgt;
  Double_t        n_usr_wgts[MAXNUWEIGHT];   //[nuwgt]
  Char_t          n_alphaspower;
  Char_t          n_part[2];

  // List of branches
  TBranch        *nb_id;          //!
  TBranch        *nb_nparticle;   //!
  TBranch        *nb_ncount;      //!
  TBranch        *nb_px;          //!
  TBranch        *nb_py;          //!
  TBranch        *nb_pz;          //!
  TBranch        *nb_E;           //!
  TBranch        *nb_alphas;      //!
  TBranch        *nb_kf;          //!
  TBranch        *nb_ps_wgt;      //!
  TBranch        *nb_weight;      //!
  TBranch        *nb_weight2;     //!
  TBranch        *nb_me_wgt;      //!
  TBranch        *nb_me_wgt2;     //!
  TBranch        *nb_x1;          //!
  TBranch        *nb_x2;          //!
  TBranch        *nb_x1p;         //!
  TBranch        *nb_x2p;         //!
  TBranch        *nb_id1;         //!
  TBranch        *nb_id2;         //!
  TBranch        *nb_id1p;        //!
  TBranch        *nb_id2p;        //!
  TBranch        *nb_fac_scale;   //!
  TBranch        *nb_ren_scale;   //!
  TBranch        *nb_nuwgt;       //!
  TBranch        *nb_usr_wgts;    //!
  TBranch        *nb_alphaspower; //!
  TBranch        *nb_part;        //!

  //--[ Overridable methods:

  int  Type();
  void Notify();
  void Init(const TSelectorReader* reader);
  bool Process();
  void SlaveBegin();
  void SlaveTerminate();

  TSelectorWrite(int multiplicity);
  ~TSelectorWrite();

  //--] Overridable methods

  //--[ Reweighting stuff:

  typedef std::vector<fastjet::PseudoJet> PseudoJetVector;
  fastjet::PseudoJet get_vec(int i) const;

  typedef std::array<int, 4> SubProcessto2;
  typedef std::array<int, 5> SubProcessto3;
  typedef std::array<int, 6> SubProcessto4;

  bool debug;
  const int multip;

  void SetFileName(string fname, string extension);
  string outfilename;
  TFile* outputfile;

  void PrepareEvent();

  int  InitializeOLP();
  void Reweight();
  
  void CopyEvent(bool fillchain);
  void TestReweighting();
  void PrintEvent(PseudoJetVector particles);
  void FinalizeStat();
  int  CheckPoint(double heft_wgt, double full_wgt);

  //--] Reweighting stuff

  //--[ Physical parameters:

  const double pi = 3.141592653589793;

  // Input parameters are VEV, mZ and alpha:
  double alpha;
  double vev;
  double Nf;
  double mZ;
  double mT;
  double mB;
  double mBMS;

  // Derived parameters:
  double e2;
  double mW;
  double sw;

  void SetParameters(double nf=4, double mt=172.3, double mb=4.75, double mbms=3.38);

  //--] Physical parameters

  //--[ Process Specific

  // H+1jet:
  void FillSubProcessMap_h1j();
  std::map<SubProcessto2, int> h1j_SubProcesses;

  // H+2jets:
  void FillSubProcessMap_h2j();
  std::map<SubProcessto3, int> h2j_SubProcesses;

  // H+3jets:
  void FillSubProcessMap_h3j();
  std::map<SubProcessto4, int> h3j_SubProcesses;

  //--] Process Specific

};

#endif
