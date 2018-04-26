#ifndef TSELECTOR_ANALYZER_H
#define TSELECTOR_ANALYZER_H

#include <fastjet/ClusterSequence.hh>
#include "TSelectorMain.h"
#include <vector>

class TSelectorReader;

class TSelectorAnalyzer : public TSelectorMain
{
 public :

  //--[ Overridable methods:

  int get_alphaspower() const { return Int_t(*input_alphaspower) - opt_extra_alphas; }

  int  Type();
  void Notify();
  void Init(const TSelectorReader* reader);
  bool Process();
  void SlaveBegin();
  void SlaveTerminate();

  TSelectorAnalyzer();
  ~TSelectorAnalyzer();

  //--] Overridable methods

  //--[ Analysis stuff:

  double call_count;
  double event_count;
  double event_binned;
  
  typedef std::vector<fastjet::PseudoJet> PseudoJetVector;
  fastjet::PseudoJet get_vec(int i) const;

  unsigned int multip;  // final state multiplicity
  unsigned int runmode = 1; // running mode
  
  void ObservablesAnalysis();
  void JetsAnalysis();
  void PrintEvent(const PseudoJetVector particles);

  // Observables vectors for observable based analysis
  vector<Double_t> mjj;
  vector<Double_t> pth;
  vector<Double_t> ptj1;
  vector<Double_t> ptj2;
  vector<Double_t> dphijj;
  vector<Double_t> yj1;
  vector<Double_t> yj2;
  vector<Double_t> yj3;
  vector<Double_t> yjj;
  vector<Double_t> zstar;
  vector<Double_t> zstarj3;
  vector<Double_t> Rptjet;
  vector<Double_t> me_weight;

  // Invariant mass
  Double_t m_inv(fastjet::PseudoJet p1, fastjet::PseudoJet p2){
     return sqrt( pow(p1.E()+p2.E(),2)-pow(p1.px()+p2.px(),2)-pow(p1.py()+p2.py(),2)-pow(p1.pz()+p2.pz(),2) );
  }

  // Observables vectors for jet based analysis
  vector<Double_t> jetsvector;

  //--] Analysis stuff:


  //--[ Member variables:
  int opt_extra_alphas;     // number of extra alphas powers

  //--] Member variables


  //--[ Reweighting variables:


  //--] Reweighting variables


};

#endif
