#ifndef TSELECTOR_MAIN_H
#define TSELECTOR_MAIN_H

// Stuff
#include <fastjet/ClusterSequence.hh>

class TSelectorReader;

class TSelectorMain
{
 public :

  //--[ NTuple access stuff:
  // Declaration of leaf types
  const Int_t*       input_id;
  const Int_t*       input_nparticle;
  const Int_t*       input_ncount;
  const Double_t*    input_px;   //[nparticle]
  const Double_t*    input_py;   //[nparticle]
  const Double_t*    input_pz;   //[nparticle]
  const Double_t*    input_E;    //[nparticle]
  const Float_t*     input_px_f; //[nparticle]
  const Float_t*     input_py_f; //[nparticle]
  const Float_t*     input_pz_f; //[nparticle]
  const Float_t*     input_E_f;  //[nparticle]
  const Double_t*    input_alphas;
  const Int_t*       input_kf;   //[nparticle]
  const Double_t*    input_ps_wgt;
  const Double_t*    input_weight;
  const Double_t*    input_weight2;
  const Double_t*    input_me_wgt;
  const Double_t*    input_me_wgt2;
  const Double_t*    input_kfac;
  const Double_t*    input_x1;
  const Double_t*    input_x2;
  const Double_t*    input_x1p;
  const Double_t*    input_x2p;
  const Int_t*       input_id1;
  const Int_t*       input_id2;
  const Int_t*       input_id1p;
  const Int_t*       input_id2p;
  const Double_t*    input_fac_scale;
  const Double_t*    input_ren_scale;
  const Int_t*       input_nuwgt;
  const Double_t*    input_usr_wgts;   //[nuwgt]
  const Char_t*      input_part;
  const Char_t*      input_alphaspower;

  // NTuples type
  const Bool_t*      input_ed_ntuples;

  // Accessor methods
  Int_t get_event_id() const { return *input_id; }
  Int_t get_nparticle() const { return *input_nparticle; }
  Int_t get_ncount() const {
    if(*(input_ed_ntuples)){
      return *input_ncount;
    }
    else {
      return event_trials;
    }
  }
  Bool_t   is_ed_ntuples() const { return *input_ed_ntuples; }
  Double_t get_px(int i) const {
    if(*(input_ed_ntuples)) {
      return input_px[i];
    }
    else {
      return(double) input_px_f[i];
    }
  }
  Double_t get_py(int i) const {
    if(*(input_ed_ntuples)) {
      return input_py[i];
    }
    else {
      return(double) input_py_f[i];
    }
  }
  Double_t get_pz(int i) const {
    if(*(input_ed_ntuples)) {
      return input_pz[i];
    }
    else {
      return(double) input_pz_f[i];
    }
  }
  Double_t get_E(int i) const {
    if(*(input_ed_ntuples)) {
      return input_E[i];
    }
    else {
      return(double) input_E_f[i];
    }
  }

  Double_t        orig_alphas() const { return *input_alphas; }
  const Int_t*    get_kf() const { return input_kf; }
  Int_t           get_kf(int i) const { return input_kf[i]; }
  Double_t        orig_ps_wgt() const { return *input_ps_wgt; }
  Double_t        orig_weight() const { return *input_weight; }
  Double_t        orig_weight2() const { return *input_weight2; }
  Double_t        orig_me_wgt() const { return *input_me_wgt; }
  Double_t        orig_me_wgt2() const { return *input_me_wgt2; }
  Double_t        get_x1() const { return *input_x1; }
  Double_t        get_x2() const { return *input_x2; }
  Double_t        get_x1p() const { return *input_x1p; }
  Double_t        get_x2p() const { return *input_x2p; }
  Int_t           get_id1() const { return *input_id1; }
  Int_t           get_id2() const { return *input_id2; }
  Int_t           get_id1p() const { return *input_id1p; }
  Int_t           get_id2p() const { return *input_id2p; }
  Double_t        orig_fac_scale() const { return *input_fac_scale; }
  Double_t        orig_ren_scale() const { return *input_ren_scale; }
  Int_t           get_nuwgt() const { return *input_nuwgt; }
  const Double_t* orig_usr_wgts() const { return input_usr_wgts; }
  Double_t        orig_usr_wgts(int i) const { return input_usr_wgts[i]; }
  virtual Int_t   get_alphaspower() const { return Int_t(*input_alphaspower); }
  const Char_t*   get_part() const { return input_part; }
  Char_t          get_part(int i) const { return input_part[i]; }

  const Char_t* originfile;

  virtual int  Type();
  virtual void Notify();
  virtual void Init(const TSelectorReader* reader);
  virtual bool Process();
  virtual void SlaveBegin();
  virtual void SlaveTerminate();

  TSelectorMain();
  ~TSelectorMain();

  //--] NTuple access stuff

  /* //--[ Analysis stuff: */

  /* typedef std::vector<fastjet::PseudoJet> PseudoJetVector; */
  /* fastjet::PseudoJet get_vec(int i) const; */

  /* void TestAnalysis(); */
  /* void PrintEvent(const PseudoJetVector particles); */

  /* //--] Analysis stuff: */


  /* //--[ Member variables: */
  /* int opt_extra_alphas;     // number of extra alphas powers */

  /* //--] Member variables */


  /* //--[ Reweighting variables: */


  /* //--] Reweighting variables */


  //--[ Counting events:
  Int_t event_prev_id;
  long  event_groups;
  double event_trials;
  bool new_event;
  //--] Counting events

};

#endif
