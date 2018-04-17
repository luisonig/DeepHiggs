#ifndef TSELECTOR_READER_H
#define TSELECTOR_READER_H

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>
//#include <array>
//#include <map>

// Stuff

#define MAXNPARTICLE 100
#define MAXNUWEIGHT 32

class TSelectorMain;

class TSelectorReader : public TSelector
{
 public :
  // -----------------------------------------------------------------------
  // ROOT stuff BEGIN         ROOT stuff BEGIN        ROOT stuff BEGIN
  // ----------------------------------------------------------------------
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  
  // Declaration of leaf types
  Int_t           id;
  Int_t           nparticle;
  Int_t           ncount;             // new 
  Double_t        px[MAXNPARTICLE];   //[nparticle]
  Double_t        py[MAXNPARTICLE];   //[nparticle]
  Double_t        pz[MAXNPARTICLE];   //[nparticle]
  Double_t        E[MAXNPARTICLE];    //[nparticle]
  Float_t         px_f[MAXNPARTICLE]; //[nparticle]
  Float_t         py_f[MAXNPARTICLE]; //[nparticle]
  Float_t         pz_f[MAXNPARTICLE]; //[nparticle]
  Float_t         E_f[MAXNPARTICLE];  //[nparticle]
  Double_t        alphas;
  Int_t           kf[MAXNPARTICLE];   //[nparticle]
  Double_t        ps_wgt;             //new
  Double_t        weight;
  Double_t        weight2;
  Double_t        me_wgt;
  Double_t        me_wgt2;
  Double_t        x1;
  Double_t        x2;
  Double_t        x1p;
  Double_t        x2p;
  Int_t           id1;
  Int_t           id2;
  Int_t           id1p;               // new
  Int_t           id2p;               // new
  Double_t        fac_scale;
  Double_t        ren_scale;
  Int_t           nuwgt;
  Double_t        usr_wgts[MAXNUWEIGHT];   //[nuwgt]
  Char_t          alphaspower;
  Char_t          part[2];
  // NTuple type
  Bool_t          ed_ntuples;
  
  // List of branches
  TBranch        *b_id;          //!
  TBranch        *b_nparticle;   //!
  TBranch        *b_ncount;      //!
  TBranch        *b_px;          //!
  TBranch        *b_py;          //!
  TBranch        *b_pz;          //!
  TBranch        *b_E;           //!
  TBranch        *b_alphas;      //!
  TBranch        *b_kf;          //!
  TBranch        *b_ps_wgt;      //!
  TBranch        *b_weight;      //!
  TBranch        *b_weight2;     //!
  TBranch        *b_me_wgt;      //!
  TBranch        *b_me_wgt2;     //!
  TBranch        *b_x1;          //!
  TBranch        *b_x2;          //!
  TBranch        *b_x1p;         //!
  TBranch        *b_x2p;         //!
  TBranch        *b_id1;         //!
  TBranch        *b_id2;         //!
  TBranch        *b_id1p;        //!
  TBranch        *b_id2p;        //!
  TBranch        *b_fac_scale;   //!
  TBranch        *b_ren_scale;   //!
  TBranch        *b_nuwgt;       //!
  TBranch        *b_usr_wgts;    //!
  TBranch        *b_alphaspower; //!
  TBranch        *b_part;        //!
  
  /* typedef std::array<int, 6> subprocess; */
  /* std::map<subprocess, int> h2jsubprocesses; */
  /* typedef std::vector<fastjet::PseudoJet> PseudoJetVector; */
  
  virtual Int_t   Version() const {
    return 2;
  }
  virtual void    Begin(TTree *tree);
  virtual void    SlaveBegin(TTree *tree);
  virtual void    Init(TTree *tree);
  virtual Bool_t  Notify();
  virtual Bool_t  Process(Long64_t entry);
  virtual Int_t   GetEntry(Long64_t entry, Int_t getall = 0) {
    return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0;
  }
  virtual void    Show(Long64_t entry = -1);
  virtual void    SetOption(const char *option) {
    fOption = option;
  }
  virtual void    SetObject(TObject *obj) {
    fObject = obj;
  }
  virtual void    SetInputList(TList *input) {
    fInput = input;
  }
  virtual TList  *GetOutputList() const {
    return fOutput;
  }
  virtual void    SlaveTerminate();
  virtual void    Terminate();
  
  ClassDef(TSelectorReader, 0);
  
  // -----------------------------------------------------------------------
  // ROOT stuff END           ROOT stuff END          ROOT stuff END
  // -----------------------------------------------------------------------
  
  // Constructor & destructor
  TSelectorReader(TTree *tree=0 );
  virtual ~TSelectorReader();
  
  //virtual void     FillDictionary();
  void addSelector(TSelectorMain* selector);
  
 protected:
  std::vector<TSelectorMain*> selectors;
  
};

#endif


/*
  void TSelectorReader::FillDictionary()
  {
  
  h2jsubprocesses[{1, 1, 25, 1, 1}] = 1;
  h2jsubprocesses[{1, -1, 25, 1, -1}] = 2;
  h2jsubprocesses[{1, -1, 25, 2, -2}] = 12;
  h2jsubprocesses[{1, -1, 25, 3, -3}] = 12;
  h2jsubprocesses[{1, -1, 25, 4, -4}] = 12;
  h2jsubprocesses[{1, -1, 25, 5, -5}] = 12;
  h2jsubprocesses[{1, -1, 25, 21, 21}] = 5;
  h2jsubprocesses[{1, 2, 25, 1, 2}] = 13;
  h2jsubprocesses[{1, -2, 25, 1, -2}] = 14;
  h2jsubprocesses[{1, 3, 25, 1, 3}] = 13;
  h2jsubprocesses[{1, -3, 25, 1, -3}] = 14;
  h2jsubprocesses[{1, 4, 25, 1, 4}] = 13;
  h2jsubprocesses[{1, -4, 25, 1, -4}] = 14;
  h2jsubprocesses[{1, 5, 25, 1, 5}] = 13;
  h2jsubprocesses[{1, -5, 25, 1, -5}] = 14;
  h2jsubprocesses[{1, 21, 25, 21, 1}] = 6;
  h2jsubprocesses[{-1, 1, 25, 1, -1}] = 4;
  h2jsubprocesses[{-1, 1, 25, 2, -2}] = 15;
  h2jsubprocesses[{-1, 1, 25, 3, -3}] = 15;
  h2jsubprocesses[{-1, 1, 25, 4, -4}] = 15;
  h2jsubprocesses[{-1, 1, 25, 5, -5}] = 15;
  h2jsubprocesses[{-1, 1, 25, 21, 21}] = 7;
  h2jsubprocesses[{-1, -1, 25, -1, -1}] = 3;
  h2jsubprocesses[{-1, 2, 25, 2, -1}] = 16;
  h2jsubprocesses[{-1, -2, 25, -1, -2}] = 17;
  h2jsubprocesses[{-1, 3, 25, 3, -1}] = 16;
  h2jsubprocesses[{-1, -3, 25, -1, -3}] = 17;
  h2jsubprocesses[{-1, 4, 25, 4, -1}] = 16;
  h2jsubprocesses[{-1, -4, 25, -1, -4}] = 17;
  h2jsubprocesses[{-1, 5, 25, 5, -1}] = 16;
  h2jsubprocesses[{-1, -5, 25, -1, -5}] = 17;
  h2jsubprocesses[{-1, 21, 25, 21, -1}] = 8;
  h2jsubprocesses[{2,1, 25, 1, 2}] = 18;
  h2jsubprocesses[{2, -1, 25, 2, -1}] = 14;
  h2jsubprocesses[{2, 2, 25, 2, 2}] = 1;
  h2jsubprocesses[{2, -2, 25, 1, -1}] = 12;
  h2jsubprocesses[{2, -2, 25, 2, -2}] = 2;
  h2jsubprocesses[{2, -2, 25, 3, -3}] = 12;
  h2jsubprocesses[{2, -2, 25, 4, -4}] = 12;
  h2jsubprocesses[{2, -2, 25, 5, -5}] = 12;
  h2jsubprocesses[{2, -2, 25, 21, 21}] = 5;
  h2jsubprocesses[{2, 3, 25, 3, 2}] = 18;
  h2jsubprocesses[{2, -3, 25, 2, -3}] = 14;
  h2jsubprocesses[{2, 4, 25, 2, 4}] = 13;
  h2jsubprocesses[{2, -4, 25, 2, -4}] = 14;
  h2jsubprocesses[{2, 5, 25, 5, 2}] = 18;
  h2jsubprocesses[{2, -5, 25, 2, -5}] = 14;
  h2jsubprocesses[{2, 21, 25, 21, 2}] = 6;
  h2jsubprocesses[{-2, 1, 25, 1, -2}] = 16;
  h2jsubprocesses[{-2, -1, 25, -1, -2}] = 19;
  h2jsubprocesses[{-2, 2, 25, 1, -1}] = 15;
  h2jsubprocesses[{-2, 2, 25, 2, -2}] = 4;
  h2jsubprocesses[{-2, 2, 25, 3, -3}] = 15;
  h2jsubprocesses[{-2, 2, 25, 4, -4}] = 15;
  h2jsubprocesses[{-2, 2, 25, 5, -5}] = 15;
  h2jsubprocesses[{-2, 2, 25, 21, 21}] = 7;
  h2jsubprocesses[{-2, -2, 25, -2, -2}] = 3;
  h2jsubprocesses[{-2, 3, 25, 3, -2}] = 16;
  h2jsubprocesses[{-2, -3, 25, -3, -2}] = 19;
  h2jsubprocesses[{-2, 4, 25, 4, -2}] = 16;
  h2jsubprocesses[{-2, -4, 25, -2, -4}] = 17;
  h2jsubprocesses[{-2, 5, 25, 5, -2}] = 16;
  h2jsubprocesses[{-2, -5, 25, -5, -2}] = 19;
  h2jsubprocesses[{-2, 21, 25, 21, -2}] = 8;
  h2jsubprocesses[{3, 1, 25, 1, 3}] = 18;
  h2jsubprocesses[{3, -1, 25, 3, -1}] = 14;
  h2jsubprocesses[{3, 2, 25, 3, 2}] = 13;
  h2jsubprocesses[{3, -2, 25, 3, -2}] = 14;
  h2jsubprocesses[{3, 3, 25, 3, 3}] = 1;
  h2jsubprocesses[{3, -3, 25, 1, -1}] = 12;
  h2jsubprocesses[{3, -3, 25, 2, -2}] = 12;
  h2jsubprocesses[{3, -3, 25, 3, -3}] = 2;
  h2jsubprocesses[{3, -3, 25, 4, -4}] = 12;
  h2jsubprocesses[{3, -3, 25, 5, -5}] = 12;
  h2jsubprocesses[{3, -3, 25, 21, 21}] = 5;
  h2jsubprocesses[{3, 4, 25, 3, 4}] = 13;
  h2jsubprocesses[{3, -4, 25, 3, -4}] = 14;
  h2jsubprocesses[{3, 5, 25, 3, 5}] = 13;
  h2jsubprocesses[{3, -5, 25, 3, -5}] = 14;
  h2jsubprocesses[{3, 21, 25, 21, 3}] = 6;
  h2jsubprocesses[{-3, 1, 25, 1, -3}] = 16;
  h2jsubprocesses[{-3, -1, 25, -1, -3}] = 19;
  h2jsubprocesses[{-3, 2, 25, 2, -3}] = 16;
  h2jsubprocesses[{-3, -2, 25, -3, -2}] = 17;
  h2jsubprocesses[{-3, 3, 25, 1, -1}] = 15;
  h2jsubprocesses[{-3, 3, 25, 2, -2}] = 15;
  h2jsubprocesses[{-3, 3, 25, 3, -3}] = 4;
  h2jsubprocesses[{-3, 3, 25, 4, -4}] = 15;
  h2jsubprocesses[{-3, 3, 25, 5, -5}] = 15;
  h2jsubprocesses[{-3, 3, 25, 21, 21}] = 7;
  h2jsubprocesses[{-3, -3, 25, -3, -3}] = 3;
  h2jsubprocesses[{-3, 4, 25, 4, -3}] = 16;
  h2jsubprocesses[{-3, -4, 25, -3, -4}] = 17;
  h2jsubprocesses[{-3, 5, 25, 5, -3}] = 16;
  h2jsubprocesses[{-3, -5, 25, -3, -5}] = 17;
  h2jsubprocesses[{-3, 21, 25, 21, -3}] = 8;
  h2jsubprocesses[{4, 1, 25, 1, 4}] = 18;
  h2jsubprocesses[{4, -1, 25, 4, -1}] = 14;
  h2jsubprocesses[{4, 2, 25, 2, 4}] = 18;
  h2jsubprocesses[{4, -2, 25, 4, -2}] = 14;
  h2jsubprocesses[{4, 3, 25, 3, 4}] = 18;
  h2jsubprocesses[{4, -3, 25, 4, -3}] = 14;
  h2jsubprocesses[{4, 4, 25, 4, 4}] = 1;
  h2jsubprocesses[{4, -4, 25, 1, -1}] = 12;
  h2jsubprocesses[{4, -4, 25, 2, -2}] = 12;
  h2jsubprocesses[{4, -4, 25, 3, -3}] = 12;
  h2jsubprocesses[{4, -4, 25, 4, -4}] = 2;
  h2jsubprocesses[{4, -4, 25, 5, -5}] = 12;
  h2jsubprocesses[{4, -4, 25, 21, 21}] = 5;
  h2jsubprocesses[{4, 5, 25, 5, 4}] = 18;
  h2jsubprocesses[{4, -5, 25, 4, -5}] = 14;
  h2jsubprocesses[{4, 21, 25, 21, 4}] = 6;
  h2jsubprocesses[{-4, 1, 25, 1, -4}] = 16;
  h2jsubprocesses[{-4, -1, 25, -1, -4}] = 19;
  h2jsubprocesses[{-4, 2, 25, 2, -4}] = 16;
  h2jsubprocesses[{-4, -2, 25, -2, -4}] = 19;
  h2jsubprocesses[{-4, 3, 25, 3, -4}] = 16;
  h2jsubprocesses[{-4, -3, 25, -3, -4}] = 19;
  h2jsubprocesses[{-4, 4, 25, 1, -1}] = 15;
  h2jsubprocesses[{-4, 4, 25, 2, -2}] = 15;
  h2jsubprocesses[{-4, 4, 25, 3, -3}] = 15;
  h2jsubprocesses[{-4, 4, 25, 4, -4}] = 4;
  h2jsubprocesses[{-4, 4, 25, 5, -5}] = 15;
  h2jsubprocesses[{-4, 4, 25, 21, 21}] = 7;
  h2jsubprocesses[{-4, -4, 25, -4, -4}] = 3;
  h2jsubprocesses[{-4, 5, 25, 5, -4}] = 16;
  h2jsubprocesses[{-4, -5, 25, -5, -4}] = 19;
  h2jsubprocesses[{-4, 21, 25, 21, -4}] = 8;
  h2jsubprocesses[{5, 1, 25, 1, 5}] = 18;
  h2jsubprocesses[{5, -1, 25, 5, -1}] = 14;
  h2jsubprocesses[{5, 2, 25, 5, 2}] = 13;
  h2jsubprocesses[{5, -2, 25, 5, -2}] = 14;
  h2jsubprocesses[{5, 3, 25, 3, 5}] = 18;
  h2jsubprocesses[{5, -3, 25, 5, -3}] = 14;
  h2jsubprocesses[{5, 4, 25, 5, 4}] = 13;
  h2jsubprocesses[{5, -4, 25, 5, -4}] = 14;
  h2jsubprocesses[{5, 5, 25, 5, 5}] = 1;
  h2jsubprocesses[{5, -5, 25, 1, -1}] = 12;
  h2jsubprocesses[{5, -5, 25, 2, -2}] = 12;
  h2jsubprocesses[{5, -5, 25, 3, -3}] = 12;
  h2jsubprocesses[{5, -5, 25, 4, -4}] = 12;
  h2jsubprocesses[{5, -5, 25, 5, -5}] = 2;
  h2jsubprocesses[{5, -5, 25, 21, 21}] = 5;
  h2jsubprocesses[{5, 21, 25, 21, 5}] = 6;
  h2jsubprocesses[{-5, 1, 25, 1, -5}] = 16;
  h2jsubprocesses[{-5, -1, 25, -1, -5}] = 19;
  h2jsubprocesses[{-5, 2, 25, 2, -5}] = 16;
  h2jsubprocesses[{-5, -2, 25, -5, -2}] = 17;
  h2jsubprocesses[{-5, 3, 25, 3, -5}] = 16;
  h2jsubprocesses[{-5, -3, 25, -3, -5}] = 19;
  h2jsubprocesses[{-5, 4, 25, 4, -5}] = 16;
  h2jsubprocesses[{-5, -4, 25, -5, -4}] = 17;
  h2jsubprocesses[{-5, 5, 25, 1, -1}] = 15;
  h2jsubprocesses[{-5, 5, 25, 2, -2}] = 15;
  h2jsubprocesses[{-5, 5, 25, 3, -3}] = 15;
  h2jsubprocesses[{-5, 5, 25, 4, -4}] = 15;
  h2jsubprocesses[{-5, 5, 25, 5, -5}] = 4;
  h2jsubprocesses[{-5, 5, 25, 21, 21}] = 7;
  h2jsubprocesses[{-5, -5, 25, -5, -5}] = 3;
  h2jsubprocesses[{-5, 21, 25, 21, -5}] = 8;
  h2jsubprocesses[{21, 1, 25, 21, 1}] = 9;
  h2jsubprocesses[{21, -1, 25, 21, -1}] = 10;
  h2jsubprocesses[{21, 2, 25, 21, 2}] = 9;
  h2jsubprocesses[{21, -2, 25, 21, -2}] = 10;
  h2jsubprocesses[{21, 3, 25, 21, 3}] = 9;
  h2jsubprocesses[{21, -3, 25, 21, -3}] = 10;
  h2jsubprocesses[{21, 4, 25, 21, 4}] = 9;
  h2jsubprocesses[{21, -4, 25, 21, -4}] = 10;
  h2jsubprocesses[{21, 5, 25, 21, 5}] = 9;
  h2jsubprocesses[{21, -5, 25, 21, -5}] = 10;
  h2jsubprocesses[{21, 21, 25, 1, -1}] = 11;
  h2jsubprocesses[{21, 21, 25, 2, -2}] = 11;
  h2jsubprocesses[{21, 21, 25, 3, -3}] = 11;
  h2jsubprocesses[{21, 21, 25, 4, -4}] = 11;
  h2jsubprocesses[{21, 21, 25, 5, -5}] = 11;
  h2jsubprocesses[{21, 21, 25, 21, 21}] = 0;
  
  return;
}

*/

/* #if defined(__MAKECINT__) */
/* #pragma link C++ class fastjet::JetDefinition; */
/* #endif */

