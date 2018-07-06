#ifndef TSELECTOR_READER_H
#define TSELECTOR_READER_H

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>

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

  void addSelector(TSelectorMain* selector);
  
 protected:
  std::vector<TSelectorMain*> selectors;
  
};

#endif
