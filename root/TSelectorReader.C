
#include <TH2.h>
#include <TStyle.h>
#include "TLeaf.h"

#ifndef NDEBUG
  #include <iostream>
#endif

//#include <TCanvas.h>
//#include "TFile.h"
//#include "TTree.h"
//#include "TH1F.h"
//#include "TLorentzVector.h"
//#include <iostream>
//#include <math.h>
//#include <string>
//#include <sstream>
//#include <cmath>

#include <fastjet/ClusterSequence.hh>

#include "TSelectorMain.h"
#include "TSelectorReader.h"
//#include "LHAGlue.h"


TSelectorReader::TSelectorReader(TTree * /*tree*/) : fChain(0)
{

}

TSelectorReader::~TSelectorReader()
{
  //if (!fChain) return;
  //   delete fChain->GetCurrentFile();
}

void TSelectorReader::Init(TTree *tree)
{

  std::cout<<"--> Init Reader:"<<std::endl;

  if (!tree) return;
  fChain = tree;
  fChain->SetMakeClass(1);

  fChain->SetBranchAddress("id", &id, &b_id);
  fChain->SetBranchAddress("nparticle", &nparticle, &b_nparticle);
  ncount = 1;
  if(fChain->GetBranch("ncount")){
    fChain->SetBranchAddress("ncount", &ncount, &b_ncount);
    std::cout<<"Using exact event recovery."<<std::endl;
  }
  if (fChain->GetLeaf("E")!=0) {
    ed_ntuples = fChain->GetLeaf("E")->GetTypeName()[0]=='D';
    std::cout<<"Auto-dectected NTuples format: ";
    std::cout<<fChain->GetLeaf("E")->GetTypeName()<<std::endl;
  }
  if(ed_ntuples){
    fChain->SetBranchAddress("px", px, &b_px);
    fChain->SetBranchAddress("py", py, &b_py);
    fChain->SetBranchAddress("pz", pz, &b_pz);
    fChain->SetBranchAddress("E", E, &b_E);}
  else{
    fChain->SetBranchAddress("px", px_f, &b_px);
    fChain->SetBranchAddress("py", py_f, &b_py);
    fChain->SetBranchAddress("pz", pz_f, &b_pz);
    fChain->SetBranchAddress("E", E_f, &b_E);
  }
  fChain->SetBranchAddress("alphas", &alphas, &b_alphas);
  fChain->SetBranchAddress("kf", kf, &b_kf);
  ps_wgt = 1.0;
  if(fChain->GetBranch("ps_wgt")){
    fChain->SetBranchAddress("ps_wgt", &ps_wgt, &b_ps_wgt);
  }
  fChain->SetBranchAddress("weight", &weight, &b_weight);
  fChain->SetBranchAddress("weight2", &weight2, &b_weight2);
  fChain->SetBranchAddress("me_wgt", &me_wgt, &b_me_wgt);
  fChain->SetBranchAddress("me_wgt2", &me_wgt2, &b_me_wgt2);
  fChain->SetBranchAddress("x1", &x1, &b_x1);
  fChain->SetBranchAddress("x2", &x2, &b_x2);
  fChain->SetBranchAddress("x1p", &x1p, &b_x1p);
  fChain->SetBranchAddress("x2p", &x2p, &b_x2p);
  fChain->SetBranchAddress("id1", &id1, &b_id1);
  fChain->SetBranchAddress("id2", &id2, &b_id2);
  id1p = 0;
  if(fChain->GetBranch("id1p")){
  fChain->SetBranchAddress("id1p", &id1p, &b_id1p);
  }
  id2p = 0;
  if(fChain->GetBranch("id2p")){
  fChain->SetBranchAddress("id2p", &id2p, &b_id2p);
  }
  fChain->SetBranchAddress("fac_scale", &fac_scale, &b_fac_scale);
  fChain->SetBranchAddress("ren_scale", &ren_scale, &b_ren_scale);
  fChain->SetBranchAddress("nuwgt", &nuwgt, &b_nuwgt);
  fChain->SetBranchAddress("usr_wgts", &usr_wgts, &b_usr_wgts);
  fChain->SetBranchAddress("alphasPower", &alphaspower, &b_alphaspower);
  fChain->SetBranchAddress("part", part, &b_part);

  std::cout<<"--<"<<std::endl;
}

Bool_t TSelectorReader::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

  if (fChain && fChain->GetCurrentFile()) {
    std::cout << "Input file: " << fChain->GetCurrentFile()->GetName() << std::endl;

    for (unsigned i = 0; i < selectors.size(); i++) {
      // std::cout<<"Processing selector of type:"<<selectors[i]->Type()<<std::endl;
      // selectors[i]->originfile=fChain->GetCurrentFile()->GetName();
      // selectors[i]->Notify();
    }
  }

  return kTRUE;
}

void TSelectorReader::Begin(TTree* /*tree*/)
{
  // The Begin() function is called at the start of the query.
  // When running with PROOF Begin() is only called on the client.
  // The tree argument is deprecated (on PROOF 0 is passed).

}

void TSelectorReader::SlaveBegin(TTree* /*tree*/)
{
  // The SlaveBegin() function is called after the Begin() function.
  // When running with PROOF SlaveBegin() is called on each slave server.
  // The tree argument is deprecated (on PROOF 0 is passed).

  fastjet::ClusterSequence::set_fastjet_banner_stream(0); // silence fastjet
}

Bool_t TSelectorReader::Process(Long64_t entry)
{
  // The Process() function is called for each entry in the tree (or possibly
  // keyed object in the case of PROOF) to be processed. The entry argument
  // specifies which entry in the currently loaded tree is to be processed.
  // It can be passed to either SelectorReader::GetEntry() or TBranch::GetEntry()
  // to read either all or the required parts of the data. When processing
  // keyed objects with PROOF, the object is already loaded and is available
  // via the fObject pointer.
  //
  // This function should contain the "body" of the analysis. It can contain
  // simple or elaborate selection criteria, run algorithms on the data
  // of thid and typically fill histograms.
  //
  // The processing can be stopped by calling Abort().
  //
  // Use fStatus to set the return value of TTree::Process().
  //
  // The return value is currently not used.

  // check for Ctrl+C
  if (gROOT->IsInterrupted()) {
    Abort("Keyboard interrupt");
  }

  GetEntry(entry);

  bool rval;
  for (unsigned i = 0; i < selectors.size(); i++) {
    rval = selectors[i]->Process();
    if (not rval) {
      std::cerr << "Selector " << i << " failed to Process() event" << std::endl;
      Abort("Analysis failed");
    }
  }

  return true;
}

void TSelectorReader::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}

void TSelectorReader::SlaveTerminate()
{
  // The SlaveTerminate() function is called after all entries or objects
  // have been processed. When running with PROOF SlaveTerminate() is called
  // on each slave server.

  for (unsigned i = 0; i < selectors.size(); i++) {
    selectors[i]->SlaveTerminate();
  }
}

void TSelectorReader::Terminate()
{
  // The Terminate() function is the last function to be called during
  // a query. It always runs on the client, it can be used to present
  // the results graphically or save the results to file.

}

void TSelectorReader::addSelector(TSelectorMain* selector)
{
  selector->Init(this);
  selectors.push_back(selector);
}
