#include <fstream>
#include <stdlib.h>
#ifndef NDEBUG
  #include <iostream>
#endif

#include "TSelectorReader.h"
#include "TSelectorAnalyzer.h"
#include <vector>
// #include "Rivet/Analysis.hh"
// #include "Rivet/Tools/Logging.hh"
// #include "Rivet/Math/Constants.hh"
// #include "Rivet/Projections/FinalState.hh"
// #include "Rivet/Projections/ChargedFinalState.hh"
// #include "Rivet/Projections/FastJets.hh"
// #include "Rivet/Projections/IdentifiedFinalState.hh"
// #include "Rivet/Projections/VisibleFinalState.hh"
// #include "Rivet/Projections/VetoedFinalState.hh"
// #include "Rivet/Projections/MissingMomentum.hh"

// --------------------------------------------------------------------------- //
// Selector
// --------------------------------------------------------------------------- //

TSelectorAnalyzer::TSelectorAnalyzer()
  : multip(0), call_count(0.), event_count(0.), event_binned(0.)
    //  , event_groups(0), new_event(1)
{

  // extra alphas powers settings
  opt_extra_alphas = 0;


}

TSelectorAnalyzer::~TSelectorAnalyzer()
{
  // if (analysis) {
  //   delete analysis;
  //   analysis = 0;
  // }
}

int TSelectorAnalyzer::Type()
{
  return 1;
}

void TSelectorAnalyzer::Init(const TSelectorReader* reader)
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
}

void TSelectorAnalyzer::Notify()
{
  return;
}

bool TSelectorAnalyzer::Process()
{

  call_count  += 1;
  event_count += get_ncount();
  
  if(runmode == 1){
    // run analysis with observables
    ObservablesAnalysis();
    
  }  else if (runmode == 2){
    // run analysis using jet momenta
    JetsAnalysis();
    
  } else {
    std::cout<<"runmode not valid, abort"<<std::endl;
    exit(EXIT_FAILURE);
  }
  
  return true;
}

void TSelectorAnalyzer::SlaveBegin()
{
  // pass
}

void TSelectorAnalyzer::SlaveTerminate()
{
  // The SlaveTerminate() function is called after all entries or objects
  // have been processed.

  //analysis->analysis_finalize(this);
}


void TSelectorAnalyzer::ObservablesAnalysis()
{ 
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
    
  //now passing final state partons to fastjet
  PseudoJetVector jetinput;
  for (Int_t i=1; i<get_nparticle(); i++){
    fastjet::PseudoJet vec = fastjet::PseudoJet(get_px(i), get_py(i), get_pz(i), get_E(i));
    jetinput.push_back(vec);
  }  

  double R(0.4);
  fastjet::JetDefinition jet_definition;
  jet_definition = fastjet::JetDefinition(fastjet::antikt_algorithm, R);  
  fastjet::ClusterSequence cs(jetinput, jet_definition);
  double jet_ptmin(25.0);
  PseudoJetVector unsortedjets = cs.inclusive_jets(jet_ptmin);
  PseudoJetVector jets = fastjet::sorted_by_pt(unsortedjets);

  // Multiplicity:
  //std::cout<<"----> "<<multip<<std::endl;

  bool accept_event = false;
  // Apply cuts:
  // multiplicity
  if (jets.size() >= multip) accept_event = true;
  if (accept_event){
    // rapidity
    for(unsigned i=0; i<jets.size(); i++){
      if (jets[i].rap() > 4.5 ) accept_event = false;
    }
    // VBF mjj
    if (m_inv(jets[0],jets[1]) < 400.0 || abs(jets[0].rap()-jets[1].rap()) < 2.8) accept_event = false;
  }
  if (accept_event){
    event_binned += 1;
   //returning Higgs pT
   pth.push_back(particles[2].pt());      
   //returning leading and subleading jet pT
   ptj1.push_back(jets[0].pt());
   ptj2.push_back(jets[1].pt());
   PseudoJetVector j1j2;
   j1j2.push_back(jets[0]+jets[1]);
   
   yj1.push_back(jets[0].rap());
   yj2.push_back(jets[1].rap());
   yjj.push_back(abs(jets[0].rap()-jets[1].rap()));
   zstar.push_back(abs(particles[2].rap()-(jets[0].rap()+jets[1].rap())/2.0)/abs(jets[0].rap()-jets[1].rap()));

   if (multip == 3){
     yj3.push_back(jets[2].rap());
     zstarj3.push_back((jets[2].rap()-(jets[0].rap()+jets[1].rap())/2.0)/abs(jets[0].rap()-jets[1].rap()));
   }

   Rptjet.push_back(j1j2[0].pt()/(jets[0].pt()+jets[1].pt()));
  
   //returning invariant jet-jet mass
   mjj.push_back(m_inv(jets[0],jets[1]));  
   
   //returning dphijj
   dphijj.push_back(abs(jets[0].phi()-jets[1].phi()));
   
   //returning weight;
   me_weight.push_back(orig_me_wgt());
  }
  
 //  NOT NEEDED HERE, BUT KEEP JUST IN CASE: //

 /*  std::map<subprocess, int>::iterator it;
     it = h2jsubprocesses.find(flav);
     if ( it != h2jsubprocesses.end()){
     if(debug) std::cout<<"subprocess = "<<h2jsubprocesses[flav]<<std::endl;
     }
     else {
     std::cerr<<"ERROR SUBPROCESS NOT FOUND!\n---> "
     <<flav[0]<<" "<<flav[1]<<" -> "
     <<flav[2]<<" "<<flav[3]<<" "<<flav[4]<<" "<<flav[5]<<std::endl;
     return;
     }
 */

}

void TSelectorAnalyzer::JetsAnalysis()
{
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
    
  //now passing final state partons to fastjet
  PseudoJetVector jetinput;
  for (Int_t i=1; i<get_nparticle(); i++){    
    fastjet::PseudoJet vec = fastjet::PseudoJet(get_px(i), get_py(i), get_pz(i), get_E(i));
    jetinput.push_back(vec);
  }  

  double R(0.4);
  fastjet::JetDefinition jet_definition;
  jet_definition = fastjet::JetDefinition(fastjet::antikt_algorithm, R);  
  fastjet::ClusterSequence cs(jetinput, jet_definition);
  double jet_ptmin(25.0);
  PseudoJetVector unsortedjets = cs.inclusive_jets(jet_ptmin);
  PseudoJetVector jets = fastjet::sorted_by_pt(unsortedjets);

  // Multiplicity:
  //std::cout<<"----> "<<multip<<std::endl;

  bool accept_event = false;
  // Apply cuts:
  // multiplicity
  if (jets.size() >= multip) accept_event = true;
  if (accept_event){
    // rapidity
    for(unsigned i=0; i<jets.size(); i++){
      if (jets[i].rap() > 4.5 ) accept_event = false;
    }
    // VBF mjj
    if (m_inv(jets[0],jets[1]) < 400.0 || abs(jets[0].rap()-jets[1].rap()) < 2.8) accept_event = false;
  }
  if (accept_event){

    event_binned += 1;

    for (unsigned i=0; i<jets.size(); i++){
      for (unsigned j=0; j<4; j++){
	jetsvector.push_back(jets[i][j]);
      }
    }
  }
}



void TSelectorAnalyzer::PrintEvent(PseudoJetVector particles)
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

