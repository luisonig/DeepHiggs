#include <fstream>
#ifndef NDEBUG
  #include <iostream>
#endif

#include "TSelectorReader.h"
#include "TSelectorMain.h"


// --------------------------------------------------------------------------- //
// Selector
// --------------------------------------------------------------------------- //

TSelectorMain::TSelectorMain()
  : event_prev_id(-1), event_groups(0),
    event_trials(1), new_event(1)
{

  // extra alphas powers settings
  //opt_extra_alphas = 0;

}

TSelectorMain::~TSelectorMain()
{
  // if (analysis) {
  //   delete analysis;
  //   analysis = 0;
  // }
}

int TSelectorMain::Type()
{
  return 0;
}

void TSelectorMain::Notify()
{
  return;
}

void TSelectorMain::Init(const TSelectorReader* reader)
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


bool TSelectorMain::Process()
{

  //TestAnalysis();

  return true;
}

void TSelectorMain::SlaveBegin()
{
  // pass
}

void TSelectorMain::SlaveTerminate()
{
  // The SlaveTerminate() function is called after all entries or objects
  // have been processed.

  //analysis->analysis_finalize(this);
}


// void TSelectorMain::TestAnalysis()
// {

//   // ALPHAS
//   //const std::string pdfset("CT10nlo");
//   //LHAPDF::initPDFSet(11000, pdfset, 0);

//   PseudoJetVector particles;
//   //fastjet::PseudoJet Hmom;
//   //PseudoJetVector partons;

//   Double_t Etot = 0.0;

//   for (Int_t j=0; j<get_nparticle(); j++) {
//     Etot+=get_E(j);
//   }

//   fastjet::PseudoJet vec1 = fastjet::PseudoJet(0., 0., get_x1()*Etot/(get_x1()+get_x2()), get_x1()*Etot/(get_x1()+get_x2()));
//   vec1.set_user_index(get_id1());
//   fastjet::PseudoJet vec2 = fastjet::PseudoJet(0., 0.,-get_x2()*Etot/(get_x1()+get_x2()), get_x2()*Etot/(get_x1()+get_x2()));
//   vec2.set_user_index(get_id2());
//   particles.push_back(vec1);
//   particles.push_back(vec2);

//   // Create and fill particle kinematic arrays:
//   for (Int_t i=0; i<get_nparticle(); i++){

//     fastjet::PseudoJet vec = fastjet::PseudoJet(get_px(i), get_py(i), get_pz(i), get_E(i));
//     vec.set_user_index(get_kf(i));
//     particles.push_back(vec);
//   }

//   PrintEvent(particles);

//  //  NOT NEEDED HERE, BUT KEEP JUST IN CASE: //

//  /*  std::map<subprocess, int>::iterator it;
//      it = h2jsubprocesses.find(flav);
//      if ( it != h2jsubprocesses.end()){
//      if(debug) std::cout<<"subprocess = "<<h2jsubprocesses[flav]<<std::endl;
//      }
//      else {
//      std::cerr<<"ERROR SUBPROCESS NOT FOUND!\n---> "
//      <<flav[0]<<" "<<flav[1]<<" -> "
//      <<flav[2]<<" "<<flav[3]<<" "<<flav[4]<<" "<<flav[5]<<std::endl;
//      return;
//      }
//  */

// }


// void TSelectorMain::PrintEvent(PseudoJetVector particles)
// {
//   cout.precision(15);
//   cout.setf(ios::scientific, ios::floatfield);

//   std::cout<<"--------------------\n";
//   std::cout<<"proc = "
// 	   <<particles[0].user_index()<<" "<<particles[1].user_index()<<" -> ";
//   for(unsigned i=2; i<particles.size(); i++){
//     std::cout<<particles[i].user_index()<<" ";
//   }
//   std::cout<<std::endl;
//   for(unsigned i=0; i<particles.size(); i++){
//     std::cout<<particles[i].E() <<"\t"
// 	     <<particles[i].px()<<"\t"
// 	     <<particles[i].py()<<"\t"
// 	     <<particles[i].pz()<<";\t m="
// 	     <<particles[i].m()<<std::endl;
//   }
// }
