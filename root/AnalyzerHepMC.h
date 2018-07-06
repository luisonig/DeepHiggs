#ifndef ANALYZER_HEPMC_H
#define ANALYZER_HEPMC_H

#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"
#include "HepMC/IO_GenEvent.h"

#include "fastjet/ClusterSequence.hh"
#include <sstream>
#include <iostream>
#include <string>
#include <vector>

class AnalyzerHepMC
{
 public :

  // Constructor and destructor
  AnalyzerHepMC();
  ~AnalyzerHepMC();

  void SetFileName(std::string filename);
  bool Process(int nevents);

  //--[ Analysis stuff:

  double call_count;
  double event_count;
  double event_binned;

  typedef std::vector<fastjet::PseudoJet> PseudoJetVector;

  unsigned int multip;  // final state multiplicity

  void Analysis(const PseudoJetVector particles);
  void PrintEvent(const PseudoJetVector particles);

  // Invariant mass
  Double_t m_inv(fastjet::PseudoJet p1, fastjet::PseudoJet p2){
     return sqrt( pow(p1.E()+p2.E(),2)-pow(p1.px()+p2.px(),2)-pow(p1.py()+p2.py(),2)-pow(p1.pz()+p2.pz(),2) );
  }

  //Partonic events for pixel analysis
  vector<Double_t> partons;
  vector<Double_t> entry;
  int nr_theta, nr_phi;

 private :

  std::string filename;
};

#endif
