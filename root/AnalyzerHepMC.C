# include "AnalyzerHepMC.h"
#include <vector>
#include <math.h>
#include <TH2.h>

AnalyzerHepMC::AnalyzerHepMC()
  :  multip(2), call_count(0.), event_count(0.), event_binned(0.),
     nr_phi(20), nr_theta(20)
{
  //
}

AnalyzerHepMC::~AnalyzerHepMC()
{
  //
}

void AnalyzerHepMC::SetFileName(std::string file)
{
  filename = file;
  return;
}

bool AnalyzerHepMC::Process(int nevents)
{
  // reset counters
  call_count  = 0;
  event_count = 0;
  event_binned = 0;

  // Define HepMC stuff:
  HepMC::GenEvent* evt = new HepMC::GenEvent;
  HepMC::IO_GenEvent persist(filename, std::ios::in);

  // Loop over events
  while ((evt = persist.read_next_event())) {

    //if (event_count%100==0) cout << "\nEvent #" << event_count+1 << ":" << endl;

    PseudoJetVector particles;

    // Loop over particle in event
    for(HepMC::GenEvent::particle_iterator part = evt->particles_begin(); part != evt->particles_end(); ){
      fastjet::PseudoJet vec = fastjet::PseudoJet((*part)->momentum().px(),
						  (*part)->momentum().py(),
						  (*part)->momentum().pz(),
						  (*part)->momentum().e());
      vec.set_user_index((*part)->pdg_id());
      particles.push_back(vec);
      ++part;
    }

    // Call the analysis
    Analysis(particles);

    // Print event
    //PrintEvent(particles);
    // Increment and check if have to stop
    event_count++;
    call_count++;
    if (event_count == nevents) break;
  }

  delete evt;

  return true;
}


void AnalyzerHepMC::Analysis(const PseudoJetVector particles)
{
  unsigned int nparticles = particles.size();

  double R(0.4);
  fastjet::JetDefinition jet_definition;
  jet_definition = fastjet::JetDefinition(fastjet::antikt_algorithm, R);
  fastjet::ClusterSequence cs(particles, jet_definition);
  double jet_ptmin(30.0);
  PseudoJetVector unsortedjets = cs.inclusive_jets(jet_ptmin);
  PseudoJetVector jets = fastjet::sorted_by_pt(unsortedjets);

  bool accept_event = false;
  if (jets.size() >= 2) accept_event = true;
  if (accept_event){
    // rapidity
    for(unsigned i=0; i<jets.size(); i++){
      if (abs(jets[i].rap()) > 4.4 ) accept_event = false;
    }
    // VBF mjj
    if (m_inv(jets[0],jets[1]) < 400.0 || abs(jets[0].rap()-jets[1].rap()) < 2.8) accept_event = false;
  }
  if (accept_event){

    double phi, theta, mass, mom;

    TH2D *pic= new TH2D("pic","pic",nr_theta, 0.0, fastjet::pi, nr_phi, 0.0, 2.0*fastjet::pi);

    //now returning final state partons

    for (unsigned int i=0; i<nparticles; i++){
      //fastjet::PseudoJet vec = fastjet::PseudoJet(particles[i]px(i), get_py(i), get_pz(i), get_E(i));
      //partons.push_back(vec);

      phi= particles[i].phi();

      mom= sqrt(pow(particles[i].E(),2)-pow(particles[i].m(),2));
      theta= acos(particles[i].pz()/mom);
      //cout<<"theta "<<theta<<" phi "<<phi<<" E "<<vec.E()<<endl;

      pic->Fill(theta,phi,particles[i].E() );
    }

    //std::pair bin_matrix;//[nr_theta][nr_phi];
    event_binned +=1;
    // 0 bin is underflow bin, therefore start at 1
    for (Int_t i=1; i<=nr_theta; i++){
        for (Int_t j=1; j<=nr_phi; j++){

	   entry.push_back(pic->GetBinContent(i,j));
        }

    }

    delete pic;
    pic=NULL;

  }
}


void AnalyzerHepMC::PrintEvent(const PseudoJetVector particles)
{
  std::cout.precision(5);
  std::cout.setf(std::ios::scientific, std::ios::floatfield);

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
  return;
}
