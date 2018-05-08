#! /usr/bin/env python

import sys
import os
import numpy as np

__all__ = ['RootData']

class RootData:
    def __init__(self, multiplicity, runmode):
        self.multip = multiplicity
        self.runmod = runmode

        self.root_init()

        
    def root_init(self):
        """
        Initializes ROOT stuff and compiles ROOT libraries
        
        Arguments:
        parameters -- InputParameter type variable with command line input parameters
        
        Returns:
        -
        
        """
    
        global ROOT
        try:
            ROOT
            return
        except NameError:
            pass
        import ROOT

        # add NtupleAnalyzer directory to macro path
        try:
            ntupleanalyzer_path = os.path.abspath(os.path.dirname(__file__))
            if not ntupleanalyzer_path:
                print ('Problem with ntuple path')
                #ntupleanalyzer_path = parameters.sourcepath
                #if not ntupleanalyzer_path:
                #    raise ValueError('Empty path to NtupleAnalyzer source: add it to input with --sourcepath=<your_path>')
        except ValueError as e:
            print (e)
            sys.exit(2)
            
        ROOT.gROOT.SetMacroPath(ROOT.gROOT.GetMacroPath().rstrip(':') + ':' + ntupleanalyzer_path)
        ROOT.gSystem.AddIncludePath("-Wno-deprecated-declarations")
        
        ROOT.gSystem.Load("libRIO.so")
        ROOT.gSystem.Load("libTreePlayer.so")
        ROOT.gPluginMgr.AddHandler("TVirtualStreamerInfo", "*", "TStreamerInfo", "RIO", "TStreamerInfo()")
        ROOT.gPluginMgr.AddHandler("TVirtualTreePlayer", "*", "TTreePlayer", "TreePlayer", "TTreePlayer()");
        
        ROOT.gSystem.Load("libfastjet.so")
        ROOT.gROOT.LoadMacro("root/TSelectorMain.C+")
        ROOT.gROOT.LoadMacro("root/TSelectorAnalyzer.C+")
        ROOT.gROOT.LoadMacro("root/TSelectorReader.C+")

        # We want to handle Ctrl+C
        sh = ROOT.TSignalHandler(ROOT.kSigInterrupt, False)
        sh.Add()
        sh.Connect("Notified()", "TROOT", ROOT.gROOT, "SetInterrupt()")


    def load_data_obs(self, ip, gp, mode):
        """
            Loads data from ROOT NTuples both for training and for test sets
        
            Arguments:
            ip -- InputParameter type variable
            gp -- GlobalParameter type variable
            mode -- string defining the running more ( can be set equal to 'train' or 'test')
            
            Returns:
            x_vec -- numpy-array of shape (n_x, m) with input vectors: 
             m  : number of examples loaded from ntuples (which depends on the applied analysis cuts) 
            n_x : input dimension (for H+2j n_x = 11, H+3j n_x = 13)
            y_vec -- numpy-array of shape (2, m) with output vectors
             m  : number of examples loaded from ntuples (which depends on the applied analysis cuts)
             -> output is a 2-dim one-hot vector (1 0) =: ggf , (0 1) =: vbf
        """

        # Define reader selector:
        TReader = ROOT.TSelectorReader()
        
        # Analysis Selectors:
        AnalyzerSelector = ROOT.TSelectorAnalyzer()
        AnalyzerSelector.multip = ip.multip
        TReader.addSelector(AnalyzerSelector)
        AnalyzerSelector.runmode = 1
        
        prc_type = []
        obs_list = []
        
        # Define chain and add file list:
        chain = ROOT.TChain("t3")
        
        if mode == 'train':
            chain.Add(ip.GGFFILE_TRAIN)
        elif mode == 'devel':
            chain.Add(ip.GGFFILE_EVAL)
        else:
            raise ValueError("Not valid mode. Mode should be 'train' or 'devel'.")
    
        chain.GetFile()  # force opening of the first file
        chain.SetMaxEntryLoop(2**60)
        if ip.events < 0:
            chain.Process(TReader, "", chain.GetMaxEntryLoop(), 0)
        else:
            chain.Process(TReader, "", 10*int(ip.events), 0)
            
        ggf_size = int(AnalyzerSelector.event_binned)
        
        chain = ROOT.TChain("t3")
        if mode == 'train':    
            chain.Add(ip.VBFFILE_TRAIN)
        elif mode == 'devel':
            chain.Add(ip.VBFFILE_EVAL)
        else:
            raise ValueError("Not valid mode. Mode should be 'train' or 'devel'.")
            
        chain.GetFile()  # force opening of the first file
        chain.SetMaxEntryLoop(2**60)
        if ip.events < 0:
            chain.Process(TReader, "", chain.GetMaxEntryLoop(), 0)
        else:
            chain.Process(TReader, "", int(ip.events), 0)
            
        tot_size = int(AnalyzerSelector.event_binned)
        vbf_size = tot_size-ggf_size
        
        gp._num_examples = tot_size
            
        for i in range(ggf_size):
            prc_type.append([1.,0.])
        for i in range(vbf_size):
            prc_type.append([0.,1.])

        for i in range(tot_size):
                
            if ip.multip == 2:
                obs_list.append([AnalyzerSelector.pth[i],
                                 AnalyzerSelector.ptj1[i],
                                 AnalyzerSelector.ptj2[i],
                                 AnalyzerSelector.mjj[i],
                                 AnalyzerSelector.dphijj[i],
                                 AnalyzerSelector.yj1[i],
                                 AnalyzerSelector.yj2[i],
                                 AnalyzerSelector.yjj[i],
                                 AnalyzerSelector.zstar[i],
                                 AnalyzerSelector.Rptjet[i],
                                 AnalyzerSelector.me_weight[i]])
                
            if ip.multip == 3:
                obs_list.append([AnalyzerSelector.pth[i],
                                 AnalyzerSelector.ptj1[i],
                                 AnalyzerSelector.ptj2[i],
                                 AnalyzerSelector.mjj[i],
                                 AnalyzerSelector.dphijj[i],
                                 AnalyzerSelector.yj1[i],
                                 AnalyzerSelector.yj2[i],
                                 AnalyzerSelector.yj3[i],
                                 AnalyzerSelector.yjj[i],
                                 AnalyzerSelector.zstar[i],
                                 AnalyzerSelector.Rptjet[i],
                                 AnalyzerSelector.zstarj3[i],
                                 AnalyzerSelector.me_weight[i]])
                
        # So far all ggf data are first and all vbf data are after
        # Reshuffle data such that ggf and vbf events appear alternating:
        perm = np.arange(gp.num_examples())
        np.random.seed(1)
        np.random.shuffle(perm)
        x_vec = np.array(obs_list)[perm]
        y_vec = np.array(prc_type)[perm]
        
        return x_vec.T, y_vec.T, ggf_size, vbf_size
    
    def load_data_jets(self, ip, gp, mode):
        """
            Loads data from ROOT NTuples both for training and for test sets
        
            Arguments:
            ip -- InputParameter type variable
            gp -- GlobalParameter type variable
            mode -- string defining the running more ( can be set equal to 'train' or 'test')
            
            Returns:
            x_vec -- numpy-array of shape (n_x, m) with input vectors: 
             m  : number of examples loaded from ntuples (which depends on the applied analysis cuts) 
            n_x : input dimension (for H+2j n_x = 11, H+3j n_x = 13)
            y_vec -- numpy-array of shape (2, m) with output vectors
             m  : number of examples loaded from ntuples (which depends on the applied analysis cuts)
             -> output is a 2-dim one-hot vector (1 0) =: ggf , (0 1) =: vbf
        """

        # Define reader selector:
        TReader = ROOT.TSelectorReader()
        
        # Analysis Selectors:
        AnalyzerSelector = ROOT.TSelectorAnalyzer()
        AnalyzerSelector.multip = ip.multip
        TReader.addSelector(AnalyzerSelector)
        AnalyzerSelector.runmode = 2
        
        prc_type = []
        jet_list = []
        
        # Define chain and add file list:
        chain = ROOT.TChain("t3")
        
        if mode == 'train':
            chain.Add(ip.GGFFILE_TRAIN)
        elif mode == 'devel':
            chain.Add(ip.GGFFILE_EVAL)
        else:
            raise ValueError("Not valid mode. Mode should be 'train' or 'devel'.")
    
        chain.GetFile()  # force opening of the first file
        chain.SetMaxEntryLoop(2**60)
        if ip.events < 0:
            chain.Process(TReader, "", chain.GetMaxEntryLoop(), 0)
        else:
            chain.Process(TReader, "", 10*int(ip.events), 0)
            
        ggf_size = int(AnalyzerSelector.event_binned)
        
        chain = ROOT.TChain("t3")
        if mode == 'train':    
            chain.Add(ip.VBFFILE_TRAIN)
        elif mode == 'devel':
            chain.Add(ip.VBFFILE_EVAL)
        else:
            raise ValueError("Not valid mode. Mode should be 'train' or 'devel'.")
            
        chain.GetFile()  # force opening of the first file
        chain.SetMaxEntryLoop(2**60)
        if ip.events < 0:
            chain.Process(TReader, "", chain.GetMaxEntryLoop(), 0)
        else:
            chain.Process(TReader, "", int(ip.events), 0)
            
        tot_size = int(AnalyzerSelector.event_binned)
        vbf_size = tot_size-ggf_size
        
        gp._num_examples = tot_size
            
        for i in range(ggf_size):
            prc_type.append([1.,0.])
        for i in range(vbf_size):
            prc_type.append([0.,1.])

        p_per_event = 4*(ip.multip+1)
            
        for i in range(tot_size):
            
            mom_list=[]

            # For energy conservation check:
            Etot = 0 
            
            #print "check: no of events=%i, no of particles per event=12, no of entries=%s" % (tot_size, str(len(AnalyzerSelector.jetsvector)))

            # G.L 25.04.2018:
            # The following works only for born type events where n_jets = n_partons
            # (one possibility could be to fill with zeros the missing momenta and
            #  increase the multiplicity by one..)            
            for j in range(i*p_per_event,(i+1)*p_per_event):

                if j%4 == 0:
                    Etot += AnalyzerSelector.jetsvector[j]
                
                mom_list.append(AnalyzerSelector.jetsvector[j])
            #mom_list.append(AnalyzerSelector.mjj[i])
            #mom_list.append(AnalyzerSelector.pth[i])
            #mom_list.append(AnalyzerSelector.ptj1[i])
            #mom_list.append(AnalyzerSelector.ptj2[i])
            #print 'momlist', mom_list

            if Etot > 1E-10:
                raise ValueError("Energy not conserved in this event, something is wrong! Abort.")
            
            jet_list.append(mom_list)

        #print jet_list
        # So far all ggf data are first and all vbf data are after
        # Reshuffle data such that ggf and vbf events appear alternating:
        perm = np.arange(gp.num_examples())
        np.random.seed(1)
        np.random.shuffle(perm)
        x_vec = np.array(jet_list)[perm]
        y_vec = np.array(prc_type)[perm]
        
        return x_vec.T, y_vec.T, ggf_size, vbf_size
    

