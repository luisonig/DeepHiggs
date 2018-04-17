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
            
        ggf_size = len(AnalyzerSelector.pth)
        
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
            
        tot_size = len(AnalyzerSelector.pth)
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

