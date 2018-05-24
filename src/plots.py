#!/usr/bin/env python
import numpy as np
#import pylab as P
import matplotlib.pyplot as plt
from matplotlib  import rc

def makeplot(ip,title, x, weights, ggf_size, y, y_NN,ggf_event_count, vbf_event_count):
    tot_size=len(y)
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')    
    
    x=np.array(x)
    y=np.array(y)
    

    #my_bins=[0.,10.,20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,210.,220.,230.,
             #240.,250.,260.,270.,280.,290.,300.,310.,320.,330.,340.,350.,360.,370.,380.,390.,400.,410.,420.,430.,440.,
             #450.,460.,470.,480.,490.,500.,510.,520.,530.,540.,550.,560.,570.,580.,590.,600.,610.,620.,630.,640.,650.,
             #660.,670.,680.,690.,700,710.,720.,730.,740.,750.,760.,770.,780.,790.,800.,810.,820.,830.,840.,850.,860.,
             #870.,880.,890.,900.,910.,920.,930.,940.,950.,960.,970.,980.,990.]
    my_bins_pt=[0.,20.,40.,60.,80.,100.,120.,140.,160.,180.,200.,220.,240.,260.,280.,300.,320.,340.,360.,380.,400.,420.,440.,
             460.,480.,500.,520.,540.,560.,580.,600.,620.,640.,660.,680.,700,720.,740.,760.,780.,800.,820.,840.,860.,
             880.,900.,920.,940.,960.,980.]
    my_bins_m=[0.,40.,80.,120.,160.,200.,240.,280.,320.,360.,400.,440.,480.,520.,560.,600.,640.,680.,720.,760.,800.,840.,880.,920.,960.,1000.,1040.,1080.,1120.,1160.,1200.,1240.,1280.,1320.,1360.,1400.,1440.,1480.,1520.,1560.,1600.,1640.,1680.,1720.,1760.,1800.,1840.,1880.,1920.,1960.]
    my_bins_y=[-4.5,-4.05,-3.6,-3.15,-2.7,-2.25,-1.8,-1.35,-0.9,-0.45,0.,0.45,0.9,1.35,1.8,2.25,2.7,3.15,3.6,4.05]
    my_bins_dphi=[0.0,0.157,0.314,0.471,0.627,0.785,0.942,1.1,1.257,1.414,1.571,1.728,1.885,2.042,2.199,2.356,2.513,2.67,2.827,2.984]
    if title.lower().find("p_")>=0:
        my_bins=my_bins_pt
        lowlimit=0.0
        uplimit=1000.0
    if title.lower().find("m_")>=0:
       my_bins=my_bins_m
       lowlimit=0.0
       uplimit=2000.0
    if title.lower().find("y_")>=0:
       my_bins=my_bins_y
       lowlimit=-4.5
       uplimit=4.5
    if title.lower().find("d ")>=0:
        my_bins=my_bins_dphi
        lowlimit=0.0
        uplimit=3.1415

    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    plt.subplots_adjust(hspace=0.6, bottom=0.1)
    ax1.set_title("GGF")
    ax1.set_yscale("log")
    ax1.set_xlim(lowlimit,uplimit)
    f.suptitle("$d\sigma/"+title+"$")
    #ax1.set_ylim(1E0,1E5)
    
    #x_ggf=x[0:ggf_size]
    #weights_ggf=np.ones(ggf_size)
    #weights_ggf=[w/ggf_event_count for w in weights[0:ggf_size]]
    
    x_ggf=[]
    x_vbf=[]
    weights_ggf=[]
    weights_vbf=[]
    #print "HIER", len(y),y.shape
    for event in range(len(y)):
        if y[event][0] == 0.0:
            #print 'ggf', weights[event]
            x_ggf.append(x[event])
            weights_ggf.append(weights[event]/ggf_event_count)
        else:
            x_vbf.append(x[event])
            weights_vbf.append(weights[event]/vbf_event_count)
    
    n_ggf, bins, patches = ax1.hist( x_ggf, bins=my_bins, weights=weights_ggf, histtype='bar', alpha=0.5)

    ax2.set_title("VBF")
    ax2.set_yscale("log")
    ax2.set_xlim(lowlimit,uplimit)
    #ax2.set_ylim(1E0,1E5)
    #x_vbf=x[ggf_size+1:tot_size]
    vbf_size=tot_size-ggf_size
    ##weights_vbf=[w / float(vbf_size) for w in weights[ggf_size+1:tot_size]]
    ##weights_vbf=np.ones(vbf_size+1)
    #weights_vbf=[w for w in weights[ggf_size+1:tot_size]]
    n_vbf, bins, patches = ax2.hist( x_vbf, bins=my_bins, weights=weights_vbf, histtype='bar', alpha=0.5)
    ##plt.show()

    xggf_rec=[]
    weights_ggf_rec=[]
    xvbf_rec=[]
    weights_vbf_rec=[]
    ggf_size_rec=0
    vbf_size_rec=0
    for i in range(len(y_NN)):
        if y_NN[i][0] == 0.0:
            xggf_rec.append(x[i])
            weights_ggf_rec.append(weights[i]/ggf_event_count)
            ggf_size_rec+=1
        else:
            xvbf_rec.append(x[i])
            weights_vbf_rec.append(weights[i]/vbf_event_count)
            vbf_size_rec+=1

    ax3.set_title("GGF REC")
    ax3.set_yscale("log")
    ax3.set_xlim(lowlimit,uplimit)
    #ax3.set_ylim(1E0,1E5)
    #weights_ggf_rec_norm=[w/float(ggf_size_rec) for w in weights_ggf_rec]
    #weights_ggf_rec_norm=np.ones(ggf_size_rec)
    n_ggf_rec, bins, patches = ax3.hist( xggf_rec, bins=my_bins, weights=weights_ggf_rec, histtype='bar', alpha=0.5)
    ##plt.show()

    ax4.set_title("VBF REC")
    ax4.set_yscale("log")
    ax4.set_xlim(lowlimit,uplimit)
    #ax4.set_ylim(1E0,1E5)
    #weights_vbf_rec_norm=[w/float(vbf_size_rec) for w in weights_vbf_rec]
    #weights_vbf_rec_norm=np.ones(vbf_size_rec)
    n_vbf_rec, bins, patches = ax4.hist( xvbf_rec, bins=my_bins, weights=weights_vbf_rec, histtype='bar', alpha=0.5)



    ratio_ggf=[]
    ratio_vbf=[]

    for i in range(len(bins)-1):
        if n_ggf_rec[i]!=0.0:
            ratio_ggf.append(n_ggf[i]/n_ggf_rec[i])
        else:
            ratio_ggf.append(0.0)
        if n_vbf_rec[i]!=0.0:
            ratio_vbf.append(n_vbf[i]/n_vbf_rec[i])
        else:
            ratio_vbf.append(0.0)


    #print "n_ggf"
    #print n_ggf
    print "integral ggf: ", sum(n_ggf)
    #print "sum ggf_weights", sum(weights_ggf)
    print "integral vbf: ", sum(n_vbf)
    print "integral ggf rec: ", sum(n_ggf_rec)
    print "integral vbf rec: ", sum(n_vbf_rec)
    #print "sum vbf_weights", sum(weights_vbf)
    #print "n_vbf"
    #print n_vbf
    #print "n_ggf_rec"
    #print n_ggf_rec
    #print "n_vbf_rec"
    #print n_vbf_rec
    #print "ratio_ggf"
    #print ratio_ggf
    #print "ratio_vbf"
    #print ratio_vbf
    ##print ratio_ggf
    ##print ratio_vbf
    ax5.set_title("Ratio GGF")
    ax5.set_xlim(lowlimit,uplimit)
    ax5.set_ylim(0.0,2.0)
    ##n_ggf_ratio, bins, patches = ax5.hist(ratio_ggf, bins=my_bins, histtype='step')
    ##print my_bins[:-1]
    ##print ratio_ggf
    ax5.plot(my_bins[:-1], ratio_ggf)

    ax6.set_title("Ratio VBF")
    ##ax6.set_xlim(0.0,1000.0)
    ax6.set_ylim(0.0,2.0)
    ##n_vbf_ratio, bins, patches = ax6.hist(ratio_vbf, bins=my_bins, histtype='step')
    ax6.plot(my_bins[:-1], ratio_vbf)

    plt.savefig('figures/Figure_'+str(ip.multip)+'_'+str(title)+'.png')
    #plt.show()

def make2dplot(ip, observables, data, ggf_size, tot_size):
    """
    Makes 2D scatter plots of data

    Arguments:
    id --
    observables --
    data --
    ggf_size --
    tot_size --

    Return:
    -
    """

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    obs_to_label={"pth": "$p_{t,\,\mathrm{H}}$",
                  "ptj1": "$p_{t,\,\mathrm{j}_{1}}$",
                  "ptj2": "$p_{t,\,\mathrm{j}_{1}}$",
                  "mjj": "$m_{j_{1}j_{2}}$",
                  "dphijj": "$d\phi_{j_{1}j_{2}}$",
                  "yj1": "$y_{j_{1}}$",
                  "yj2": "$y_{j_{2}}$",
                  "yj3": "$y_{j_{3}}$",
                  "yjj": "$y_{j_{1}j_{2}}$",
                  "zstar": "$z^{\star}$",
                  "zstar3j": "$z^{\star}_{3j}$",
                  "Rptjet": "$R_{p_{t,\,j}}$",
                  "Weight": "Weight"}

    for j in range(0, len(observables)):
        plt.figure(j+1, figsize=(15,10))
        k=0
        for i in range(0, len(observables)):
            if i == j:
                continue
            k+=1
            ax=plt.subplot(3,4,k)
            plt.plot(data[j, 0:ggf_size], data[i, 0:ggf_size], 'r.', label='GGF', markersize=1)
            plt.plot(data[j, ggf_size:tot_size], data[i, ggf_size:tot_size], 'b.', label='VBF', markersize=1)
            plt.xlabel(r""+obs_to_label[observables[j]]+"")
            plt.ylabel(r""+obs_to_label[observables[i]]+"")
            if observables[j] in ["pth", "ptj1", "ptj2", "mjj", "Weight"]:
                ax.set_xscale("log")
            if observables[i] in ["pth", "ptj1", "ptj2", "mjj", "Weight"]:
                ax.set_yscale("log")
            plt.title(r"Plot: "+obs_to_label[observables[j]]+" vs. "+obs_to_label[observables[i]])
            plt.legend(loc='best', ncol=1)
        plt.tight_layout()
        plt.savefig('figures/Figure_'+str(ip.multip)+'j_'+str(j+1)+'.png')
    #plt.show()
