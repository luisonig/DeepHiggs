#!/usr/bin/env python

import numpy as np

def compute_XS(x_devel, y_devel, ggf_size, vbf_size, ggf_event_count, vbf_event_count):

    #print x_devel.shape
    #print y_devel.shape

    #print("ggf_size %i " % ggf_size)
    #print("vbf_size %i " % vbf_size)

    tot_xs_ggf = 0
    tot_xs_vbf = 0

    #print x_devel.transpose()[0][10]
    
    for event in range(y_devel.shape[1]):

        if y_devel.transpose()[event][0] == 1.0:
            # this is a ggf event
            tot_xs_ggf += x_devel.transpose()[event][-1]
        else:
            # this is a vbf event
            tot_xs_vbf += x_devel.transpose()[event][-1]

    
    tot_xs_ggf /= ggf_event_count
    tot_xs_vbf /= vbf_event_count

    print("GGF XS: %s" % str(tot_xs_ggf))
    print("VBF XS: %s" % str(tot_xs_vbf))
    
    return tot_xs_ggf, tot_xs_vbf
