import numpy as np
import matplotlib.pyplot as plt
import itertools
from pymatching import Matching
import time


l = 3 # number of sublattice points (2 for toric code) or primal/dual
Nrep = 1000 # number of iterations
repeat = 4
L_list = [8,10,12,14]
p1_list = np.linspace(0.003,0.007,10)

for i_rep in np.arange(4,4+repeat):
    for i_L, r in enumerate(L_list):
        
        tic = time.time()
        print("L= ", r, " rep= ", i_rep)
        fail_prob_z = np.zeros(len(p1_list))

        r1 = r # dimension of cube
        r2 = r # dimension of cube
        r3 = r # dimension of cube

        # star stabilzers
        Sx = np.zeros((r1*r2*r3,l*r1*r2*r3))
        for ix in range(r1):
            for iy in range(r2):
                for iz in range(r3):
                    Sx[ix + r1*(iy+ r2*iz), l*(ix + r1*(iy+ r2*iz))] = 1
                    Sx[ix + r1*(iy+ r2*iz), l*(ix + r1*(iy+ r2*iz))+1] = 1
                    Sx[ix + r1*(iy+ r2*iz), l*(ix + r1*(iy+ r2*iz))+2] = 1
                    Sx[ix + r1*(iy+ r2*iz), l*( ((ix-1)%r1) + r1*(iy+ r2*iz))] = 1
                    Sx[ix + r1*(iy+ r2*iz), l*(ix + r1*( ((iy-1)%r2)+ r2*iz) )+1] = 1
                    Sx[ix + r1*(iy+ r2*iz), l*(ix + r1*(iy+ r2* ((iz-1)%r3)) )+2] = 1

        logicals = np.zeros((3,l*r1*r2*r3))
        ## x ops
        for i1 in range(r2):
            logicals[0,np.ix_(3*np.arange(i1*r1,r1*r2*r3,r1*r2))] = np.ones(r3)
        for i1 in range(r3):
            logicals[1,np.ix_(1+ 3*(i1*r1*r2+ np.arange(0,r1) ) )] = np.ones(r1)
        logicals[2,2:3*r1*r2+1:3] = np.ones(r1*r2) 

        m_orig = Matching(Sx)


        # print(np.dot(logicals,logicals.T))
        for i_p, p1 in enumerate(p1_list):
            # z flip error
            p2 = p1
            prob_x_axis = (2+2/3)*p1 + 40/15*p2
            prob_y_axis = 2*p1 + 32/15*p2
            prob_z_axis = prob_y_axis
            p2_x = 12/15*p2
            p2_y = 4/15*p2
            p2_z = p2_y
            
            for i_n in range(Nrep):
                
                error_tot = np.zeros(l*r1*r2*r3,dtype=int)

                error_x = np.random.rand(r1*r2*r3) 
                zflip_x_inds = np.argwhere(error_x < prob_x_axis)
                error_tot[3*zflip_x_inds] = 1

                error_y = np.random.rand(r1*r2*r3) 
                zflip_y_inds = np.argwhere(error_y < prob_y_axis)
                error_tot[3*zflip_y_inds+1] = 1

                error_z = np.random.rand(r1*r2*r3) 
                zflip_z_inds = np.argwhere(error_z < prob_z_axis)
                error_tot[3*zflip_z_inds+2] = 1

                corr_err_x_z = np.random.rand(r1*r2*r3)
                zflip_x_inds = np.argwhere(corr_err_x_z < p2_x)
                x2 = ((zflip_x_inds%(r1*r2))%r1) 
                y2 = np.floor((zflip_x_inds%(r1*r2))/r1)
                z2 = (np.floor(zflip_x_inds/(r1*r2))+1)%r3
                q2 = (z2*r2+y2)*r1 + x2
                error_tot[3*zflip_x_inds] += 1
                error_tot[3*q2.astype(int)] += 1

                corr_err_x_y = np.random.rand(r1*r2*r3) 
                zflip_x_inds = np.argwhere(corr_err_x_y < p2_x)
                x2 = ((zflip_x_inds%(r1*r2))%r1) 
                y2 = (np.floor((zflip_x_inds%(r1*r2))/r1)+1)%r2
                z2 = np.floor(zflip_x_inds/(r1*r2))       
                q2 = (z2*r2+y2)*r1 + x2
                error_tot[3*zflip_x_inds] += 1
                error_tot[3*q2.astype(int)] += 1

                corr_err_y_z = np.random.rand(r1*r2*r3) 
                zflip_y_inds = np.argwhere(corr_err_y_z < p2_y)
                x2 = ((zflip_y_inds%(r1*r2))%r1) 
                y2 = np.floor((zflip_y_inds%(r1*r2))/r1)
                z2 = (np.floor(zflip_y_inds/(r1*r2))+1)%r3
                q2 = (z2*r2+y2)*r1 + x2
                error_tot[3*zflip_y_inds+1] += 1
                error_tot[3*q2.astype(int)+1] += 1

                corr_err_z_y = np.random.rand(r1*r2*r3) 
                zflip_z_inds = np.argwhere(corr_err_z_y < p2_z)
                x2 = ((zflip_z_inds%(r1*r2))%r1) 
                y2 = (np.floor((zflip_z_inds%(r1*r2))/r1)+1)%r2
                z2 = np.floor(zflip_z_inds/(r1*r2))       
                q2 = (z2*r2+y2)*r1 + x2
                error_tot[3*zflip_z_inds+2] += 1
                error_tot[3*q2.astype(int)+2] += 1

                error_tot %= 2 
                # find syndrome
                syndrome_x = np.dot(error_tot,Sx.T) % 2
                # solve decoder
                rec_x = m_orig.decode(syndrome_x)
                np.dot( (rec_x + error_tot )%2 , logicals.T) % 2

                assert np.sum(np.dot( (rec_x + error_tot )%2 , Sx.T) % 2) == 0

                ###########
                if np.sum(np.dot( (rec_x + error_tot )%2, logicals.T) %2)  > 0:
                    fail_prob_z[i_p] +=  1
        toc = time.time()
        print("Finished in %d secs" % (toc-tic))
        fname = "data_qdot/" + "p1_eq_p2_L_%d_i_%d.npz" % (r,i_rep)
        np.savez(fname, p1_list=p1_list, fail_prob_z=fail_prob_z, Nrep=Nrep)

print("Done!")