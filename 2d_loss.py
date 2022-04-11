import numpy as np
from pymatching import Matching
import time

repeat = 4
Nrep = 4000 # number of iterations
L_list = [6,8,10]
prob_l = 0.2 # loss rate
pz_list = np.linspace(0.06,0.1,10)
fail_prob_z = np.zeros((len(L_list),len(pz_list)))

for i_rep in range(repeat):
    for i_L, r in enumerate(L_list):
        tic = time.time()
        print("L= ", r, " rep= ", i_rep)
        fail_prob_z = np.zeros(len(pz_list))

        l = 2 # number of sublattice points (2 for toric code) or primal/dual
    #     r = 5 # number of columns
        r1 = r # number of rows
        r2 = r # number of rows

      # star stabilzers
        Sx = np.zeros((r1*r2,l*r1*r2),dtype=int)
        for i_y in range(r2):
            for i_x in range(r1):
                Sx[i_y*r1 + i_x, 2*(i_y*r1 + i_x)] = 1
                Sx[i_y*r1 + i_x, 2*(i_y*r1 + i_x)+1] = 1
                Sx[i_y*r1 + i_x, 2*(i_y*r1+(i_x-1)%r1 )] = 1
                Sx[i_y*r1 + i_x, 2*(((i_y-1)%r1)*r1+i_x)+1] = 1

        for i_p, prob_z in enumerate(pz_list):
            for i_n in range(Nrep):
                   # # loss error
                error_loss = np.random.rand(l*r1*r2) 
                loss_inds = np.argwhere(error_loss < prob_l)[:,0]
                remain_inds = np.argwhere(error_loss >= prob_l)[:,0]
                error_loss[loss_inds] = 1
                error_loss[remain_inds] = 0

                Sx_new = []
                inds_new = []
                Sx_old = np.copy(Sx)
                inds_old = list(range(r1*r2))
                for loss_index in loss_inds:
                    st_ind = np.argwhere(Sx_old[:,loss_index]>0)
                    st_ind = list(st_ind[:,0])
                    if len(st_ind)==2:
                        inds_new.append(st_ind)
                        Stot = np.zeros(l*r1*r2)
                        for i_remove in st_ind:
                            inds_old.remove(i_remove)
                            Stot += Sx_old[i_remove,:]
                        Sx_new.append(list(Stot%2)) 
                        Sx_old[st_ind,:] = 0
                    elif len(st_ind)==1:
                        st_new_ind = np.argwhere(np.array(Sx_new)[:,loss_index]>0)[0,0]
                        inds_new[st_new_ind][:] += st_ind
                        inds_old.remove(st_ind[0])
                        Sx_new[st_new_ind] = list((np.array(Sx_new)[st_new_ind,:]+Sx_old[st_ind[0],:]) %2)
                        Sx_old[st_ind,:] = 0
                    else:
                        st_new_ind = np.argwhere(np.array(Sx_new)[:,loss_index]>0)
                        if len(st_new_ind)>1:
                            st_new_ind = list(st_new_ind[:,0])
                            inds_new[st_new_ind[0]][:] += inds_new[st_new_ind[1]][:]
                            Stot = np.zeros(l*r1*r2)
                            for i_remove in st_new_ind:
                                Stot += np.array(Sx_new)[i_remove,:]        
                            Sx_new[st_new_ind[0]] = list(Stot%2)
                            del inds_new[st_new_ind[1]]
                            del Sx_new[st_new_ind[1]]
                Sx_new = np.array(Sx_new, dtype=int)
                for loss_index in loss_inds:
                    st_ind = np.argwhere(Sx_old[:,loss_index]>0)
                    st_new_ind = np.argwhere(Sx_old[:,loss_index]>0)
                    assert len(st_ind)+len(st_new_ind)==0

                num_Sx_red = len(inds_new)+len(inds_old)
                Sx_red = np.zeros((num_Sx_red,len(remain_inds)),dtype=int)
                # print(np.shape(Sx_red[0:len(inds_old),:]),np.shape(Sx_old[np.ix_(inds_old,remain_inds)]))
                Sx_red[0:len(inds_old),:] = Sx_old[np.ix_(inds_old,remain_inds)]
                if len(inds_new)>0:
                    Sx_red[len(inds_old):,:] = Sx_new[:,remain_inds]

                keep_cols = np.argwhere(np.sum(Sx_red,axis=0)>0)[:,0]
                Sx_red = Sx_red[:,keep_cols]

                # print(np.shape(Sx_red))
                # print(Sx_red.T@Sx_red)

                overlap = Sx_red.T@Sx_red
                inds_to_keep = list(range(np.size(Sx_red,1)))
                nl = []
                counter = 0
                i = 0 
                while counter < np.size(Sx_red,1):
                    edge = inds_to_keep[i]
                    ovlp_inds = np.argwhere(overlap[edge,inds_to_keep[i+1:]]==2)
                    if len(ovlp_inds)>0:
                        # print("edge= ", edge)
                        for j in ovlp_inds[::-1,0]:
                            # print(inds_to_keep[i+1+j])
                            inds_to_keep.remove(inds_to_keep[i+1+j])
                        # print(inds_to_keep)
                        counter += (len(ovlp_inds)+1)
                        nl.append(len(ovlp_inds)+1)
                    else:
                        counter += 1
                        nl.append(1)
                    i += 1

                Sx_red_netx = Sx_red[:,inds_to_keep]
                remain_qubits = remain_inds[keep_cols[inds_to_keep]]
                num_edge = len(remain_qubits) #np.size(Sx_red_netx,1)
                nl = np.array(nl)


                ## z flip error
                pl = (1-(1-2*prob_z)**nl)/2
                error_table = np.random.rand(num_edge) < pl
                zflip_inds = np.argwhere(error_table == True)[:,0]
                no_zflip_inds = np.argwhere(error_table == False)[:,0]
                error_z = np.zeros(num_edge,dtype=int)
                error_z[zflip_inds] = 1
                error_z[no_zflip_inds] = 0

                if num_edge > 1:
                    m = Matching(Sx_red_netx,spacelike_weights=np.log((1-pl)/pl))
                else:
                    continue

                # find syndrome
                syndrome_x = Sx_red_netx@error_z % 2
                synd_x_inds = np.argwhere(syndrome_x > 0)
                if len(synd_x_inds)>0:
                    rec2 = m.decode(syndrome_x)
                else:
                    continue

                i2 = int(r2/2)
                i_h = (i2*r1+np.arange(r1))
                logical_h = np.zeros(l*r1*r2)
                i_new_used = np.zeros(len(inds_new),dtype = bool)
                for h_pos in i_h:
                    if h_pos in inds_old:
                        logical_h += Sx[h_pos,:] 
                    else:   
                        for i_new in range(len(inds_new)):
                            if h_pos in inds_new[i_new] and i_new_used[i_new]==False:
                                logical_h += Sx_new[i_new,:] 
                                i_new_used[i_new] = True

                logical_h = (logical_h% 2)
                ind_logic = np.argwhere(logical_h>0)[:,0]

                for i in range(len(ind_logic)):
                    latt_pos = int(ind_logic[i]/2)
                    if ind_logic[i] % 2 == 0 :
                        ry = int(latt_pos/r1) 
                    else:
                        ry = int(latt_pos/r1)  + 0.5 #+0.1
                    if ry>= i2:
                        logical_h[ind_logic[i]] = 0

                i1 = int(r1/2)
                i_v = (i1+r1*np.arange(r2))
                logical_v = np.zeros(l*r1*r2)
                i_new_used = np.zeros(len(inds_new),dtype = bool)
                for v_pos in i_v:
                    if v_pos in inds_old:
                        logical_v += Sx[v_pos,:] 
                    else:   
                        for i_new in range(len(inds_new)):
                            if v_pos in inds_new[i_new] and i_new_used[i_new]==False:
                                logical_v += Sx_new[i_new,:] 
                                i_new_used[i_new] = True

                logical_v = (logical_v% 2)
                ind_logic = np.argwhere(logical_v>0)[:,0]

                for i in range(len(ind_logic)):
                    latt_pos = int(ind_logic[i]/2)
                    if ind_logic[i] % 2 == 0 :
                        rx = (latt_pos%r1) + 0.5   
                    else:
                        rx = (latt_pos%r1)   
                    if rx>= i1:
                        logical_v[ind_logic[i]] = 0

                error_rec = (rec2 + error_z )%2
                s_h = np.dot( error_rec , logical_h[remain_qubits].T) %2 
                s_v = np.dot( error_rec, logical_v[remain_qubits].T) %2 

                assert np.sum(np.dot( error_rec , Sx_red_netx.T) % 2) == 0

                ###########
                if s_h + s_v  > 0:
                    fail_prob_z[i_p] +=  1
        
        toc = time.time()
        print("Finished in %d secs" % (toc-tic))
        fname = "data_2d_loss/" + "loss_%.2f_L_%d_i_%d.npz" % (prob_l,r,i_rep)
        np.savez(fname, pz_list=pz_list, fail_prob_z=fail_prob_z, Nrep=Nrep)


print("Done!")
