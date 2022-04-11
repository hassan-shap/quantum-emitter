import numpy as np
import matplotlib.pyplot as plt
import itertools
from pymatching import Matching
import networkx as nx
import time

repeat = 4
Nrep_loss = 2000 # number of iterations
Nrep_flip = 1 # number of iterations
L_list = [8,12,16,20]
prob_l = 0.2 # loss rate
pz_list = np.linspace(0.06,0.1,5)

for i_rep in np.arange(repeat):
    for i_L, r in enumerate(L_list):
        print("L= ", r, " rep= ", i_rep)
        fail_prob_z = np.zeros(len(pz_list))
        loss_prob = 0

        l = 2 # number of sublattice points (2 for toric code) or primal/dual
        r1 = r # number of columns
        r2 = r # number of rows

        tic = time.time()


        def does_loss_percolate(loss_inds):
            Gy = nx.Graph()
            Gy.add_nodes_from(np.arange(r1*r2))
            Gx = nx.Graph()
            Gx.add_nodes_from(np.arange(r1*r2))


            for i2 in range(r2):
                for i1 in range(r1):
                    ind1 = i2*r1+ i1
                    ind2 = i2*r1 + ((i1+1)%r1)
                    # cylinder along y
                    if 2*ind1 in loss_inds:
                        Gy.add_edge(ind1,ind2)
                    if ind1 +r1 < r1*r2 and 2*ind1+1 in loss_inds:
                        Gy.add_edge(ind1,ind1+r1)
                    # cylinder along x
                    ind2 = ((i2+1)%r2)*r1 + i1
                    if 2*ind1+1 in loss_inds:
                        Gx.add_edge(ind1,ind2)
                    if i1+1 < r1 and 2*ind1 in loss_inds:
                        Gx.add_edge(ind1,ind1+1)

            first_row = np.arange(r1)
            last_row = np.arange((r2-1)*r1,r2*r1)
            path_y = False
            for i_first in first_row:
                for i_last in last_row:
                    if nx.has_path(Gy,i_first,i_last):
                        if (i_first - i_last )%r1 ==0: # or 2*i_last+1 in loss_inds
                            path_y = True
                            break
                if path_y:
                    break

            first_col = np.arange(0,(r2-1)*r1+1,r1)
            last_col = np.arange(r1-1,r2*r1+1,r1)
            path_x = False
            for i_first in first_col:
                for i_last in last_col:
                    if nx.has_path(Gx,i_first,i_last):
                        if  int(i_first/r1) == int(i_last/r1): # or 2*i_last in loss_inds
                            path_x = True
                            break
                if path_x:
                    break

            for i in range(len(loss_inds)):
                ind1 = int(loss_inds[i]/2)
                if loss_inds[i] % 2 == 0 :
                    ind2 = int(int(loss_inds[i]/2)/r1)*r1 + (((int(loss_inds[i]/2)%r1)+1)%r1)
                    assert Gy.has_edge(ind1,ind2)
                    ind2 = ind1 + 1
                    if (int(loss_inds[i]/2)%r1)+1 < r1:
                        assert Gx.has_edge(ind1,ind2)     
                else:
                    ind2 = ind1 + r1
                    if ind2 < r1*r2:
                        assert Gy.has_edge(ind1,ind2)
                    ind2 = ((int(int(loss_inds[i]/2)/r1)+1)%r2)*r1 + (int(loss_inds[i]/2)%r1)
                    assert Gx.has_edge(ind1,ind2)
            return path_y,path_x

        def compute_eff_Sx(Sx,loss_inds,remain_inds):
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
                st_new_ind = np.argwhere(Sx_new[:,loss_index]>0)
                assert len(st_ind)+len(st_new_ind)==0

            num_Sx_red = len(inds_new)+len(inds_old)
            Sx_red = np.zeros((num_Sx_red,len(remain_inds)),dtype=int)
            Sx_red[0:len(inds_old),:] = Sx_old[np.ix_(inds_old,remain_inds)]
            if len(inds_new)>0:
                Sx_red[len(inds_old):,:] = Sx_new[:,remain_inds]

            keep_cols = np.argwhere(np.sum(Sx_red,axis=0)>0)[:,0]
            Sx_red = Sx_red[:,keep_cols]

            qubits_to_plot = remain_inds[keep_cols]
            return Sx_red, Sx_new, inds_old, inds_new, qubits_to_plot

        ##################
        def find_logical_ops(Sx,Sx_new,inds_old,inds_new,qubits_to_plot):
            num_qubits = len(qubits_to_plot)
            
            def horizontal_logic(i2):
                # i2 = int(r2/2)
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

                return logical_h
            logic_graph = []
            for i2 in range(r2):
                logical_h = horizontal_logic(i2)
                inds_logic_h = np.argwhere(logical_h[qubits_to_plot] > 0 )[:,0]
                Gq_x = nx.Graph()
                for qubit in qubits_to_plot[inds_logic_h]:
                    if qubit % 2 == 1:
                        if (int(qubit/2)%r1)-1 >= 0:
                            Gq_x.add_edge(int(qubit/2),int(qubit/2)-1)
                    else:
                        q2 = ((int(int(qubit/2)/r1)-1)%r2)*r1 + (int(qubit/2)%r1)
                        Gq_x.add_edge(int(qubit/2),q2)

                components = [Gq_x.subgraph(c).copy() for c in nx.connected_components(Gq_x)]
                if len(components)==2:
                    logic_graph = components[0].nodes()
                    break
            if len(components)!=2:
                print("could not find horz logical op")

            logical_qs = []
            for vertex in logic_graph:
                if 2*vertex in qubits_to_plot[inds_logic_h]:
                    logical_qs.append(2*vertex)
                if 2*vertex+1 in qubits_to_plot[inds_logic_h]:
                    logical_qs.append(2*vertex+1)
                v2 = int(vertex/r1)*r1 + ((vertex%r1)+1)%r1
                if 2*v2+1 in qubits_to_plot[inds_logic_h]:
                    logical_qs.append(2*v2+1)
                v2 = ((int(vertex/r1)+1)%r2)*r1 + (vertex%r1)
                if 2*v2 in qubits_to_plot[inds_logic_h]:
                    logical_qs.append(2*v2)

            logical_x = np.zeros(l*r1*r2)
            logical_x[logical_qs] = 1

            ################
            def vertical_logic(i1):
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
                used_inds = np.argwhere(i_new_used==True)[:,0]

                return logical_v
            logic_graph = []
            for i1 in range(r1):
                logical_v = vertical_logic(i1)
                inds_logic_v = np.argwhere(logical_v[qubits_to_plot] > 0 )[:,0]
                Gq_y = nx.Graph()
                for qubit in qubits_to_plot[inds_logic_v]:
                    if qubit % 2 == 0:
                        if int(qubit/2)-r1 >= 0:
                            Gq_y.add_edge(int(qubit/2),int(qubit/2)-r1)
                    else:
                        q2 = int(int(qubit/2)/r1)*r1 + ((int(qubit/2)%r1-1)%r1)
                        Gq_y.add_edge(int(qubit/2),q2)

                components = [Gq_y.subgraph(c).copy() for c in nx.connected_components(Gq_y)]
                if len(components)==2:
                    logic_graph = components[0].nodes()
                    break
            if len(components)!=2:
                print("could not find vert logical op")
           
            logical_qs = []
            for vertex in logic_graph:
                if 2*vertex in qubits_to_plot[inds_logic_v]:
                    logical_qs.append(2*vertex)
                if 2*vertex+1 in qubits_to_plot[inds_logic_v]:
                    logical_qs.append(2*vertex+1)
                v2 = int(vertex/r1)*r1 + ((vertex%r1)+1)%r1
                if 2*v2+1 in qubits_to_plot[inds_logic_v]:
                    logical_qs.append(2*v2+1)
                v2 = ((int(vertex/r1)+1)%r2)*r1 + (vertex%r1)
                if 2*v2 in qubits_to_plot[inds_logic_v]:
                    logical_qs.append(2*v2)

            logical_y = np.zeros(l*r1*r2)
            logical_y[logical_qs] = 1

            return logical_x, logical_y
           
        def netx_Sx(Sx_red,qubits_to_plot):
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
            # remain_qubits = remain_inds[keep_cols[inds_to_keep]]
            remain_qubits = qubits_to_plot[inds_to_keep]
            nl = np.array(nl)
            return Sx_red_netx,remain_qubits, nl

        ##################
        # star stabilzers
        Sx = np.zeros((r1*r2,l*r1*r2),dtype=int)
        for i_y in range(r2):
            for i_x in range(r1):
                Sx[i_y*r1 + i_x, 2*(i_y*r1 + i_x)] = 1
                Sx[i_y*r1 + i_x, 2*(i_y*r1 + i_x)+1] = 1
                Sx[i_y*r1 + i_x, 2*(i_y*r1+(i_x-1)%r1 )] = 1
                Sx[i_y*r1 + i_x, 2*(((i_y-1)%r1)*r1+i_x)+1] = 1

        for i_loss in range(Nrep_loss):

           ## loss error
            error_loss = np.random.rand(l*r1*r2) 
            loss_inds = np.argwhere(error_loss < prob_l)[:,0]
            remain_inds = np.argwhere(error_loss >= prob_l)[:,0]
            percolate_y, percolate_x = does_loss_percolate(loss_inds) 
            loss_percolate = (percolate_x or percolate_y)
            if loss_percolate:
                # fail_prob_z +=  1
                loss_prob +=  1
                continue
            error_loss[loss_inds] = 1
            error_loss[remain_inds] = 0

            Sx_red, Sx_new, inds_old, inds_new, qubits_to_plot = compute_eff_Sx(Sx,loss_inds,remain_inds)

            lost_qubits = np.array(list(set(np.arange(l*r1*r2)) - set(qubits_to_plot)))
            percolate_y, percolate_x = does_loss_percolate(lost_qubits)
            loss_percolate = (percolate_x or percolate_y)
            if loss_percolate:
                # fail_prob_z +=  1
                loss_prob +=  1
                continue

            logical_x, logical_y = find_logical_ops(Sx,Sx_new,inds_old,inds_new,qubits_to_plot)
            # logic_exist, logical_x, logical_y = find_logical_ops(qubits_to_plot)
            # if not logic_exist:
            #     fail_prob_z[i_L,i_p] +=  1
            #     print("how?")
            #     continue

            Sx_red_netx, remain_qubits, nl = netx_Sx(Sx_red,qubits_to_plot)
            num_edge = len(remain_qubits) 
            ################

            for i_p, prob_z in enumerate(pz_list):
                for i_n in range(Nrep_flip):

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
                        print("how?")
                        fail_prob_z[i_p] +=  1
                        continue

                    # find syndrome
                    syndrome_x = Sx_red_netx@error_z % 2
                    synd_x_inds = np.argwhere(syndrome_x > 0)
                    if len(synd_x_inds)>0:
                        rec2 = m.decode(syndrome_x)

                        error_rec = (rec2 + error_z )%2
                        s_h = np.dot( error_rec , logical_x[remain_qubits].T) %2 
                        s_v = np.dot( error_rec, logical_y[remain_qubits].T) %2 

                        assert np.sum(np.dot( error_rec , Sx_red_netx.T) % 2) == 0

                        ###########
                        if s_h + s_v  > 0:
                            fail_prob_z[i_p] +=  1

        toc = time.time()
        print("Finished in %d secs" % (toc-tic))
        fname = "data_loss_toric/" + "two_p_%.2f_L_%d_i_%d.npz" % (prob_l,r,i_rep)
        np.savez(fname, pz_list=pz_list, loss_prob=loss_prob, fail_prob_z=fail_prob_z, Nrep_loss=Nrep_loss, Nrep_flip=Nrep_flip)

    print("Done!")