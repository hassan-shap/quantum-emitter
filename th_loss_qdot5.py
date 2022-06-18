import numpy as np
import matplotlib.pyplot as plt
import itertools
from pymatching import Matching
import networkx as nx
import time

repeat = 24
Nrep_loss = 1000 # number of iterations
L_list = [8,10,12,14,16]
prob_l = 0.01 # loss rate
p1_list = np.linspace(4.2e-3,6.2e-3,8) # smaller than 0.18
l = 3 # number of links per node


from joblib import Parallel, delayed
import multiprocessing
# what are your inputs, and what operation do you want to
# perform on each input. For example...
num_cores = 12#multiprocessing.cpu_count()                                     

for r in L_list:    
    print("L= %d" % (r))
    def runner(i_rep):
        fail_prob_z = np.zeros(len(p1_list))

        r1 = r # dimension of cube
        r2 = r # dimension of cube
        r3 = r # dimension of cube
        logicals = np.zeros((3,l*r1*r2*r3))
        ## x ops
        for i1 in range(r2):
            logicals[0,np.ix_(3*np.arange(i1*r1,r1*r2*r3,r1*r2))] = np.ones(r3)
        for i1 in range(r3):
            logicals[1,np.ix_(1+ 3*(i1*r1*r2+ np.arange(0,r1) ) )] = np.ones(r1)
        logicals[2,2:3*r1*r2+1:3] = np.ones(r1*r2) 

        def compute_eff_Sx(Sx,loss_inds,remain_inds):
            G_loss = nx.Graph()
            for loss_index in loss_inds:
                if loss_index%l ==0:
                    x2 = (((int(loss_index/3)%(r1*r2))%r1) +1 ) % r1
                    y2 = int((int(loss_index/3)%(r1*r2))/r1)
                    z2 = int(int(loss_index/3)/(r1*r2))
                elif loss_index%l == 1:
                    x2 = ((int(loss_index/3)%(r1*r2))%r1) 
                    y2 = (int((int(loss_index/3)%(r1*r2))/r1)+1)%r2
                    z2 = int(int(loss_index/3)/(r1*r2))            
                else:
                    x2 = ((int(loss_index/3)%(r1*r2))%r1) 
                    y2 = int((int(loss_index/3)%(r1*r2))/r1)
                    z2 = (int(int(loss_index/3)/(r1*r2))+1)%r3
                q2 = (z2*r2+y2)*r1 + x2
                G_loss.add_edge(int(loss_index/l),q2)

            components = [G_loss.subgraph(c).copy() for c in nx.connected_components(G_loss)]
            lost_vs = []
            for i_c, c in enumerate(components):
                lost_vs += c.nodes()

            remain_vs = list(set(range(r1*r2*r3)) - set(lost_vs))
            num_stab = len(components)+len(remain_vs)
            Sx_red2 = np.zeros((num_stab,len(remain_inds)))
            Sx_red2[len(components):,:] = Sx[np.ix_(remain_vs,remain_inds)]
            for i_c, c in enumerate(components):
                Sx_red2[i_c,:] = np.sum(Sx[np.ix_(c.nodes(),remain_inds)],axis = 0)%2

            keep_cols = np.argwhere(np.sum(Sx_red2,axis=0)>0)[:,0]
            Sx_red2 = Sx_red2[:,keep_cols]
            qubits_to_plot = remain_inds[keep_cols]

            return Sx_red2, qubits_to_plot

        ##################
        def netx_Sx(Sx_red,overlap,qubits_to_plot):
            inds = np.argwhere(overlap>1)
            rep_edges = []
            for i_v in inds:
                if i_v[1]>i_v[0]:
                    if not (i_v[0] in rep_edges):
                        rep_edges.append(i_v[0])
                    if not (i_v[1] in rep_edges):
                        rep_edges.append(i_v[1])

            rep_edges = np.sort(rep_edges).astype(int)
            not_rep_qubits = np.array(list(set(range(np.size(Sx_red,1))) - set(rep_edges)))
            num_not_rep = len(not_rep_qubits)

            ql2 = []
            nl2 = []
            nl_x2 = []
            nl_y2 = []
            counter = 0
            i = 0 
            overlap2 = overlap[np.ix_(rep_edges,rep_edges)]
            inds_to_keep2 = list(range(len(rep_edges)))
            while counter < len(rep_edges):
                edge = inds_to_keep2[i]
                ovlp_inds = np.argwhere(overlap2[edge,inds_to_keep2[i+1:]]==2)
                if qubits_to_plot[rep_edges[edge]] %l ==0:
                    nl_i_x = 1
                    nl_i_y = 0 
                else:
                    nl_i_x = 0 
                    nl_i_y = 1
                nl_i = len(ovlp_inds)+1

                qlist = qubits_to_plot[rep_edges[np.ix_([ inds_to_keep2[k] for k in i+1+ovlp_inds[:,0]])]]
                ovlp_inds_x = np.argwhere(qlist %l==0)
                ovlp_inds_y = np.argwhere(qlist %l >0)
                nl_i_x += len(ovlp_inds_x)
                nl_i_y += len(ovlp_inds_y)
                for j in ovlp_inds[::-1,0]:
                    inds_to_keep2.remove(inds_to_keep2[i+1+j])
                ql2.append(np.concatenate(([qubits_to_plot[rep_edges[edge]]],qlist)))
                counter += nl_i
                nl2.append(nl_i)
                nl_x2.append(nl_i_x)
                nl_y2.append(nl_i_y)
                i += 1

            num_rep = len(inds_to_keep2)
            nl_x_tot = np.zeros(num_rep+num_not_rep)
            nl_x_tot[ np.argwhere(qubits_to_plot[not_rep_qubits]%l ==0)  ] = 1 
            nl_x_tot[len(not_rep_qubits):] = nl_x2

            nl_y_tot = np.zeros(num_rep+num_not_rep)
            nl_y_tot[ np.argwhere(qubits_to_plot[not_rep_qubits]%l >0)  ] = 1 
            nl_y_tot[num_not_rep:] = nl_y2

            nl_tot = np.concatenate((np.ones(num_not_rep),nl2))

            inds_to_keep2 = rep_edges[inds_to_keep2]
            comb_inds = np.concatenate((not_rep_qubits,inds_to_keep2))
            inds_sorted = np.argsort(comb_inds)
            inds_to_keep2 = comb_inds[inds_sorted]
            nl_x_tot = nl_x_tot[inds_sorted]
            nl_y_tot = nl_y_tot[inds_sorted]
            nl_tot = nl_tot[inds_sorted]
            remain_qubits = qubits_to_plot[inds_to_keep2]
            # Sx_red_netx = Sx_red[:,inds_to_keep]

            ql_tot = []
            rep_count = 0
            for i in inds_sorted:
                if i< num_not_rep:
                    ql_tot.append([qubits_to_plot[not_rep_qubits[i]]])
                else:
                    ql_tot.append(ql2[rep_count][:])
                    rep_count += 1

            return remain_qubits, inds_to_keep2, ql_tot, nl_x_tot, nl_y_tot
    #             def netx_Sx(Sx_red,overlap,qubits_to_plot):
    #                 inds_to_keep = list(range(np.size(Sx_red,1)))
    #                 ql = []
    #                 nl = []
    #                 nl_x = []
    #                 nl_y = []
    #                 counter = 0
    #                 i = 0 
    #                 while counter < np.size(Sx_red,1):
    #                     edge = inds_to_keep[i]
    #                     ovlp_inds = np.argwhere(overlap[edge,inds_to_keep[i+1:]]==2)
    #                     if qubits_to_plot[edge] %l ==0:
    #                         nl_i_x = 1
    #                         nl_i_y = 0 
    #                     else:
    #                         nl_i_x = 0 
    #                         nl_i_y = 1
    #                     nl_i = len(ovlp_inds)+1

    #                     if len(ovlp_inds)>0:
    #                         qlist = qubits_to_plot[np.ix_([ inds_to_keep[k] for k in i+1+ovlp_inds[:,0]])]
    #                         ovlp_inds_x = np.argwhere(qlist %l==0)
    #                         ovlp_inds_y = np.argwhere(qlist %l >0)
    #                         nl_i_x += len(ovlp_inds_x)
    #                         nl_i_y += len(ovlp_inds_y)
    #                         for j in ovlp_inds[::-1,0]:
    #                             inds_to_keep.remove(inds_to_keep[i+1+j])
    #                         ql.append(np.concatenate(([qubits_to_plot[edge]],qlist)))
    #                     else:
    #                         ql.append([qubits_to_plot[edge]])

    #                     counter += nl_i
    #                     nl.append(nl_i)
    #                     nl_x.append(nl_i_x)
    #                     nl_y.append(nl_i_y)
    #                     i += 1

    #                 Sx_red_netx = Sx_red[:,inds_to_keep]
    #                 remain_qubits = qubits_to_plot[inds_to_keep]
    #                 nl = np.array(nl)
    #                 nl_x = np.array(nl_x)
    #                 nl_y = np.array(nl_y)

    #                 return remain_qubits, inds_to_keep, ql, nl_x, nl_y

        ##################
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

        tic = time.time()

        for i_loss in range(Nrep_loss):

           ## loss error
            error_loss = np.random.rand(l*r1*r2*r3) 
            loss_inds = np.argwhere(error_loss < prob_l)[:,0]
            remain_inds = np.argwhere(error_loss >= prob_l)[:,0]

            Sx_red, qubits_to_plot = compute_eff_Sx(Sx,loss_inds,remain_inds)

            overlap = Sx_red.T@Sx_red
            remain_qubits, inds_to_keep, ql, nl_x, nl_y = netx_Sx(Sx_red,overlap,qubits_to_plot)
            num_edge = len(remain_qubits) 
            ################
            for i_p, p1 in enumerate(p1_list):
                # z flip error
                p2 = p1
                prob_x_axis = (2+2/3)*p1 + 40/15*p2
                prob_y_axis = 2*p1 + 32/15*p2
                prob_z_axis = prob_y_axis
                p2_x = 12/15*p2
                p2_y = 4/15*p2
                p2_z = p2_y

                pl = (1-(1-2*prob_x_axis)**nl_x * (1-2*prob_y_axis)**nl_y)/2
                # ########## weights on square lattice ############
                weights = np.zeros(l*r1*r2*r3)
                weights[remain_qubits] = np.log((1-pl)/pl) 

                inds_to_keep_2 = list(range(np.size(Sx_red,1)))
                for i in range(num_edge):
                    edge = inds_to_keep[i]
                    ovlp_inds = np.argwhere(overlap[edge,inds_to_keep_2[i+1:]]==2)
                    if len(ovlp_inds)>0:
                        for j in ovlp_inds[::-1,0]:
                            weights[qubits_to_plot[inds_to_keep_2[i+1+j]]] = weights[qubits_to_plot[edge]]

                assert len(np.argwhere(weights>0))== len(qubits_to_plot)

                if num_edge > 1:
                    m_orig = Matching(Sx,spacelike_weights=weights)
                else:
                    print("percolate")
                    fail_prob_z[i_p] +=  1
                    continue

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

                error_z = np.zeros(num_edge)
                for i_q, qubits in enumerate(ql):
                    error_z[i_q] = np.sum(error_tot[qubits])%2

                zflip_inds = np.argwhere(error_z > 0)[:,0]
                error_z_orig = np.zeros(l*r1*r2*r3,dtype=int)
                error_z_orig[remain_qubits[zflip_inds]] = 1

                # find syndrome
                syndrome_x_orig = (Sx@error_z_orig) % 2
                synd_x_inds = np.argwhere(syndrome_x_orig > 0)
                if len(synd_x_inds)>0:
                    rec2_orig = m_orig.decode(syndrome_x_orig)
                    rec2_orig_inds = np.argwhere(rec2_orig > 0)[:,0]

                    error_rec_orig = (rec2_orig + error_z_orig )%2
                    s_orig = np.dot( error_rec_orig , logicals.T) %2 
                    if np.sum(s_orig)  > 0:
                        fail_prob_z[i_p] +=  1

        toc = time.time()
        print("Finished L= %d, r=%d in %d secs" % (r,i_rep,toc-tic))
        fname = "draft_loss_qdot/" + "p1_eq_p2_p_%.2f_L_%d_i_%d.npz" % (prob_l,r,i_rep)
        np.savez(fname, p1_list=p1_list, fail_prob_z=fail_prob_z, Nrep_loss=Nrep_loss)

        return 0
    
    results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(repeat))


