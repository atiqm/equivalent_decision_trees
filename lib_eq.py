#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:23:02 2019

@author: mounir
"""
import numpy as np
import copy
from sklearn.tree import DecisionTreeClassifier
import lib_tree



def coherent_new_split(phi,th,rule):

    inrule, sense = lib_tree.isinrule(rule,(phi,th))

    if inrule:
        return 0,sense
    
    feats, ths, bools = rule
    
    if phi not in feats:
        return 1,0
    else:
        if np.sum((feats == phi)*(bools==-1)) != 0:
            max_th = np.amin(ths[(feats == phi)*(bools==-1)])
        else:
            max_th = np.inf
        
        if np.sum((feats == phi)*(bools==1)) != 0:
            min_th = np.amax(ths[(feats == phi)*(bools==1)])
        else:
            min_th = - np.inf
        
        if th >= max_th :
            return 0,-1
        elif th <= min_th:
            return 0,1
        else:
            return 1,0
      
def all_coherent_splits(rule,all_splits):

    inds = np.zeros(all_splits.shape[0],dtype=bool)
    splits = copy.copy(all_splits)
    for c,split in enumerate(all_splits):
        phi, th = split
        coh,sense = coherent_new_split(phi,th,rule)
        if coh:
            inds[c] = 1
       
    return splits[inds]

def new_random_split(p,all_splits):
    inds = np.arange(0,all_splits.shape[0])
    ind = int(np.random.choice(inds,p=p))
    return all_splits[ind]


def filter_feature(splits,feats):
    positive_splits = list()
    negative_splits = list()
    
    for s in splits :
        phi,th = s
        if phi in feats:
            positive_splits.append(s)
        else:
            negative_splits.append(s)
            
    return np.array(positive_splits), np.array(negative_splits)

# =============================================================================
# 
# =============================================================================
def liste_non_coherent_splits(dtree,rule,node=0):
    
    indexes_subtree = lib_tree.sub_nodes(dtree.tree_,node)
    
    phis = list()
    ths = list()
    b = list()
    indexes = list()
    
    for n in indexes_subtree:
        phi,th = dtree.tree_.feature[n],dtree.tree_.threshold[n]
        coh,non_coherent_sense = coherent_new_split(phi,th,rule)
        if not coh :
            phis.append(phi)
            ths.append(th)
            b.append(non_coherent_sense)
            indexes.append(n)
            
    return indexes,phis,ths,b

def ForceCoherence(dtree,rule,node=0,Translate=False,indexes_nodes=list(),drifts=list()):
    
    if Translate :
        if len(indexes_nodes) != len(drifts):
            print('Error in parameter size for drifts')
            return node
        else:
            for k,n in enumerate(indexes_nodes):
                dtree.tree_.threshold[n] += drifts[k]
    
    phis,ths,bs = rule
    non_coherent_sense = 0
    
    phi,th = dtree.tree_.feature[node],dtree.tree_.threshold[node]
    
    if phi != -2:
        coh,non_coherent_sense = coherent_new_split(phi,th,rule)
        
        if not coh:
            if Translate :
                print('Warning:this translation made incoherent subtree')
                
            node = lib_tree.cut_from_left_right(dtree, node, non_coherent_sense)
    
        phi,th = dtree.tree_.feature[node],dtree.tree_.threshold[node]
        
        phis_l = np.array(list(phis) + [dtree.tree_.feature[node]])
        phis_r = phis_l
        ths_l = np.array(list(ths) + [dtree.tree_.threshold[node]])
        ths_r = ths_l
        bs_l = np.array(list(bs) + [-1])
        bs_r = np.array(list(bs) + [1])  

        rule_l = phis_l,ths_l,bs_l
        rule_r = phis_r,ths_r,bs_r

        node_l = dtree.tree_.children_left[node]

        if dtree.tree_.feature[node_l] != -2 :
            node = ForceCoherence(dtree,rule_l,node_l)
        
        node_r = dtree.tree_.children_right[node]
        
        if dtree.tree_.feature[node_r] != -2 :
            node = ForceCoherence(dtree,rule_r,node_r)
            
        return node
    

def CoherentFusionDecisionTree(dTree1, node, dTree2):
    """adding the coherent part of tree dTree2 to node 'node' of tree dTree1"""   
    dtree1 = copy.deepcopy(dTree1)
    dtree2 = copy.deepcopy(dTree2)
    
    leaf = lib_tree.cut_into_leaf2(dtree1, node)
    rule = lib_tree.extract_rule(dtree1,leaf)

    ForceCoherence(dtree2,rule,node = 0)
    lib_tree.fusionDecisionTree(dtree1,leaf,dtree2)
    
    return  dtree1

# =============================================================================
# 
# =============================================================================


def Cl_entropy_gain(rule,reaching_class,split,o_l_p,n_cl=2):

    phi,th = split

    if rule is not None:
        phis, ths, bs = rule
        H0 = 1 - pow(1/len(reaching_class),2)
    else:
        phis, ths, bs = np.array([]), np.array([]), np.array([])
        H0 = 1 - 1/n_cl
    

    new_rule_l = np.concatenate((phis,np.array([phi]))),np.concatenate((ths,np.array([th]))),np.concatenate((bs,np.array([-1])))
    new_rule_r = np.concatenate((phis,np.array([phi]))),np.concatenate((ths,np.array([th]))),np.concatenate((bs,np.array([1])))
    
    reach_class_l = list()
    reach_class_r = list()
    
    for c in range(n_cl):
        for r in o_l_p[c]:
            if not lib_tree.isdisj(r,new_rule_l):
                reach_class_l.append(c)
            if not lib_tree.isdisj(r,new_rule_r):
                reach_class_r.append(c)
                
    reach_class_l = list(set(reach_class_l))
    reach_class_r = list(set(reach_class_r))       
        
    L = len(reach_class_l)
    R = len(reach_class_r)
    
    H_l = 1 - pow(1/L,2)
    H_r = 1 - pow(1/R,2)
    
    G = H0 - 0.5*H_l - 0.5*H_r
    
    return G

    
def EntropyGainFromClasses(rule,reaching_class,considered_splits,o_l_p,n_cl=2):
    
    gains = np.zeros(considered_splits.shape[0])
    
    for k,split in enumerate(considered_splits):
        
        gains[k] = Cl_entropy_gain(rule,reaching_class,split,o_l_p,n_cl=n_cl)
        
    return gains

# =============================================================================
# 
# =============================================================================

def CreateFullNewTree(dtree):
    d = dict()
    d['node_count'] = 1
    d['max_depth'] = 0 
    d['nodes'] = np.zeros(1,dtype=[('left_child', '<i8'),
                             ('right_child', '<i8'), ('feature', '<i8'),
                             ('threshold', '<f8'), ('impurity', '<f8'),
                             ('n_node_samples', '<i8'), ('weighted_n_node_samples', '<f8')])

    d['values'] = np.zeros((1,dtree.tree_.value.shape[1],dtree.tree_.value.shape[2]))
    (Tree,(n_f,n_c,n_o),b) = dtree.tree_.__reduce__()
    new_tree = Tree(n_f, n_c, n_o)

    new_tree.__setstate__(d)  

    new_dtree = DecisionTreeClassifier()
    new_dtree.n_features_ = n_f
    new_dtree.n_classes_ = n_c[0]

    new_dtree.classes_ = np.linspace(0,n_c[0]-1,n_c[0]).astype(int)
    new_dtree.n_outputs_ = n_o
    new_dtree.tree_ = new_tree
    
    return new_dtree

# =============================================================================
# 
# =============================================================================


    
