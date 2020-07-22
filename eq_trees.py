#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:58:22 2019

@author: mounir
"""
import sys
import numpy as np
import copy
from lib_eq import CreateFullNewTree,new_random_split,all_coherent_splits, EntropyGainFromClasses, filter_feature
from lib_tree import extract_rule,extract_leaves_rules,add_to_parents,add_child_leaf,fill_with_samples,isdisj,sub_tree,search_rule,fusionDecisionTree,cut_into_leaf2,nodes_in_depth


def eq_rec_tree(dtree_or, actual_new_node, dtree_new = None, actual_rule = None, K_union_rules = None, actual_reaching_class = None, considered_splits = None, max_depth = None, from_depth = None, on_subtrees = False, subtrees_nodes = None, finishing_features = list(), smallest_tree = False):

    if from_depth is not None:
        from_depth = int(from_depth)
        if from_depth < 1:
            print('WARNING : Given depth < 1 !')
        else: 
            nodes_depth = np.array(nodes_in_depth(dtree_or,from_depth))
            on_subtrees = True
            subtrees_nodes = nodes_depth
        
    if actual_new_node == 0:
        if on_subtrees:
            if subtrees_nodes is None :
                print('WARNING : No specified subtrees !')

            dtree_new = copy.deepcopy(dtree_or)
            
            for i in subtrees_nodes:
                r = extract_rule(dtree_or,i)
                subtree = sub_tree(dtree_or,i)
                
                cut_node,b_ = search_rule(dtree_new,r)
                fusion_node = cut_into_leaf2(dtree_new,cut_node)
                
                subeqtree = eq_rec_tree(subtree,0,max_depth=max_depth,finishing_features=finishing_features,smallest_tree=smallest_tree)
                dtree_new = fusionDecisionTree(dtree_new,fusion_node,subeqtree)
                
            return dtree_new
        else:
            leaves, rules = extract_leaves_rules(dtree_or)
                    
            if K_union_rules is None:
                K_union_rules = np.zeros(dtree_or.n_classes_,dtype=object)
                for c,K in enumerate(dtree_or.classes_.astype(int)):
                   
                    K_union_rules[c] = list()
                for k,l in enumerate(leaves):
                    c = int(np.argmax(dtree_or.tree_.value[l,:,:]))
                    K_union_rules[c].append(rules[k])
    
            all_splits = np.zeros(dtree_or.tree_.node_count - leaves.size,dtype=[("phi",'<i8'),("th",'<f8')])
            
            compt = 0
            for i in range(dtree_or.tree_.node_count):
                if i not in leaves:
                    all_splits[compt]= (dtree_or.tree_.feature[i],dtree_or.tree_.threshold[i])
                    compt = compt + 1
            
            considered_splits = all_splits
            
            actual_reaching_class = dtree_or.classes_.astype(int)
            dtree_new = CreateFullNewTree(dtree_or)
    
    
    if actual_rule is not None:
        phi_actual , th_actual, b_actual = actual_rule
    else:
        phi_actual , th_actual, b_actual = np.array([]), np.array([]), np.array([])

    if len(actual_reaching_class) > 1 :

        #Warning : no equivalence waranty if a max_depth is specified
        if ( max_depth is not None ) and ( actual_rule is not None ) and actual_rule[0].size >= int(max_depth):

            for c in actual_reaching_class :
                dtree_new.tree_.value[actual_new_node,:,c] = 1
                add_to_parents(dtree_new, actual_new_node, dtree_new.tree_.value[actual_new_node])
                
            dtree_new.tree_.n_node_samples[actual_new_node] = len(actual_reaching_class)
            dtree_new.tree_.weighted_n_node_samples[actual_new_node] = len(actual_reaching_class)
            
        else:

            if len(finishing_features) > 0:
                particular_splits, other_considered_splits = filter_feature(considered_splits,finishing_features)
                
                if other_considered_splits.size == 0:
                    considered_splits = particular_splits
                else:
                    considered_splits = other_considered_splits
            ###
            if smallest_tree:
                gains = EntropyGainFromClasses(actual_rule,actual_reaching_class,considered_splits,K_union_rules,n_cl = dtree_or.n_classes_)
                p = np.zeros(considered_splits.size)
                p[gains == np.amax(gains)] = 1
                p = p/sum(p)
            else:
                p = np.ones(considered_splits.size)
                p = p/sum(p)
            ###

                    
            phi,th = new_random_split(p,considered_splits)
            
            dtree_new.tree_.feature[actual_new_node] = phi
            dtree_new.tree_.threshold[actual_new_node] = th
            
            new_rule_l = np.concatenate((phi_actual,np.array([phi]))),np.concatenate((th_actual,np.array([th]))),np.concatenate((b_actual,np.array([-1])))
            new_rule_r = np.concatenate((phi_actual,np.array([phi]))),np.concatenate((th_actual,np.array([th]))),np.concatenate((b_actual,np.array([1])))


            if len(finishing_features) > 0:

                if particular_splits.size == 0:
                    considered_splits = other_considered_splits
                elif other_considered_splits.size == 0:
                    considered_splits = particular_splits
                else:
                    considered_splits = np.concatenate((particular_splits,other_considered_splits))
                    
            considered_splits_l = all_coherent_splits(new_rule_l,considered_splits)
            considered_splits_r = all_coherent_splits(new_rule_r,considered_splits)
            
            reach_class_l = list()
            reach_class_r = list()
            
            for c,K in enumerate(dtree_or.classes_.astype(int)):
                for r in K_union_rules[c]:
                    if not isdisj(r,new_rule_l):
                        reach_class_l.append(c)
                    if not isdisj(r,new_rule_r):
                        reach_class_r.append(c)
                        
            reach_class_l = list(set(reach_class_l))
            reach_class_r = list(set(reach_class_r))            
            
                        
            dtree_new, child_l = add_child_leaf(dtree_new,actual_new_node,-1)
            dtree_new = eq_rec_tree(dtree_or,child_l,dtree_new,actual_rule=new_rule_l, K_union_rules=K_union_rules,actual_reaching_class=reach_class_l,considered_splits=considered_splits_l,max_depth=max_depth,from_depth=from_depth,on_subtrees=on_subtrees,subtrees_nodes=subtrees_nodes,finishing_features=finishing_features,smallest_tree=smallest_tree)

            dtree_new, child_r = add_child_leaf(dtree_new,actual_new_node,1)
            dtree_new = eq_rec_tree(dtree_or,child_r,dtree_new,actual_rule=new_rule_r, K_union_rules=K_union_rules,actual_reaching_class=reach_class_r,considered_splits=considered_splits_r,max_depth=max_depth,from_depth=from_depth,on_subtrees=on_subtrees,subtrees_nodes=subtrees_nodes,finishing_features=finishing_features,smallest_tree=smallest_tree)

    elif len(actual_reaching_class) == 1:

        c = actual_reaching_class[0]
        
        dtree_new.tree_.value[actual_new_node,:,c] = 1
        add_to_parents(dtree_new, actual_new_node, dtree_new.tree_.value[actual_new_node])
        dtree_new.tree_.n_node_samples[actual_new_node] = 1
        dtree_new.tree_.weighted_n_node_samples[actual_new_node] = 1
   
    else:
        print('ERREUR 0 données !')

    if actual_new_node == 0:
        dtree_new.max_depth = dtree_new.tree_.max_depth
        
    return dtree_new

# =============================================================================
# 
# =============================================================================


def eqtree_rec_rf(rf_or,node,dtree_new=None,actual_rule=None,original_leaves_path=None,reaching_class_by_tree=None,considered_splits=None, max_depth = None, from_depth = None, on_subtrees = False, subtrees_nodes = None, finishing_features = list(), smallest_tree = True):

    if node == 0:
        
        if original_leaves_path is None:
            original_leaves_path = np.zeros(rf_or.n_estimators,dtype=object)
            
            for k,dt in enumerate(rf_or.estimators_):
                leaves, rules = extract_leaves_rules(dt)
                original_leaves_path[k] = np.zeros(rf_or.n_classes_,dtype=object)

                for c,K in enumerate(rf_or.classes_.astype(int)):   
                    original_leaves_path[k][c] = list()
                    
                for i,l in enumerate(leaves):
                    c = int(np.argmax(dt.tree_.value[l,:,:]))
                    original_leaves_path[k][c] .append(rules[i])
                
        all_splits = list()
        
        for dt in rf_or.estimators_:
            for i in range(dt.tree_.node_count):
                if dt.tree_.feature[i] != -2:
                    all_splits.append((dt.tree_.feature[i],dt.tree_.threshold[i]))

        
        considered_splits = np.array(all_splits,dtype=[("phi",'<i8'),("th",'<f8')])
        reaching_class_by_tree = np.zeros(rf_or.n_estimators,dtype=object)
        
        for i in range(rf_or.n_estimators):
            reaching_class_by_tree[i] = rf_or.classes_.astype(int)
        
        dtree_new = CreateFullNewTree(rf_or.estimators_[0])    

    
    if actual_rule is not None:
        phi_actual , th_actual, b_actual = actual_rule
    else:
        phi_actual , th_actual, b_actual = np.array([]), np.array([]), np.array([])


    union = list()
    for cl in reaching_class_by_tree:
        union += list(cl)
        
    union = list(set(union))
        

    if len(union) > 1 :
        final_path = 1
        
        for c in reaching_class_by_tree :
            if len(c) > 1:
                final_path = 0
                
                
        if final_path :
            #New leaf
            
            for ct in reaching_class_by_tree:
                dtree_new.tree_.value[node,:,ct] += 1
                add_to_parents(dtree_new, node, dtree_new.tree_.value[node])
                
                dtree_new.tree_.n_node_samples[node] += 1
                dtree_new.tree_.weighted_n_node_samples[node] += 1
  
        else:
    
            #Attention à cette option : plus de garantie d'équivalence
            if max_depth != None and ( actual_rule is not None ) and actual_rule[0].size >= int(max_depth):
            
                for ct in reaching_class_by_tree:
                    dtree_new.tree_.value[node,:,ct] += 1
                    add_to_parents(dtree_new, node, dtree_new.tree_.value[node])
                    
                    dtree_new.tree_.n_node_samples[node] += 1
                    dtree_new.tree_.weighted_n_node_samples[node] += 1
           
            else:

                if smallest_tree:
                    w_gains = np.zeros(considered_splits.size)
                    for k in range(rf_or.n_estimators):
                        w_gains += EntropyGainFromClasses(actual_rule,reaching_class_by_tree[k],considered_splits,original_leaves_path[k],n_cl = dtree_new.n_classes_)
                    
                    p = np.zeros(considered_splits.size)
                    p[w_gains == np.amax(w_gains)] = 1
                    p = p/sum(p)                   
                else:
                    p = np.ones(considered_splits.size)
                    p = p/sum(p)
                
                phi,th = new_random_split(p,considered_splits)
                
                dtree_new.tree_.feature[node] = phi
                dtree_new.tree_.threshold[node] = th
                
                new_rule_l = np.concatenate((phi_actual,np.array([phi]))),np.concatenate((th_actual,np.array([th]))),np.concatenate((b_actual,np.array([-1])))
                new_rule_r = np.concatenate((phi_actual,np.array([phi]))),np.concatenate((th_actual,np.array([th]))),np.concatenate((b_actual,np.array([1])))
    
                considered_splits_l = all_coherent_splits(new_rule_l,considered_splits)
                considered_splits_r = all_coherent_splits(new_rule_r,considered_splits)

                reach_class_by_tree_l = np.zeros(rf_or.n_estimators,dtype=object)
                reach_class_by_tree_r = np.zeros(rf_or.n_estimators,dtype=object)
                
                
                for k in range(rf_or.n_estimators):
                    reach_class_by_tree_l[k]  = list()
                    reach_class_by_tree_r[k] = list()
                    for c in rf_or.classes_.astype(int):
                        for r in original_leaves_path[k][c]:
                            if not isdisj(r,new_rule_l):
                                reach_class_by_tree_l[k].append(c)
                            if not isdisj(r,new_rule_r):
                                reach_class_by_tree_r[k].append(c) 
                                
                    reach_class_by_tree_l[k] = list(set(reach_class_by_tree_l[k]))
                    reach_class_by_tree_r[k] = list(set(reach_class_by_tree_r[k]))    

                dtree_new, child_l = add_child_leaf(dtree_new,node,-1)
                dtree_new = eqtree_rec_rf(rf_or,child_l,dtree_new,actual_rule=new_rule_l, original_leaves_path=original_leaves_path,reaching_class_by_tree=reach_class_by_tree_l,
                                          considered_splits=considered_splits_l,max_depth=max_depth,from_depth=from_depth,on_subtrees=on_subtrees,subtrees_nodes=subtrees_nodes,
                                          finishing_features=finishing_features,smallest_tree=smallest_tree)

                dtree_new, child_r = add_child_leaf(dtree_new,node,1)
                dtree_new = eqtree_rec_rf(rf_or,child_r,dtree_new,actual_rule=new_rule_r,original_leaves_path=original_leaves_path,reaching_class_by_tree=reach_class_by_tree_r,
                                          considered_splits=considered_splits_r,max_depth=max_depth,from_depth=from_depth,on_subtrees=on_subtrees,subtrees_nodes=subtrees_nodes,
                                          finishing_features=finishing_features,smallest_tree=smallest_tree)


    elif len(union) == 1:
        
        dtree_new.tree_.value[node,:,union] += 1
        add_to_parents(dtree_new, node, dtree_new.tree_.value[node])
        
        dtree_new.tree_.n_node_samples[node] += 1
        dtree_new.tree_.weighted_n_node_samples[node] += 1
    
    else:
        print('0 données !')

    return dtree_new


def eq_dec_RF(rf_or, max_depth = None, from_depth = None, on_subtrees = False, subtrees_nodes = None, finishing_features = list(), smallest_tree = False):
    rf_new = copy.deepcopy(rf_or)
    
    for k,t in enumerate(rf_new.estimators_):
        rf_new.estimators_[k] = eq_rec_tree(t,0,max_depth=max_depth,from_depth=from_depth,on_subtrees=on_subtrees,subtrees_nodes=subtrees_nodes,finishing_features=finishing_features,smallest_tree=smallest_tree)
    
    return rf_new


    
