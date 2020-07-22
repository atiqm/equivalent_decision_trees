#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 19:11:42 2019

@author: mounir
"""
import numpy as np
import copy 


from lib_eq import CreateFullNewTree

def depth_tree(dt,node=0):
    
    if dt.tree_.feature[node] == -2:
        return 0
    else:
        nl = dt.tree_.children_left[node]
        nr = dt.tree_.children_right[node]
        
        return max(depth_tree(dt,nl),depth_tree(dt,nr)) + 1
    
def depth_rf(rf):
    d = 0
    for p in rf.estimators_:
        d = d + p.tree_.max_depth
    return d/len(rf.estimators_)

def depth(dtree,node):
    p,t,b = extract_rule(dtree,node)
    return len(p)

def depth_array(dtree, inds):
    depths = np.zeros(np.array(inds).size)
    for i, e in enumerate(inds):
        depths[i] = depth(dtree, i)
    return depths


def leaf_error(tree, node):
    if np.sum(tree.value[node]) == 0:
        return 0
    else:
        return 1 - np.max(tree.value[node]) / np.sum(tree.value[node])


def error(tree, node):
    if node == -1:
        return 0
    else:

        if tree.feature[node] == -2:
            return leaf_error(tree, node)
        else:
            nr = np.sum(tree.value[tree.children_right[node]])
            nl = np.sum(tree.value[tree.children_left[node]])

            if nr + nl == 0:
                return 0
            else:
                er = error(tree, tree.children_right[node])
                el = error(tree, tree.children_left[node])

                return (el * nl + er * nr) / (nl + nr)

# =============================================================================
# 
# =============================================================================
    
def isinrule(rule, split):
    f,t = split
    
    feats, ths, bools = rule
    for k,f2 in enumerate(feats):

        if f2 == f and t == ths[k]:
            return 1,bools[k]
    return 0,0

def isdisj_feat(ths1,bools1,ths2,bools2):
    if np.sum(bools1 == -1) != 0:
        max_th1 = np.amin(ths1[bools1==-1])
    else:
        max_th1 = np.inf
        
    if np.sum(bools1 == 1) != 0:
        min_th1 = np.amax(ths1[bools1==1])
    else:
        min_th1 = - np.inf
    
    if np.sum(bools2 == -1) != 0:
        max_th2 = np.amin(ths2[bools2==-1])
    else: 
        max_th2 = np.inf
        
    if np.sum(bools2 == 1) != 0:
        min_th2 = np.amax(ths2[bools2==1])  
    else:
        min_th2 = - np.inf
    
    if ( min_th2> min_th1 and min_th2< max_th1 ) or ( max_th2> min_th1 and max_th2< max_th1 ) or ( max_th1> min_th2 and max_th1< max_th2 ) or ( min_th1> min_th2 and min_th1< max_th2 ) or ( min_th1 == min_th2 and max_th1 == max_th2 )   :
        return 0
    else:
        return 1
    
def isdisj(rule1,rule2):
    feats1, ths1, bools1 = rule1
    feats2, ths2, bools2 = rule2
    if np.array(rule1).size == 0 or np.array(rule2).size == 0 :
        return 0
    isdj = 0

    for phi in feats1:
        
        if phi in feats2:
            
            ths1_f = ths1[ feats1 == phi ]
            ths2_f = ths2[ feats2 == phi ]
            bools1_f = bools1[ feats1 == phi ]
            bools2_f = bools2[ feats2 == phi ]
            
            if isdisj_feat(ths1_f,bools1_f,ths2_f,bools2_f):
                isdj = 1

    
    return isdj

# =============================================================================
# 
# =============================================================================
def extract_rule(dtree,node):

    feats = list()
    ths = list()
    bools = list()
    nodes = list()
    b = 1
    if node != 0:
        while b != 0:
           
            feats.append(dtree.tree_.feature[node])
            ths.append(dtree.tree_.threshold[node])
            bools.append(b)
            nodes.append(node)
            node,b = find_parent(dtree,node)
            
        feats.pop(0)
        ths.pop(0)
        bools.pop(0)
        nodes.pop(0)
  
    return np.array(feats), np.array(ths), np.array(bools)


def extract_leaves_rules(dtree):
    leaves = np.where(dtree.tree_.feature == -2)[0]
    
    rules = np.zeros(leaves.size,dtype = object)
    for k,f in enumerate(leaves) :
        rules[k] = extract_rule(dtree,f)
        
    return leaves, rules

def find_parent(dtree, i_node):
    p = -1
    b = 0
    if i_node != 0 and i_node != -1:

        try:
            p = list(dtree.tree_.children_left).index(i_node)
            b = -1
        except:
            p = p
        try:
            p = list(dtree.tree_.children_right).index(i_node)
            b = 1
        except:
            p = p

    return p, b


def sub_nodes(tree, node):
    if (node == -1):
        return list()
    if (tree.feature[node] == -2):
        return [node]
    else:
        return [node] + sub_nodes(tree, tree.children_left[node]) + sub_nodes(tree, tree.children_right[node])

def nodes_in_depth(dtree, D,node =0):

    d = depth(dtree,node)
    if d < D:
            
        if dtree.tree_.feature[node] != -2:
            node_left = dtree.tree_.children_left[node] 
            node_right = dtree.tree_.children_right[node] 
            
            nodes_left = nodes_in_depth(dtree,D,node = node_left)
            nodes_right = nodes_in_depth(dtree,D,node = node_right) 
            
            return nodes_left + nodes_right
    else:
        if d == D:
            return [node]
        
def is_same_rule(r1,r2):
    
    phis1,ths1,bs1 = r1
    phis2,ths2,bs2 = r2
    s1 = phis1.size
    s2 = phis2.size
    
    if s2 != s1 :
        return False
    else:
        s = s1
        bool_phis = ( sum(phis1 == phis2) == s )
        bool_ths = ( sum(ths1 == ths2) == s )
        bool_bs = ( sum(bs1 == bs2) == s )
        
        if bool_phis and bool_ths and bool_bs:
            return True
        else:
            return False
        
def search_rule(dtree, rule, node = 0):
    
    new_rule = extract_rule(dtree,node)

    if is_same_rule(new_rule,rule) :
        return node,1
    else:
        if dtree.tree_.feature[node] != -2:
            node_left = dtree.tree_.children_left[node] 
            node_right = dtree.tree_.children_right[node] 

            node_l,b_l = search_rule(dtree,rule,node = node_left)
            node_r,b_r = search_rule(dtree,rule,node = node_right) 
            
            if b_l == 1:
                return node_l,1
            elif b_r == 1:
                return node_r,1
            else:
                return 0,0
        else: 
            return 0,0


def corresponding_nodes(dtree,dtree_new,nodes):
    nodes = np.array(nodes)
    new_nodes = np.zeros(nodes.size)

    for k,i in enumerate(nodes) :
        r = extract_rule(dtree,i)
        new_nodes[k] = search_rule(dtree_new,r)
        
    return new_nodes
        
def prune_nodes(dtree,inds,dtree_copy=None,node=0):
    rules = list()

    for i in inds :
        rules.append(extract_rule(dtree,i))
        
    if dtree_copy is None:
        dtree_copy = copy.deepcopy(dtree)
        
    rule = extract_rule(dtree_copy,node)
    
    if rule in rules :
        node = cut_into_leaf2(dtree_copy,node)
        
    if dtree_copy.tree_.feature[node] != -2:
        node_left = dtree.tree_.children_left[node] 
        node_right = dtree.tree_.children_right[node] 
        prune_nodes(dtree_copy,inds,dtree_copy=dtree_copy,node = node_left)
        prune_nodes(dtree_copy,inds,dtree_copy=dtree_copy,node = node_right)
    
    return dtree_copy

#def cut_depth(dtree,node,depth,dtree_copy=None):
#    
#    if dtree_copy is None:
#        dtree_copy = copy.deepcopy(dtree)
#        
#    d = depth(dtree,node)
#    if dtree.tree_.feature[node] != -2:
#        if d < depth:
#            node_left = dtree.tree_.children_left[node] 
#            node_right = dtree.tree_.children_right[node] 
#            cut_depth(dtree,node_left,depth,dtree_copy=dtree_copy)
#            cut_depth(dtree,node_right,depth,dtree_copy=dtree_copy)
#        else:
#            cut_into_leaf2(dtree_copy,node)
#    
#    return dtree_copy
    
def sub_tree(dtree,i_node):
    subdtree = CreateFullNewTree(dtree)
    if (i_node == -1):
        return subdtree
    else:
        subdtree.tree_.feature[0] = dtree.tree_.feature[i_node]
        subdtree.tree_.threshold[0] = dtree.tree_.threshold[i_node]
        subdtree.tree_.value[0] = dtree.tree_.value[i_node]
        subdtree.tree_.impurity[0] = dtree.tree_.impurity[i_node]
        subdtree.tree_.n_node_samples[0] = dtree.tree_.n_node_samples[i_node]
        subdtree.tree_.weighted_n_node_samples[0] = dtree.tree_.weighted_n_node_samples[i_node]
        
        if dtree.tree_.feature[i_node] != -2:
            add_child_leaf(subdtree,0,-1)
            add_child_leaf(subdtree,0,1)
            subdtree.tree_.value[subdtree.tree_.children_left[0]] = dtree.tree_.value[dtree.tree_.children_left[i_node]]
            subdtree.tree_.value[subdtree.tree_.children_right[0]] = dtree.tree_.value[dtree.tree_.children_right[i_node]]
            
            i_left = dtree.tree_.children_left[i_node] 
            i_right = dtree.tree_.children_right[i_node] 
            subdtree_left = sub_tree(dtree,i_left)
            subdtree_right = sub_tree(dtree,i_right)
            
            subdtree = fusionDecisionTree(subdtree, subdtree.tree_.children_left[0], subdtree_left)
            subdtree = fusionDecisionTree(subdtree, subdtree.tree_.children_right[0], subdtree_right)
        
        return subdtree



# =============================================================================
# 
# =============================================================================

def add_child_leaf(dtree,node,lr):
    dtree_copy = copy.deepcopy(dtree)
    d = dtree_copy.tree_.__getstate__()
    new_node = np.zeros(1,dtype=[('left_child', '<i8'),
                                 ('right_child', '<i8'), ('feature', '<i8'),
                                 ('threshold', '<f8'), ('impurity', '<f8'),
                                 ('n_node_samples', '<i8'), ('weighted_n_node_samples', '<f8')])
    
    new_node[0]['left_child'] = -1
    new_node[0]['right_child'] = -1
    new_node[0]['feature'] = -2
    new_node[0]['threshold'] = -1
    
    #new_node[0]['impurity'] = 
    
    new_node[0]['n_node_samples'] = 0
    new_node[0]['weighted_n_node_samples'] = 0
    
    d['nodes'] = np.concatenate((d['nodes'],new_node))
    d['values'] = np.concatenate(( d['values'],np.zeros((1,d['values'].shape[1],d['values'].shape[2]))))
    
    if lr == -1:
        d['nodes']['left_child'][node] = d['nodes'].size - 1
    if lr == 1:
        d['nodes']['right_child'][node] = d['nodes'].size - 1
        
    d['node_count'] = d['node_count'] + 1
    
    dep = depth(dtree,node)
    if ( dep > d['max_depth'] ):
        d['max_depth'] = dep
    
    (Tree,(n_f,n_c,n_o),b) = dtree_copy.tree_.__reduce__()
    new_tree = Tree(n_f, n_c, n_o)
    new_tree.__setstate__(d)  

    dtree.tree_ = new_tree
        
    return dtree, d['nodes'].size - 1

# =============================================================================
#            
# =============================================================================

def fusionTree(tree1, f, tree2):
    """adding tree tree2 to leaf f of tree tree1"""

    dic = tree1.__getstate__().copy()
    dic2 = tree2.__getstate__().copy()

    size_init = tree1.node_count

#    if depth_vtree(tree1, f) + dic2['max_depth'] > dic['max_depth']:
#        dic['max_depth'] = depth_vtree(tree1, f) + tree2.max_depth

    dic['capacity'] = tree1.capacity + tree2.capacity - 1
    dic['node_count'] = tree1.node_count + tree2.node_count - 1

    dic['nodes'][f] = dic2['nodes'][0]

    if (dic2['nodes']['left_child'][0] != - 1):
        dic['nodes']['left_child'][f] = dic2[
            'nodes']['left_child'][0] + size_init - 1
    else:
        dic['nodes']['left_child'][f] = -1
    if (dic2['nodes']['right_child'][0] != - 1):
        dic['nodes']['right_child'][f] = dic2[
            'nodes']['right_child'][0] + size_init - 1
    else:
        dic['nodes']['right_child'][f] = -1

    # Warning : mpurity vector is not updated

    dic['nodes'] = np.concatenate((dic['nodes'], dic2['nodes'][1:]))
    dic['nodes']['left_child'][size_init:] = (dic['nodes']['left_child'][
                                              size_init:] != -1) * (dic['nodes']['left_child'][size_init:] + size_init) - 1
    dic['nodes']['right_child'][size_init:] = (dic['nodes']['right_child'][
                                               size_init:] != -1) * (dic['nodes']['right_child'][size_init:] + size_init) - 1

    values = np.concatenate((dic['values'], np.zeros((dic2['values'].shape[
                            0] - 1, dic['values'].shape[1], dic['values'].shape[2]))), axis=0)

    dic['values'] = values

    (Tree, (n_f, n_c, n_o), b) = tree1.__reduce__()

    tree1 = Tree(n_f, n_c, n_o)

    tree1.__setstate__(dic)
    return tree1


def fusionDecisionTree(dTree1, f, dTree2):
    """adding tree dTree2 to leaf f of tree dTree1"""

    size_init = dTree1.tree_.node_count
    dTree1.tree_ = fusionTree(dTree1.tree_, f, dTree2.tree_)
    if depth(dTree1, f) + dTree2.tree_.max_depth > dTree1.tree_.max_depth:
        dTree1.tree_.max_depth = depth(dTree1, f) + dTree2.tree_.max_depth

    try:
        dTree1.tree_.value[size_init:, :, dTree2.classes_.astype(
            int)] = dTree2.tree_.value[1:, :, :]
    except IndexError as e:
        print("IndexError : size init : ", size_init,
              "\ndTree2.classes_ : ", dTree2.classes_)
        print(e)
    dTree1.max_depth = dTree1.tree_.max_depth
    return dTree1

def cut_from_left_right(dTree, node, bool_left_right):
    dic = dTree.tree_.__getstate__().copy()

    node_to_rem = list()
    size_init = dTree.tree_.node_count

    p, b = find_parent(dTree, node)

    if bool_left_right == 1:
        repl_node = dTree.tree_.children_left[node]
        node_to_rem = [node, dTree.tree_.children_right[node]]
    elif bool_left_right == -1:
        repl_node = dTree.tree_.children_right[node]
        node_to_rem = [node, dTree.tree_.children_left[node]]

    inds = list(
        set(np.linspace(0, size_init - 1, size_init).astype(int)) - set(node_to_rem))

    dic['capacity'] = dTree.tree_.capacity - len(node_to_rem)
    dic['node_count'] = dTree.tree_.node_count - len(node_to_rem)

    if b == 1:
        dic['nodes']['right_child'][p] = repl_node
    elif b == -1:
        dic['nodes']['left_child'][p] = repl_node

    dic_old = dic.copy()
    left_old = dic_old['nodes']['left_child']
    right_old = dic_old['nodes']['right_child']

    dic['nodes'] = dic['nodes'][inds]
    dic['values'] = dic['values'][inds]

    for i, new in enumerate(inds):
        if (left_old[new] != -1):
            dic['nodes']['left_child'][i] = inds.index(left_old[new])
        else:
            dic['nodes']['left_child'][i] = -1
        if (right_old[new] != -1):
            dic['nodes']['right_child'][i] = inds.index(right_old[new])
        else:
            dic['nodes']['right_child'][i] = -1

    (Tree, (n_f, n_c, n_o), b) = dTree.tree_.__reduce__()
    del dTree.tree_

    dTree.tree_ = Tree(n_f, n_c, n_o)
    dTree.tree_.__setstate__(dic)
    depths = depth_array(dTree, np.linspace(
        0, dTree.tree_.node_count - 1, dTree.tree_.node_count).astype(int))
    dTree.tree_.max_depth = np.max(depths)

    return inds.index(repl_node)


def cut_into_leaf2(dTree, node):
    dic = dTree.tree_.__getstate__().copy()

    node_to_rem = list()
    size_init = dTree.tree_.node_count

    node_to_rem = node_to_rem + sub_nodes(dTree.tree_, node)[1:]
    node_to_rem = list(set(node_to_rem))

    inds = list(
        set(np.linspace(0, size_init - 1, size_init).astype(int)) - set(node_to_rem))
    depths = depth_array(dTree, inds)
    dic['max_depth'] = np.max(depths)

    dic['capacity'] = dTree.tree_.capacity - len(node_to_rem)
    dic['node_count'] = dTree.tree_.node_count - len(node_to_rem)

    dic['nodes']['feature'][node] = -2
    dic['nodes']['left_child'][node] = -1
    dic['nodes']['right_child'][node] = -1

    dic_old = dic.copy()
    left_old = dic_old['nodes']['left_child']
    right_old = dic_old['nodes']['right_child']

    dic['nodes'] = dic['nodes'][inds]
    dic['values'] = dic['values'][inds]


    for i, new in enumerate(inds):
        if (left_old[new] != -1):
            dic['nodes']['left_child'][i] = inds.index(left_old[new])
        else:
            dic['nodes']['left_child'][i] = -1
        if (right_old[new] != -1):
            dic['nodes']['right_child'][i] = inds.index(right_old[new])
        else:
            dic['nodes']['right_child'][i] = -1

    (Tree, (n_f, n_c, n_o), b) = dTree.tree_.__reduce__()
    del dTree.tree_

    dTree.tree_ = Tree(n_f, n_c, n_o)
    dTree.tree_.__setstate__(dic)

    return inds.index(node)

def add_to_parents(dTree, node, values):

    p, b = find_parent(dTree, node)

    if b != 0:
        dTree.tree_.value[p] = dTree.tree_.value[p] + values
        add_to_parents(dTree, p, values)


def add_to_child(dTree, node, values):

    l = dTree.tree_.children_left[node]
    r = dTree.tree_.children_right[node]

    if r != -1:
        dTree.tree_.value[r] = dTree.tree_.value[r] + values
        add_to_child(dTree, r, values)
    if l != -1:
        dTree.tree_.value[l] = dTree.tree_.value[l] + values
        add_to_child(dTree, l, values)
        
# =============================================================================
#         
# =============================================================================
def get_children_distributions(decisiontree,
                               node_index):
    tree = decisiontree.tree_
    child_l = tree.children_left[node_index]
    child_r = tree.children_right[node_index]
    Q_source_l = tree.value[child_l]
    Q_source_r = tree.value[child_r]
    return [np.asarray(Q_source_l), np.asarray(Q_source_r)]


def get_node_distribution(decisiontree,
                          node_index):
    tree = decisiontree.tree_
    Q = tree.value[node_index]
    return np.asarray(Q)


def compute_class_distribution(classes,
                               class_membership):
    unique, counts = np.unique(class_membership,
                               return_counts=True)
    classes_counts = dict(zip(unique, counts))
    classes_index = dict(zip(classes, range(len(classes))))
    distribution = np.zeros(len(classes))
    for label, count in classes_counts.items():
        class_index = classes_index[label]
        distribution[class_index] = count
    return distribution


def compute_Q_children_target(X_target_node,
                              Y_target_node,
                              phi,
                              threshold,
                              classes):
    # Split parent node target sample using the threshold provided
    # instances <= threshold go to the left
    # instances > threshold go to the right
    decision_l = X_target_node[:, phi] <= threshold
    decision_r = np.logical_not(decision_l)
    Y_target_child_l = Y_target_node[decision_l]
    Y_target_child_r = Y_target_node[decision_r]
    Q_target_l = compute_class_distribution(classes, Y_target_child_l)
    Q_target_r = compute_class_distribution(classes, Y_target_child_r)
    return Q_target_l, Q_target_r



def KL_divergence(class_counts_P,
                  class_counts_Q):
    # KL Divergence to assess the difference between two distributions
    # Definition: $D_{KL}(P||Q) = \sum{i} P(i)ln(\frac{P(i)}{Q(i)})$
    # epsilon to avoid division by 0
    epsilon = 1e-8
    class_counts_P += epsilon
    class_counts_Q += epsilon
    P = class_counts_P * 1. / class_counts_P.sum()
    Q = class_counts_Q * 1. / class_counts_Q.sum()
    Dkl = (P * np.log(P * 1. / Q)).sum()
    return Dkl

def H(class_counts):
    # Entropy
    # Definition: $H(P) = \sum{i} -P(i) ln(P(i))$
    epsilon = 1e-8
    class_counts += epsilon
    P = class_counts * 1. / class_counts.sum()
    return - (P * np.log(P)).sum()


def IG(class_counts_parent,
       class_counts_children):
    # Information Gain
    H_parent = H(class_counts_parent)
    H_children = np.asarray([H(class_counts_child)
                             for class_counts_child in class_counts_children])
    N = class_counts_parent.sum()
    p_children = np.asarray([class_counts_child.sum(
    ) * 1. / N for class_counts_child in class_counts_children])
    information_gain = H_parent - (p_children * H_children).sum()
    return information_gain


def JSD(P, Q):
    M = (P + Q) * 1. / 2
    Dkl_PM = KL_divergence(P, M)
    Dkl_QM = KL_divergence(Q, M)
    return (Dkl_PM + Dkl_QM) * 1. / 2


def DG(Q_source_l,
       Q_source_r,
       Q_target_l,
       Q_target_r):
    # compute proportion of instances at left and right
    p_l = Q_target_l.sum()
    p_r = Q_target_r.sum()
    total_counts = p_l + p_r
    p_l /= total_counts
    p_r /= total_counts
    # compute the DG
    return 1. - p_l * JSD(Q_target_l, Q_source_l) - p_r * JSD(Q_target_r, Q_source_r)



def GINI(class_distribution):
    if class_distribution.sum():
        p = class_distribution / class_distribution.sum()
        return 1 - (p**2).sum()
    return 0
        
        
# =============================================================================
# 
# =============================================================================


def bootstrap(X,y,class_wise=True):
    s = y.size
    
    if class_wise:
        indexes = np.random.choice(np.linspace(0, s - 1, s).astype(int), s, replace=True)
    else:
        indexes = np.array([])
        for k in set(y):
            inds = np.where(y==k)[0]
            size_k = inds.size
            indexes_k = np.random.choice(inds, size_k, replace=True)
            indexes = np.concatenate((indexes,indexes_k))
        
    return X[indexes],y[indexes],indexes
    
def fill_rec(node,tree,X,y,refill=0):
    if y.size > 0 :
        if refill:
            tree.value[node,:,:] = 0
            tree.n_node_samples[node] = 0
            tree.weighted_n_node_samples[node] = 0
            
        for k,lab in enumerate(set(y.astype(int))):

            tree.value[node,:,lab] = tree.value[node,:,lab] + np.sum(y == lab)
            tree.n_node_samples[node] = tree.n_node_samples[node] + np.sum(y == lab)
            tree.weighted_n_node_samples[node] = tree.weighted_n_node_samples[node] + np.sum(y == lab)
          
        if tree.feature[node] != -2:
            bool_test = X[:,tree.feature[node]] <= tree.threshold[node]
            not_bool_test = X[:,tree.feature[node]] > tree.threshold[node]
            
            ind_left = np.where(bool_test)[0]
            ind_right = np.where(not_bool_test)[0]
        
            X_node_left = X[ind_left]
            y_node_left = y[ind_left]
        
            X_node_right = X[ind_right]
            y_node_right = y[ind_right]
            
            fill_rec(tree.children_left[node],tree,X_node_left,y_node_left)
            fill_rec(tree.children_right[node],tree,X_node_right,y_node_right)


def fill_with_samples(dtree,X,y,refill = 0):
    fill_rec(0,dtree.tree_,X,y,refill = refill)



# =============================================================================
#
# =============================================================================


def vote_rf(rf,X,y):
    Z = np.zeros((y.size,rf.n_estimators))
    vote = np.zeros(y.size)
    for i,dt in enumerate(rf.estimators_):
        Z[:,i] = dt.predict(X)
    
    for i in range(y.size):
        votek = np.zeros(rf.n_classes_)
        for k in range(rf.n_classes_):
            votek[k] = list(Z[i,:]).count(k)
        vote[i] = np.argmax(votek)
    
    return vote

def predict_vote_rf(rf,X):
    
    Z = np.zeros((X.shape[0],rf.n_estimators))
    vote = np.zeros(X.shape[0])
    for i,dt in enumerate(rf.estimators_):
        Z[:,i] = dt.predict(X)
    
    for i in range(X.shape[0]):
        votek = np.zeros(rf.n_classes_)
        for k in range(rf.n_classes_):
            votek[k] = list(Z[i,:]).count(k)
        vote[i] = np.argmax(votek)
    
    return vote

def error_vote_rf(rf,X,y,balanced=False):
    comp = ( predict_vote_rf(rf,X) != y )
    
    err = sum(comp)/y.size
    
    return err

