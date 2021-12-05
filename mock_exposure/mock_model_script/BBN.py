#!/usr/bin/env python
# coding: utf-8

# In[28]:


from mesa import Agent, Model
from mesa.time import StagedActivation, RandomActivation
from mesa.datacollection import DataCollector
 #to collect features during the #simulation
from mesa.space import MultiGrid
 #to generate the environment
#for computation and visualization purpose
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import math
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
import networkx as nx
from itertools import combinations
import pandas as pd
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
from pybbn.graph.factory import Factory


# In[29]:


# This function helps to calculate probability distribution, which goes into BBN (note, can handle up to 2 parents)
def probs(data, child, parents):
    if parents==None:
        # Calculate probabilities
        prob=pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index().to_numpy().reshape(-1).tolist()
    elif parents!=None:
            # Check if child node has 1 parent or 2 parents
            if len(parents)==1:
                # Caclucate probabilities
                prob=pd.crosstab(data[parents[0]],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
            else:    
                # Caclucate probabilities
                prob=pd.crosstab([data[parents[i]] for i in range(len(parents))],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else: print("Error in Probability Frequency Calculations")
    return prob


# In[30]:



class BBN():
    
    def __init__(self,structure):
        '''
        Bayesian Belief Network structure for Agent outcome dynamics: a Bbn based object. 
        
        '''
        self.init_network=Bbn()#create the initial empty Bbn
        self.structure=structure #from the dependancy structure specified in the model
        self.score=0 #for future development
        
    
    def initialise_from_cpt(self):
        """
        Method to initialise the join tree of the Bbn with the conditional probabilities specified as an assumption in the model. 
        From the dictionnary self.structure, create the nodes with their possible outcome and the probability tables initially defined. 
        """
        for key in self.structure.keys():
            #create node from keys in the structure defined in the model
            node=BbnNode(Variable(key, key, self.structure[key]["outcome"]), 
                         self.structure[key]['probs'])#probability defined in the model: list of conditional probs (see reading order)
            self.init_network.add_node(node)
        
        for key in self.structure.keys():
            #create the edges specified in the model
            for pa in self.structure[key]['parents']:
                self.init_network.add_edge(Edge(self.init_network.nodes[key],self.init_network.nodes[pa],EdgeType.DIRECTED))
        
        #initialise the network tree structure
        self.init_tree = InferenceController.apply(self.init_network)
    
    
    def initialise_from_data(self,data):
        """
        Method to initialise the join tree of the Bbn with the conditional probabilities specified as an assumption in the model. 
        From the dictionnary self.structure, create the nodes with their possible outcome and the probability tables initially defined. 
        """
        for key in self.structure.keys():
            #create node from keys in the structure defined in the model
            node=BbnNode(Variable(key, key, self.structure[key]["outcome"]), 
                         probs(data, child=key,parents=self.structure[key]["parents"]))#probability defined in the model: list of conditional probs (see reading order)
            self.init_network.add_node(node)
        
        for key in self.structure.keys():
            #create the edges specified in the model
            for pa in self.structure[key]['parents']:
                self.init_network.add_edge(Edge(self.init_network.nodes[key],self.init_network.nodes[pa],EdgeType.DIRECTED))
        
        #initialise the network tree structure
        self.init_tree = InferenceController.apply(self.init_network)


    def learn_from_data(self,data):
        #checks methods to learn bbn structure aswell as cpt
        pass
    
    
    def update_from_step(self,data,step):
        '''
        Define another tree (update the self.current_tree) from the initial structure with the simulation data collected at a given step: the previous one for instance. This is useful in the case where other dynamics are at play than the simple BNN decision. 
        '''
        #ideally this would learn the structure aswell! see ways to make it learn.
        new_data={}
        
        #create a new dictionnary with only variable names and new conditional probabilities extracted from the data
        for key in self.structure.keys():
            new_data[key]=probs(data[data.index.get_level_values('Step')==step], child=key,parents=self.structure[key]["parents"])
        
        #update (or recreate jointree) with the reapply() method (try to find some documentation on this)
        self.current_tree=InferenceController.reapply(self.init_tree,new_data)
        
    
      
    def potential_to_series(self,node):
        '''
        Get the conditional probabilities from the join tree 
        '''
        #get the potentials (marginal probabilities) for a given node. 
        
       
        n=self.init_tree.get_bbn_node_by_name(node)
        p=self.init_tree.get_bbn_potential(n)

        vals = []
        index = []
        for pe in p.entries:
            try:
                v = pe.entries.values()[0]
            except:
                v = list(pe.entries.values())[0]
            p = pe.value

            vals.append(p)
            index.append(v)

        return pd.Series(vals, index=index)

    
    def try_for_evidence(self,agent,variable):
        #from initial structure find marginal probabilities to take decisions
        #the case when agent's state is given by the DAG
         for name in self.structure[variable]['parents']:
            ev = EvidenceBuilder() \
                .with_node(self.init_tree.get_bbn_node_by_name(name)) \
                .with_evidence(getattr(agent,name), 1.0) \#set the likelihood of effective observation of variables to 1
                .build()
            self.init_tree.set_observation(ev)
        
        #get the marginal prob given the evidence of the agent in the initial tree for decision: 
        outcome=self.potential_to_series(variable,current="False")
        
        #set back to initial state: 
        self.init_tree.unobserve_all()
        
        return(outcome)

    
    def print_probs(self,join_tree=self.current_tree):
            for node in join_tree.get_bbn_nodes():
                potential = join_tree.get_bbn_potential(node)
                print("Node:", node)
                print("Values:")
                print(potential)
                print('----------------')
    
    
    def draw_bbn(self):
        options = {"font_size": 16,
        "node_size": 4000,
        "node_color": "white",
        "edgecolors": "black",
        "edge_color": "red",
        "linewidths": 5,
        "width": 5,}
    
        # Generate graph: for now same structure as current or learned one so use intial bbn to draw. 
        n, d = self.init_network.to_nx_graph()
        nx.draw(n, with_labels=True, labels=d, **options)

        # Update margins and print the graph
        ax = plt.gca()
        ax.margins(0.10)
        plt.axis("off")
        plt.show()



