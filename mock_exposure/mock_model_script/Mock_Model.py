#!/usr/bin/env python
# coding: utf-8

# In[6]:

import mock_model_script.Agent as ma
#import mock_model_script.parameters as par
#from mock_model_script.BBN import BBN
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



class M_Model(Model):
    def __init__(self,params,data_model,data_agent):
        
        #generic attributes
        self.num_agents = params['N']
        self.G = nx.Graph()
        self.grid = MultiGrid(params['width'], params['height'], True)
        self.schedule = RandomActivation(self)
        self.running = True 
        
        #model parameters
        self.network_density=params['network']['density']
        self.network_type=params['network']['type']
        self.network_dynamic=params['network']['dynamic']
        self.odds_behaviour_death = params['odds_behaviour_death']#1.1
        self.odds_group=params['odds_group_adopt']#1.2
        self.change_exposure_rate= params['change_exposure_rate']#0.2
        self.odds_exposure_adopt = params['odds_exposure_adopt']#np.random.uniform(0,0.5,7)#1 will be added
        self.odds_exposure_death = params['odds_exposure_death']#np.random.uniform(0,0.5,7)#1 will be added
        self.adopt_prob = params['adopt_probability']#0.075
        self.drop_prob = params['drop_probability']#0.05
        self.death_prob = params['death_probability']#0.005
        self.birth_rate=params['birth_rate']#0.02
        self.params=params
        self.dependancies=params['dependancies']
        #model variables
        self.dead_in_step=0
        self.dead_agents=[]
        self.dead_from_behaviour=0

        
        # Network formation method
        def set_edges(self,t,p,seed=None):
            #keep option to set seed
            if seed is not None: 
                random.seed(seed) 
            #keep option to have other forms of networks
            if t=="random":
                for u, v in combinations(self.G,2):
                    if random.random() < p:
                        self.G.add_edge(u, v)
            
            if (t=="homophily"):
                for u, v in combinations(self.G,2):
                    d=u.difference_from_self(v,'deprivation')+(u.age not in range(v.age-5,v.age+5))
                    pr=1/(d+1)**params['network']['density'] # no division by 0
                    if (random.random() < pr):
                        self.G.add_edge(u,v)

            
            if t=="spatial" :
                #small world: connect with close agents
                for u,v in combinations(self.G,2):
                    d=u.distance_from_self(v)
                    pr=1/(d+1)**params['network']['density'] # no division by 0
                    if random.random() < pr:
                        self.G.add_edge(u,v)
            if t=="area":
                #connect with neighborhood (check mesa for get neighborhood function)
                for u,v in combinations(self.G,2):
                    if u.pos==v.pos:
                        self.G.add_edge(u,v)
                
                        
        # Create agents
        for i in range(self.num_agents):
            a = ma.NormalAgent(i, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x,y))
            
            #Create the graph and links and neighborhood
            if self.network_type is not None:
                self.G.add_node(a, behaviour=a.behaviour, deprivation=a.deprivation,group=a.group,age=a.age)

        if (self.network_type is not None) and (self.network_dynamic=='fixed'):
            set_edges(self,self.network_type, self.network_density)
            for a in self.schedule.agents:
                a.find_friends()
                a.find_neighbors()
                
        elif self.network_dynamic=='progressive':
            for a in self.schedule.agents:
                a.find_friends()
                a.find_neighbors    
        #define what data is to be collected
        self.datacollector = DataCollector(
            model_reporters = data_model,
            agent_reporters= data_agent
        )
        
        
    #The attributes of the nodes are in the model, so they have to be updated from the model.
    def update_node_attributes(self):
        for a in self.G.nodes:
            self.G.nodes[a]['deprivation']=a.deprivation
            self.G.nodes[a]['behaviour']=a.behaviour
            self.G.nodes[a]['age']=a.age
                                
    
    #update global variables
    def update_model_attributes(self):
        self.dead_in_step=sum([(a.state-a.previous_state) for a in self.schedule.agents])
        self.num_agents=sum([(a.state==0) for a in self.schedule.agents])
    
    
    #model method
    def step(self):
        self.datacollector.collect(self)
        self.update_node_attributes()
        self.schedule.step()
        self.update_model_attributes()
        
        


# In[ ]:




