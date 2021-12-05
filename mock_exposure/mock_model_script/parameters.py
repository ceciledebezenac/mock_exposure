#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random 
import numpy as np


# In[4]:


#######################
# Agent data retrieval
######################


#Raw data:
########################
def get_exposure0(agent):
    return(agent.exposure[0])

def get_exposure1(agent):
    return(agent.exposure[1])

def get_exposure2(agent):
    return(agent.exposure[2])


def get_friends(agent):
    return(len(agent.friends))

def get_adoption_friends(agent):
    return(len([a for a in agent.friends if a.behaviour==1]))



#Creating indices for analysis
##############################
def get_friend_deprivation(agent):
    return(np.mean([a.deprivation for a in agent.friends]))

def relative_deprivation_network(agent):
    friend_deprivation=np.mean([a.deprivation for a in agent.friends])
    relative_deprivation=max(0,agent.deprivation-friend_deprivation)
    #or could be the count of friends that are less deprived weighted by their deprivation
    return(relative_deprivation)

def relative_deprivation_neighbors(agent):
    neighbor_deprivation=np.mean([a.deprivation for a in agent.neighbors])
    relative_deprivation=max(0,agent.deprivation-neighbor_deprivation)
    return(relative_deprivation)

def area_deprivation(agent):
    area_dep=np.mean([a.deprivation for a in agent.model.grid.get_cell_list_contents([agent.pos])])
    return(area_deprivation)
    #same as neighbor but agent counts himself with it. 


# In[ ]:


############################
# Model data retrieval
############################

def behaviour_prevalence(model):
    proportion=sum([a.behaviour for a in model.schedule.agents if a.state==0])/model.num_agents
    return(proportion)

def count_dead(model):
    dead=sum([a.state for a in model.schedule.agents])
    return (dead)


def get_mean_age(model):
    return(np.mean([a.age for a in model.schedule.agents if a.state==0]))
    
def count_behaviour_alive(model):
    adopters=sum([a.behaviour for a in model.schedule.agents if a.state==0])
    return (adopters)

def get_group1(model):
    return(sum([a.group for a in model.schedule.agents if a.state==0]))

def count_behaviour_dead(model):
    dead_adopters=sum([a.behaviour for a in model.schedule.agents if a.state==1])
    return(dead_adopters)


def compute_death_rate(model):
    return(model.dead_in_step/model.num_agents)


def compute_adoption_rate(model):
    return(sum([a.behaviour for a in model.schedule.agents if a.time==1])/model.num_agents)



def compute_global_deprivation(model):
    return(np.mean([a.deprivation for a in model.schedule.agents if a.state==0]))


# In[ ]:


params={
    'N':100,
    'width':10,
    'height':10,
    'network':{'type':'random','dynamic':'progressive','density':0.01},
    'death_probability':0.005,
    'adopt_probability':0.01,
    'drop_probability':0.01,
    'birth_rate':0.02,
    'odds_exposure_death':np.random.uniform(0,0.5,3),#1 will be added,
    'odds_exposure_adopt':np.random.uniform(0,0.5,3),#1 will be added,
    'odds_relative_deprivation':1,
    'change_exposure_rate':0.1,
    'odds_group_adopt':1.5,
    'odds_behaviour_death':1.5 ,
    'dependancies':[]
}  


# In[ ]:


model_reporters = {"population":"num_agents",
                   "deprivation":compute_global_deprivation,
                   "mean_age":get_mean_age,
                   "group1":get_group1,
                   "behaviour": count_behaviour_alive,
                   "behaviour_prevalence":behaviour_prevalence,
                   "adoption_rate":compute_adoption_rate,
                   "dead_in_step":"dead_in_step",
                   "cumulative_death":count_dead,
                   "death_rate":compute_death_rate,
                   "network":"G",#for network measures if needed
                   "odds_exposure_death":"odds_exposure_death",
                   "odds_exposure_adopt":"odds_exposure_adopt",
                   "odds_group_adopt":"odds_group",
                   "odds_behaviour_death":"odds_behaviour_death",
                   "birth_rate":"birth_rate",
                   "dead_from_behaviour":"dead_from_behaviour"
                  }



agent_reporters= {
                  "age":"age",
                  "exposure":"exposure",
                  "group":"group",
                  "behaviour":"behaviour",
                  "state": "state", 
                  "time":"time",
                  "friends":get_friends,
                  "adopter_friends":get_adoption_friends,
                  "area":"pos",
                  "deprivation":"deprivation",
                  "true_relative_deprivation":"true_relative_deprivation",
                  "friend_deprivation":get_friend_deprivation,
                  "relative_deprivation_network":relative_deprivation_network,#relative deprivation in area?or network?
                  "relative_deprivation_neighbors":relative_deprivation_neighbors,
                  "area_deprivation":area_deprivation,#neighborhood effect
                  "exposure0":get_exposure0,
                  "exposure1":get_exposure1,
                  "exposure2":get_exposure2,
                  "prob_die":"prob_die",
                  "prob_adopt":"prob_adopt"
                  }

