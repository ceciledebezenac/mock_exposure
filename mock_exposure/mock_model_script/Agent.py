#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mesa import Agent, Model
#from mock_model_script.BBN import BBN 
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

from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController


def init_from_BBN(agent,BBN,analysis):
    ''''
    Initialise the agent from a Bayesian network if given in the model: example of a more complex underlying bn with confounders that are not availbe in the data. The question raised here is what is the effect of aggregation or forgetting confounders. The attributes will be the ones in the dag structure as well as the approximations that are available in the data and are potentially the result of aggregation or other bias. 
    Input:
    - agent:
    -BN
    - analysis: dictionary 
    '''
    #True attributes: see if it works for area (coordinates)
    for key in BBN.structure.keys():
        #initialise with one of the possible outcome randomly (children will be changed later: it avoids having to search in the network for parents and children)
        value=np.random.choice(np.array(BBN.structure[key]['outcome']),p=BBN.structure[key]['prob'])#get BN posteriors instead
        setattr(agent,key,value)
        
    #proxies that will be analysed with statistical method: probably not useful if all the data is collected in the dataframe
    for proxy in analysis.keys():
        value=proxy[key](agent)#see to sparse string for call of functions: DOES NOT WORK YET
        setattr(agent,proxy,value)

        
        
class NormalAgent(Agent):
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.age=random.randint(1,100)
        self.behaviour=0
        self.previous_behaviour=[]
        self.social_max=max(0,random.gauss(self.model.num_agents*self.model.network_density,1))
        self.state=0
        self.group=random.randint(0,2)
        self.friends=[]
        self.neighbors=[]
        self.exposure=np.random.randint(0,2,3)
        self.deprivation=sum(self.exposure)
        self.state=0
        self.previous_state=0
        self.previous_exposure=np.random.randint(0,2,3)
        self.time=0
        self.previous_behaviour=[]
        self.sensitivity_behaviour=random.random()
        self.prob_die=0
        self.prob_adopt=0
        self.prob_drop=0
        self.true_relative_deprivation=0
        

        
    def distance_from_self(self,other):
        '''
        Euclidean distance from other agent or located object
        '''
        return (math.sqrt((self.pos[0]-other.pos[0])**2+(self.pos[1]-other.pos[1])**2))
  


    def difference_from_self(self, other):
        #difference based on exposure and group affiliation: goes from 0 to 2
        vector_self=np.array(list(self.exposure)+[self.group])
        vector_other=np.array(list(other.exposure)+[other.group])#add group in the characteristics
        cos_similarity=np.dot(vector_self, vector_other)/(np.linalg.norm(vector_self)*np.linalg.norm(vector_other))
        #return(abs(getattr(other,variable) - getattr(self,variable)))
        return(1-cos_similarity)
        
    
    
              
    def find_friends(self):
        '''
        Network method: (determined)
        Get neihgbours on the Graph initialised with the model.
        
        
        Input: 
            - model.G : the graph class given in the model.
        
        
        Return: 
            - friends: list of agent instances
                       list of friends linked to the agent in the model.graph
        '''
        
        #find the neighbors in the graph not in the grid
        
        if nx.is_empty(self.model.G)==False:
            self.friends=self.model.G[self]#calculate again in case deaths: able to create new ties
        else:
            self.friends=[]
        
    def make_friends(self):

        if len(self.friends)<self.social_max:
            potential=np.random.choice([a for a in self.model.schedule.agents if a.state==0])#choose potential alive friend

            if (len(potential.friends)<potential.social_max) :
                # Creating a dissimilarity measure between agents for edge formation
                if self.model.network_type=='homophily+spatial':
                    d=self.distance_from_self(potential)+self.difference_from_self(potential)+(potential.age not in range(self.age-5,self.age+5))#more likely to make friends in the same age group
                elif self.model.network_type=='spatial':
                    d=self.distance_from_self(potential)+(potential.age not in range(self.age-5,self.age+5))
                elif self.model.network_type=='homophily':
                    d=self.difference_from_self(potential)+(potential.age not in range(self.age-5,self.age+5))    
                else: 
                    d=0#the agent will create the link either way

                #Creating the edge:
                if (random.random()<(1/(d+1)**2)): 
                    #if close enough create a link
                    self.model.G.add_edge(self,potential)
                    #update friend status for both agents
                    self.find_friends()
                    potential.find_friends()
        else: 
            pass


    
    
    def find_neighbors(self):
        '''
        Spatial method: (variable)
        Get the neighbors in the Mesa grid. 
        Input: 
            - model.grid: the mesa grid object given in the model
            - pos: the cell position of the agent
            
        Return: 
            - neighbors: the list of agent that are in same cell as the agent
            
        '''
        
        #the agents that are in the same "area" as the agent self
        self.neighbors=self.model.grid.get_cell_list_contents([self.pos])
        
    
    
        
    def find_similar_friends(self):
        '''
        (Optional) Can be replaced by a homophily method for edge formation. 
        Agents will keep their network throughout the simulation but because exposure can change, similarity between agents can evolve. 
        Hypothesis: 
            - agents relate more to similar agents in their network.
            - similarity is based on general deprivation and group affiliation
            - similarity based on a threshold: threshold_similar
            
        Input:
            - friends: the list of agents that are connected to the agent in the model graph.
            - deprivation: the sum of exposure of an agent that serves as basis of comparison given a threshold of acceptable difference. 
            - group: the generic group the agent belongs to, serves as basis for comparison
            
        Returns: 
            - similar_friends: the list of agents that are connected to the agent and are also similar in respect to group and deprivation. 
            
        Model parameters: 
            - model.threshold_similar
        
        '''
        #assumption of similar: based on deprivation, could be group could be both
        similar_friends=[]
        
        for a in self.friends:
            if (abs(a.deprivation - self.deprivation) <=self.model.threshold_similar) & (a.group==self.group):
                similar_friends.append(a)
        
        self.similar_friends=similar_friends
        
        
    def find_similar_neighbors(self):
        '''
        (Optional) Hypothesis: agents relate to similar neighbors. Can be replaced by a homophily method for cell choice. 
        Agents may change cell throughout the simulation and their neighbros may change location or exposure. 
        Input:
            - neighbors: the list of agents that are in the same cell as the agent.
            - deprivation: the sum of exposure of an agent that serves as basis of comparison given a threshold of acceptable difference. 
            - group: the generic group the agent belongs to, serves as basis for comparison
            
        Returns: 
            - similar_neighbors: the list of agents that are in the same cell and are also similar in respect to group and deprivation. 
            
        Model parameters: 
            - model.threshold_similar
        
        '''
        similar_neighbors=[]
        
        for a in self.neighbors:
            if (abs(a.deprivation - self.deprivation) <=self.model.threshold_similar) & (a.group==self.group):
                similar_neighbors.append(a)
        
        self.similar_neighbors=similar_neighbors


        
    def sprout(self):
        if random.random()<self.model.birth_rate:
            a=NormalAgent(len(self.model.schedule.agents),self.model)
            a.group=self.group
            a.exposure=self.exposure
           
            self.model.schedule.add(a)
            #position in the same cell as agent
            self.model.grid.place_agent(a,self.pos)
            self.model.G.add_node(a,state=a.state, behaviour=a.behaviour, deprivation=a.deprivation)
            

            self.model.G.add_edge(self,a)#link to parent
            a.make_friends()
                

            
        
    ######################################
    # Updating Exogenous attributes
    ######################################
            
    def switch_exposure(self,index="random"):
        '''
        Enable agents to spontaneously change a given dimension of deprivation. This relates to the hypothesis that agents can move change their conditions of living or their income etc. In the model, this change in exogenous in the sense that the process is not dependant on any attributes or vairalbes of the model.

        Hypothesis: 
            - agent's deprivation dimensions can change throughout a life course
            - the causes of the change are not specified in the model, the process is random
            - life course deprivation can be a risk factor for behaviour or state
            - the different dimensions may have different impact

        Input: 
            - model.change_rate: parameter in the model defining the particular probability of mutation
            - index: the specific dimension of the agent exposure that changes. This could be random or not. 

        Updates: 
            - previous_exposure: accounts for evolution of the exposure in case temporal dependancies are added in the model.
            - exposure: the binary exposure array with mutation or not (switch from one possible state to the other)

        Model_paramters: 
            - change_rate between 0 and 1
        '''
        #store exposure in history of the agent: see what to do with it or it can be accessed some other way
        self.previous_exposure.append(self.exposure)
        
        #spatial component of exposure: exposure in one dimension depends on the area: seperated in two:
        if 'spatial' in self.model.dependancies:
            self.exposure[0]=(self.pos[0]<self.model.width/2)
            
        if self.age in [25,60]:#agent can switch 2 times exposure, once at 25, once at 60 
            if index=="random":#change the other two randomly
                #Allow one dimension to change with a given change_rate: change randomly up or down
                ind=random.randint(int('spatial' in self.model.dependancies),len(self.exposure))
                if random.random()<self.model.change_exposure_rate:
                    #change the state of the exposure in dimension index. 
                    self.exposure[ind]=int(self.exposure[ind]==0)
            else:
                #change a specific exposure
                self.exposure[index]=int(self.exposure[index]==0)

            

            
            
    def move_agent(self, position="random"):
        ''''
        (Optional) Enable agents to move cell or neighbors randomly or given rules specified in the model.
        
        Hypothesis: 
            - agents have long term mobility
        Input: 
            - position: a new position to move agents to. 
            
        Updates:
            - pos: the agent is in a new position on the grid
        '''
        #next step: make it possible to move randomly, with maximum likelihood etc
        self.model.grid.move_agent(self,position)

        
        
    
    
    def switch_neighborhood(self):
        '''
        (Optional) Enable agents to move neighborhood based on the similarity of their neighbors. 
        
        Hypothesis: 
            - agents are mobile
            - agents rather live near similar agents
            - agents have a threshold of similar wanted in a neighborhood. 
            
        Input:
            - similar_neighbors
            - neighbors
            - model.threshold_segregation
        Update:
            - pos: position updated given the similarity of neighbors. The process is random. 
        Model paramter: 
            - model.threshold_segregation
        '''
        
        self.find_similar_neighbors()
        percent_similar=self.similar_neighbors/self.neighbors
        #if percent_similar<self.model.threshold_segregation :
            #self.move(random): CHOOSE RANDOM LOCATION
            
            
    ######################################
    # Measuring risk
    ######################################   
    
    def vulnerability_exposure(self,outcome="behaviour"):
        '''
        Compute the odds of a particular exposure with respect to a deprivation of 0. The increase in likelihood of an outcome (behaviour or death) given the exposure array of an agent. The odds are parameters given by the model
        Hypothesis: 
            - Deprivation can increase the probability of adopting a behaviour
            - Deprivation can increase the probability of dying
            - specific dimensions of deprivation will have different impact on the odds of outcome
            - The effects are additive (the odds are multiplied)

        Input: 
            - exposure: the state of the agent on the different dimensions
            - model.exposure_death_odds: percentage of "increase": odds - 1 (in order to add 1 to all elements in array for product)
            - model.exposure_behaviour_odds: percentage of increase of odds for adopting behaviour (odds - 1)

        Returns: 
            - product of odds for different dimensions of the exposure. 

        Model parameters: 
            - model.exposure_adoption_odds

        '''
        
        if outcome=="behaviour":     
            odds_array=(self.exposure*self.model.odds_exposure_adopt)+1#add one to all dimensions 
        elif outcome=="death":
            odds_array=(self.exposure*self.model.odds_exposure_death)+1#add one to all dimensions
        else:
            print(outcome, "The only possible outcomes are death and behaviour")
        return (odds_array.prod())
    
    
    
    def compute_relative_deprivation(self):
        #hypothesis: only effect behaviour: on all dimensions where the person is deprived, they will compare with their friends that are not deprived. the deprivation is minimised if they also have a lot of friends that are deprived. 
        self.true_relative_deprivation=sum(self.exposure*(len(self.friends)-sum([a.exposure for a in self.friends]))/(sum([a.exposure for a in self.friends])+1))
        
    def vulnerability_relative_deprivation(self):
        self.compute_relative_deprivation()
        if (self.true_relative_deprivation>0) and ("relative_deprivation" in self.model.dependancies):
            return((self.true_relative_deprivation*self.model.odds_relative_deprivation))
        else: 
            return(1)
    
    def vulnerability_age(self,outcome="behaviour",trend="gauss"):
        '''
        Compute the odds of adopting or dying based on the age of the agent. The increase in likelihood of one of these actions. 
        Hypothesis: 
            - a specific age group is more vulnerable to the dangerous behaviour
            - the older the agent the more likely they are to die
            - the odds of dying converge to infinity for age close to 100 (probability -> 1)
            - the vulnerability to behaviour follows a gaussien (or lognormal) around 25 years old. 
            
        Input: 
            - trend: the type of function of sensitivity to behaviour
            - outcome: what vulnerability is being computed, action of dying or adopting behaviour
            - age: the age of the agent
        Returns: 
            - odds: the increase in likelihood of dying or adopting the behaviour compared to age group with no risk (an reference agent with odds=1 such as age=0 for example)
        '''
        
        if outcome=="behaviour":

            if trend=="gauss":
                #see to set parameters later
                odds=(math.exp(-(self.age-25)**2/98))+1
            elif trend=="lognormal":
                #see to set parameters later
                odds=((1/self.age)*math.exp(-(math.log(self.age)-math.log(40))**2/2))+1
            elif trend=='stair':
                odds=int(self.age in [15,40])+1
                
        elif outcome=="death":
            odds=(1/(1-(self.age/100)**2))
        else: 
            print(outcome, "outcome for age can only be death and behavior")
            
        return(odds)
    
    
    def vulnerability_network(self):
        '''
        (Optional) Compute the odds of adopting a behaviour given the adoption rate of friends (that are similar, or all friends)
        Hypothesis: 
            - adoption is more probable if friends have already adopted the behaviour
            - the increase in likelihood is linearly dependant on the percentage of friends that have adopted the beahivour
            - the percentage could be computed for only similar friends: to be determined
        Input: 
            - friends
        Returns: 
            - odds: the increase in likelihood given the percentage of friend adopters. 
        
        '''
        odds=1
        if (len(self.friends)>0) and ("network"in self.model.dependancies) :
            friend_behaviour=[a.behaviour for a in self.friends]
            percent_adopt=sum(friend_behaviour)/len(self.friends)
            odds=percent_adopt+1
        else:
            odds=1
            #think this through!!!!!!
        return (odds)
    
    def vulnerability_behaviour(self):
        '''
        Compute the odds of dying when behaviour =1 as opposed to 0. 
        Input: 
            - behaviour
            - model.odds_behaivour_death
        REturn: 
            - odds: the increase in likelihood of dying when behaviour =1
        
        Model parameters: 
            - model.odds_behaivour_death
       
        '''
        odds=1
        if self.behaviour==1: 
            odds=self.model.odds_behaviour_death
        else:
            odds=1
    
        return(odds)
    
    
    ###################################
    # Updating agent state
    ###################################
    
    
    def compute_prob_adopt(self):
        '''
        Compute the probability of adopting the behaviour given the vulnaribility of the agent. The probability is calculated from the different odds ratio with the reference agent being a non deprived, age 0 , group 0, no friends (or none who have adopted), etc. The susceptibility will be the same for all agents with the same characteristics. 
        
        Hypothesis: 
            - all of the above: the odds are additive: no interaction in this version
            - if age below 15 or above 50, the probability of adopting if it has not yet been adopted is null.
            - adopting depends on exposure, age, network, group
        Input: 
            - model.odds_group: the odds ratio of group 1 in regards to group 0. 
            - model.adopt_prob: the reference prob of adopting the behaviour with no risk factor: the "natural" risk.
            
        Return: 
            - prob: the probability of adopting the behaviour at step t
            
        Model_parameter: 
            - model.adopt_prob
        '''
        
        constant_rate=self.model.adopt_prob/(1 - self.model.adopt_prob)
        total_odds=constant_rate\
        *self.vulnerability_exposure()\
        *self.vulnerability_relative_deprivation()\
        *self.vulnerability_age(outcome="behaviour",trend="gauss")\
        *self.vulnerability_network()\
        *self.model.odds_group
        prob=(self.age in range(15,60))*total_odds/(1+total_odds)
        #prob=(self.age in range(0,100))*self.model.adopt_prob/(self.model.adopt_prob + (1/total_odds)*(1-self.model.adopt_prob))
        #interpretation: the agent is total_odds more likely to adopt the behaviour than the reference person: no dep, no friends with drugs, 0 years old... (add the option that people less than 15 cannot adopt and people more than 50 cannot)
        return(prob)
            
        
    def compute_prob_drop(self):
        '''
        Compute the probability of dropping the behaviour given the vulnaribility of the agent. The probability is calculated from the different odds ratio with the reference agent being a non deprived, age 0 , group 0, no friends (or none who have adopted), etc. The susceptibility will be the same for all agents with the same characteristics. 
        
        Hypothesis: 
            - it is possible to drop the behaivour once adopted
            - the drop rate will depend on the same risk factors as for the adoption
            - the odds are inverse for the probability of dropping: odd_drop=1/odd_adopt
        Input: 
            - model.odds_group: the odds ratio of group 1 in regards to group 0. 
            - model.drop_prob: the reference prob of dropping the behaviour with no risk factor: the "natural" probability of dropping out.
            
        Return: 
            - prob: the probability of dropping the behaviour at step t
            
        Model_parameters: 
            - model.drop_prob
        '''
        constant_rate=self.model.drop_prob/(1 - self.model.drop_prob)
        total_odds=self.vulnerability_exposure()\
                   *self.vulnerability_relative_deprivation()\
                   *self.vulnerability_age("behaviour")\
                   *self.vulnerability_network()\
                   *self.model.odds_group
        prob=constant_rate*(1/total_odds)/(1+constant_rate*(1/total_odds))
        #prob=(self.model.drop_prob/(self.model.drop_prob + total_odds*(1-self.model.adopt_prob)))
        #interpretation: all the factors that make the agent more susceptible to bheaviour also make them less susceptible to drop the same behaviour. Should also add things like social deprivation and such later. 
        return(prob)
         
        

    def update_behaviour(self):
        """
        Decision to adopt the behaviour or drop the beahviour.
        Hypothesis: 
            - each agent has a personal sensitivity to the behaviour (between 0 and 1)
            - time of behaviour is accounted for (to be used in temporal version)
            - an agent's attitude towards dropping the behaviour is inverse to adoption
        Input: 
            - prob_adopt: from agent method prob_adopt()
            - sensitivity_behaviour
        Update: 
            - behaviour: 0 or 1
            - time: +1 if not dropped or 0 if dropped. 
        
        """
        #try to isolate the test for different death causes: is it behaviour or other.
        #self.update_var_from_BNN('behaviour'):check what to do about the drop rates etc
        self.previous_behaviour.append(self.behaviour)
        
        if self.behaviour==0:
            prob_adopt=self.sensitivity_behaviour*self.compute_prob_adopt()
            self.prob_adopt=prob_adopt
            #print(prob_adopt)
            if random.random()<self.prob_adopt:
                self.behaviour=1
                self.time=1
        else:
            #possibility of drop rate depending on the time of the behaviour?
            prob_drop=(1/self.sensitivity_behaviour)*self.compute_prob_drop()
            self.prob_drop=prob_drop
            if random.random()<self.prob_drop:
                self.behaviour=0
                self.time=0
            else:
                self.time+=1
                

            
    def die(self):
        '''
        Possibility of dying for all agents at each step. 
        Hypothesis: 
            - death is impacted by age, behaviour and deprivation 
            - probability of dying at 100 is 1
            - 
        '''
        if self.age<100:
                   
            constant_rate=self.model.death_prob/(1 - self.model.death_prob)
            
            total_odds=constant_rate\
                   *self.vulnerability_exposure(outcome="death")\
                   *self.vulnerability_age(outcome="death",trend="gauss")\
                   *self.vulnerability_behaviour()
            
            odds_without_behaviour=total_odds/self.vulnerability_behaviour()
            
            prob=total_odds/(1+total_odds)#(self.model.death_prob/(self.model.death_prob + (1/total_odds)*(1-self.model.death_prob)))
            prob_without_behaviour=odds_without_behaviour/(1+odds_without_behaviour)
        
        else:
                   
            prob=1#(self.model.death_prob/(self.model.death_prob + (1/total_odds)*(1-self.model.death_prob)))
            prob_without_behaviour=1#all die at 100
                   
        self.prob_die=prob  
        random_test=random.random()
                   
        if random_test<self.prob_die:
            self.state=1
            #remove from network if there is one       
            if nx.is_empty(self.model.G)==False:
                self.model.G.remove_node(self)
            #add to dead agents       
            self.model.dead_agents.append(self)
            #check cause of death: is it behaviour or other     
            if random_test>=prob_without_behaviour:
                self.model.dead_from_behaviour+=1
            
     
              
    def step(self):
        if self.state==0:
            self.age+=1
            self.find_friends()
            if self.model.network_dynamic=='progressive':
                self.make_friends()
            self.sprout()
            self.update_behaviour()
            self.die()
        else:
            self.previous_state=1#set previous state to 1
            



            
            
            
            


        
class DAGAgent(Agent):
    
    """ An agent with fixed initial wealth."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        #create attributes of agent that are in the initial bn, including past states if these have an effect of the outcome 
        init_from_BBN(self,self.model.BBN,self.model.analysis)
          
    def find_friends(self):
        '''
        Network method: 
        Get neihgbours on the Graph initialised with the model.
        
        Return: 
            - friends: list of agent instances
                       list of friends linked to the agent in the model.graph
        '''
        
        #find the neighbors in the graph not in the grid
        if nx.is_empty(self.model.G)==False:
            self.friends=self.model.G[self]
            
        else:
            print("The model graph is empty")
    
    
    def find_neighbors(self):
        #the agents that are in the same "area" as the agent self
        self.neighbors=self.model.grid.get_cell_list_contents(self.pos)
        
         
    ######################################
    # Updating Exogenous attributes
    ######################################
            
    def switch_exposure(self,change_rate,index="random"):
        '''
        Enable agents to get out or in deprivation by changing one of their states based on some probability: probably need to update the others if there is some dependancy structure in there. 
        '''
        #store exposure in history of the agent: see what to do with it or it can be accessed some other way
        self.previous_exposure.append(self.exposure)
        
        if index=="random":
            #Allow one dimension to change with a given change_rate: change randomly up or down
            ind=random.randint(0,len(self.exposure))
            if random.random()<change_rate:
                #change the state of the exposure in dimension index. 
                self.exposure[ind]=int(self.exposure[ind]==0)
        else:
            #change a specific exposure
            self.exposure[index]=int(self.exposure[index]==0)
            
            

    def move_agent(self, position="random"):
        ''''
        For visualisation: move the agent to the location of the bar. 
        '''
        #next step: make it possible to move randomly, with maximum likelihood etc
        self.model.grid.move_agent(self,position)

     
    
    ##########################################
    # Updating dependant variables
    ##########################################
    
    
    def update_dependant_from_BNN(self,variable):
        #possibly if time one version where the abm is based on a more complicated dag to validate the framework and create possible score. If not score can be linked to the prediction and the 
        #make it as a "prediction" instead of an update!! and not from the intial one but from the learned one!
        """
        Update a state of agent for a particular variable given the BNN: this is the same as using the initial structure and cpt, but creates a framework for more complex decisions and evolving BNN. 
        """
        #keep the previous value in an attribute if the previous value is a parent in the BBN: 
        previous='past_'+variable
        if previous in self.model.BBN.structure[variable]['parents']:
            setattr(self,previous, self.variable)
        
       #infer next value for the variable outcome 
        outcome_prob=self.model.BBN.try_for_evidence(self,variable)#is a series so check if it works with np.ranomd.choice
        
        #choose one of outcome with the marignal probablities found
        outcome=np.random.choice(self.model.BBN.structure[variable]['outcome'],p=outcome_prob)
 

    def update_time(self):
        if self.behaviour==1:
            self.time_behaviour+=1
        else:
            self.time_behaviour=0
    
    
    def update_deprivation_measures(self):
        self.deprivation=sum(self.exposure)#weighted or not change formula for that
        self.area_deprivation=np.mean([a.deprivation for a in self.neighbors])#check the get neighbors function!!
        self.network_deprivation=np.mean([a.deprivation for a in self.social_network])#get the neighbors in the network
        self.relative_deprivation=(self.deprivation-self.network_deprivation)/self.network_deprivation #check the literature for the typical structure of relative deprivation. 
        
        
        
    def logit_behaviour(self):
        
        logit= (b_0 + b_age*self.age 
        + b_group*self.group 
        + b_e[0]*self.exposure[0] )#for instance suppose only one of them actually accounts for drug use. 
        
        prob=math.exp(logit)/(1+math.exp(logit))
        return(prob)
        
    def logit_exposure(self):
        logit=e0 + e_1*self.exposure[1] #= others depending on assumptions
        
        
        
        
        
        
        
        
        
        
        
    def update_behaviour(self,adopt_rate,drop_rate):
        """
        BASIC METHOD. Agent adopts dangerous beahviour based on probabilities calculated a random way
        """
        #self.update_var_from_BNN('behaviour'):check what to do about the drop rates etc
        
        if self.behavior==0:
            if random.random()<adopt_rate:
                self.behaviour==1
                self.time=1
        else:
            if randrm.random()<drop_rate:
                self.behaviour==0
                self.phases.append(self.time)
                self.time=0
            else:
                self.time+=1

            
                
      
    
    def prediction_logistic(self):
        #
        pass
    
    
    def other_methods(self):
        pass
        
       
    def distance_from_self(self,other):
        '''
        Euclidean distance from other agent or located object
        '''
        return (math.sqrt((self.pos[0]-other.pos[0])**2+(self.pos[1]-other.pos[1])**2))
    
    
            
                     
    
        '''    def adopt_drug(self):
                #think of adding the possibility of leaving drugs aswell: maybe in the network version
                #think of adding a BBN instead of the two step decision compute probability and adopt drug: the BBN can decide by "predicting"  what it will be: probabilistic BNN. just as a structure in the decision process: when process is added, is it possible to measure how "far" from the BBN we have gone for the individual agent? like a possible inerval over probabilities or maximum expected difference of the probability computed for decision. 
        '''

    
   


