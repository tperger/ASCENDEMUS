# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:51:30 2021

@author: perger
"""

import numpy as np
import pandas as pd
from pyomo.environ import *

def run_LP(
        load, PV, emissions, p_grid_in, p_grid_out, prosumer_data, time_steps,
        prosumer, weight, solver_name, distances):

    # Define some parameters and variables
    SoC_max = 'Maximum Storage|Electricity|Energy Storage System'
    SoC_min = 'Minimum Storage|Electricity|Energy Storage System'
    q_bat_max = 'Maximum Charge|Electricity|Energy Storage System'
    q_bat_min = 'Maximum Discharge|Electricity|Energy Storage System'
    # PV_capacity = 'Maximum Active power|Electricity|Solar'
    w = 'Price|Carbon'
    
    eta_battery = 0.9
    
    index_time = list(range(len(time_steps)))
    
    # deactivate BESS
    # if battery == False:
    #     for i in prosumer:
    #         prosumer_data.loc[SoC_max,i] = 0
    #         prosumer_data.loc[q_bat_max,i] = 0 
    #         prosumer_data.loc[q_bat_min,i] = 0 
    
    # Define model as concrete model
    model = ConcreteModel()
    
    #Define optimization variables 
    model.q_grid_in = Var(time_steps, 
                          prosumer, 
                          within = NonNegativeReals)
    model.q_grid_out = Var(time_steps, 
                           prosumer, 
                           within = NonNegativeReals)
    model.q_share = Var(time_steps, 
                        prosumer, 
                        prosumer, 
                        within = NonNegativeReals)
    model.q_bat_in = Var(time_steps, 
                         prosumer, 
                         within = NonNegativeReals)
    model.q_bat_out = Var(time_steps, 
                          prosumer, 
                          within = NonNegativeReals)
    model.SoC = Var(time_steps, 
                    prosumer, 
                    within = NonNegativeReals)
    
    # Define constraints
    def load_constraint_rule(model, i, t):    
        return (model.q_grid_in[t,i] 
                + model.q_bat_out[t,i] 
                + sum(model.q_share[t,j,i] for j in prosumer)
                - load.loc[t,i] == 0)
    model.load_con = Constraint(prosumer, 
                                time_steps, 
                                rule = load_constraint_rule)
    
    def PV_constraint_rule(model, i, t):    
        return (model.q_grid_out[t,i] 
                + model.q_bat_in[t,i] 
                + sum(model.q_share[t,i,j] for j in prosumer) 
                - PV.loc[t,i] == 0)
    model.PV_con = Constraint(prosumer, 
                              time_steps, 
                              rule = PV_constraint_rule)
    
    def SoC_min_constraint_rule(model, i, t):
        return (model.SoC[t,i] >= prosumer_data.loc[SoC_min][i])
    model.SoC_min_con = Constraint(prosumer, 
                                   time_steps, 
                                   rule = SoC_min_constraint_rule)
    
    def SoC_max_constraint_rule(model, i, t):
        return (model.SoC[t,i] <= prosumer_data.loc[SoC_max][i])
    model.SoC_max_con = Constraint(prosumer, 
                                   time_steps, 
                                   rule = SoC_max_constraint_rule)
    
    def q_bat_in_constraint_rule(model, i, t):
        return (model.q_bat_in[t,i] <= prosumer_data.loc[q_bat_max][i])
    model.q_bat_in_con = Constraint(prosumer, 
                                    time_steps, 
                                    rule = q_bat_in_constraint_rule)
    
    def q_bat_out_constraint_rule(model, i, t):
        return (model.q_bat_out[t,i] <= prosumer_data.loc[q_bat_min][i])
    model.q_bat_out_con = Constraint(prosumer, 
                                     time_steps, 
                                     rule = q_bat_out_constraint_rule)
    
    def SoC_constraint_rule(model, i, t):
        if t == 0:
            return (model.SoC[time_steps[-1],i] 
                    + model.q_bat_in[time_steps[t],i]*eta_battery 
                    - model.q_bat_out[time_steps[t],i]/eta_battery
                    - model.SoC[time_steps[t],i] == 0)
        elif t > 0:
            return (model.SoC[time_steps[t-1],i] 
                    + model.q_bat_in[time_steps[t],i]*eta_battery 
                    - model.q_bat_out[time_steps[t],i]/eta_battery
                    - model.SoC[time_steps[t],i] == 0)
    model.SoC_con = Constraint(prosumer, 
                               index_time, 
                               rule = SoC_constraint_rule)
    
    # Objective function
    community_welfare = {new_list: [] for new_list in prosumer}
    prosumer_welfare = {new_list: [] for new_list in prosumer}
    prosumer_welfare2 = {new_list: [] for new_list in prosumer}
       
    
    for i in prosumer:
        community_welfare[i] = sum((- p_grid_in * model.q_grid_in[t,i]
                                   + p_grid_out * model.q_grid_out[t,i])
                                   * weight.loc[t, 'weight']
                                   for t in time_steps)
        prosumer_welfare[i] = sum((p_grid_in 
                                   + (prosumer_data.loc[w,j]
                                      * (1 - distances.loc[i,j]))
                                   * emissions.loc[t,'Emissions'] / 1000000)
                                  * model.q_share[t,i,j]
                                  * weight.loc[t, 'weight']
                                  for j in prosumer 
                                  for t in time_steps)
        prosumer_welfare2[i] = sum((p_grid_in 
                                    + (prosumer_data.loc[w,i]
                                       * (1 - distances.loc[j,i]))
                                    * emissions.loc[t,'Emissions'] / 1000000)
                                   * model.q_share[t,j,i]
                                   * weight.loc[t, 'weight']
                                   for j in prosumer 
                                   for t in time_steps)
    
        # 1. prosumer i sells to prosumer j
        # 2. prosumer i buys from prosumer j
    
    model.obj = Objective(
        expr = sum(community_welfare[i] 
                   + prosumer_welfare2[i] 
                   for i in prosumer), 
        sense = maximize)
    
    opt = SolverFactory(solver_name)
    opt_success = opt.solve(model)
    
    # Evaluate the results
    social_welfare = value(model.obj)
    
    q_share_total = pd.DataFrame(index=prosumer)
    for j in prosumer:
        a = []
        for i in prosumer:
            a.append(value(sum(model.q_share[t,i,j] * weight.loc[t, 'weight'] 
                               for t in time_steps)))
        q_share_total[j] = a
    
    results= pd.DataFrame(index=prosumer)
    for i in prosumer:
        results.loc[i,'buying grid'] = value(sum(model.q_grid_in[t,i]
                                                 * weight.loc[t, 'weight']
                                                 for t in time_steps))
        results.loc[i,'selling grid'] = value(sum(model.q_grid_out[t,i]
                                                  * weight.loc[t, 'weight']
                                                  for t in time_steps))
        results.loc[i,'battery charging'] = value(sum(model.q_bat_in[t,i]
                                                      * weight.loc[t, 'weight']
                                                      for t in time_steps))
        results.loc[i,'battery discharging'] = value(sum(model.q_bat_out[t,i]
                                                         * weight.loc[t, 'weight']
                                                         for t in time_steps))
        results.loc[i,'self-consumption'] = q_share_total.loc[i,i]
        results.loc[i,'buying community'] = (sum(q_share_total.loc[j,i] 
                                                 for j in prosumer) 
                                             - q_share_total.loc[i,i])
        results.loc[i,'selling community'] = (sum(q_share_total.loc[i,j] 
                                                  for j in prosumer) 
                                              - q_share_total.loc[i,i])
        results.loc[i,'emissions'] = value(sum(model.q_grid_in[t,i]
                                               * weight.loc[t, 'weight']
                                               * emissions.loc[t,'Emissions']
                                               / 1000000 
                                                for t in time_steps))
        results.loc[i,'costs'] = (value(-community_welfare[i]) 
                                      - value(prosumer_welfare[i]) 
                                      + value(prosumer_welfare2[i]))
    
    return results, q_share_total, social_welfare