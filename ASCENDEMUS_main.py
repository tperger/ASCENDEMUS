# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:53:12 2022

@author: perger
"""

import pandas as pd
import numpy as np
import math
from pathlib import Path
#import statistics
#from datetime import timedelta, datetime
#import pyam
import FRESH_LP
#import FRESH_clustering
import ASCENDEMUS_functions as af

""" 

1. FRESH_define_community.py 
    input arguments: region, scenario, settlement pattern
2. FRESH_LP.py
    LP is executed for all settlement patterns and results are saved 
3. Settlement pattern algorithm.py
    number of energy communities per region (theoretical)
    input arguments: region
4. adapt number of communities per scenario?? 
5. evaluate results (savings etc.)

""" 

country = 'Austria'
scenario_name = 'Default scenario'
model_name = 'FRESH:COM v2.0'

solver_name = 'gurobi'

building_types = ['SH', 'SAB', 'LAB']
settlement_patterns = ['city', 'town', 'suburban', 'rural']

buildings_per_SP ={'city': {'SH': 0, 'SAB': 0, 'LAB': 10}, #city 
                   'town': {'SH': 0, 'SAB': 10, 'LAB': 0}, #town
                   'suburban': {'SH': 10, 'SAB': 0, 'LAB': 2}, #suburban
                   'rural': {'SH': 10, 'SAB': 0, 'LAB': 0} #rural
                   }

    

df, df_SP, results_per_SP, demand_buildings = af.settlement_pattern_algorithm(
    building_types,
    settlement_patterns,
    buildings_per_SP,
    country,
    level=5)

cols = ['buying grid', 'selling grid', 'battery charging',
       'battery discharging', 'self-consumption', 'buying community',
       'selling community', 'emissions', 'costs']
results_wo_comm = pd.DataFrame(index=settlement_patterns, columns=cols)
results_with_comm = pd.DataFrame(index=settlement_patterns, columns=cols)

for SP in settlement_patterns:

    (load, PV, prosumer_data, prosumer, emissions, 
      p_grid_in, p_grid_out, weight, distances, time_steps) = af.define_community(
        SP, 
        buildings_per_SP, 
        model_name, 
        scenario_name, 
        country, 
        2019, 
        False)
    
    if SP == 'rural':
        PV_peak = [5,5,0,3,5,5,0,3,5,0]      
        load *= demand_buildings['SH']
        prosumer_data.loc['Maximum Storage|Electricity|Energy Storage System'] *= 3
        prosumer_data.loc['Maximum Charge|Electricity|Energy Storage System'] *= 2
        prosumer_data.loc['Maximum Discharge|Electricity|Energy Storage System'] *= 2
    elif SP == 'city':
        PV_peak = [15,15,0,10,15,15,0,10,15,0]  
        load *= demand_buildings['LAB']
        prosumer_data.loc['Maximum Storage|Electricity|Energy Storage System'] *= 8
        prosumer_data.loc['Maximum Charge|Electricity|Energy Storage System'] *= 5
        prosumer_data.loc['Maximum Discharge|Electricity|Energy Storage System'] *= 5
    elif SP == 'town':
        PV_peak = [8,8,0,5,8,8,0,5,8,0]  
        load *= demand_buildings['SAB']
        prosumer_data.loc['Maximum Storage|Electricity|Energy Storage System'] *= 5
        prosumer_data.loc['Maximum Charge|Electricity|Energy Storage System'] *= 3
        prosumer_data.loc['Maximum Discharge|Electricity|Energy Storage System'] *= 3
    elif SP == 'suburban':
        PV_peak = [15,10,5,5,0,3,5,5,0,3,5,0]  
        load *= (demand_buildings['LAB'][4:6] + demand_buildings['SH'])
        prosumer_data.loc['Maximum Storage|Electricity|Energy Storage System'] *= 3
        prosumer_data.loc['Maximum Charge|Electricity|Energy Storage System'] *= 2
        prosumer_data.loc['Maximum Discharge|Electricity|Energy Storage System'] *= 2
        
    load /= 1000
    PV *= PV_peak
    
    results, q_share_total, social_welfare = FRESH_LP.run_LP(
        load, PV, emissions, p_grid_in, p_grid_out, prosumer_data, time_steps,
        prosumer, weight, solver_name, distances, sharing=False)
        
    for j in cols:
        results_wo_comm.loc[SP, j] = sum(results[j])
        
    results, q_share_total, social_welfare = FRESH_LP.run_LP(
        load, PV, emissions, p_grid_in, p_grid_out, prosumer_data, time_steps,
        prosumer, weight, solver_name, distances, sharing=True)
        
    for j in cols:
        results_with_comm.loc[SP, j] = sum(results[j])

            