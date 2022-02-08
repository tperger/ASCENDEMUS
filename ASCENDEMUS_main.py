# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:53:12 2022

@author: perger
"""

import pandas as pd
import numpy as np
from numpy import matlib
import math
from pathlib import Path
import statistics
from datetime import timedelta, datetime
import pyam
import glob
import FRESH_LP
import FRESH_clustering


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

country = 'Greece'
scenario_name = 'Default scenario'
model_name = 'FRESH:COM v2.0'

solver_name = 'gurobi'

building_types = ['SH', 'SAB', 'LAB']
settlement_patterns = ['city', 'town', 'mixed', 'rural']

buildings_per_SP ={'city': {'SH': 0, 'SAB': 0, 'LAB': 10}, #city 
                   'town': {'SH': 0, 'SAB': 10, 'LAB': 0}, #town
                   'mixed': {'SH': 10, 'SAB': 0, 'LAB': 2}, #mixed
                   'rural': {'SH': 10, 'SAB': 0, 'LAB': 0} #rural
                   }

def settlement_pattern_algorithm(building_types=None,
                                 settlement_patterns=None,
                                 buildings_per_SP=None,
                                 country=None,
                                 level=None):
    
    #n = pd.DataFrame(data=buildings_per_SP, 
    #                 index=settlement_patterns)
    
    n_city = buildings_per_SP['city']
    n_town = buildings_per_SP['town']
    n_mixed = buildings_per_SP['mixed']
    n_rural = buildings_per_SP['rural']

    # Percentage to distribute LABs between 'city' and 'mixed'
    p_LAB_city = 0.75 # default value, might be higher if needed
    p_LAB_mixed = 1 - p_LAB_city

    th_LAB_city = 0.04 # threshold for city communites in municipality

    # load country data
    if country == 'Austria':
        filename_buildings = 'Buildings_Austria_2011.csv'
        level_districts = 4
    if country == 'Greece':
        filename_buildings = 'Buildings_Greece_2011.csv'
        level_districts = 5
    
    filename_demand = 'Electricity_demand_households.csv'

    COUNTRY_PATH = Path(__file__).parent / country

    df = pd.read_csv(COUNTRY_PATH / filename_buildings, sep=';')

    df_SP = pd.DataFrame(columns=['Level','Code'])
    df_SP['Level'] = df.Level
    df_SP['Code'] = df.Code
    
    # PART 1
    # iterating through political districts or regions (actual SP algorithm)
    for index, row in df.iterrows():
        
        N_SH = row['SH']
        N_SAB = row['SAB']
        N_LAB = row['LAB']
        N_total = N_SH + N_SAB + N_LAB
        
        # town
        N_town = math.floor(N_SAB / n_town['SAB'])
                
        # mixed and city 
        if N_LAB / N_total < th_LAB_city:
            N_city = 0
            N_mixed = math.floor(N_LAB / n_mixed['LAB'])
        else:
            if N_SH / n_mixed['SH'] < math.floor(p_LAB_mixed 
                                                 * N_LAB
                                                 / n_mixed['LAB']):
                p_LAB_mixed = (N_SH / N_LAB * n_mixed['LAB'] / n_mixed['SH']) 
                p_LAB_city = 1 - p_LAB_mixed
                
            N_mixed = math.floor(
                p_LAB_mixed * N_LAB / n_mixed['LAB'])        
            N_city = math.floor(
                 (N_LAB - N_mixed * n_mixed['LAB']) 
                 / n_city['LAB'])
               
        # rural
        N_rural = math.floor(
            (N_SH - N_mixed * n_mixed['SH']) / n_rural['SH'])
        
        # assignment
        df_SP.loc[index, 'city'] = N_city
        df_SP.loc[index, 'town'] = N_town
        df_SP.loc[index, 'mixed'] = N_mixed
        df_SP.loc[index, 'rural'] = N_rural
        
    # PART 2
    # checking electricity demand of the country

    demand_data = pd.read_csv(filename_demand,
                              sep=';', 
                              index_col=0)

    # number of ECs per SP (whole country)
    # total building demand within the settlement pattern (whole country)
    results_per_SP = pd.DataFrame(index=settlement_patterns, 
                                  columns=['number of ECs', 
                                           'demand per SP' ])
    
    for i in settlement_patterns:
        results_per_SP.loc[i, 'number of ECs'] = sum(
            df_SP[(df.Level==level_districts)][i])
        results_per_SP.loc[i, 'demand per SP'] = (
            results_per_SP.loc[i, 'number of ECs'] 
            * sum(
                buildings_per_SP[i][j] 
                * demand_data[demand_data.country == country].loc[
                    'Average dwellings per building', j]
                * demand_data[demand_data.country == country].loc[
                    'Average electricity consumption per dwelling (kWh/a)', j]
                for j in building_types
                )
            )
    
    demand_buildings = {} # contains annual demand values per building type
    for i in building_types:
        mu = (
            demand_data[demand_data.country == country].loc[
                'Average dwellings per building', i] 
            * demand_data[demand_data.country == country].loc[
                'Average electricity consumption per dwelling (kWh/a)', i]
            )
        sigma = 0.3 * mu
        norm_dist = statistics.NormalDist(mu=mu, sigma=sigma)
        demand_buildings[i] = [norm_dist.inv_cdf(i/10+0.05) 
                               for i in range(0,10)]
        
    return df, df_SP, results_per_SP, demand_buildings

def define_community(settlement_pattern=None,
                     buildings_per_SP=None,
                     model_name=None,   
                     scenario_name=None,
                     region_name=None,
                     year=None,
                     clustering=False):
    
    
    """
    Settlement pattern options: city, town, mixed, rural
    Region options: Austria, Greece, Norway, Spain, UK
        
    """
    
    if region_name not in ['Austria', 'Greece', 'Norway', 'Spain', 'UK']:
        raise Exception('Selected country not in list of available countries')
        
    if settlement_pattern not in ['city', 'town', 'mixed', 'rural']:
        raise Exception('Selected settlement pattern not available')
        
    # Read Input Data (from the IAMC Format)

    # input data of prosumer
    
    PATH_FILES = Path(__file__).parent / 'Community data'
            
    n = buildings_per_SP[settlement_pattern]
    prosumer = (['Prosumer LAB '+str(i+1) for i in range(n['LAB'])]
                + ['Prosumer SAB '+str(i+1) for i in range(n['SAB'])]
                + ['Prosumer SH '+str(i+1) for i in range(n['SH'])]
                )    
                   
    # IAMC variable names: Electricity demand, PV generation, other prosumer data
    # load_var = 'Final Energy|Residential and Commercial|Electricity'
    # PV_var = 'Secondary Energy|Electricity|Solar|PV'
    SoC_max = 'Maximum Storage|Electricity|Energy Storage System'
    SoC_min = 'Minimum Storage|Electricity|Energy Storage System'
    q_bat_max = 'Maximum Charge|Electricity|Energy Storage System'
    q_bat_min = 'Maximum Discharge|Electricity|Energy Storage System'
    PV_capacity = 'Maximum Active power|Electricity|Solar'
    w = 'Price|Carbon'
    prosumer_var = [w, SoC_max, SoC_min, q_bat_max, q_bat_min, PV_capacity]
    
    load = pd.DataFrame()
    PV = pd.DataFrame()
    prosumer_data = pd.DataFrame()
    emissions = pd.DataFrame()
    
    # Prosumer data
    for i in prosumer:
        _filename = i+'.csv'
        _df = pyam.IamDataFrame(PATH_FILES / _filename)
        _data = (_df
            .filter(region=region_name)
            .filter(model=model_name)
            .filter(scenario=scenario_name)
            .filter(year=year))
        load[i] = (_data
            .filter(
                variable='Final Energy|Residential and Commercial|Electricity')
            .as_pandas().set_index('time').value)
        PV[i] = (_data
            .filter(
                variable='Secondary Energy|Electricity|Solar|PV')
            .as_pandas().set_index('time').value)
        # prosumer data DataFrame
        prosumer_data[i] = (_data
            .filter(variable=prosumer_var)
            .as_pandas().set_index('variable').value)
        
    # Grid data
    _df = pyam.IamDataFrame(data='Grid_data.csv', sep=';')
    _data = (_df
            .filter(region=region_name)
            .filter(model=model_name)
            .filter(scenario=scenario_name)
            .filter(year=year))
    p_grid_in = (_data
                 .filter(
                     variable='Price|Final Energy|Residential|Electricity')['value']
                 .values[0]/1000) # price EUR/kWh
    p_grid_out = (_data
                  .filter(
                      variable='Price|Secondary Energy|Electricity')['value']
                  .values[0]/1000) # price EUR/kWh
    emissions['Emissions'] = (_data
                 .filter(variable='Emissions|CO2')
                 .as_pandas().set_index('time').value)
    
    time_steps = load.index.tolist()
    
    if clustering:
        k = 3 # number of representative days
        hours = 24 # time steps of representative days       
        (emissions, load, PV, 
         time_steps, counts) = FRESH_clustering.cluster_input(prosumer, 
                                                              emissions, 
                                                              load, 
                                                              PV, 
                                                              k, 
                                                              hours)
        _data = np.repeat(counts, k*[hours])
        weight = pd.DataFrame(_data, index=time_steps, columns=['weight'])
    else:
        _data = [1]*8760
        weight = pd.DataFrame(_data, index=time_steps, columns=['weight'])
        
    # Other values
    distances = pd.read_csv('Distances_'+settlement_pattern+'.csv',
                          sep=';',
                          header=0, 
                          index_col='Prosumer')
    
    time_steps = load.index.tolist()
    
    return (load, PV, prosumer_data, prosumer, 
            emissions, p_grid_in, p_grid_out, weight, distances, time_steps)

df, df_SP, results_per_SP, demand_buildings = settlement_pattern_algorithm(
    building_types,
    settlement_patterns,
    buildings_per_SP,
    country,
    level=4)

(load, PV, prosumer_data, prosumer, emissions, 
  p_grid_in, p_grid_out, weight, distances, time_steps) = define_community(
    'rural', 
    buildings_per_SP, 
    model_name, 
    scenario_name, 
    country, 
    2019, 
    False)
            
# results, q_share_total, social_welfare = FRESH_LP.run_LP(
#     load, PV, emissions, p_grid_in, p_grid_out, prosumer_data, time_steps,
#     prosumer, weight, solver_name, distances)
            