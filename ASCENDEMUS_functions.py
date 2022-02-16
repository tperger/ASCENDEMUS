# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:23:50 2022

@author: perger
"""
import pandas as pd
import numpy as np
import math
from pathlib import Path
import statistics
import pyam
import FRESH_clustering

def settlement_pattern_algorithm(building_types=None,
                                 settlement_patterns=None,
                                 buildings_per_SP=None,
                                 country=None,
                                 level=None):
    
    
    n_city = buildings_per_SP['city']
    n_town = buildings_per_SP['town']
    n_suburban = buildings_per_SP['suburban']
    n_rural = buildings_per_SP['rural']

    # Percentage to distribute LABs between 'city' and 'suburban'
    p_LAB_city = 0.75 # default value, might be higher if needed
    p_LAB_suburban = 1 - p_LAB_city

    th_LAB_city = 0.04 # threshold for city communites in municipality

    # load country data
    if country == 'Austria':
        filename_buildings = 'Buildings_Austria_2011.csv'
        level_districts = 4
    if country == 'Greece':
        filename_buildings = 'Buildings_Greece_2011.csv'
        level_districts = 5
    if country == 'Spain':
        filename_buildings = 'Buildings_Spain_2011.csv'
        level_districts = 3
    
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
                
        # suburban and city 
        if N_LAB / N_total < th_LAB_city:
            N_city = 0
            N_suburban = math.floor(N_LAB / n_suburban['LAB'])
        else:
            if N_SH / n_suburban['SH'] < math.floor(p_LAB_suburban 
                                                 * N_LAB
                                                 / n_suburban['LAB']):
                p_LAB_suburban = (N_SH / N_LAB * n_suburban['LAB'] / n_suburban['SH']) 
                p_LAB_city = 1 - p_LAB_suburban
                
            N_suburban = math.floor(
                p_LAB_suburban * N_LAB / n_suburban['LAB'])        
            N_city = math.floor(
                 (N_LAB - N_suburban * n_suburban['LAB']) 
                 / n_city['LAB'])
               
        # rural
        N_rural = math.floor(
            (N_SH - N_suburban * n_suburban['SH']) / n_rural['SH'])
        
        # assignment
        df_SP.loc[index, 'city'] = N_city
        df_SP.loc[index, 'town'] = N_town
        df_SP.loc[index, 'suburban'] = N_suburban
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
    Settlement pattern options: city, town, suburban, rural
    Region options: Austria, Greece, Norway, Spain, UK
        
    """
    
    if region_name not in ['Austria', 'Greece', 'Norway', 'Spain', 'UK']:
        raise Exception('Selected country not in list of available countries')
        
    if settlement_pattern not in ['city', 'town', 'suburban', 'rural']:
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