# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:21:57 2021

@author: david.cheung1
"""
import pandas as pd
from io import StringIO
import glob
import math
from date import date
import logging
from tkinter import Tk     
from tkinter.filedialog import askopenfilename
from datetime import datetime
import os.path

# function #################################################################################################################################################
def convert_dotout_or_excel_to_df(source_fileName: str , grid: int, header_pollutant: str) -> pd.DataFrame:
    if '.out' in source_fileName:
        f = open(f'{directory}/{grid}/{source_fileName}', 'r')
        data = f.read()[2:]
        df = pd.read_csv(StringIO(data))
        df.reset_index(inplace=True)
        df = df[list(df.columns[4:df.shape[1]-1])] # drop unwanted columns
        df['date'] = date_index.loc[:,('date')].astype('int64')
        # df['date'] = date_index['date'].astype('int64')
        df.set_index('date', inplace = True)
        
        # get header from reference excel
        for header_file in glob.glob(f'{directory}/{grid}/{grid}_{header_pollutant}*xlsx'):
            continue
        df1 = pd.read_excel(header_file)        
        df =  df.set_axis(list(df1)[1:], axis=1, inplace=False)

    elif '.xlsx' in source_fileName:
        df = pd.read_excel(f'{directory}/{grid}/{source_fileName}')
        df = df[list(df.columns[1:])]
        df['date'] = date_index.loc[:,('date')].astype('int64')
        # df['date'] = date_index['date'].astype('int64')
        df.set_index('date', inplace = True)
    return(df)

def get_unique_pollutant_in_a_grid(grid: int) -> pd.array:
    totalPollutant = pd.unique(input1Table['AQO_Pollutant'][input1Table.Grid == grid])
    return totalPollutant

def get_background_CONC(pollutant:str, PATH_filename:str, Grid:int) -> pd.DataFrame:    
    background_CONC = pd.read_csv(f'{directory}/{Grid}/{PATH_filename}', delim_whitespace=True)
    background_CONC = background_CONC.drop([0,1])
    background_CONC = background_CONC[background_CONC.columns[4:]]
    background_CONC = background_CONC.astype(str).astype(float)
    background_CONC.reset_index(inplace = True, drop = True)
    background_CONC['NOx'] = background_CONC['NO2'] + background_CONC['NO']
    background_CONC['date'] = date_index['date'].astype('int64')
    background_CONC.set_index('date', inplace = True)
    
    resultdf = background_CONC[f'{pollutant}'].to_frame()
    return resultdf

def jenkin_calculation(CONC: float) -> list:
    OX = float(inputPara['Value'][inputPara.Parameter == 'OX'])
    JK = float(inputPara['Value'][inputPara.Parameter == 'j/k'])
    NO2 = ((CONC + OX + JK) - math.sqrt((CONC + OX + JK)**2 - 4*CONC*OX))/2
    return NO2

def OLM_calculation(pollu_val_df:pd.DataFrame, NO_df:pd.DataFrame) -> pd.DataFrame:
    PATH_filename = pd.unique(pollu_val_df["Revised_PATH_Level1"])[0]
    Grid = pd.unique(pollu_val_df["Grid"])[0]
    pollutant = "O3"
    PATH_background_CONC_df = get_background_CONC(pollutant, PATH_filename, Grid).apply(lambda x: x*(46/48))
    PATH_background_CONC_df = PATH_background_CONC_df.rename(columns={'O3':'PATH_03'})
    NO_df = pd.concat([NO_df, PATH_background_CONC_df], axis=1)
    
    # case: lv1 + lv2 + lv3
    if str(pd.unique(pollu_val_df["ASR_Height_for_PATH_Level3"])[0]) != 'nan':
        PATH2_height = pd.unique(pollu_val_df["ASR_Height_for_PATH_Level2"])[0]
        PATH2_filename = pd.unique(pollu_val_df["PATH_Level2"])[0]
        PATH2_background_CONC_df = get_background_CONC(pollutant, PATH2_filename, Grid).apply(lambda x: x*(46/48))
        PATH2_background_CONC_df = PATH2_background_CONC_df.rename(columns={'O3':'PATH2_03'})
        NO_df = pd.concat([NO_df, PATH2_background_CONC_df], axis=1)
        
        PATH3_height = pd.unique(pollu_val_df["ASR_Height_for_PATH_Level3"])[0]
        PATH3_filename = pd.unique(pollu_val_df["PATH_Level_3"])[0]
        PATH3_background_CONC_df = get_background_CONC(pollutant, PATH3_filename, Grid).apply(lambda x: x*(46/48))
        PATH3_background_CONC_df = PATH3_background_CONC_df.rename(columns={'O3':'PATH3_03'})
        NO_df = pd.concat([NO_df, PATH3_background_CONC_df], axis=1)
        # logging.info('OLM calculation()  case2: lv1 + lv2 + lv3')
        for columns in NO_df:
            # sum up with lv1 PATH
            if float(columns.split('_')[-1]) < PATH2_height:
                NO_df[columns] = NO_df.apply(lambda row: min(row[columns], row["PATH_03"]), axis = 1)
                
            # sum up with lv2 PATH
            elif PATH2_height <= float(columns.split('_')[-1]) and float(columns.split('_')[-1]) < PATH3_height:
                NO_df[columns] = NO_df.apply(lambda row: min(row[columns], row["PATH2_03"]), axis = 1)

            # sum up with lv3 PATH
            elif PATH3_height <= float(columns.split('_')[-1]):
                NO_df[columns] = NO_df.apply(lambda row: min(row[columns], row["PATH3_03"]), axis = 1)              
        
        NO_df = NO_df.drop(columns = ['PATH_03','PATH2_03','PATH3_03'])
    
    # case: lv1 + lv2    
    elif str(pd.unique(pollu_val_df["ASR_Height_for_PATH_Level2"])[0]) != 'nan':
        PATH2_height = pd.unique(pollu_val_df["ASR_Height_for_PATH_Level2"])[0]
        PATH2_filename = pd.unique(pollu_val_df["PATH_Level2"])[0]
        PATH2_background_CONC_df = get_background_CONC(pollutant, PATH2_filename, Grid).apply(lambda x: x*(46/48))
        PATH2_background_CONC_df = PATH2_background_CONC_df.rename(columns={'O3': 'PATH2_03'})
        NO_df = pd.concat([NO_df, PATH2_background_CONC_df], axis=1)
        # logging.info('OLM calculation() case2: lv1 + lv2')
        for columns in NO_df:
            # sum up with lv1 PATH
            if float(columns.split('_')[-1]) < PATH2_height:
                NO_df[columns] = NO_df.apply(lambda row: min(row[columns], row["PATH_03"]), axis = 1)
                
            # sum up with lv2 PATH
            else:
                NO_df[columns] = NO_df.apply(lambda row: min(row[columns], row["PATH2_03"]), axis = 1)
       
        NO_df = NO_df.drop(columns = ['PATH_03','PATH2_03'])

    # case: lv1 only
    else:
        for columns in NO_df:
            NO_df[columns] = NO_df.apply(lambda row: min(row[columns], row["PATH_03"]), axis = 1)
            
        NO_df = NO_df.drop(columns = ['PATH_03'])

    return NO_df
        
def classify_pollutant_from_input1Table(pollu_val: str, jenkin: str) -> str:
    if pollu_val == 'NO2':
        if jenkin == 'Y':
            pollutant = 'NOx'
        else:
            pollutant = 'NO2'
    elif pollu_val == 'RSP':
        pollutant = 'RSP'
    elif pollu_val == 'FSP':
        pollutant = 'FSP'
        
    return pollutant

def classify_NOx_NO(OLM_df: pd.DataFrame)->pd.DataFrame:
    final_df = pd.DataFrame()
    NO_df = pd.DataFrame()
    NO2_df = pd.DataFrame()
    for i in OLM_df.index:
        source_fileName = str(OLM_df['Filename'].loc[i])
        grid = int(OLM_df['Grid'].loc[i])
        header_pollutant = str(OLM_df['AQO_Pollutant'].loc[i])
        ratio = float(OLM_df['NO/NO2_ratio'].loc[i])
        temp_df = convert_dotout_or_excel_to_df(source_fileName, grid, header_pollutant)
        # filter NOx, NO, NO2 in Source_Pollutant column
        if  str(OLM_df['Source_Pollutant'].loc[i]) == 'NOx':

            # get NO from NOx first
            df1 = temp_df.applymap(lambda x: x*ratio)
            if NO_df.shape[0] < 1:
                NO_df = df1
            else:
                NO_df = NO_df + df1
            # get NO2 from NOx
            df1 = temp_df.applymap(lambda x: x*(1 - ratio))                
            if NO2_df.shape[0] < 1:
                NO2_df = df1
            else:
                NO2_df = NO2_df + df1
           
        elif str(OLM_df['Source_Pollutant'].loc[i]) == 'NO':
            if NO_df.shape[0] < 1:
                NO_df = temp_df
            else:
                NO_df = NO_df + temp_df
        
        else:
          
            if NO2_df.shape[0] < 1:
                NO2_df = temp_df
            else:
                NO2_df = NO2_df + temp_df

    if NO_df.shape[0] > 1:
        NO_df = OLM_calculation(OLM_df, NO_df)
        if final_df.shape[0] > 1:
            if NO2_df.shape[0] > 1:
                final_df = final_df + NO_df + NO2_df
            else:
                final_df = final_df + NO_df
        else:
            if NO2_df.shape[0] > 1:
                final_df = NO_df + NO2_df
            else:
                final_df = NO_df
    else:
        if final_df.shape[0] > 1:
            final_df = final_df + NO2_df
        else:
            final_df = NO2_df

    return final_df
    
def PATH_PATH2_PATH3(pollu_val_df: pd.DataFrame, process_df: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    Grid = pd.unique(pollu_val_df["Grid"])[0]

    PATH_filename = pd.unique(pollu_val_df["Revised_PATH_Level1"])[0]
    PATH_background_CONC_df = get_background_CONC(pollutant, PATH_filename, Grid)
    PATH2_height = pd.unique(pollu_val_df["ASR_Height_for_PATH_Level2"])[0]
    PATH2_filename = pd.unique(pollu_val_df["PATH_Level2"])[0]
    PATH2_background_CONC_df = get_background_CONC(pollutant, PATH2_filename, Grid)
    PATH3_height = pd.unique(pollu_val_df["ASR_Height_for_PATH_Level3"])[0]
    PATH3_filename = pd.unique(pollu_val_df["PATH_Level_3"])[0]
    PATH3_background_CONC_df = get_background_CONC(pollutant, PATH3_filename, Grid)
    # logging.info('case: PATH + PATH2 + PATH3')
    for columns in process_df:
        # sum up with lv1 PATH
        if float(columns.split('_')[-1]) < PATH2_height:
            process_df[columns] = process_df[columns] + PATH_background_CONC_df[pollutant]
            
        # sum up with lv2 PATH
        elif PATH2_height <= float(columns.split('_')[-1]) and float(columns.split('_')[-1]) < PATH3_height:
            process_df[columns] = process_df[columns] + PATH2_background_CONC_df[pollutant]

            
        # sum up with lv3 PATH
        elif PATH3_height <= float(columns.split('_')[-1]):
            process_df[columns] = process_df[columns] + PATH3_background_CONC_df[pollutant]
            
    return process_df

def PATH_PATH2(pollu_val_df: pd.DataFrame, process_df: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    Grid = pd.unique(pollu_val_df["Grid"])[0]

    PATH_filename = pd.unique(pollu_val_df["Revised_PATH_Level1"])[0]
    PATH_background_CONC_df = get_background_CONC(pollutant, PATH_filename, Grid)
    PATH2_height = pd.unique(pollu_val_df["ASR_Height_for_PATH_Level2"])[0]
    PATH2_filename = pd.unique(pollu_val_df["PATH_Level2"])[0]
    PATH2_background_CONC_df = get_background_CONC(pollutant, PATH2_filename, Grid)
    # logging.info('case: PATH + PATH2')
    for columns in process_df:
        # sum up with lv1 PATH
        if float(columns.split('_')[-1]) < PATH2_height:
            process_df[columns] = process_df[columns] + PATH_background_CONC_df[pollutant]
            
        # sum up with lv2 PATH
        else:
            process_df[columns] = process_df[columns] + PATH2_background_CONC_df[pollutant]

    return process_df

def jenkin_and_other_pollutant(pollu_val_df: pd.DataFrame) -> pd.DataFrame:
    AQO_Pollutant = str(pd.unique(pollu_val_df['AQO_Pollutant'])[0])
    # 1. fliter input dataframe to NO2 + jenkin if there is Jenkin
    if AQO_Pollutant == 'NO2':
        jenkin = "Y"
        pollu_val_df = pollu_val_df[(pollu_val_df.Jenkin_Method != "O")]
    else:
        jenkin = 'N'
    
    #  2. combine all the raw data file together as process_df  
    pollutant = classify_pollutant_from_input1Table(AQO_Pollutant, jenkin)
    process_df = pd.DataFrame()
    for i in pollu_val_df.index:
        source_fileName = str(pollu_val_df['Filename'].loc[i])
        grid = int(pollu_val_df['Grid'].loc[i])
        header_pollutant = str(pollu_val_df['AQO_Pollutant'].loc[i])
        # logging.info(f'f(x)_jenkin_and_other_pollutant part 2: import data from {source_fileName}')
        if process_df.shape[0] < 1:
            process_df = convert_dotout_or_excel_to_df(source_fileName, grid, header_pollutant)
            
        else:
            process_df = process_df + convert_dotout_or_excel_to_df(source_fileName, grid, header_pollutant)
           
            
    # 3. get background concentration from PATH file + add background concentreation
    PATH_filename = pd.unique(pollu_val_df["Revised_PATH_Level1"])[0]
    Grid = pd.unique(pollu_val_df["Grid"])[0]
    PATH_background_CONC_df = get_background_CONC(pollutant, PATH_filename, Grid)
    
    # case: lv1 + lv2 + lv3
    if str(pd.unique(pollu_val_df["ASR_Height_for_PATH_Level3"])[0]) != 'nan':
        process_df = PATH_PATH2_PATH3(pollu_val_df, process_df, pollutant)
    # case: lv1 + lv2    
    elif str(pd.unique(pollu_val_df["ASR_Height_for_PATH_Level2"])[0]) != 'nan':
        process_df = PATH_PATH2(pollu_val_df, process_df, pollutant)
    # case: lv1 only    
    else:
        for columns in process_df:
            process_df[columns] = process_df[columns] + PATH_background_CONC_df[pollutant]
        
        
    # 4. apply Jenkin calculation if there is Jenkin
    if jenkin == 'Y':
        # logging.info('f(x)_jenkin_and_other_pollutant part 4: apply Jenkin calculation')
        process_df = process_df.applymap(lambda x : jenkin_calculation(x))
       

    return process_df    

def OLM(pollu_val_df: pd.DataFrame) -> pd.DataFrame:
    # 0. create OLM df
    OLM_df = pollu_val_df[(pollu_val_df.AQO_Pollutant == 'NO2') & (pollu_val_df.Jenkin_Method != 'J')]
    OLM_df.reset_index(inplace = True,drop=True)
    # 1. classify into NO OLM, OLM1 only, OLM2 only, OLM3 only into 4 dataframe
    # NO OLM
    OLM0_df = OLM_df[(OLM_df.OLM1 != 'Y') & (OLM_df.OLM2 != 'Y') & (OLM_df.OLM3 != 'Y')]
    OLM0_df.reset_index(inplace = True,drop=True)
    # OLM1 only
    OLM1_df = OLM_df[(OLM_df.OLM1 == 'Y')]
    OLM1_df.reset_index(inplace = True,drop=True)
    # OLM2 only
    OLM2_df = OLM_df[(OLM_df.OLM2 == 'Y')]
    OLM2_df.reset_index(inplace = True,drop=True)
    # OLM3 only
    OLM3_df = OLM_df[(OLM_df.OLM3 == 'Y')]
    OLM3_df.reset_index(inplace = True,drop=True)
    
    # 2. append OLM df to dfs if OLM df has value
    lists = [OLM0_df,OLM1_df,OLM2_df,OLM3_df]
    dfs = []
    for df in lists:
        if df.shape[0] > 1:
            dfs.append(df)
            
    # 3. import raw data from source file in each dataframe
    result_df = pd.DataFrame() 
    for df in dfs:
        if result_df.shape[0] < 1:
            result_df = classify_NOx_NO(df)
        else:
            result_df = result_df + classify_NOx_NO(df)
    # 4. sum up PATH background concentration
    PATH_filename = pd.unique(pollu_val_df["Revised_PATH_Level1"])[0]
    Grid = pd.unique(OLM_df["Grid"])[0]
    pollutant = "NO2"
    PATH_background_CONC_df = get_background_CONC(pollutant, PATH_filename, Grid)
    
    # case: lv1 + lv2 + lv3
    if str(pd.unique(OLM_df["ASR_Height_for_PATH_Level3"])[0]) != 'nan':
        result_df = PATH_PATH2_PATH3(OLM_df, result_df, pollutant)
    
    # case: lv1 + lv2
    elif str(pd.unique(OLM_df["ASR_Height_for_PATH_Level2"])[0]) != 'nan':
        result_df = PATH_PATH2(OLM_df, result_df, pollutant)
    # case: lv1 only
    else:
        for columns in result_df:
            result_df[columns] = result_df[columns] + PATH_background_CONC_df[pollutant]
    
    return result_df

def check_output_file_exist(export_filename:str) -> str:
    if os.path.isfile(f'{export_filename}'):
        mode = "a"
    else:
        mode = "w"
        
    return mode

def check_result_df(result_df:pd.DataFrame):
    if result_df.shape[0] < 1:
        logging.error(f'Group {i}, no result df reture in OLM part')
        
def export_result_df_to_excel(result_df:pd.DataFrame, export_filename:str, group_id:int, group_pollutant:str):
    check_result_df(result_df)
    mode = check_output_file_exist(export_filename)
    # export to output.xlsx        
    with pd.ExcelWriter(f'{export_filename}',mode = mode) as writer:  
                result_df.to_excel(writer, sheet_name=f'{group_id}_{group_pollutant}')
    

if __name__ == '__main__':        
    # logging #################################################################################################################################################
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Script start")
    
    # variable #################################################################################################################################################
    Tk().withdraw()
    logging.info("Please select Cumulative_Input excel in file dialog.")
    filename = askopenfilename()
    inputfile = filename
    directory = inputfile.rsplit('/',1)[0]
    date_index = pd.read_csv(StringIO(date))
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M")
    export_filename = f'{directory}/Output_{dt_string}.xlsx'
    
    input1Table = pd.read_excel(inputfile)
    input1Table.columns = input1Table.iloc[4]
    input1Table = input1Table.iloc[5:,0:18]
    input1Table.reset_index(inplace = True, drop = True)
    input1Table = input1Table.dropna(how='all')
    input1Table.columns = [c.replace(' ', '_') for c in input1Table.columns]
    
    
    inputPara = pd.read_excel(inputfile)
    inputPara = inputPara.iloc[5:,19:21]
    inputPara.reset_index(inplace = True, drop = True)
    inputPara = inputPara.rename(columns={'Unnamed: 19': 'Parameter', 'Unnamed: 20': 'Value'})
    
    totalId = pd.unique(input1Table['id'])
    
    # main script #################################################################################################################################################
    for i in totalId:
        pollu_val_df = input1Table[input1Table.id == i]
        
        # Step 1: based on group id (id column) to filter the input1Table into pollu_val_df
        # Step 2: input {pollu_val_df} to OLM() or jenkin_and_other_pollutant(), output would be result_df
        # Step 3: check the output {result_df} and export it to Output_(current_date)_(current_time).xlsx
        if "A" in pd.unique(pollu_val_df['Jenkin_Method']):
            # OLM + Jenkin
            logging.info(f'Group {i}, OLM + Jenkin')
            logging.info(f'Group {i}, OLM part start.')
            result_df = OLM(pollu_val_df)
            export_result_df_to_excel(result_df, export_filename, i, "NO2_OLM")
            logging.info(f'Group {i}, OLM part is done.')
            
            logging.info(f'Group {i}, Jenkin part start.')
            Jenkin_df = pollu_val_df[(pollu_val_df.AQO_Pollutant == 'NO2') & (pollu_val_df.Jenkin_Method != 'O')]
            result_df = jenkin_and_other_pollutant(Jenkin_df)
            export_result_df_to_excel(result_df, export_filename, i, "NO2_Jenkin")
            logging.info(f'Group {i}, Jenkin part is done.')
           
        else:
            # only OLM
            if "O" in pd.unique(pollu_val_df['Jenkin_Method']):
                logging.info(f'Group {i}, only OLM.')
                logging.info(f'Group {i}, OLM part start.')
                pollu_val_df = input1Table[(input1Table.AQO_Pollutant == 'NO2') & (input1Table.Jenkin_Method == 'O')]
                result_df = OLM(pollu_val_df)
                export_result_df_to_excel(result_df, export_filename, i, "NO2_OLM")
                logging.info(f'Group {i}, OLM part is done.')
            # only Jenkin
            elif "J" in pd.unique(pollu_val_df['Jenkin_Method']):
                logging.info(f'Group {i}, only Jenkin.')
                logging.info(f'Group {i}, Jenkin part start.')
                pollu_val_df = input1Table[(input1Table.AQO_Pollutant == 'NO2') & (input1Table.Jenkin_Method == 'J')]
                result_df = jenkin_and_other_pollutant(pollu_val_df)
                export_result_df_to_excel(result_df, export_filename, i, "NO2_Jenkin")
                logging.info(f'Group {i}, Jenkin part is done.')
            # other pollutant    
            else:
                pollu_val = pd.unique(pollu_val_df['AQO_Pollutant'])[0]
                logging.info(f'Group {i}, other pollutant: {pollu_val}.')
                logging.info(f'Group {i}, {pollu_val} start calculation.')
                result_df = jenkin_and_other_pollutant(pollu_val_df)
                export_result_df_to_excel(result_df, export_filename, i, pollu_val)
                logging.info(f'Group {i}, {pollu_val} calculation is done.')
        
    logging.info("Script End")
    logging.info(f'Output file is in: {export_filename}')