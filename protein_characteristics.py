import pandas as pd
import numpy as np

import os
import sys
import requests
import re
from io import StringIO
import csv

import json
from collections import OrderedDict


def ProtParam(uniprot_id):

    '''
    The aim of this function is to extract protein characteristics, based on their UniProt ID.

    input: 
        - input_csv, which contains UniProt IDs
        - output_csv, which is used to store appended new data to the input dataset
    output:

    '''

    web_url = "https://web.expasy.org/cgi-bin/protparam/protparam_bis.cgi?"
    
    protein_url = web_url+uniprot_id+"@@" # dodala sam @ i onda je radio ahahahh LOLL
    files ={
        'file': uniprot_id
    }
    response = requests.get(protein_url)
    # response = requests.get(web_url)
    if response.status_code==200:
        # print(response.text)
        xml_data = response.text

        # Extract and print the relevant information
        molecular_weight = re.findall(r'<strong>Molecular weight:</strong>\s*(-?\d*\.?\d+)', xml_data)[0]
        theoretical_pI = re.findall(r'<strong>Theoretical pI:</strong>\s*(-?\d*\.?\d+)', xml_data)[0]
        ext_coeff_abs = re.findall(r'Abs 0.1% \(=1 g/l\)\s*(-?\d*\.?\d+)', xml_data)
        instability_index = re.findall(r'The instability index \(II\) is computed to be\s*(-?\d*\.?\d+)', xml_data)[0]
        aliphatic_index = re.findall(r'<strong>Aliphatic index:</strong>\s*(-?\d*\.?\d+)', xml_data)[0]
        hydrophaticity = re.findall(r'<strong>Grand average of hydropathicity \(GRAVY\):</strong>(-?\d*\.?\d+)', xml_data)[0]
        print(f"Molecular weight: {molecular_weight}")
        print('Theoretical pI: ', theoretical_pI)
        print('Ext.coef abs: ',  ext_coeff_abs)
        print('Instability index: ', instability_index)
        print("Aliphatic index: ", aliphatic_index)
        print('GRAVI: ', hydrophaticity)
        
        # You can extract other properties similarly
        # Example:

        # Add more properties as needed
    else:
        print(f"Failed to retrieve data: {response.status_code}")

    return float(molecular_weight), float(theoretical_pI), float(ext_coeff_abs[0]), float(ext_coeff_abs[1]), float(instability_index[0]), float(aliphatic_index[0]), float(hydrophaticity)

def ProtParam_from_sequence(sequence):

    # URL for the ProtParam tool
    url = 'https://web.expasy.org/cgi-bin/protparam/protparam'

    # Prepare the payload for submission
    payload = {
        'sequence': sequence,
        'compute': 'Compute parameters'
    }

    # Submit the sequence to the server
    response = requests.post(url, data=payload)

    # Check if the request was successful
    if response.status_code == 200:
        # print(response.text)
        # Parse the response
        xml_data = response.text

        # Extract and print the relevant information
        molecular_weight = re.findall(r'<strong>Molecular weight:</strong>\s*(-?\d*\.?\d+)', xml_data)[0]
        theoretical_pI = re.findall(r'<strong>Theoretical pI:</strong>\s*(-?\d*\.?\d+)', xml_data)[0]
        ext_coeff_abs = re.findall(r'Abs 0.1% \(=1 g/l\)\s*(-?\d*\.?\d+)', xml_data)
        instability_index = re.findall(r'The instability index \(II\) is computed to be\s*(-?\d*\.?\d+)', xml_data)[0]
        aliphatic_index = re.findall(r'<strong>Aliphatic index:</strong>\s*(-?\d*\.?\d+)', xml_data)[0]
        hydrophaticity = re.findall(r'<strong>Grand average of hydropathicity \(GRAVY\):</strong>(-?\d*\.?\d+)', xml_data)[0]
        print(f"Molecular weight: {molecular_weight}")
        print('Theoretical pI: ', theoretical_pI)
        print('Ext.coef abs: ',  ext_coeff_abs)
        print('Instability index: ', instability_index)
        print("Aliphatic index: ", aliphatic_index)
        print('GRAVI: ', hydrophaticity)
        
        # You can extract other properties similarly
        # Example:

        # Add more properties as needed
    else:
        print(f"Failed to retrieve data: {response.status_code}")

    return float(molecular_weight), float(theoretical_pI), float(ext_coeff_abs[0]), float(ext_coeff_abs[1]), float(instability_index[0]), float(aliphatic_index[0]), float(hydrophaticity)


def download_GO_for_protein(uniprot_id, output_csv, download_limit = 100):

    """
    Using QuickGO API, extracting GO annotations for a specific protein, using its UniProt id

    input:
        - uniprot_id, identification for a certain protein
        - output_csv where the data will be stored
        - download_limit, the maximal number of annotatins per protein to be extracted. Default number is 100. 
    output:
        - list of GO annotations for a specific protein, based on the UniProt ID.
    """

    requestURL = "https://www.ebi.ac.uk/QuickGO/services/annotation/downloadSearch?includeFields=goName&selectedFields=qualifier,goId&downloadLimit={}&geneProductId={}".format(download_limit, uniprot_id)

    r = requests.get(requestURL, headers={ "Accept" : "text/tsv"})

    if not r.ok:
        r.raise_for_status()
        sys.exit()

    responseBody = r.text
    data = StringIO(responseBody)
    df = pd.read_csv(data, sep='\t')

    # Display the DataFrame
    # print(df)
    GO = df['GO TERM'].to_list()
    QUALIFIER = df['QUALIFIER'].to_list()
    fields = [uniprot_id, GO, QUALIFIER]

    with open(output_csv, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    return GO  

if __name__ == "__main__":
    prot_dir = 'data/davis/proteins.txt'
    proteins = json.load(open(prot_dir), object_pairs_hook=OrderedDict)
    prots = []
    columns = ['GeneID', 'Sequence', 'molecular_weight', 'theoretical_pI', 'ext_coeff_abs', 'ext_coeff_abs_reduced_Cys', 'instability_index', 'aliphatic_index', 'hydropathicity']
    # df = pd.DataFrame(columns=columns)
    data = []
    i = 0
    for gene, sequence in proteins.items():
        molwt, th_pi, ext_co_ab0, ext_co_ab1, inst, aliph, hyd = ProtParam_from_sequence(sequence)
        row = {
            'GeneID': gene,
            'Sequence': sequence,
            'molecular_weight': molwt,
            'theoretical_pI': th_pi,
            'ext_coeff_abs': ext_co_ab0,
            'ext_coeff_abs_reduced_Cys': ext_co_ab1,
            'instability_index': inst,
            'aliphatic_index': aliph,
            'hydropathicity': hyd
        }
        data.append(row)
        i += 1
        print(i)

    df = pd.DataFrame(data, columns=columns)
    df.to_csv('interpretability/protein_parameters/davis_proteins_ProtParam.csv', index=False)

        
