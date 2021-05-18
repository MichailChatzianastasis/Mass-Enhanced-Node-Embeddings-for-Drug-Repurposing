import numpy as np
import sys
from msi.msi import MSI

from diff_prof.diffusion_profiles import DiffusionProfiles
import multiprocessing
import pickle
import networkx as nx

from tests.msi import test_msi
from tests.diff_prof import test_diffusion_profiles
from numpy import linalg as LA
from scipy.spatial import distance
import pandas as pd
from operator import itemgetter
import pickle


def construct_disease_drug_tsv():
    '''
    Construct a tsv with disease-drug pairs. ( We currently have drug-disease pairs)
    '''
    drug_disease_tsv = pd.read_csv("/data/multiscale-interactome/data/6_drug_indication_df.tsv",sep='\t', header=0)
    disease_drug_tsv = drug_disease_tsv.sort_values(by='indication')
    disease_drug_tsv = disease_drug_tsv[["indication","indication_name","drug","drug_name"]]
    disease_drug_tsv.to_csv("/data/multiscale-interactome/data/disease_drug_df.tsv",sep="\t", index=False)
    
    # create dictonary with diseases as keys. 
    dict_to_save = {}
    for index,row in disease_drug_tsv.iterrows():
        if(row[0] in dict_to_save):
            dict_to_save[row[0]].append([row[1],row[2],row[3]])
        else:
            dict_to_save[row[0]] = [[row[1],row[2],row[3]]]
    
    with open('/data/multiscale-interactome/data/disease_to_drug_dict.pickle', 'wb') as handle:
        pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # construct list of drugs
    drug_list_codes = {}
    for index,row in drug_disease_tsv.iterrows():
        drug_list_codes[row[0]]=1
    with open('/data/multiscale-interactome/data/drug_codes_dict.pickle', 'wb') as handle:
        pickle.dump(drug_list_codes, handle, protocol=pickle.HIGHEST_PROTOCOL)

def evaluate_model(model="diffusion_profiles",metric="L1 norm"):
        ''' 
        For each disease, the model produces a ranked list of drugs. We identify the drugs approved to treat the disease (drug-disease labels). For each
        disease, we then compute the model’s AUROC, Average Precision, and Recall@50 values based on the ranked list of drugs. We report the model’s performance across
        diseases by reporting the median of the AUROC, the mean of the Average Precision, and the mean of the Recall@50 values across diseases.
        Five-cross validation splitting.
        '''

        # construct msi 
        msi = MSI()
        msi.load()
        
        #load diffusion profile
        if(model=="diffusion_profiles"):
            print("Loading diffusion profiles..")
            dp_saved = DiffusionProfiles(alpha = None, max_iter = None, tol = None, weights = None, num_cores = None, save_load_file_path = "results/")
            msi.load_saved_node_idx_mapping_and_nodelist(dp_saved.save_load_file_path)
            dp_saved.load_diffusion_profiles(msi.drugs_in_graph + msi.indications_in_graph)
            print("Diffusion profiles loaded.")
            
            #load disease drug ditct
            with open('/data/multiscale-interactome/data/disease_to_drug_dict.pickle', 'rb') as handle:
                disease_to_drug_dict = pickle.load(handle)

            #load drug list
            with open('/data/multiscale-interactome/data/drug_codes_dict.pickle', 'rb') as handle:
                drug_codes_dict = pickle.load(handle)

            # for each disease and drug compute the diffusion profiles r. Then for each disease, rank the drugs based on the simildistancearity of the diffusion profile r_disease and r_drug.
            for key, value in disease_to_drug_dict.items():  #key is disease code, value is a list of ["disease_name","drug_code","drug_name"] cure the disease. 
                #diffusion profile of the disease
                disease_r = dp_saved.drug_or_indication2diffusion_profile[key]

                distance_disease_drug = {}
                #for every drug compute the diffusion profile distance
                for drug_code in drug_codes_dict:
                    #diffusion profile of the drug
                    drug_r = dp_saved.drug_or_indication2diffusion_profile[drug_code] 
                    #compute distance between disease:key and drug:drug_code
                    distance_disease_drug[drug_code] = diffusion_profile_distance(disease_r,drug_r,metric)
                
                #sort drugs based on distance with the given disease.
                distance_disease_drug = dict(sorted(distance_disease_drug.items(), key=lambda item: item[1]))
                print(distance_disease_drug)
                exit()

def diffusion_profile_distance(r1,r2,metric="L1 norm"):
        '''
        Calculate the distance between two diffusion profiles based on the given metric.
        Available metrics: L2 norm, L1 norm, Canberra distance, Cosine distance, Correlation distance. (in paper extra two metrics)
        '''
        if(metric=="L2 norm"):
            return LA.norm(r1-r2,ord=2)
        if(metric=="L1 norm"):
            return LA.norm(r1-r2,ord=1)
        if(metric=="Canberra distance"):
            return distance.canberra(r1,r2)
        if(metric=="Cosine distance"):
            return distance.cosine(r1,r2)
        if(metric=="Correlation distance"):
            return distance.correlation(r1,r2)
        




if __name__ == "__main__":
    #construct_disease_drug_tsv()
    evaluate_model()
    exit()
    msi = MSI()
    msi.load()
    # Test against reference
    #test_msi()

    #dp = DiffusionProfiles(alpha = 0.8595436247434408, max_iter = 1000, tol = 1e-06, weights = {'down_biological_function': 4.4863053901688685, 'indication': 3.541889556309463, 'biological_function': 6.583155399238509, 'up_biological_function': 2.09685000906964, 'protein': 4.396695660380823, 'drug': 3.2071696595616364}, num_cores = int(multiprocessing.cpu_count()/2) - 4, save_load_file_path = "results/")
    #dp.calculate_diffusion_profiles(msi)
    dp_saved = DiffusionProfiles(alpha = None, max_iter = None, tol = None, weights = None, num_cores = None, save_load_file_path = "results/")
    msi.load_saved_node_idx_mapping_and_nodelist(dp_saved.save_load_file_path)
    dp_saved.load_diffusion_profiles(msi.drugs_in_graph + msi.indications_in_graph)

    # Diffusion profile for Rosuvastatin (DB01098)
    dp_saved.drug_or_indication2diffusion_profile["DB01098"]
    # Test against reference
    #test_diffusion_profiles("data/10_top_msi/", "results/")

    distance1 = diffusion_profile_distance(dp_saved.drug_or_indication2diffusion_profile["DB01098"],dp_saved.drug_or_indication2diffusion_profile["DB01098"])
    distance2 = diffusion_profile_distance(dp_saved.drug_or_indication2diffusion_profile["DB01098"],dp_saved.drug_or_indication2diffusion_profile["DB00978"])
    print(distance1,distance2)



