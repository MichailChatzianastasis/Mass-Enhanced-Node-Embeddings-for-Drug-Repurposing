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


    #drug to protein dict
    drug_to_protein_tsv = pd.read_csv("/data/multiscale-interactome/data/1_drug_to_protein.tsv",sep='\t', header=0)
    dict_to_save = {}
    for index,row in drug_to_protein_tsv.iterrows():
        if(row[0] in dict_to_save):
            dict_to_save[row[0]].append(row[1])
        else:
            dict_to_save[row[0]] = [row[1]]
    with open('/data/multiscale-interactome/data/drug_to_protein_dict.pickle', 'wb') as handle:
        pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #disease to protein dict
    disease_to_protein_tsv = pd.read_csv("/data/multiscale-interactome/data/2_indication_to_protein.tsv",sep='\t', header=0)
    dict_to_save = {}
    for index,row in disease_to_protein_tsv.iterrows():
        if(row[0] in dict_to_save):
            dict_to_save[row[0]].append(row[1])
        else:
            dict_to_save[row[0]] = [row[1]]
    with open('/data/multiscale-interactome/data/disease_to_protein_dict.pickle', 'wb') as handle:
        pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

def evaluate_model(model="diffusion_profiles",metric="Correlation distance"):
        ''' 
        For each disease, the model produces a ranked list of drugs. We identify the drugs approved to treat the disease (drug-disease labels).
        For each disease, we then compute the model’s AUROC, Average Precision,and Recall@50 values based on the ranked list of drugs. 
        We report the model’s performance across diseases by reporting the median of the AUROC, the mean of the Average Precision, and the mean of the Recall@50 values across diseases.
        Five-cross validation splitting.
        '''

        # construct msi 
        msi = MSI()
        msi.load()
        average_precision = []
        recall50 = []

        #load disease drug dict
        with open('/data/multiscale-interactome/data/disease_to_drug_dict.pickle', 'rb') as handle:
            disease_to_drug_dict = pickle.load(handle)

        #load drug to protein dict
        with open('/data/multiscale-interactome/data/drug_to_protein_dict.pickle', 'rb') as handle:
            drug_to_protein_dict = pickle.load(handle)

        #load disease to protein dict
        with open('/data/multiscale-interactome/data/disease_to_protein_dict.pickle', 'rb') as handle:
            disease_to_protein_dict = pickle.load(handle)
        #load drug list
        with open('/data/multiscale-interactome/data/drug_codes_dict.pickle', 'rb') as handle:
            drug_codes_dict = pickle.load(handle)
        
        display_counter = 0

        if(model=="diffusion_profiles"):
            print("Loading diffusion profiles..")
            #load best diffusion profile
            #dp_saved = DiffusionProfiles(alpha = None, max_iter = None, tol = None, weights = None, num_cores = None, save_load_file_path = "results/")
            #msi.load_saved_node_idx_mapping_and_nodelist(dp_saved.save_load_file_path)
            #dp_saved.load_diffusion_profiles(msi.drugs_in_graph + msi.indications_in_graph)

            #compute diffusion profiles with above parameters
            #dp_saved = DiffusionProfiles(alpha = 0.5595436247434408, max_iter = 1000, tol = 1e-06, weights = {'down_biological_function': 7.4863053901688685, 
            #                'indication': 1.541889556309463, 'biological_function': 4.583155399238509, 'up_biological_function': 3.09685000906964, 'protein': 1.396695660380823, 
            #                'drug': 3.2071696595616364}, num_cores = int(multiprocessing.cpu_count()/2) - 4, save_load_file_path = "/data/multiscale-interactome/results/")
            #dp_saved.calculate_diffusion_profiles(msi)

            dp_saved = DiffusionProfiles(alpha = None, max_iter = None, tol = None, weights = None, num_cores = None, save_load_file_path = "/data/multiscale-interactome/data/10_top_msi/")
            msi.load_saved_node_idx_mapping_and_nodelist(dp_saved.save_load_file_path)
            dp_saved.load_diffusion_profiles(msi.drugs_in_graph + msi.indications_in_graph)
            print("Diffusion profiles loaded.")

            # for each disease and drug compute the diffusion profiles r. Then for each disease, rank the drugs based on the distance of the diffusion profile r_disease and r_drug.
            for key, value in disease_to_drug_dict.items():  #key is disease code, value is a list of ["disease_name","drug_code","drug_name"] cure the disease. 
                #diffusion profile of the disease
                if(display_counter % 10 == 0):
                    print("index",display_counter)
                display_counter+=1
                disease_r = dp_saved.drug_or_indication2diffusion_profile[key]

                distance_disease_drug = {}
                #for every drug compute the diffusion profile distance
                for drug_code in drug_codes_dict:
                    #diffusion profile of the drug
                    drug_r = dp_saved.drug_or_indication2diffusion_profile[drug_code] 
                    #compute distance between disease:key and drug:drug_code
                    distance_disease_drug[drug_code] = calculate_diffusion_profile_distance(disease_r,drug_r,metric)
                
                #sort drugs based on distance with the given disease.
                distance_disease_drug = dict(sorted(distance_disease_drug.items(), key=lambda item: item[1]))
                # compute metrics (average_precision,recall etc from predictions and true label pairs. ) 
                true_ranking = {}
                for drug_list in value:
                    true_ranking[(drug_list[1])] = 1 # append the drug code
                av_p,r50 = evaluate_disease(distance_disease_drug,true_ranking)
                average_precision.append(av_p)
                recall50.append(r50)
            print("Mean Average Precision",sum(average_precision) / len(average_precision))
            print("Mean Recall@50", sum(recall50) / len(recall50))
              
        elif(model=="protein_overlap"):
            #define as the Jaccard Similarity between the set of drug targets T and the set of disease proteins S:
            # 1) compute disease proteins S
            for key,value in disease_to_drug_dict.items():
                if(display_counter % 10 ==0):
                    print("index",display_counter)
                display_counter+= 1
                S_proteins = disease_to_protein_dict[key]
                distance_disease_drug = {}

                for drug_code in drug_codes_dict:
                    T_proteins = drug_to_protein_dict[drug_code]
                    distance_disease_drug[drug_code] = calculate_jaccard_similarity(S_proteins,T_proteins)
                
                distance_disease_drug = dict(sorted(distance_disease_drug.items(), key=lambda item: item[1] , reverse=True))

                # compute metrics (average_precision,recall etc from predictions and true label pairs. ) 
                true_ranking = {}
                for drug_list in value:
                    true_ranking[(drug_list[1])] = 1 # append the drug code
                av_p,r50 = evaluate_disease(distance_disease_drug,true_ranking)
                average_precision.append(av_p)
                recall50.append(r50)
            print("Mean Average Precision",sum(average_precision) / len(average_precision))
            print("Mean Recall@50", sum(recall50) / len(recall50))
            

def evaluate_disease(ranked_predictions,true_ranking):
    '''
    Evaluate the AUROC,Average Precision, and Recall@50  of the predictions of what drugs will treat a disease.
    ranked_predictions: Sorted dictionary with keys:drug code and values:distance between drug and disease
    true_ranking: list of approved drug_codes for the disease
    '''
    #calculate average precision,recall@50
    average_precision,recall50 = calculate_average_precision_recall(ranked_predictions,true_ranking) 

    return average_precision,recall50


def calculate_jaccard_similarity(S, T):
    intersection = len(list(set(S).intersection(T)))
    union = (len(S) + len(T)) - intersection
    return float(intersection) / union    


def calculate_average_precision_recall(ranked_predictions,true_ranking):
    '''
     1)Calculate the average precision using the formula :  $\frac{1}{m}\sum_{k=1}^{N} {P(k) \cdot rel(k)}$,
     $m$ number of relevant items, $N$ number of recommended items, $P(k)$ is the precision calculated only the recoomendations from rank 1 to k, $rel(k)$ is
     an indicator of whether that $k^{th}$ item was relevant ($rel(k)=1$) or not ($rel(k)=0$).
     2)Calculate recall@50.
    '''
    m = len(true_ranking)
    average_precision = 0
    correct_predictions = 0
    correct_50 = 0
    for k,drug_prediction in enumerate(ranked_predictions): 
        if(drug_prediction in true_ranking):
            correct_predictions += 1
            average_precision += correct_predictions / (k+1)
            if(k<50):
                correct_50+= 1
    average_precision = average_precision / m
    recall50 = correct_50 / m
    return average_precision,recall50


def calculate_diffusion_profile_distance(r1,r2,metric="Correlation distance"):
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
    construct_disease_drug_tsv()
    evaluate_model(model="protein_overlap")
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

    distance1 = calculate_diffusion_profile_distance(dp_saved.drug_or_indication2diffusion_profile["DB01098"],dp_saved.drug_or_indication2diffusion_profile["DB01098"])
    distance2 = calculate_diffusion_profile_distance(dp_saved.drug_or_indication2diffusion_profile["DB01098"],dp_saved.drug_or_indication2diffusion_profile["DB00978"])
    print(distance1,distance2)



