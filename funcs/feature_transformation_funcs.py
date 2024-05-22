import numpy as np
import spacy
nlp = spacy.load("en_core_web_lg")
import en_core_web_lg

def str_common_word(str1, str2):
	if type(str1) == str and type(str2) == str:
		return sum(int(str2.find(word)>=0) for word in str1.split())
	else:
		return

def calc_query_legth(df_all):
    
    # CALCULATE THE LENGTH OF THE SEARCH QUERY
    
    df_all['len_of_query'] = df_all['search_term_stem'].map(lambda x:0 if type(x) != str else len(x.split())).astype(np.int64)
    
    return df_all

def count_common_words(df_all):

	# COUNT THE COMMON WORDS BETWEEN THE SEARCH TERM AND EACH ATTRIBUTE (STEMMED)
	df_all['word_in_title_stemmed'] = df_all['product_info_stemmed'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]) if type(x) == str else 0)
	df_all['word_in_description_stemmed'] = df_all['product_info_stemmed'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]) if type(x) == str else 0)

	# COUNT THE COMMON WORDS BETWEEN THE SEARCH TERM AND EACH ATTRIBUTE (UNSTEMMED)
	df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]) if type(x) == str else 0)
	df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]) if type(x) == str else 0)

	# CODE BLOCK 4: COMPUTING NEW COUNT FEATURES
	# ADDING ATTRIBUTES - COUNT OF PRODUCT ATTRIBUTES IN THE SEARCH TERM
	df_all['word_in_attributes'] = df_all['search_term_and_all_attributes'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]) if type(x)==str else 0)

	return df_all

def spacy_similarity(doc_1_col, doc_2_col):
    # doc_1 = nlp(doc_1_col)
    # doc_2 = nlp(doc_2_col)
    doc_1 = nlp(str(doc_1_col))
    doc_2 = nlp(str(doc_2_col))
    return doc_1.similarity(doc_2)

def calc_spacy_title_query(df_all):
    nlp = en_core_web_lg.load()
    
    # Spacy similarity score between just title and search term
    df_all["title_spacy_similarity"] = df_all.apply(lambda x: spacy_similarity(x["product_title"], x["search_term"]), axis = 1)
    
    return df_all

def calc_spacy_long(df_all):
	## CODE BLOCK 13: DISTANCE MEASURES WITH SPACY ##
	# NOTE: This code block (which was split into smaller code blocks) originally took a few hours to run due to spacy being time consuming out-of-the-box. We suggest you skip to the import statements at the end to
	# run the models

	def apply_feature_score(newcolname, col1, col2, feature_scorer):
		df_all[newcolname] = df_all.apply(lambda x: feature_scorer(str(x[col1]), str(x[col2])), axis = 1)

	# Spacy Similarity -- Unstemmed; used only for comparison
	apply_feature_score("spacy_title_and_assembled_measurement", "title_and_assembled_measurement", "search_term", spacy_similarity)
	apply_feature_score("spacy_title_and_measurement_LWD","title_and_measurement_LWD", "search_term", spacy_similarity)
	apply_feature_score("spacy_title_and_measurement_DWH","title_and_measurement_DWH", "search_term", spacy_similarity)
	apply_feature_score("spacy_title_and_color","title_and_color", "search_term", spacy_similarity)
	apply_feature_score("spacy_title_and_color_fam","title_and_color_fam", "search_term", spacy_similarity)

	# Spacy Similarity -- Stemmed
	apply_feature_score("spacy_description_stemmed","product_description_stem", "search_term_stem", spacy_similarity)
	apply_feature_score("spacy_title_stem_and_assembled_measurement", "title_and_assembled_measurement_stem", "search_term_stem", spacy_similarity)
	apply_feature_score("spacy_title_and_measurement_LWD_stem", "title_and_measurement_LWD_stem", "search_term_stem", spacy_similarity)
	apply_feature_score("spacy_title_and_measurement_DWH_stem", "title_and_measurement_DWH_stem", "search_term_stem", spacy_similarity)
	apply_feature_score("spacy_title_and_color_stem","title_and_color_stem", "search_term_stem", spacy_similarity)
	apply_feature_score("spacy_title_and_color_fam_stem","title_and_color_fam_stem", "search_term_stem", spacy_similarity)
	
	df_all.to_csv('long spacy checkpoint.csv')

	return df_all

def finalise_and_export(df_all):
    ## CODE BLOCK 14: FINALISE & EXPORT DATASETS
    unstemmed_all = df_all[["id","relevance", "len_of_query", "word_in_title", "word_in_description", "word_in_attributes", "product_uid", "title_spacy_similarity", "spacy_title_and_assembled_measurement", "spacy_title_and_measurement_LWD", "spacy_title_and_measurement_DWH", "spacy_title_and_color", "spacy_title_and_color_fam"]]
    stemmed_all = df_all[["id","relevance", "len_of_query", "word_in_title_stemmed", "word_in_description_stemmed", "word_in_attributes", "product_uid", "spacy_description_stemmed", "spacy_title_stem_and_assembled_measurement", "spacy_title_and_measurement_LWD_stem", "spacy_title_and_measurement_DWH_stem", "spacy_title_and_color_stem", "spacy_title_and_color_fam_stem"]]
    counts_only = df_all[["id","relevance", "len_of_query", "word_in_title", "word_in_description", "product_uid"]]
    spacy_only = df_all[["id","relevance", "product_uid", "spacy_description_stemmed", "spacy_title_stem_and_assembled_measurement", "spacy_title_and_measurement_LWD_stem", "spacy_title_and_measurement_DWH_stem", "spacy_title_and_color_stem", "spacy_title_and_color_fam_stem"]]
    
    unstemmed_all.to_csv("unstemmed_attributes.csv")
    stemmed_all.to_csv("stemmed_all.csv")
    counts_only.to_csv("count_features_only.csv")
    spacy_only.to_csv("spacy_only_features.csv")
    
    return unstemmed_all, stemmed_all, counts_only, spacy_only

def longest_subsequence(df_all, attributes):
    
    combined = pd.concat((df_all, attributes), axis = 1)
    
    df_all['lcs_title'] = combined.apply(lambda x: pylcs.lcs_sequence_length(str(x['search_term_stem']), str(x['product_title_stem'])), axis = 1)
    df_all['lcs_brand'] = combined.apply(lambda x: pylcs.lcs_sequence_length(str(x['search_term_stem']), str(x['MFG Brand Name'])), axis = 1)
    df_all['lcs_title_string'] = combined.apply(lambda x: pylcs.lcs_string_length(str(x['search_term_stem']), str(x['product_title_stem'])), axis = 1)
    df_all['lcs_brand_string'] = df_all.apply(lambda x: pylcs.lcs_string_length(str(x['search_term_stem']), str(x['MFG Brand Name'])), axis = 1)
    
    return df_all

def unique_str_common_word(str1, str2):
    if type(str1) == str and type(str2) == str:
        return sum(int(str2.find(word)>=0) for word in set(str1.split()))
    else:
        return
    
def count_unique_common_words(df_all):
    df_all['unique_word_in_title_stemmed'] = df_all['product_info_stemmed'].map(lambda x:unique_str_common_word(x.split('\t')[0],x.split('\t')[1]) if type(x) == str else 0)
    df_all['unique_word_in_description_stemmed'] = df_all['product_info_stemmed'].map(lambda x:unique_str_common_word(x.split('\t')[0],x.split('\t')[2]) if type(x) == str else 0)

    # uncomment to include unstemmed
    # df_all['unique_word_in_title'] = df_all['product_info'].map(lambda x:unique_str_common_word(x.split('\t')[0],x.split('\t')[1]) if type(x) == str else 0)
    # df_all['unique_word_in_description'] = df_all['product_info'].map(lambda x:unique_str_common_word(x.split('\t')[0],x.split('\t')[2]) if type(x) == str else 0)
    
    df_all['unique_word_in_attributes'] = df_all['search_term_and_all_attributes'].map(lambda x:unique_str_common_word(x.split('\t')[0],x.split('\t')[2]) if type(x)==str else 0)

    return df_all

def count_unique_common_letter(df_all):

    df_all['unique_letter_in_title_stemmed'] = df_all['product_info_stemmed'].map(lambda x:unique_str_common_letters(x.split('\t')[0],x.split('\t')[1]) if type(x) == str else 0)
    df_all['unique_letter_in_description_stemmed'] = df_all['product_info_stemmed'].map(lambda x:unique_str_common_letters(x.split('\t')[0],x.split('\t')[2]) if type(x) == str else 0)

    # uncomment to include unstemmed
    # df_all['unique_word_in_title'] = df_all['product_info'].map(lambda x:unique_str_common_letters(x.split('\t')[0],x.split('\t')[1]) if type(x) == str else 0)
    # df_all['unique_word_in_description'] = df_all['product_info'].map(lambda x:unique_str_common_letters(x.split('\t')[0],x.split('\t')[2]) if type(x) == str else 0)
 
    df_all['unique_letter_in_attributes'] = df_all['search_term_and_all_attributes'].map(lambda x:unique_str_common_letters(x.split('\t')[0],x.split('\t')[2]) if type(x)==str else 0)

    return df_all

def calc_frequency_ratios(df_all):
    df_all['len_search_term_stem_word'] = df_all['search_term_stem'].map(lambda x: len(x.split()))
    df_all['len_search_term_stem_letter'] = df_all['search_term_stem'].map(lambda x: len(x))
    
    df_all['search_title_uniqe_words'] = df_all['unique_word_in_title_stemmed']/ df_all['len_search_term_stem_word']
    df_all['search_description_uniqe_words'] = df_all['unique_word_in_description_stemmed']/ df_all['len_search_term_stem_word']
    df_all['search_attributes_uniqe_words'] = df_all['unique_word_in_attributes']/ df_all['len_search_term_stem_word']
    
    df_all['search_title_uniqe_letters'] = df_all['unique_letter_in_title_stemmed']/ df_all['len_search_term_stem_letter']
    df_all['search_description_uniqe_letters'] = df_all['unique_letter_in_description_stemmed']/ df_all['len_search_term_stem_letter']
    df_all['search_attributes_uniqe_letters'] = df_all['unique_letter_in_attributes']/ df_all['len_search_term_stem_letter']
    
    return df_all