import pandas
from data_preprocessing import *
from feature_transformation_funcs import *

load_save = False

if load_save:
    unstemmed_all = pd.read_csv('unstemmed_attributes.csv')
    stemmed_all = pd.read_csv("stemmed_all.csv")
    counts_only = pd.read_csv("count_features_only.csv")
    spacy_only = pd.read_csv("spacy_only_features.csv")
else:
	calc_query_legth(data)
	count_common_words(data)
	calc_spacy_title_query(data)
	print('starting long spacy')
	calc_spacy_long(data)
	unstemmed_all, stemmed_all, counts_only, spacy_only = finalise_and_export(data)