import pandas as pd
import nltk
# nltk.download('punkt')
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from google_spelling_checker_dict import * # file downloaded from Chenglong public repo shared on Kaggle forum
from wordsegment import load, segment



def load_data(train_path, test_path, attributes_path, product_descriptions_path, train_len = None, bool_dev = False):
    df_train = pd.read_csv(train_path, encoding = 'ISO-8859-1')
    df_test = pd.read_csv(test_path, encoding = 'latin-1')
    df_attr = pd.read_csv(attributes_path, encoding = 'ISO-8859-1')
    df_pro_desc = pd.read_csv(product_descriptions_path, encoding = 'latin-1')

    if train_len == None:
        len_training = df_train.shape[0]
    else:
        len_training = num_samples
    df_train = df_train[:len_training]
    df_test = df_test[len_training:]
    pro_desc = df_pro_desc[:len_training]

    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df_all = pd.merge(df_all, pro_desc, how='left', on='product_uid')
    
    if bool_dev:
        return df_all[:400], df_attr[:400]
    else:
        return df_all, df_attr
    
def str_stemmer(s):
    stemmer = SnowballStemmer('english')
    
    if type(s) == str:
        return " ".join([stemmer.stem(word) for word in s.lower().split()])
    else:
        return

def stem(df_all):
    stemmer = SnowballStemmer('english')
    
    # STEM THE ATTRIBUTES WE WANT TO TURN INTO COUNT FEATURES
    df_all['search_term_stem'] = df_all['search_term'].map(lambda x:str_stemmer(x))
    df_all['product_title_stem'] = df_all['product_title'].map(lambda x:str_stemmer(x))
    df_all['product_description_stem'] = df_all['product_description'].map(lambda x:str_stemmer(x))

    # COMBINE ALL STEMMED ATTRIBUTES INTO ONE COLUMN
    df_all['product_info_stemmed'] = df_all['search_term_stem']+"\t"+df_all['product_title_stem']+"\t"+df_all['product_description_stem']

    # COMBINE ALL UNSTEMMED ATTRIBUTES INTO ONE COLUMN
    df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']
    
    return df_all

def add_init_attributes(df_all, df_attr):
    stemmer = SnowballStemmer('english')

    # This adds attributes (name, value, all, values, search_term_and_all_attributes)

    # ADDING ATTRIBUTES - COUNT OF PRODUCT ATTRIBUTES IN THE SEARCH TERM
    original_attributes = df_attr.copy()
    
    df_attr['all_values'] = df_attr[['product_uid', 'value']].groupby(['product_uid'])['value'].transform(lambda x : ','.join((word_tokenize(stemmer.stem(str(x))))))
    df_all_w_attr = df_all.join(df_attr, how = 'left', lsuffix = '_left', rsuffix='_right')
    df_all_w_attr['search_term_and_all_attributes'] = df_all_w_attr['search_term']+"\t"+df_all_w_attr['name'].transform(lambda x : ','.join(word_tokenize(str(x))))+"\t"+df_all_w_attr['all_values']
    
    return df_all_w_attr, original_attributes

def get_common_attributes(attributes, n: int = 20):
    # CODE BLOCK 5: ATTRIBUTE - VALUE PAIRS PRE-PROCESSING
    attributes = attributes.astype({"name":str})
    grouped_attr = attributes.groupby(["product_uid", "name"], group_keys=False).agg(lambda x: x)
    grouped_attr = grouped_attr.reset_index()
    grouped_attr.columns = ["product_uid", "attr_name", "attr_value"]
    
    # CODE BLOCK 7: EXTRACT MOST COMMON ATTRIBUTES

    types_of_attributes = grouped_attr["attr_name"].value_counts()
    types_of_attributes = types_of_attributes.to_frame()
    types_of_attributes = types_of_attributes.reset_index()
    bullets = types_of_attributes["attr_name"].str.contains("Bullet")
    types_of_attributes = types_of_attributes[~bullets]

    most_used_attributes = types_of_attributes.iloc[0:20]
    most_used_attribute_list = []

    for i in most_used_attributes["attr_name"]:
        most_used_attribute_list.append(i)
        # print(len(most_used_attribute_list))


    grouped_attr = grouped_attr[grouped_attr["attr_name"].isin(most_used_attribute_list)]
    attributes_as_cols = grouped_attr.pivot(index = "product_uid", columns="attr_name", values="attr_value")
    
    return attributes_as_cols

def combine_dimensions(attributes_as_cols):
    # CODE BLOCK 8: PROCESS MEASUREMENT FEATURES

    def cleanup_text_in_df(text, dataframe, column):
        dataframe[column] = dataframe[column].str.replace(text, '')
        return dataframe


    columns_to_clean = ["Assembled Height (in.)", "Assembled Width (in.)", "Product Depth (in.)", "Product Height (in.)", "Product Length (in.)", "Product Width (in.)"]
    attributes_as_cols = cleanup_text_in_df("in", attributes_as_cols, "Assembled Depth (in.)")
    for column in columns_to_clean:
        attributes_as_cols = cleanup_text_in_df("in", attributes_as_cols, column)


    def make_one_measurement(depth_col, width_col, length_col):
        def check_nan(col_name):
            nan = type(float('nan'))
            if type(col_name) != nan:
                return True
        measurements = []
        if check_nan(depth_col):
            depth = depth_col
            measurements.append(depth)
        if check_nan(width_col):
            width = width_col
            measurements.append(width)
        if check_nan(length_col):
            length = length_col
            measurements.append(length)
        measurement_str = (" x ").join(measurements)
        return measurement_str


    attributes_as_cols["Assembled Measurement (DxWxH)"] = attributes_as_cols.apply(lambda x: make_one_measurement(x["Assembled Depth (in.)"], x["Assembled Height (in.)"], x["Assembled Width (in.)"]), axis = 1)
    attributes_as_cols["Product Measurement (LxWxD)"] = attributes_as_cols.apply(lambda x: make_one_measurement(x["Product Length (in.)"], x["Product Width (in.)"], x["Assembled Depth (in.)"]), axis = 1)
    attributes_as_cols["Product Measurement (DxWxH)"] = attributes_as_cols.apply(lambda x: make_one_measurement(x["Product Length (in.)"], x["Product Width (in.)"], x["Assembled Height (in.)"]), axis = 1)
    
    return attributes_as_cols

def integrate_prod_attr(df_all, df_attr):
    # adding attributes (init)
    df_w_attr, original_attributes = add_init_attributes(df_all, df_attr)
    # get 20 most common attributes
    common_attributes = get_common_attributes(original_attributes)
    # combine dimensions into one value
    combined_dim_attr = combine_dimensions(common_attributes)
    # add title-dims combo
    df_all = add_title_plus_dims(df_w_attr, combined_dim_attr)
    # add title-color combo
    df_all = add_title_plus_color(df_all, combined_dim_attr)
    
    return df_all, combined_dim_attr

def add_title_plus_dims(df_all_w_attr, attributes_as_cols):
    # CODE BLOCK 10: MAKE PRODUCT TITLE + MEASUREMENTS FEATURE

    measurements_in_product = attributes_as_cols[["Assembled Measurement (DxWxH)", "Product Measurement (LxWxD)", "Product Measurement (DxWxH)"]]
    measurements_in_product.reset_index(inplace = True)
    measurements_in_product.columns = ["product_uid", "Assembled Measurement (DxWxH)", "Product Measurement (LxWxD)", "Product Measurement (DxWxH)"]
    df_all_w_attr = df_all_w_attr.join(measurements_in_product, how = 'left')

    # Unstemmed
    df_all_w_attr["title_and_assembled_measurement"] = df_all_w_attr["product_title"] + " " + df_all_w_attr["Assembled Measurement (DxWxH)"]
    df_all_w_attr["title_and_measurement_LWD"] = df_all_w_attr["product_title"] + " " + df_all_w_attr["Product Measurement (LxWxD)"]
    df_all_w_attr["title_and_measurement_DWH"] = df_all_w_attr["product_title"] + " " + df_all_w_attr["Product Measurement (DxWxH)"]

    # Stemmed
    df_all_w_attr["title_and_assembled_measurement_stem"] = df_all_w_attr["product_title_stem"] + " " + df_all_w_attr["Assembled Measurement (DxWxH)"]
    df_all_w_attr["title_and_measurement_LWD_stem"] = df_all_w_attr["product_title_stem"] + " " + df_all_w_attr["Product Measurement (LxWxD)"]
    df_all_w_attr["title_and_measurement_DWH_stem"] = df_all_w_attr["product_title_stem"] + " " + df_all_w_attr["Product Measurement (DxWxH)"]
    
    return df_all_w_attr
    
def add_title_plus_color(df_all_w_attr, attributes_as_cols):
    stemmer = SnowballStemmer('english')
    
    # CODE BLOCK 11: MAKE PRODUCT TITLE + COLOUR FEATURE
    colours = attributes_as_cols[["Color Family", "Color/Finish"]]
    colours = colours.reset_index()
    # Make Colour Stems
    colours['Color Family Stem'] = colours['Color Family'].map(lambda x:str_stemmer(x))
    colours['Color/Finish Stem'] = colours['Color/Finish'].map(lambda x:str_stemmer(x))

    df_all_w_attr = df_all_w_attr.merge(colours, how = "left", on = "product_uid")
    # Non-stemmed
    df_all_w_attr["title_and_color"] = df_all_w_attr["product_title"] + " " + df_all_w_attr["Color/Finish"]
    df_all_w_attr["title_and_color_fam"] = df_all_w_attr["product_title"] + " " + df_all_w_attr["Color Family"]
    # Stemmed
    df_all_w_attr["title_and_color_stem"] = df_all_w_attr["product_title_stem"] + " " + df_all_w_attr["Color/Finish Stem"]
    df_all_w_attr["title_and_color_fam_stem"] = df_all_w_attr["product_title_stem"] + " " + df_all_w_attr["Color Family Stem"]
    
    return df_all_w_attr

def remove_stopwords(df_all, df_attr):
    '''
    CAUTION: this modifies the original inputs
    This removes stopwords from title, product description, attributes, and search term.
    
    df_all: pandas dataframe incl product_title, search_term, and product_description as columns
    df_attr: pandas dataframe incl 
    '''

    stop_words_incl_in = set(stopwords.words(("english")))
    non_stopwords = ['in', 'no']
    stop_words = set([word for word in stop_words_incl_in if word not in non_stopwords]) # removing in from stop_words since dimensions use in. to denote inches
    
    columns = ['search_term','product_title', 'product_description']
    for column in columns:
        df_all[column] = df_all[column].apply(lambda x: x if type(x) != str else ' '.join([word for word in x.split() if word not in stop_words]))
    
    df_attr['value'] = df_attr['value'].apply(lambda x: x if type(x) != str else ' '.join([word for word in x.split() if word not in stop_words]))
    
def remove_words_in_brackets(df_all):
    '''
    This is only used on the title!
    '''
    
    df_all["product_title"] = df_all["product_title"].str.replace(r"(\(.*?\))", "",regex=True)
    
    return df_all
    
def space_before_capitals(df_all, df_attr):
    load()
    columns = ['product_description']
    for column in columns:
        df_all[column] = df_all[column].str.replace(r"([a-z])([A-Z])",r"\1 \2", regex = True)
        
    df_attr['value'] = df_attr['value'].str.replace(r"([a-z])([A-Z])",r"\1 \2", regex = True)
    
    return df_all

def spell_check(df_all, df_attr):
    dictionary = spelling_checker_dict
    
    for column in df_all.columns:  
        if column not in ['id','product_uid','relevance']:
            for phrase in df_all[column]:
                if type(phrase) != str:
                    continue                
                for key in dictionary.keys():
                    phrase.replace(key, dictionary[key])
                    
    for phrase in df_attr['value']:
        if type(phrase) != str:
            continue
        for key in dictionary.keys():
            phrase.replace(key, dictionary[key])
    
    return df_all, df_attr

def convert_to_lowercase(df_all, df_attr):
    
    for column in df_all.columns:
        if column != 'id' and column != 'product_uid' and column != 'relevance':
            df_all[column] = df_all[column].str.lower()
            
    df_attr['value'] = df_attr['value'].str.lower()
    
    return df_all, df_attr