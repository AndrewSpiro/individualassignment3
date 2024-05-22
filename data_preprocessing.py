from funcs.preprocessing_funcs import *
import nltk

load_save = False
load_small = False

if load_save == True:
    data = pd.read_csv('full preprocessed data.csv')
    if load_small == True:
        data = data[:400]

else:
    train_path = 'data/home-depot-product-search-relevance/train.csv/train.csv'
    test_path = 'data/home-depot-product-search-relevance/test.csv/test.csv'
    attributes_path = 'data/home-depot-product-search-relevance/attributes.csv/attributes.csv'
    product_descriptions_path = 'data/home-depot-product-search-relevance/product_descriptions.csv/product_descriptions.csv'

    df_all, df_attr = load_data(train_path, test_path, attributes_path, product_descriptions_path, bool_dev = True)

    print('starting')
    # spell correction
    spell_check(df_all, df_attr)
    print('finished spell check')
    # remove word in brackets from title
    remove_words_in_brackets(df_all)
    print('finished removing brackets')
    # segment words
    space_before_capitals(df_all, df_attr)
    print('finished adding spaces')
    # remove stop words
    remove_stopwords(df_all, df_attr)
    print('finished removing stop words')
    # lowercasing
    convert_to_lowercase(df_all, df_attr)
    print('finished converting to lowercase')
    # stemming
    stem(df_all)
    print('finished stemming')
    # adding product attribute information
    df_all, attributes_as_columns = integrate_prod_attr(df_all, df_attr)
    print('finished integrating product attributes')

    df_all.to_csv("data/preprocessed data/test preprocessed data.csv")
    attributes_as_columns.to_csv("data/preprocessed data/test attributes as cols.csv")
    
    # test change