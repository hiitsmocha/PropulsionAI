# install pandas and spacy library

import pandas as pd
#review the spacy NLP library
import spacy
import re
from top2vec import Top2Vec
from gensim import corpora
from gensim.models import LdaModel


# Extract company names and locations
def extract_company_names_and_locations(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    companies = []
    locations = []
    ideas = []
    for ent in doc.ents:
        if ent.label_ == "ORG":
            companies.append(ent.text)
        elif ent.label_ == "GPE":
            locations.append(ent.text)
        elif ent.label_ == "NORP":
            ideas.append(ent.text)
    return companies, locations, ideas

# Example usage
with open("bain_fashion.txt", "r") as file:
    text = file.read()
with open("bain_coffee.txt", "r") as file:
    text += file.read()

companies, locations, ideas = extract_company_names_and_locations(text)
print("Companies:", companies)
print("Locations:", locations)
print("Ideas:", ideas)

# Was able to generate a list of companies, locations and ideas from the text data. But this 
# is not the final goal.

# The problem is that the speed and accuracy is not yet optimal. The model is not able to extract
# concepts and ideas, rather it extracts companies and locations by specific keywords.

# To fix the problem, integrate with the LDA gensim model or the Top2Vec model to train base on
# a larger dataset.

# To-do: Obtain larger datasets with more words.

# To-develop: Develop a graph reading model to read info from graphs and tables.



