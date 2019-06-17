import pandas as pd
import xml.etree.ElementTree as ET
from lxml import etree
import sys
from random import shuffle

category = ""

def __init__(self, category):
    self.category = category


def define_file_names_pos_neg(category):
    file_name = '../data/test_data_raw/'+category+'.xml'
    return file_name


def parseFiles(file_name):
    
    with open(file_name, 'r') as file_data:
        data_xml_string = file_data.read()
    file_data.close()

    xmltree_parser = etree.XMLParser(encoding = "UTF-8", recover = True)

    data_root = etree.fromstring(data_xml_string, parser = xmltree_parser)

    review_data = data_root.findall('review')

    return review_data

def helpful_conversion(text):
    try:
        new_text = text.replace('of', '|')
        scores = new_text.split('|')
        helpful_score = (float(scores[0].replace("\n", "").replace("\t", "").replace(" ", "")) / float(scores[1].replace("\n", "").replace("\t", "").replace(" ", "")))
        helpful_score *= 100
        return int(helpful_score)
    except:
        return 0

def add_to_review_dict(reviews_list, review):
    count = 0
    while count < 100:
        try:
            rev = review[count + 1]
            uniq_id = rev[0].text.strip()
            product_name = rev[2].text.strip()
            helpful = helpful_conversion(rev[4].text)
            summary = rev[6].text.strip()
            review_text = rev[10].text.strip()
            reviewer = rev[8].text.strip()

            reviews_list.append([uniq_id, product_name, summary, review_text, reviewer, helpful])

        except:
            print("Unexpected error:", sys.exc_info()[0])
            continue
            
        count = count + 1

    return reviews_list


def test_data_results(reviews_list, review):
    count = 0
    while count < 100:
        try:
            rev = review[count + 1]
            uniq_id = rev[0].text.strip()
            rating = int(float(rev[5].text.replace("\n", "").replace("\t", "").replace(" ", "")))

            reviews_list.append([uniq_id, rating])

        except:
            print("Unexpected error:", sys.exc_info()[0])
            continue
            
        count = count + 1

    return reviews_list


def saveAsCSV(reviews, category):
    shuffle(reviews)
    dataframe = pd.DataFrame(reviews, columns = ['uniq_id', 'product_name', 'summary', 'review_text', 'reviewer', 'helpful'])
    dataframe.to_csv(r'data/test_data/test_'+category+'.csv')
    return

def saveResCSV(reviews, category):
    shuffle(reviews)
    dataframe = pd.DataFrame(reviews, columns = ['uniq_id', 'rating'])
    dataframe.to_csv(r'data/solution_data/'+category+'.csv')
    return

if __name__ == "__main__":

    file_name = define_file_names_pos_neg(category)
    data_review = parseFiles(file_name)

    reviews = []
    reviewAns = []

    reviews = add_to_review_dict(reviews, data_review)
    reviewAns = test_data_results(reviewAns, data_review)

    saveAsCSV(reviews, category)
    saveResCSV(reviewAns, category)