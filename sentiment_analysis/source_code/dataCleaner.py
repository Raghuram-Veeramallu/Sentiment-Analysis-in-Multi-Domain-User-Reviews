import pandas as pd
import xml.etree.ElementTree as ET
from lxml import etree
import sys
from random import shuffle

category = ""

def __init__(self, category):
    self.category = category


def define_file_names_pos_neg(category):
    positive_file = '../data/train_data_raw/'+category+'/reviews_positive.xml'
    negative_file = '../data/train_data_raw/'+category+'/reviews_negative.xml'
    return (positive_file, negative_file)


def parseFiles(positive_file, negative_file):
    with open(positive_file, 'r') as pos_file:
        positive_xml_string = pos_file.read()
    pos_file.close()

    with open(negative_file, 'r') as neg_file:
        negative_xml_string = neg_file.read()
    neg_file.close()

    pos_parser = etree.XMLParser(encoding = "UTF-8", recover = True)
    neg_parser = etree.XMLParser(encoding = "UTF-8", recover = True)

    positive_root = etree.fromstring(positive_xml_string, parser = pos_parser)
    negative_root = etree.fromstring(negative_xml_string, parser = neg_parser)

    positive_reviews = positive_root.findall('review')
    negative_reviews = negative_root.findall('review')

    return (positive_reviews, negative_reviews)

def helpful_conversion(text):
    try:
        new_text = text.replace('of', '|')
        scores = new_text.split('|')
        helpful_score = (float(scores[0].replace("\n", "").replace("\t", "").replace(" ", "")) / float(scores[1].replace("\n", "").replace("\t", "").replace(" ", "")))
        helpful_score *= 100
        return int(helpful_score)
    except:
        return 0

def add_to_review_dict(reviews_list, review, sentiment):
    count = 0
    while count < (len(review)-1):
        try:
            rev = review[count + 1]
            uniq_id = rev[0].text.strip()
            product_name = rev[2].text.strip()
            helpful = helpful_conversion(rev[4].text)
            rating = int(float(rev[5].text.replace("\n", "").replace("\t", "").replace(" ", "")))
            summary = rev[6].text.strip()
            review_text = rev[10].text.strip()
            reviewer = rev[8].text.strip()
            sentiment = sentiment

            reviews_list.append([uniq_id, product_name, summary, review_text, reviewer, helpful, rating, sentiment])

        except:
            print("Unexpected error:", sys.exc_info()[0])
            continue
            
        count = count + 1

    return reviews_list


def saveAsCSV(reviews, category):
    shuffle(reviews)
    dataframe = pd.DataFrame(reviews, columns = ['uniq_id', 'product_name', 'summary', 'review_text', 'reviewer', 'helpful', 'rating', 'sentiment'])
    dataframe.to_csv(r'data/full_data/'+category+'.csv')
    return

if __name__ == "__main__":

    positive_file, negative_file = define_file_names_pos_neg(category)
    pos_rev, neg_rev = parseFiles(positive_file, negative_file)

    reviews = []

    reviews = add_to_review_dict(reviews, pos_rev, "1")
    reviews = add_to_review_dict(reviews, neg_rev, "0")

    saveAsCSV(reviews, category)