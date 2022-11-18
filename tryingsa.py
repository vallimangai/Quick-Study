import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')



import random
import numpy as np



import nltk

nltk.download('punkt')
nltk.download('brown')
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize


def postprocesstext(content):
    final = ""
    for sent in sent_tokenize(content):
        sent = sent.capitalize()
        final = final + " " + sent
    return final


def summarizer(text, model, tokenizer):
    text = text.strip().replace("\n", " ")
    text = "summarize: " + text
    # print (text)
    max_len = 512
    encoding = tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=False, truncation=True,
                                     return_tensors="pt")

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(input_ids=input_ids,
                          attention_mask=attention_mask,
                          early_stopping=True,
                          num_beams=3,
                          num_return_sequences=1,
                          no_repeat_ngram_size=2,
                          min_length=75,
                          max_length=300)

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    summary = dec[0]
    summary = postprocesstext(summary)
    summary = summary.strip()
    return summary




# In[5]:


import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import spacy
import pke
import traceback


def get_nouns_multipartite(content):
    out = []
    try:
        e = nlp = spacy.load('en_core_web_sm')
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=e(content), language='en')
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'PROPN', 'NOUN'}
        # pos = {'PROPN','NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        # extractor.candidate_selection(pos=pos, stoplist=stoplist)
        extractor.candidate_selection(pos=pos)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.75,
                                      method='average')
        keyphrases = extractor.get_n_best(n=15)

        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()

    return out


# In[6]:


from flashtext import KeywordProcessor


def get_keywords(originaltext):
    keywords = get_nouns_multipartite(originaltext)
    print("keywords unsummarized: ", keywords)
    keyword_processor = KeywordProcessor()
    for keyword in keywords:
        keyword_processor.add_keyword(keyword)

    # keywords_found = keyword_processor.extract_keywords(summarytext)
    # keywords_found = list(set(keywords_found))
    # print("keywords_found in summarized: ", keywords_found)

    important_keywords = list(set(keywords))
    print(list(set(important_keywords)))
    return list(set(important_keywords))


# In[7]:





def get_question(context, answer, model, tokenizer):
    text = "context: {} answer: {}".format(context, answer)
    encoding = tokenizer.encode_plus(text, max_length=384, pad_to_max_length=False, truncation=True,
                                     return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    outs = model.generate(input_ids=input_ids,
                          attention_mask=attention_mask,
                          early_stopping=True,
                          num_beams=5,
                          num_return_sequences=1,
                          no_repeat_ngram_size=2,
                          max_length=72)

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

    Question = dec[0].replace("question:", "")
    Question = Question.strip()
    return Question

from transformers import DistilBertTokenizerFast,DistilBertForQuestionAnswering
tokenizer = DistilBertTokenizerFast.from_pretrained('Valli/distillbert_custom_answer')
model = DistilBertForQuestionAnswering.from_pretrained("Valli/distillbert_custom_answer")
from transformers import pipeline,AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
qa=pipeline('question-answering',model=model,tokenizer=tokenizer)
def get_answer(text,question):
    ans = qa(question, text)['answer']
    return ans