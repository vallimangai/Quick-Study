import streamlit as st
import pandas as pd
import numpy as np
from tryingsa import *
import random
st.title("Short Question and Answer Generator")
if("text" not in st.session_state or st.session_state['text']==""):
    st.write("text" not in st.session_state )
    st.write("Please enter the text")
else:
    text = st.session_state['text']
    imp_keywords = get_keywords(text)
    #st.write(imp_keywords)
    print(len(imp_keywords))
    question=[]
    for answer in imp_keywords:
        ques = get_question(text, answer,question_model, question_tokenizer)
        question.append(ques)
    answer=[]
    for ques in list(set(question)):
        ans=get_answer(text,ques)
        #if ans not in answer:
        st.write(ques)
        answer.append(ans)
        st.success("       "+ans)
