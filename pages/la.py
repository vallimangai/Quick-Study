import streamlit as st
import pandas as pd
import numpy as np
from tryingla import *
import random
st.title("trying....")
if("text" not in st.session_state or st.session_state['text']==""):
    st.write("text" not in st.session_state )
    st.write("Please enter the text")
else:
    text = st.session_state['text']
    imp_keywords = get_keywords(text)
    question=[]
    for answer in imp_keywords:
        ques = get_question(text, answer,question_model, question_tokenizer)
        question.append(ques)
    answer=[]
    print(len(list(set(question))))
    for ques in list(set(question)):
        st.write("getting answer")
        ans=get_answer(text,ques)
        if ans not in answer:
            st.write(ques)
            answer.append(ans)
            st.success("       "+ans)
