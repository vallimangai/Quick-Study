import streamlit as st
import pandas as pd
import numpy as np
from final_mcq import *
import random
st.title("trying....")
if("text" not in st.session_state or st.session_state['text']==""):
    st.write("text" not in st.session_state )
    st.write("Please enter the text")
else:
    option=[]
    text=st.session_state['text']
    summarized_text = summarizer(text, summary_model, summary_tokenizer)
    imp_keywords = get_keywords(text, summarized_text)
    for answer in imp_keywords:
        ques = get_question(summarized_text, answer, question_model, question_tokenizer)
            # st.write(ques)
        print(answer.capitalize())
        print("\n")
        l=get_distractors(answer, ques, s2v, sentence_transformer_model, 40, 0.2)
        if(len(l)>3):
            l = l[:3]
            opt=['a','b','c','d']
            l.append(answer)
            random.shuffle(l)
            st.write(ques)
            for i in range(len(l)):
                if(l[i]==answer):
                    st.success("   "+opt[i]+") "+l[i])
                else:
                    st.write("    "+opt[i]+") "+l[i])
            # c = ["Please select an answer"]
            # c.extend(l)
            # st.text(f"Solve: {ques}")
            # a = st.selectbox('Answer:', c)
            #
            # if a != "Please select an answer":
            #     st.write(f"You chose {a}")
            #     if (answer == a):
            #         st.write("Correct!")
            #     else:
            #         st.write(f"Wrong!, the correct answer is {answer}")
