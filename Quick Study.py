import streamlit as st
import ptt as pt
import img2txt as tt
import os
from transformers import DistilBertTokenizerFast,DistilBertForQuestionAnswering
tokenizer = DistilBertTokenizerFast.from_pretrained('Valli/distillbert_custom_answer')
model = DistilBertForQuestionAnswering.from_pretrained("Valli/distillbert_custom_answer")
import torch
from transformers import pipeline,AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration, T5Tokenizer

qa=pipeline('question-answering',model=model,tokenizer=tokenizer)
summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')

# tokenizer_cus = AutoTokenizer.from_pretrained("result_summarise/checkpoint-1500")
# model_cus = AutoModelForSeq2SeqLM.from_pretrained("result_summarise/checkpoint-1500")
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

conptt=""
st.set_page_config(
    page_title="Quick Study",
    page_icon=":)"
)



st.title('Quick Study')

def save_uploadedfile(uploadedfile):
    with open( uploadedfile.name, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("File uploaded successfully👍"+uploadedfile.name)


# File upload
uploaded_file = st.file_uploader('Choose your .pdf file', type=["pdf","png","jpg"])
if uploaded_file is not None:
    # print(type(uploaded_file))
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}

    save_uploadedfile(uploaded_file)
    print(uploaded_file.type)
    if(uploaded_file.type=="application/pdf"):
        conptt = pt.pdfTotext(file_details["FileName"])
    else:
        r= uploaded_file.name

        conptt=tt.img2txt(r)

if "text" not in st.session_state:
    st.session_state["text"]=""
with st.form("my_form"):
   text=st.text_area("Text", value=conptt)
   question=st.text_input(label="Question")


   # Every form must have a submit button.
   st.session_state["text"] = text
   col1,col2,col3 = st.columns(3)
   with col1:
       sum = st.form_submit_button("summarise")

   with col2:
       sa = st.form_submit_button("short answer")

       # if mcq:
       #     st.session_state["text"] = text
       #     st.write("Check the MCQ tab")
   with col3:
       la = st.form_submit_button("Long answer question")
   if sum:
       summarized_text = summarizer(text, summary_model, summary_tokenizer)
       st.write(summarized_text)
   # if mcq:
   #     st.session_state["text"] = text
   #     st.write("Check the MCQ tab")
   if la:
       model_name = "vblagoje/bart_lfqa"
       query_and_docs = "question: {} context: {}".format(question, text)
       tokenizer_lq = AutoTokenizer.from_pretrained(model_name)
       model_lf = AutoModelForSeq2SeqLM.from_pretrained(model_name)
       model_input = tokenizer_lq(query_and_docs, truncation=True, padding=True, return_tensors="pt")
       generated_answers_encoded = model_lf.generate(input_ids=model_input["input_ids"],
                                                  attention_mask=model_input["attention_mask"],
                                                  min_length=64,
                                                  max_length=256,
                                                  do_sample=False,
                                                  early_stopping=True,
                                                  num_beams=8,
                                                  temperature=1.0,
                                                  top_k=None,
                                                  top_p=None,
                                                  eos_token_id=tokenizer_lq.eos_token_id,
                                                  no_repeat_ngram_size=3,
                                                  num_return_sequences=1)
       st.write(tokenizer_lq.batch_decode(generated_answers_encoded, skip_special_tokens=True,clean_up_tokenization_spaces=True))

   if sa:
       qa = pipeline('question-answering', model=model, tokenizer=tokenizer)
       ans=qa(question,text)['answer']
       st.write(ans)
