import streamlit as st
from transformers import DistilBertTokenizerFast,DistilBertForQuestionAnswering
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-custom')
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-custom")
import torch
from transformers import pipeline,AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
qa=pipeline('question-answering',model=model,tokenizer=tokenizer)

st.set_page_config(
    page_title="Multipage APP",
    page_icon=":)"
)

st.title('MCQ generator')
if "text" not in st.session_state:
    st.session_state["text"]=""
with st.form("my_form"):
   text=st.text_area("Inside the form",st.session_state["text"])
   question=st.text_input("Enter the question")
   sa=st.form_submit_button("short answer")
   la=st.form_submit_button("Long answer question")
   # Every form must have a submit button.
   mcq = st.form_submit_button("mcq")
   if mcq:
       st.session_state["text"]=text
       st.write("Check the MCQ tab")
   if la:
       model_name = "bart_lfqa"
       query_and_docs = "question: {} context: {}".format(question, text)
       tokenizer_lq = AutoTokenizer.from_pretrained(model_name)
       model_lf = AutoModelForSeq2SeqLM.from_pretrained(model_name)
       model_input = tokenizer_lq(query_and_docs, truncation=True, padding=True, return_tensors="pt")
       generated_answers_encoded = model_lf.generate(input_ids=model_input["input_ids"].to(device),
                                                  attention_mask=model_input["attention_mask"].to(device),
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
#     summarized_text = summarizer(text, summary_model, summary_tokenizer)
#     imp_keywords = get_keywords(text, summarized_text)
#     for answer in imp_keywords:
#         ques = get_question(summarized_text, answer, question_model, question_tokenizer)
#         # st.write(ques)
#         print(answer.capitalize())
#         print("\n")
#         l=get_distractors(answer, ques, s2v, sentence_transformer_model, 40, 0.2)
#         if(len(l)>3):
#             l=l[:4]
#             l.append(answer)
#             random.shuffle(l)
#             c=["Please select an answer"]
#             c.extend(l)
#             st.text(f"Solve: {ques}")
#             a = st.selectbox('Answer:', c)
#
#             if a != "Please select an answer":
#                 st.write(f"You chose {a}")
#                 if (answer == a):
#                     st.write("Correct!")
#                 else:
#                     st.write(f"Wrong!, the correct answer is {answer}")
#
#             # st.write(ques)
#             # ans=st.radio(" ", (i for i in l))
#            # if ans == answer:
#            #     st.write('You selected Correctly.')
#            # else:
#            #     st.write("Try again")
#         print("\n\n")

st.write( st.session_state["text"])

