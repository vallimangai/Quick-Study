# Quick-Study
It is NLP model for Question and Answer based on text
### Introduction
QuickStudy is an educational tool designed to help people grasp concepts quickly while avoiding getting stuck. Users can upload pdfs, images or give the text and receive summarised text, allowing them to understand the concept by reading a few sentences rather than a pool of sentences from the original text. Users can also get short and long answers to their text-related questions by simply typing them in. QuickStudy can also generate MCQs and questions along with corresponding short and long answers to assist users in assessing their knowledge of the topic. 

### Features:
QuickStudy contains multiple features which are listed below
#### Summarisation
Text summarization is the process of creating a short, coherent, and fluent summary of a longer text document and involves the outlining of the text’s major points. Extracted text from pdf, image or the text given by the user is taken as input for the model and the summarised text is the expected output of the model.

#### Short answer
Given a question by the user, the model in this module must be able to generate corresponding short answer from the text.

#### Long answer
Given a question by the user, the model in this module must be able to generate corrsponding long answer from the text.
#### Short question and answer
In this module question and answer generation is automated. It takes text as input for the model and generates both question and short answer.

#### Long question and answer
In this module question and answer generation is automated. It takes text as input for the model and generates both question and long answer.

#### MCQ
The model for MCQ generation takes original text and summarised text as input and process to display questions, distractors, correct answer as output.
#### Text extraction
The text is extracted from the uploaded pdf and the image.

### Pre-requisite:
Fast internet to download the model that is available in hugging face

Tesseract <a href="https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.2.0.20220712.exe"> To download </a>



### How to run it

#### Clone the repository to your local directory

git clone https://github.com/vallimangai/Quick-Study

#### Activate your virtual environment. Follow steps in this link to create your virtual environment : Click here

pip install virtualenv

virtualenv env

env\Scripts\activate

#### Install packages from req.txt

pip install -r req.txt

#### Run app.py file

streamlit run "Quick Study.py"

Now you can see our app running on http://localhost:8501/! Register with an account and try it out for yourself.

### Outputs:
#### uploading the image to get the text
![image](https://user-images.githubusercontent.com/55653139/202740254-6638a13f-34cf-4232-85ee-7d68134ddc89.png)
#### uploading the pdf to get the text
![image](https://user-images.githubusercontent.com/55653139/202745150-d446cd4e-6c23-4846-9426-869b0237fcbd.png)
#### summarisation
![image](https://user-images.githubusercontent.com/55653139/202745246-c1da67f4-960d-485e-8b06-1403e020c419.png)
#### Short answer for questions
![image](https://user-images.githubusercontent.com/55653139/202745323-509264ca-302f-45a5-a944-03a94ca4d595.png)
#### Long answer for questions
![image](https://user-images.githubusercontent.com/55653139/202745381-5516064b-6a6d-4c23-b2e7-f3191d48286d.png)
#### Long question and answer generations
![image](https://user-images.githubusercontent.com/55653139/202745472-0d05ef82-79e1-43a8-8782-a0f458ab9750.png)
![image](https://user-images.githubusercontent.com/55653139/202745588-1009d916-1281-4ad8-aa45-6874d7af4955.png)
#### Short question and answer generations
![image](https://user-images.githubusercontent.com/55653139/202745628-ed8afe27-8ed4-48e0-8da5-a37853080d79.png)
![image](https://user-images.githubusercontent.com/55653139/202745661-dfcf3db4-4e37-4d88-82fd-af79942d9653.png)
#### MCQ generation
![image](https://user-images.githubusercontent.com/55653139/202745744-3836c062-64fd-47b4-b6d5-01c340979aa2.png)
![image](https://user-images.githubusercontent.com/55653139/202745809-0a48710e-6600-4c8e-b5c4-2a3655d1e183.png)


