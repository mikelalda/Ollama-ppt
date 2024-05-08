import gradio as gr
from pptx import Presentation
import subprocess
import io
from io import BytesIO
import PyPDF2
import os
import subprocess
import requests
from bs4 import BeautifulSoup
import os
from langchain_community.llms import Ollama
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.embeddings.ollama import OllamaEmbeddings


DESCRIPTION = """\
# Ollama ppt creator
This Space demonstrates models integrated in [Ollama](https://ollama.com) with the model [llama3](https://huggingface.co/meta-llama/Meta-Llama-3-8B) by Meta, a Llama 3 model with 8B parameters fine-tuned for chat instructions. Feel free to play with it, or duplicate to run generations without a queue! If you want to run your own service, you can also [deploy the model on Inference Endpoints](https://huggingface.co/inference-endpoints).
"""

LICENSE = """
---
As a derivate work of [Ollama](https://ollama.com) and [llama3](https://huggingface.co/meta-llama/Meta-Llama-3-8B),
this demo is governed by the original [license](https://github.com/ollama/ollama?tab=MIT-1-ov-file) and [license](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/LICENSE.txt).
"""

model="ppt"
pipe = Ollama(model=model)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

prompt = PromptTemplate.from_template(
            """ 
            Contexto: {context} 
            Pregunta: {question}
            """
        )
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False,)

def filter_custom(choice):
    if choice == "custom":
        ppt_file = gr.File(type="filepath", label="Upload PDF", visible=True)
    else:
        ppt_file = gr.File(type="filepath", label="Upload PDF", visible=False)
    return ppt_file
def filter_input(choice):
    if choice == "Text":
        input_text = gr.TextArea(lines=5, placeholder="Enter TEXT", visible=True)
        pdf_file = gr.File(type="filepath", label="Upload PDF", visible=False)
        input_url = gr.TextArea(lines=5, placeholder="Enter TEXT", visible=False)
    if choice == "url":
        input_text = gr.TextArea(lines=5, placeholder="Enter TEXT", visible=False)
        pdf_file = gr.File(type="filepath", label="Upload PDF", visible=False)
        input_url = gr.TextArea(lines=5, placeholder="Enter TEXT", visible=True)
    if choice == "PDF":
        input_text = gr.TextArea(lines=5, placeholder="Enter TEXT", visible=False)
        pdf_file = gr.File(type="filepath", label="Upload PDF", visible=True)
        input_url = gr.TextArea(lines=5, placeholder="Enter TEXT", visible=False)

    return input_text, pdf_file, input_url

def generate_text2ppt_input_prompt(input_type, input_value, input_pages):
    header = """Realizaz un PPt de %s paginas sobre el contexto. Intenta filtrar las cosas interesantes utilizando solamente el contexto.""" % input_pages
    if input_type == "Text":
        with open('./file.txt', 'w') as f:
            f.write(input_value)
        docs_list = TextLoader('./file.txt').load()
        os.remove('./file.txt')
        docs = filter_complex_metadata(text_splitter.split_documents(docs_list))
    elif input_type == "PDF":
        # with open(input_value, 'rb') as pdf_file:
            # pdf_reader = PyPDF2.PdfReader(pdf_file)
            # # Convert the content of each page to a string.
            # text = ""
            # for page_num in range(len(pdf_reader.pages)):
            #     page = pdf_reader.pages[page_num]
            #     summary_value += page.extract_text()
        docs_list = PyPDFLoader(input_value, extract_images=True).load()
        docs = filter_complex_metadata(text_splitter.split_documents(docs_list))

    elif input_type == "url":
        urls_list = input_value.split("\n")
        doc = [WebBaseLoader(url).load() for url in urls_list]
        docs_list = [item for sublist in doc for item in sublist]
        docs = filter_complex_metadata(text_splitter.split_documents(docs_list))

    else:
        print("ERROR: Invalid input")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    # simp_vector = vectorstore.similarity_search_with_score(self.msg, k=5)
    summary_value = vectorstore.as_retriever()
    # vectorstore.persist() # To sabe the vector in a db
    return header, summary_value

def find_images(search_query,num_results):
    # Construct the Google Images search URL
    search_url = f'https://www.google.com/search?q={search_query}&source=lnms&tbm=isch'
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all image links in the search results
    image_links = [link.get('src') for link in soup.find_all('img')]
    return image_links[:num_results]


def images2slides(text, num_images):
    slides = text.split("===")
    output = ""
    for num, slide in enumerate(slides):
        output += slide
        images = find_images(slide.split('\n')[-1],num_images)
        for i, image in enumerate(images):
            output += "![Image{}]({})\n".format(i,image) + '\n ==='
    return output

# Function to execute text2ppt
def text2ppt(header, input_prompt, input_type, input_theme, num_images):
    # llamada al modelo
    chain = ({"context": input_prompt, "question": RunnablePassthrough()}
                    | prompt 
                    | pipe
                    | StrOutputParser())

    result = chain.invoke({"query": header})
    md_text = result
    md_text =images2slides(md_text, num_images)
    md_text_list = md_text.split('===')

    f = io.open("text2ppt_input.md", 'w', encoding="utf-8")
    for i in range(0, len(md_text_list)):
        data = md_text_list[i].strip() + "\n"
        f.write(data)
    f.close()

    if input_theme == 'default':
        subprocess.call(["./pandoc.exe", "text2ppt_input.md", "-t", "pptx", "-o", "text2ppt_output.pptx"])
    else:
        ppt_theme = "--reference-doc=./template/"+input_theme+".pptx"
        subprocess.call(["./pandoc.exe", "text2ppt_input.md", "-t", "pptx", ppt_theme, "-o", "text2ppt_output.pptx"])


def create_ppt(choice, page_choice, thema_select, input_text, pdf_file,num_images, input_url):
    if choice == "Text":
        input_ = input_text
    elif choice == "PDF":
        input_ = pdf_file
    elif choice == "url":
        input_ = input_url
    header, input_ = generate_text2ppt_input_prompt(choice, input_, page_choice)
    text2ppt(header, input_, choice, thema_select, num_images)
    prs = Presentation("text2ppt_output.pptx")
    binary_output = BytesIO()
    prs.save(binary_output)
    return "./text2ppt_output.pptx"

def interface():
    
    with gr.Tab("Text2PPT"):
        with gr.Row("Template selection"):
            template = gr.Dropdown(
                label="Please select the template you want.",
                choices=['default', 'yellow', 'gradation_green', 'blue', 'green', 'custom'],
                value='default'
            )
            
            ppt_file = gr.File(type="filepath", label="Upload PDF", visible=False)
        with gr.Row("Page cuantity selection"):
            pages = gr.Slider(minimum=2, maximum=30, step=1, value=5, label="Number of PPT pages")
            images = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="Number of images in each PPT page")
        with gr.Row("Input selection"):
            my_order = ['Text', 'PDF', 'url']
            radio_from = gr.Radio(my_order, label="Please select the file type and enter the content!", value='Text')
            input_text = ''
            input_text = gr.TextArea(lines=5,label="Text", placeholder="Enter TEXT", visible=True)
            input_url = gr.TextArea(lines=5, label="Url", placeholder="Enter URL", visible=False)
            pdf_file = gr.File(type="filepath", label="Upload PDF", visible=False)   
        with gr.Row("Output ppt"):
            output_ppt = gr.File(
                label='Output PPT'
            )  
                

        confirm = gr.Button(value="Confirm",)
        
    template.change(fn=filter_custom,inputs=[template], outputs=[ppt_file])
    radio_from.change(fn=filter_input,inputs=[radio_from], outputs=[input_text, pdf_file,input_url])
    confirm.click(fn=create_ppt,inputs=[radio_from,pages,template,input_text, pdf_file, images, input_url],outputs=output_ppt)



with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    interface()
    gr.Markdown(LICENSE)

if __name__ == "__main__":
    # env_set()
    demo.queue(max_size=20).launch()