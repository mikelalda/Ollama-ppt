import gradio as gr
import json
import requests
from bs4 import BeautifulSoup
from langchain_community.llms import Ollama
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders import UnstructuredPDFLoader
from slide_gen import SlideGen


DESCRIPTION = """\
# Ollama ppt creator
This Space demonstrates models integrated in [Ollama](https://ollama.com) with the model [llama3](https://huggingface.co/meta-llama/Meta-Llama-3-8B) by Meta, a Llama 3 model with 8B parameters fine-tuned for chat instructions. Feel free to play with it, or duplicate to run generations without a queue! If you want to run your own service, you can also [deploy the model on Inference Endpoints](https://huggingface.co/inference-endpoints).
"""

LICENSE = """
---
As a derivate work of [Ollama](https://ollama.com) and [llama3](https://huggingface.co/meta-llama/Meta-Llama-3-8B),
this demo is governed by the original [license](https://github.com/ollama/ollama?tab=MIT-1-ov-file) and [license](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/LICENSE.txt).
"""

QUERY_PROMPT = PromptTemplate(
    template="""
    Eres un asistente de modelo de lenguaje de IA que crea PPTs.
    """,
)

model = Ollama(model='ppt')
pipe = Ollama(model='llama3.1')
embeddings = OllamaEmbeddings(model="mxbai-embed-large",show_progress=True)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
vectorstore = None

def filter_custom(choice):
    if choice == "custom":
        ppt_file = gr.File(type="filepath", label="Upload PPT", visible=True)
    else:
        ppt_file = gr.File(type="filepath", label="Upload PPT", visible=False)
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

def generate_text2ppt_input_prompt(input_type, input_value):
    global vectorstore
    if input_type == "Text":
        summary_value = input_value
    elif input_type == "PDF":
        if vectorstore != None:
            vectorstore.delete_collection()
        loader = UnstructuredPDFLoader(file_path=input_value)
        data = loader.load()
        chunks = text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name="local-rag")
        summary_value = MultiQueryRetriever.from_llm(vectorstore.as_retriever(), pipe, prompt=QUERY_PROMPT)
    elif input_type == "url":
        if vectorstore != None:
            vectorstore.delete_collection()
        urls_list = input_value.split("\n")
        doc = [WebBaseLoader(url).load() for url in urls_list]
        docs_list = [item for sublist in doc for item in sublist]
        chunks = text_splitter.split_documents(docs_list)
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name="local-rag")
        summary_value = MultiQueryRetriever.from_llm(vectorstore.as_retriever(), pipe, prompt=QUERY_PROMPT)
    else:
        print("ERROR: Invalid input")
    return summary_value

def find_images(search_query,num_results):
    # Construct the Google Images search URL
    search_url = f'https://www.google.com/search?q={search_query}&source=lnms&tbm=isch'
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all image links in the search results
    image_links = [link.get('src') for link in soup.find_all('img') if link.get('src').startswith('http')]
    return image_links[:num_results]


def images2slides(slides, num_images):
    for num, slide in enumerate(slides):
        slide["img_path"] = find_images(slide.get("title_text", ""), num_images)
        slides[num] = slide
    return slides

# Function to execute text2ppt
def text2ppt(input_prompt, input_type, input_theme, num_images, ppt_file):
    # llamada al modelo
    template = """
    {context}

    Eres un asistente de modelo de lenguaje de IA que crea PPTs utilizando el formato json.

    Resuma el texto de entrada y organícelo en una matriz de objetos JSON para que sea adecuado para una presentación de PowerPoint.
    Determine la cantidad necesaria de objetos JSON (diapositivas) en función de la longitud del texto.
    Cada punto clave de una diapositiva debe tener un máximo de 10 palabras.
    Considere un máximo de 5 viñetas por diapositiva.
    Devuelva la respuesta como una matriz de objetos JSON.
    El primer elemento de la lista debe ser un objeto JSON para la diapositiva del título. Este es un ejemplo de un objeto json de este tipo:
    {question}

    Asegúrese de que el objeto json sea correcto y válido.
    No muestre ninguna explicación. Solo necesito la matriz JSON como salida.
    Solo la matriz. No en formato Markdown.

    
    """
    question = """[{
        "id": 1,
        "title_text": "Título de mi presentación",
        "subtitle_text": "Subtítulo de mi presentación",
        "is_title_slide": "yes"
        },
        Y aquí está el ejemplo de datos json para diapositivas:
        {"id": 2, "title_text": "Título de la diapositiva 1", "text": ["Viñeta 1", "Viñeta 2"]},
        {"id": 3, "title_text": "Título de la diapositiva 2", "text": ["Viñeta 1", "Viñeta 2", "Viñeta 3"]}

        ...]"""

    if input_type == "Text":
        prompt = PromptTemplate.from_template("{topic}")
        chain = (prompt | model | StrOutputParser())
        result = chain.invoke({"topic": input_prompt})
    else:
        prompt = ChatPromptTemplate.from_template(template)
        chain = ({"context": input_prompt, "question": RunnablePassthrough()}
                        | prompt 
                        | model
                        | StrOutputParser())

        result = chain.invoke({"query": question})
    json_object = json.loads(result.encode('utf8'))
    if input_theme == 'default':
        deck = SlideGen()
    else:
        deck = SlideGen(ppt_file)
    title_slide_data = json_object[0]
    slides_data = json_object[1:]
    slides_data = images2slides(slides_data, num_images)
    return deck.create_presentation(title_slide_data, slides_data)

def create_ppt(choice, thema_select, input_text, pdf_file,num_images, input_url, ppt_file):
    if choice == "Text":
        input_ = input_text
    elif choice == "PDF":
        input_ = pdf_file
    elif choice == "url":
        input_ = input_url
    input_ = generate_text2ppt_input_prompt(choice, input_)
    prs = text2ppt(input_, choice, thema_select, num_images, ppt_file)
    return prs

def interface():
    
    with gr.Tab("Text2PPT"):
        with gr.Row("Template selection"):
            template = gr.Dropdown(
                label="Please select the template you want.",
                choices=['default', 'custom'],
                value='default'
            )
            
            ppt_file = gr.File(type="filepath", label="Upload PDF", visible=False)
        with gr.Row("Page cuantity selection"):
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
    confirm.click(fn=create_ppt,inputs=[radio_from,template,input_text, pdf_file, images, input_url, ppt_file],outputs=output_ppt)



with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    interface()
    gr.Markdown(LICENSE)

if __name__ == "__main__":
    # env_set()
    demo.queue(max_size=20).launch()