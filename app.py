import gradio as gr
from pptx import Presentation
import subprocess
import io
from io import BytesIO
import PyPDF2
import os
import subprocess
import torch
from langchain.llms import Ollama
import requests
from bs4 import BeautifulSoup
import os


MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = 4096
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache() 

sampling_params = dict(do_sample=True, temperature=0.3, top_k=50, top_p=0.9)


DESCRIPTION = """\
# Ollama ppt creator
This Space demonstrates models integrated in [Ollama](https://ollama.com) with the model [llama2](https://huggingface.co/meta-llama/Llama-2-7b) by Meta, a Llama 2 model with 7B parameters fine-tuned for chat instructions. Feel free to play with it, or duplicate to run generations without a queue! If you want to run your own service, you can also [deploy the model on Inference Endpoints](https://huggingface.co/inference-endpoints).
"""

LICENSE = """
---
As a derivate work of [Ollama](https://ollama.com) and [llama2](https://huggingface.co/meta-llama/Llama-2-7b),
this demo is governed by the original [license](https://github.com/ollama/ollama?tab=MIT-1-ov-file) and [license](https://huggingface.co/meta-llama/Llama-2-7b/blob/main/LICENSE.txt).
"""

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"


pipe = Ollama(model="ppt")

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
    if choice == "PDF":
        input_text = gr.TextArea(lines=5, placeholder="Enter TEXT", visible=False)
        pdf_file = gr.File(type="filepath", label="Upload PDF", visible=True)

    return input_text, pdf_file

def generate_text2ppt_input_prompt(input_type, input_value, input_pages):
    header = """
    Write a PPT of %s pages about:
    +++
    """ % input_pages

    summary_value = ""

    if input_type == "Text":
        header = """
        Write a PPT of %s pages about:
        +++
        """ % input_pages
        summary_value += input_value
        summary_value += "\n"
    elif input_type == "PDF":
        header = """
        Write a PPT of %s pages about this text:
        +++
        """ % input_pages
        with open(input_value, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)

        # Convert the content of each page to a string.
        text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        summary_value += text
        summary_value += "\n"
    else:
        print("ERROR: Invalid input")


    return header + summary_value
def find_images(search_query,num_results):
    # Construct the Google Images search URL
    search_url = f'https://www.google.com/search?q={search_query}&source=lnms&tbm=isch'
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all image links in the search results
    image_links = [link.get('src') for link in soup.find_all('img')]
    return image_links[:num_results]


def images2slides(text, num_images):
    slides = text.split("\n\nSlide ")
    output = ""

    for i, slide in enumerate(slides):
        if i == 0:
            output += f"Slide {i}: {slide.split(':')[0]}\n\n**\\================**\n\n"
        else:
            output += f"\nSlide {i}: {slide.split(':')[0]}\n\n**\\=====================**\n\n"

        slide_parts = slide.split("\n\n")
        for part in slide_parts[1:]:
            output += part.strip() + "\n\n"

        if i != len(slides) - 1:
            output += "!\\[Image\\](https://via.placeholder.com/350x200)\n\n**\\------------------**\n"
    return output

# Function to execute text2ppt
def text2ppt(input_prompt, input_theme, num_images):
    # llamada al modelo local de llama-2
    outputs = pipe.invoke(input_prompt)
    result = outputs.split('+++')[-1]
    md_text = result[4:] if result[:3] == "---" else result
    # md_text =images2slides(md_text, num_images)
    md_text_list = md_text.split('\n')

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


def create_ppt(choice, page_choice, thema_select, input_text, pdf_file,num_images):
    if choice == "Text":
        input_ = input_text
    elif choice == "PDF":
        input_ = pdf_file
    input_ = generate_text2ppt_input_prompt(choice, input_, page_choice)
    text2ppt(input_, thema_select, num_images)
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
            pages = gr.Slider(minimum=2, maximum=12, step=1, value=5, label="Number of PPT pages")
            images = gr.Slider(minimum=1, maximum=12, step=1, value=1, label="Number of images in each PPT page")
        with gr.Row("Input selection"):
            my_order = ['Text', 'PDF']
            radio_from = gr.Radio(my_order, label="Please select the file type and enter the content!", value='Text')
            input_text = ''
            input_text = gr.TextArea(lines=5, placeholder="Enter TEXT", visible=True)
            pdf_file = gr.File(type="filepath", label="Upload PDF", visible=False)   
        with gr.Row("Output ppt"):
            output_ppt = gr.File(
                label='Output PPT'
            )  
                

        confirm = gr.Button(value="Confirm",)
        
    template.change(fn=filter_custom,inputs=[template], outputs=[ppt_file])
    radio_from.change(fn=filter_input,inputs=[radio_from], outputs=[input_text, pdf_file])
    confirm.click(fn=create_ppt,inputs=[radio_from,pages,template,input_text, pdf_file, images],outputs=output_ppt)



with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    interface()
    gr.Markdown(LICENSE)

if __name__ == "__main__":
    # env_set()
    demo.queue(max_size=20).launch()