# PPT CREATOR

This is a ppt creator chat. It will open a web application for ppt creation.
PPT Creator Ollama_→ Phi3 or llama3

## PREPARATION

First of all, clone the repo by running this command.

```shell
git clone https://github.com/mikelalda/Ollama-ppt
cd Ollama-ppt
```

The download pandoc executable to this folder. You can download from [this](https://github.com/jgm/pandoc/releases/tag/3.1.13) page or links ([linux-amd](https://github.com/jgm/pandoc/releases/download/3.1.13/pandoc-3.1.13-linux-amd64.tar.gz), [linux-arm](https://github.com/jgm/pandoc/releases/download/3.1.13/pandoc-3.1.13-linux-arm64.tar.gz), [windows](https://github.com/jgm/pandoc/releases/download/3.1.13/pandoc-3.1.13-windows-x86_64.zip), [mac](https://github.com/jgm/pandoc/releases/download/3.1.13/pandoc-3.1.13-x86_64-macOS.zip)), download the zip file of your computer operating system and extract the pandoc.exe file to the folder created.

### Create a virtual environment

Create a virtual env by running this command

```shell
python -m venv .venv\ppt
.venv\ppt\Scripts\activate.bat
pip install -r requirements.txt
```

### Install ollama and run model

[ollama](https://ollama.com/https:/) instalation from here.

Ollama supports a list of models available on [ollama.com/library](https://ollama.com/library "ollama model library")

Then pull any mnodel:


| Model              | Parameters | Size  | Download                        |
| -------------------- | ------------ | ------- | --------------------------------- |
| Llama 3            | 8B         | 4.7GB | `ollama pull llama3`            |
| Llama 3            | 70B        | 40GB  | `ollama pull llama3:70b`        |
| Phi-3              | 3,8B       | 2.3GB | `ollama pull phi3`              |
| Mistral            | 7B         | 4.1GB | `ollama pull mistral`           |
| Neural Chat        | 7B         | 4.1GB | `ollama pull neural-chat`       |
| Starling           | 7B         | 4.1GB | `ollama pull starling-lm`       |
| Code Llama         | 7B         | 3.8GB | `ollama pull codellama`         |
| Llama 2 Uncensored | 7B         | 3.8GB | `ollama pull llama2-uncensored` |
| LLaVA              | 7B         | 4.5GB | `ollama pull llava`             |
| Gemma              | 2B         | 1.4GB | `ollama pull gemma:2b`          |
| Gemma              | 7B         | 4.8GB | `ollama pull gemma:7b`          |
| Solar              | 10.7B      | 6.1GB | `ollama pull solar`             |

Then customize de model

```shell
ollama create ppt -f ./Modelfile
```

### Running the app

Finally execute de app by running this script.

```shell
python app.py
```

## TODO

* [ ] Busqueda imagen → Web + ImageBird
* [ ] Creacion de ppt →
  * [X] Crear en base al 1
  * [X] Crear con RAG
  * [ ] Crear con metodo STORM
* [ ] Corregir cambios entre RAG y tema
