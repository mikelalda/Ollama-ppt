# PPT CREATOR

This is a ppt creator chat. It will open a web application for ppt creation.

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

### Running the app

Finally execute de app by running this script.

```shell
python app.py
```
