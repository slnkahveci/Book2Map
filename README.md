# Book2Map

Project submission for GSERM 2025: Deep Learning for Generative AI 

Book2Map is a tool that extracts and visualizes locations mentioned in books or stories. By uploading a text or PDF file, the app identifies place names and references, geocodes them, and displays them on an interactive map. This helps users explore the geographical journey of a narrative.

## User Story
As a reader, I want to visualize the locations mentioned in a book or story on a map, so that I can better understand the geographical context and journey of the characters.

## Features
- Extracts explicit and inferred locations from text using LLM (Gemini API)
- Geocodes locations with Google Maps
- Visualizes locations in narrative order on an interactive map
- Supports PDF and TXT file uploads

## Usage
1. Paste the text or upload a PDF or TXT file.
2. Select the chapters or chunks of text to process.
3. Click on the "Process" button.
4. Click on map markers to see details about each location.

## Requirements
- Python 3.10+
- [Poetry](https://python-poetry.org/docs/) for dependency management
- See `pyproject.toml` for dependencies

## Run
After cloning the repository, install the dependencies using Poetry:
```bash
poetry install
```
Then, you can run the application with:
```bash
poetry run python3 gradio_instance.py
```

## TODOS
- [x] Interactive list of locations under the map that allows ~~removing~~ hiding locations from the map. ~~and reordering them (drag and drop)~~. Default order should be the order of mentions in the text.
- [ ] ~~Google Maps export.~~   -> There is no native support for this. I have to do KML export and then import it into Google Maps.
- [x] .txt, .pdf, .epub file upload support.
- [x] Better chunking with sliding overlapping windows.
- [ ] Hierarchical chunk selection for large texts.


### What to improve in the future:
- [ ] Text reference column to contain summary of the events in each location. 
- [ ] Visualizing the map without plotly which keeps resetting the map state.
- [ ] Better chapter splitting for different languages. (currently english regex only to reduce computational cost)

## Choosing the right model
I have not used smaller distilled model or NER because they are not as good at understanding context and relationships between entities. And part of my user story is to extract not only explicit mentions of locations, but also implicit ones, like descriptions of places without naming them. This requires a model that can understand the context and relationships between entities, which is why I chose larger model like Gemini.

