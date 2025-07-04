# Book2Map

Project submission for GSERM 2025: Deep Learning for Generative AI 

Book2Map is a tool that extracts and visualizes locations mentioned in books or stories. By uploading a text or PDF file, the app identifies place names and references, geocodes them, and displays them on an interactive map. This helps users explore the geographical journey of a narrative.

## Features
- Extracts explicit and inferred locations from text using AI (Gemini API)
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
- See `pyproject.toml` for dependencies

## Run
You can launch the Gradio interface with:
```bash
python gradio_instance.py
```

## TODOS
- [x] Interactive list of locations under the map that allows ~~removing~~ hiding locations from the map. ~~and reordering them (drag and drop)~~. Default order should be the order of mentions in the text.
- [ ] ~~Google Maps export.~~   -> There is no native support for this. I have to do KML export and then import it into Google Maps.
- [ ] .txt, .pdf, .epub file upload support.
- [ ] Better chunking with sliding overlapping windows.
- [ ] MVPv2: Summary of events in each location.