# --- Enhanced Location Extraction: Chapter & Scale Selection UI ---
import asyncio
import plotly.graph_objects as go
from gemini_extractor import (
    TextPreprocessor,
    GeminiExtractor,
    GoogleMapsExtractor,
    GEMINI_API_KEY,
    GOOGLE_MAPS_KEY,
    LocationMention,
    extract_and_geocode_locations,
)
from typing import List
import nest_asyncio

nest_asyncio.apply()
import gradio as gr

# Step 1: Analyze chapters/chunks
def analyze_chapters(text):
    preprocessor = TextPreprocessor()
    chapters = preprocessor.split_by_chapter(text)
    if len(chapters) > 1:
        chapter_titles = []
        for i, ch in enumerate(chapters):
            # Try to extract a title (first line or up to 60 chars)
            first_line = ch.strip().split("\n")[0]
            title = first_line[:60] if first_line else f"Chapter {i+1}"
            chapter_titles.append(f"{i+1}: {title}")
        return {
            "num_chapters": len(chapters),
            "num_chunks": len(chapters),
            "chapter_titles": chapter_titles,
            "chunks": chapters,
        }
    else:
        # fallback: split by tokens
        chunks = preprocessor.split_by_tokens(text)
        return {
            "num_chapters": 1,
            "num_chunks": len(chunks),
            "chapter_titles": [f"Chunk {i+1}" for i in range(len(chunks))],
            "chunks": chunks,
        }

def map_and_table_from_geocoded_locations(geocoded_locations):
    if not geocoded_locations:
        return go.Figure(), []
    lats = [loc["lat"] for loc in geocoded_locations]
    lngs = [loc["lng"] for loc in geocoded_locations]
    names = [loc["name"] for loc in geocoded_locations]
    text_refs = [loc["text_reference"] for loc in geocoded_locations]
    confidences = [loc["confidence"] for loc in geocoded_locations]
    scales = [loc["scale"] for loc in geocoded_locations]
    customdata = list(zip(names, text_refs, confidences, scales))
    fig = go.Figure(
        go.Scattermapbox(
            customdata=customdata,
            lat=lats,
            lon=lngs,
            mode="markers",
            marker=go.scattermapbox.Marker(size=10),
            name="",
            hoverinfo="skip",
            hovertemplate="<b>Name</b>: %{customdata[0]}<br><b>Confidence</b>: %{customdata[2]}<br><b>Scale</b>: %{customdata[3]}",
        )
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        hovermode="closest",
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=sum(lats) / len(lats), lon=sum(lngs) / len(lngs)
            ),
            pitch=0,
            zoom=2,
        ),
    )
    # Prepare locations list for display (ordered by mention)
    locations_list = [
        [loc["name"], loc["text_reference"], loc["confidence"], loc["scale"]]
        for loc in geocoded_locations
    ]
    return fig, locations_list

# Gradio UI: Two-step process
def chapter_scale_ui():
    with gr.Blocks() as demo3:
        gr.Markdown("""# Step 1: Paste Text and Analyze Chapters""")
        text_input = gr.Textbox(label="Book/Story Text", lines=10)
        analyze_btn = gr.Button("Analyze Chapters")
        num_chapters = gr.Number(label="Number of Chapters", interactive=False)
        num_chunks = gr.Number(label="Number of Chunks", interactive=False)
        chapter_select = gr.CheckboxGroup(
            label="Select Chapters/Chunks to Process", choices=[], interactive=True
        )
        scale_select = gr.CheckboxGroup(
            label="Select Scales to Include",
            choices=[
                "country",
                "state",
                "city",
                "neighborhood",
                "landmark",
                "building",
                "other",
            ],
            value=["neighborhood", "landmark", "building", "other"],
            interactive=True,
        )
        gr.Markdown("""# Step 2: Extract & Map Locations for Selected Chapters and Scales""")
        map_plot = gr.Plot()
        locations_table = gr.Dataframe(
            headers=["Name", "Text Reference", "Confidence", "Scale"],
            datatype=["str", "str", "number", "str"],
            label="Locations (in order of mention)",
            interactive=False,
            visible=True,
            wrap=True
        )
        extract_btn = gr.Button("Extract & Map Locations")

        # Prepare chapter labels with preview
        # Chapters are not preselected so the user is encouraged to select the chapters they are actually interested in
        def get_chapter_labels(info):
            labels = []
            for i, (title, chunk) in enumerate(zip(info["chapter_titles"], info["chunks"])):
                preview = chunk.strip().replace("\n", " ")[:100]
                labels.append(f"{title} — {preview}…")
            return labels

        def analyze_callback(text):
            info = analyze_chapters(text)
            labels = get_chapter_labels(info)
            return (
                info["num_chapters"],
                info["num_chunks"],
                gr.update(choices=labels, value=[]),
            )

        analyze_btn.click(
            analyze_callback,
            inputs=text_input,
            outputs=[num_chapters, num_chunks, chapter_select],
        )

        def extract_callback(text, selected, scales):
            info = analyze_chapters(text)
            indices = [get_chapter_labels(info).index(s) for s in selected]
            selected_chunks = [info["chunks"][i] for i in indices]
            geocoded_locations = extract_and_geocode_locations(selected_chunks, scales)
            fig, table_data = map_and_table_from_geocoded_locations(geocoded_locations)
            return fig, table_data

        extract_btn.click(
            extract_callback,
            inputs=[text_input, chapter_select, scale_select],
            outputs=[map_plot, locations_table],
        )

        # Make chapter_select scrollable
        demo3.css = """
        .svelte-1ipelgc { /* Gradio's default class for CheckboxGroup container */
            max-height: 300px;
            overflow-y: auto;
        }
        """
    return demo3

chapter_scale_ui().launch()
