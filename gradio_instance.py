# --- Enhanced Location Extraction: Two-Tab UI with Auto-Collapse ---
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


def map_and_table_from_geocoded_locations(geocoded_locations, selected_index=None):
    """Create map and table from geocoded locations - simple version"""
    if not geocoded_locations:
        return go.Figure(), []

    lats = [loc["lat"] for loc in geocoded_locations]
    lngs = [loc["lng"] for loc in geocoded_locations]
    names = [loc["name"] for loc in geocoded_locations]
    text_refs = [loc["text_reference"] for loc in geocoded_locations]
    confidences = [loc["confidence"] for loc in geocoded_locations]
    scales = [loc["scale"] for loc in geocoded_locations]
    customdata = list(zip(names, text_refs, confidences, scales))

    # Set marker colors: highlight selected
    colors = [
        "red" if i == selected_index else "blue" for i in range(len(geocoded_locations))
    ]
    sizes = [15 if i == selected_index else 10 for i in range(len(geocoded_locations))]

    fig = go.Figure(
        go.Scattermap(
            customdata=customdata,
            lat=lats,
            lon=lngs,
            mode="markers",
            marker=go.scattermap.Marker(size=sizes, color=colors),
            name="",
            hoverinfo="skip",
            hovertemplate="<b>Name</b>: %{customdata[0]}<br><b>Confidence</b>: %{customdata[2]}<br><b>Scale</b>: %{customdata[3]}",
        )
    )

    # Simple center calculation
    map_center_lat = sum(lats) / len(lats)
    map_center_lon = sum(lngs) / len(lngs)

    fig.update_layout(
        mapbox_style="open-street-map",
        hovermode="closest",
        mapbox=dict(
            bearing=0,
            center=dict(lat=map_center_lat, lon=map_center_lon),
            pitch=0,
            zoom=2,
        ),
        height=500,
    )

    # Prepare locations list for display
    locations_list = [
        [loc["name"], loc["text_reference"], loc["confidence"], loc["scale"]]
        for loc in geocoded_locations
    ]
    return fig, locations_list


# Gradio UI: Two-tab interface
def chapter_scale_ui():
    with gr.Blocks() as demo3:
        gr.Markdown("# 📍 Location Extraction Tool")

        with gr.Tabs() as tabs:
            # Tab 1: Setup and Analysis
            with gr.TabItem("📝 Step 1: Text Analysis", id="setup_tab") as tab1:
                with gr.Accordion(
                    "Text Input & Chapter Analysis", open=True
                ) as setup_accordion:
                    gr.Markdown("### Paste your text and analyze its structure")
                    text_input = gr.Textbox(
                        label="Book/Story Text",
                        lines=10,
                        placeholder="Paste your text here...",
                    )

                    with gr.Row():
                        analyze_btn = gr.Button(
                            "🔍 Analyze Chapters", variant="primary"
                        )

                    with gr.Row():
                        num_chapters = gr.Number(
                            label="Number of Chapters", interactive=False
                        )
                        num_chunks = gr.Number(
                            label="Number of Chunks", interactive=False
                        )

                    chapter_select = gr.CheckboxGroup(
                        label="Select Chapters/Chunks to Process",
                        choices=[],
                        interactive=True,
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

                    extract_btn = gr.Button(
                        "🗺️ Extract & Map Locations", variant="primary", size="lg"
                    )

            # Tab 2: Results - Map and Table
            with gr.TabItem("🗺️ Step 2: Location Mapping", id="results_tab") as tab2:
                gr.Markdown("### Interactive Location Map & Data")

                with gr.Row():
                    with gr.Column(scale=2):
                        map_plot = gr.Plot(label="Location Map")

                with gr.Row():
                    locations_table = gr.Dataframe(
                        headers=["Name", "Text Reference", "Confidence", "Scale"],
                        datatype=["str", "str", "number", "str"],
                        label="📋 Locations (click a row to highlight on map)",
                        interactive=False,
                        visible=True,
                        wrap=True,
                    )

                # Instructions
                gr.Markdown(
                    """
                **Instructions:**
                - Click on any row in the table to highlight the corresponding location on the map
                - Red markers indicate selected locations, blue markers are unselected
                """
                )

        # Simple state management
        geocoded_locations_state = gr.State([])
        analysis_info_state = gr.State({})

        # Helper functions
        def get_chapter_labels(info):
            labels = []
            for i, (title, chunk) in enumerate(
                zip(info["chapter_titles"], info["chunks"])
            ):
                preview = chunk.strip().replace("\n", " ")[:100]
                labels.append(f"{title} — {preview}…")
            return labels

        def analyze_callback(text):
            if not text.strip():
                return (
                    0,
                    0,
                    gr.update(choices=[], value=[]),
                    {},
                    gr.update(),
                )

            info = analyze_chapters(text)
            labels = get_chapter_labels(info)

            return (
                info["num_chapters"],
                info["num_chunks"],
                gr.update(choices=labels, value=labels),
                info,
                gr.update(),
            )

        def extract_callback(text, selected, scales, analysis_info):
            if not selected or not text.strip():
                return (
                    go.Figure(),
                    [],
                    [],
                    gr.update(selected="results_tab"),
                    gr.update(open=False),
                )

            # Use stored analysis info if available, otherwise re-analyze
            if analysis_info:
                info = analysis_info
            else:
                info = analyze_chapters(text)

            labels = get_chapter_labels(info)
            indices = [labels.index(s) for s in selected if s in labels]
            selected_chunks = [info["chunks"][i] for i in indices]

            geocoded_locations = extract_and_geocode_locations(selected_chunks, scales)
            fig, table_data = map_and_table_from_geocoded_locations(geocoded_locations)

            return (
                fig,
                table_data,
                geocoded_locations,
                gr.update(selected="results_tab"),
                gr.update(open=False),
            )

        def highlight_location(evt: gr.SelectData, geocoded_locations):
            """Simple highlight function - just change marker colors"""
            if evt is None or not geocoded_locations:
                return gr.update()

            selected_index = (
                evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            )
            print(f"Selected row index: {selected_index}")

            # Create updated map with highlighted marker
            fig, _ = map_and_table_from_geocoded_locations(
                geocoded_locations, selected_index=selected_index
            )

            return fig

        # Event handlers
        analyze_btn.click(
            analyze_callback,
            inputs=[text_input],
            outputs=[
                num_chapters,
                num_chunks,
                chapter_select,
                analysis_info_state,
                tabs,
            ],
        )

        extract_btn.click(
            extract_callback,
            inputs=[text_input, chapter_select, scale_select, analysis_info_state],
            outputs=[
                map_plot,
                locations_table,
                geocoded_locations_state,
                tabs,
                setup_accordion,
            ],
        )

        # Simple row selection handler
        locations_table.select(
            highlight_location,
            inputs=[geocoded_locations_state],
            outputs=[map_plot],
            show_progress=False,
            queue=False,
        )

        # Custom CSS for better UX
        demo3.css = """
        /* Smooth scrolling and transitions */
        .gradio-container {
            scroll-behavior: smooth;
        }
        
        /* Make checkbox group scrollable */
        .gr-checkbox-group {
            max-height: 250px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
        }
        
        /* Improve plot container */
        .js-plotly-plot {
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        /* Better dataframe styling */
        .gr-dataframe {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        /* Tab styling */
        .gradio-tabs {
            border-radius: 8px;
        }
        
        /* Accordion styling */
        .gr-accordion {
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
        
        /* Button improvements */
        .gr-button-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }
        
        .gr-button-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        """

    return demo3


chapter_scale_ui().launch()
