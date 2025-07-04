
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
    MASTER_PROMPT,
)
from typing import List
import nest_asyncio

nest_asyncio.apply()
import gradio as gr
import io
import os
from pathlib import Path

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Try to import better PDF libraries
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
except ImportError:
    ebooklib = None
    BeautifulSoup = None


# FIXED: Convert function that handles Gradio file objects properly
def convert_to_text(doc_input):
    """Convert uploaded files to text - handles both file paths and file objects"""
    if doc_input is None:
        return ""

    try:
        # Handle different Gradio file input types
        if isinstance(doc_input, str):
            # It's a file path string
            file_path = doc_input
        elif hasattr(doc_input, "name"):
            # It's a file-like object or NamedString
            file_path = doc_input.name
        else:
            # Try to convert to string (fallback)
            file_path = str(doc_input)

        # Verify file exists
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"

        file_extension = Path(file_path).suffix.lower().lstrip(".")

        if file_extension == "txt":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()

        elif file_extension == "pdf":
            text_parts = []

            # Try pdfplumber first
            if pdfplumber is not None:
                try:
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            text = page.extract_text()
                            if text and text.strip():
                                text_parts.append(text.strip())
                except:
                    pass

            # Try PyMuPDF if pdfplumber didn't work
            if not text_parts and fitz is not None:
                try:
                    doc = fitz.open(file_path)
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        text = page.get_text("text")
                        if text and text.strip():
                            text_parts.append(text.strip())
                    doc.close()
                except:
                    pass

            # Fallback to PyPDF2
            if not text_parts and PyPDF2 is not None:
                try:
                    with open(file_path, "rb") as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text and text.strip():
                                text_parts.append(text.strip())
                except:
                    pass

            if text_parts:
                return "\n\n".join(text_parts)
            else:
                if not any([pdfplumber, fitz, PyPDF2]):
                    return (
                        "Error: No PDF libraries installed. Run: pip install pdfplumber"
                    )
                return "No text found in PDF. May be scanned/image-based."

        elif file_extension == "epub":
            if ebooklib is None or BeautifulSoup is None:
                return "Error: Required libraries missing. Run: pip install ebooklib beautifulsoup4"

            book = epub.read_epub(file_path)
            text_parts = []
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), "html.parser")
                    text = soup.get_text().strip()
                    if text:
                        text_parts.append(text)
            return "\n\n".join(text_parts) if text_parts else "No text found in EPUB"

        else:
            return f"Error: Unsupported file extension: {file_extension}"

    except Exception as e:
        return f"Error processing file: {str(e)}"


# Step 1: Analyze chapters/chunks
def analyze_chapters(text):
    try:
        preprocessor = TextPreprocessor()
        chapters = preprocessor.process(text)
        return chapters
    except Exception as e:
        print(f"Error in analyze_chapters: {e}")
        return {"num_chapters": 0, "num_chunks": 0, "chunks": []}


def map_and_table_from_geocoded_locations(
    geocoded_locations, visible_indices=None, selected_index=None
):
    """Create map and table from geocoded locations with visibility control"""
    if not geocoded_locations:
        fig = go.Figure()
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(bearing=0, center=dict(lat=0, lon=0), pitch=0, zoom=1),
            height=400,
        )
        return fig, []

    # If no visible indices specified, show all
    if visible_indices is None:
        visible_indices = list(range(len(geocoded_locations)))

    # Filter locations based on visibility
    visible_locations = [
        geocoded_locations[i] for i in visible_indices if i < len(geocoded_locations)
    ]

    if not visible_locations:
        fig = go.Figure()
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(bearing=0, center=dict(lat=0, lon=0), pitch=0, zoom=1),
            height=400,
        )
        return fig, []

    # FIXED: Add error checking for location data
    lats = []
    lngs = []
    names = []
    text_refs = []
    confidences = []
    scales = []

    for loc in visible_locations:
        if all(
            key in loc
            for key in ["lat", "lng", "name", "text_reference", "confidence", "scale", "first_mention_order"]
        ):
            lats.append(loc["lat"])
            lngs.append(loc["lng"])
            names.append(loc["name"])
            text_refs.append(loc["text_reference"])
            confidences.append(loc["confidence"])
            scales.append(loc["scale"])

    if not lats:  # No valid locations
        fig = go.Figure()
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(bearing=0, center=dict(lat=0, lon=0), pitch=0, zoom=1),
            height=400,
        )
        return fig, []

    customdata = list(zip(names, text_refs, confidences, scales))

    # Set marker colors: highlight selected
    colors = []
    sizes = []
    for i, visible_idx in enumerate(visible_indices):
        if visible_idx == selected_index:
            colors.append("red")
            sizes.append(15)
        else:
            colors.append("blue")
            sizes.append(10)

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

    # Prepare locations list for display (all locations, but mark visible ones)
    locations_list = []
    for i, loc in enumerate(geocoded_locations):
        if all(key in loc for key in ["name", "text_reference", "confidence", "scale", "first_mention_order"]):
            visible_status = "✓" if i in visible_indices else "✗"
            locations_list.append(
                [
                    visible_status,
                    loc["name"],
                    loc["text_reference"],
                    loc["confidence"],
                    loc["scale"],
                    loc["first_mention_order"] + 1,  # Add 1 to make it 1-based for display
                ]
            )

    return fig, locations_list


# Gradio UI: Two-tab interface
def chapter_scale_ui():
    with gr.Blocks(
        css = """
        /* Dynamic height adjustment */
        .js-plotly-plot {
            height: calc(100vh - 400px) !important;
            min-height: 300px;
        }
        
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
            border-radius: 3px;
            box-shadow: 0 2px 3px rgba(0,0,0,0.1);
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
        
        /* Control panel styling */
        .gr-column:has(.gr-checkbox-group) {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-right: 10px;
        }
        
        /* Small button styling */
        .gr-button[data-size="sm"] {
            padding: 5px 12px;
            font-size: 0.875rem;
        }
        
        /* Prompt editor styling */
        .gr-textbox[data-testid="textbox"] {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        
        /* Status indicator styling */
        .gr-markdown:has(strong:contains("Prompt:")) {
            background: #f8f9fa;
            padding: 8px 12px;
            border-radius: 6px;
            border-left: 4px solid #007bff;
            margin: 10px 0;
        }
        
        /* Test result styling */
        .gr-json {
            background: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 10px;
        }
        
        /* Model selection styling */
        .gr-dropdown {
            border-radius: 6px;
        }
        
        /* Model status indicator styling */
        .gr-markdown:has(strong:contains("Model:")) {
            background: #f8f9fa;
            padding: 8px 12px;
            border-radius: 6px;
            border-left: 4px solid #28a745;
            margin: 10px 0;
        }
        """
    ) as demo3:

        gr.Markdown("# 📍 Location Extraction Tool")

        with gr.Tabs() as tabs:
            # Tab 1: Setup and Analysis
            with gr.TabItem("📝 Step 1: Text Analysis", id="setup_tab") as tab1:
                with gr.Accordion(
                    "Text Input & Chapter Analysis", open=True
                ) as setup_accordion:
                    gr.Markdown("### Paste your text and analyze its structure")
                    with gr.Row():
                        with gr.Column():
                            text_input = gr.Textbox(
                                label="Book/Story Text",
                                lines=10,
                                placeholder="Paste your text here...",
                            )

                        with gr.Column():
                            with gr.Row():
                                doc_input = gr.File(
                                    label="Upload a TXT, PDF or EPUB file",
                                    file_types=[".txt", ".pdf", ".epub"],
                                    interactive=True,
                                )
                            with gr.Row():
                                convert_btn = gr.Button("🔄 Convert to Text", variant="primary")

                    analyze_btn = gr.Button("🔍 Analyze Chapters", variant="primary")

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
                        value=[],
                        interactive=True,
                    )

                    with gr.Row():
                        select_all_chapters_btn = gr.Button("Select All Chapters", size="sm")
                        deselect_all_chapters_btn = gr.Button("Deselect All Chapters", size="sm")

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
                    
                    prompt_status_indicator = gr.Markdown("**Prompt:** Using default prompt")
                    model_status_indicator = gr.Markdown("**Model:** Using gemini-2.5-flash")

            # Tab 2: Results - Map and Table
            with gr.TabItem("🗺️ Step 2: Location Mapping", id="results_tab") as tab2:
                gr.Markdown("### Interactive Location Map & Data")

                with gr.Row():
                    with gr.Column(scale=2):
                        map_plot = gr.Plot(label="Location Map")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🎛️ Visibility Controls")
                        locations_visibility = gr.CheckboxGroup(
                            label="Select locations to show on map",
                            choices=[],
                            value=[],
                            interactive=True,
                        )

                        with gr.Row():
                            select_all_btn = gr.Button("Select All", size="sm")
                            deselect_all_btn = gr.Button("Deselect All", size="sm")

                    with gr.Column(scale=2):
                        locations_table = gr.Dataframe(
                            headers=[
                                "Visible",
                                "Name",
                                "Text Reference",
                                "Confidence",
                                "Scale",
                                "Order of Mention",
                            ],
                            datatype=["str", "str", "str", "number", "str", "number"],
                            label="📋 Locations (click a row to highlight on map)",
                            interactive=False,
                            visible=True,
                            wrap=True,
                        )

                gr.Markdown(
                    """
                **Instructions:**
                - **Show/Hide**: Use the checkboxes on the left to control which locations appear on the map
                - **Highlight**: Click on any row in the table to highlight the corresponding location on the map
                - **Quick Select**: Use "Select All" or "Deselect All" buttons for convenience
                - Red markers indicate selected locations, blue markers are unselected
                """
                )

            # Tab 3: Prompt Configuration
            with gr.TabItem("⚙️ Optional: Prompt Configuration", id="prompt_tab") as tab3:
                gr.Markdown("### Customize the AI Model and Prompt for Location Extraction")
                
                with gr.Accordion("🤖 Model Selection", open=True) as model_accordion:
                    gr.Markdown("""
                    **Choose the AI Model:**
                    Select which Gemini model to use for location extraction. Different models may have varying performance and capabilities.
                    
                    **Available Models:**
                    - **gemini-2.5-flash**: Fast and efficient (default)
                    - **gemini-2.0-flash**: Alternative option
                    - **gemini-1.5-flash**: Legacy model
                    """)
                    
                    model_select = gr.Dropdown(
                        label="Select Model",
                        choices=[
                            "gemini-2.5-flash",
                            "gemini-2.0-flash", 
                            "gemini-1.5-flash"
                        ],
                        value="gemini-2.5-flash",
                        interactive=True,
                    )
                    
                    model_status = gr.Markdown("**Model Status:** Using gemini-2.5-flash")
                
                with gr.Accordion("📝 Master Prompt Editor", open=True) as prompt_accordion:
                    gr.Markdown("""
                    **About the Master Prompt:**
                    This prompt instructs the AI model on how to extract locations from your text. 
                    You can customize it to focus on specific types of locations or adjust the extraction criteria.
                    
                    **Key Sections:**
                    - **EXTRACTION GUIDELINES**: What to extract and what to ignore
                    - **CONFIDENCE SCORING**: How to rate location confidence (0.0-1.0)
                    - **OUTPUT REQUIREMENTS**: JSON format specifications
                    - **EXAMPLES**: Sample inputs and expected outputs
                    """)
                    
                    prompt_editor = gr.Textbox(
                        label="Master Prompt",
                        value=MASTER_PROMPT,
                        lines=25,
                        placeholder="Enter your custom prompt here...",
                        interactive=True,
                    )
                    
                    with gr.Row():
                        save_prompt_btn = gr.Button("💾 Save Prompt", variant="primary")
                        reset_prompt_btn = gr.Button("🔄 Reset to Default", variant="secondary")
                        test_prompt_btn = gr.Button("🧪 Test Prompt", variant="secondary")
                    
                    prompt_status = gr.Markdown("**Status:** Ready to use default prompt")
                    
                    with gr.Accordion("📋 Prompt Testing", open=False) as test_accordion:
                        gr.Markdown("Test your custom prompt with a sample text:")
                        
                        test_text = gr.Textbox(
                            label="Test Text",
                            lines=5,
                            placeholder="Enter a sample text to test your prompt...",
                            value="I am in Berlin. There's a huge gate in front of me. And there are a bunch of embassies.",
                        )
                        
                        test_result = gr.JSON(
                            label="Test Result",
                        )

        # State management
        geocoded_locations_state = gr.State([])
        analysis_info_state = gr.State({})
        selected_location_index = gr.State(None)
        custom_prompt_state = gr.State(MASTER_PROMPT)
        selected_model_state = gr.State("gemini-2.5-flash")

        def get_chapter_labels(info):
            """Extract chapter labels safely"""
            labels = []
            try:
                for chunk in info.get("chunks", []):
                    title = chunk.get("parent_label", "Unknown Chapter")
                    preview = chunk.get("preview", "").replace("\n", " ")[:100]
                    labels.append(f"{title} — {preview}…")
            except Exception as e:
                print(f"Error getting chapter labels: {e}")
            return labels

        def create_location_choices(geocoded_locations):
            """Create choices for the visibility checkbox group"""
            choices = []
            try:
                for i, loc in enumerate(geocoded_locations):
                    name = loc.get("name", "Unknown")
                    scale = loc.get("scale", "unknown")
                    order = loc.get("first_mention_order", 0) + 1  # 1-based for display
                    text_ref = loc.get("text_reference", "")[:50]
                    choice_text = f"#{order} {name} ({scale}) - {text_ref}..."
                    choices.append(choice_text)
            except Exception as e:
                print(f"Error creating location choices: {e}")
            return choices

        def analyze_callback(text):
            """Analyze text and return chapter info"""
            try:
                if not text.strip():
                    return 0, 0, gr.update(choices=[], value=[]), {}, gr.update()

                info = analyze_chapters(text)
                labels = get_chapter_labels(info)

                return (
                    info.get("num_chapters", 0),
                    info.get("num_chunks", 0),
                    gr.update(choices=labels, value=[]),
                    info,
                    gr.update(),
                )
            except Exception as e:
                print(f"Error in analyze_callback: {e}")
                return 0, 0, gr.update(choices=[], value=[]), {}, gr.update()

        def extract_callback(text, selected, scales, analysis_info, custom_prompt, selected_model):
            """Extract locations and create map"""
            try:
                if not selected or not text.strip():
                    return (
                        go.Figure(),
                        [],
                        [],
                        gr.update(choices=[], value=[]),
                        gr.update(selected="results_tab"),
                        gr.update(open=False),
                    )

                # Use stored analysis info if available, otherwise re-analyze
                if analysis_info and analysis_info.get("chunks"):
                    info = analysis_info
                else:
                    info = analyze_chapters(text)

                labels = get_chapter_labels(info)
                indices = [labels.index(s) for s in selected if s in labels]
                selected_chunks = [
                    info["chunks"][i] for i in indices if i < len(info["chunks"])
                ]

                try:
                    # Use custom prompt and model if provided, otherwise use defaults
                    geocoded_locations = extract_and_geocode_locations(
                        selected_chunks, scales, custom_prompt, selected_model
                    )
                except Exception as e:
                    print(f"Error extracting locations: {e}")
                    geocoded_locations = []

                # Create visibility choices and set all as visible initially
                visibility_choices = create_location_choices(geocoded_locations)
                visible_indices = list(range(len(geocoded_locations)))

                fig, table_data = map_and_table_from_geocoded_locations(
                    geocoded_locations, visible_indices=visible_indices
                )

                return (
                    fig,
                    table_data,
                    geocoded_locations,
                    gr.update(choices=visibility_choices, value=visibility_choices),
                    gr.update(selected="results_tab"),
                    gr.update(open=False),
                )
            except Exception as e:
                print(f"Error in extract_callback: {e}")
                return (
                    go.Figure(),
                    [],
                    [],
                    gr.update(choices=[], value=[]),
                    gr.update(),
                    gr.update(),
                )

        def update_map_visibility(
            selected_visibility, geocoded_locations, selected_index
        ):
            """Update map based on visibility selections"""
            try:
                if not geocoded_locations:
                    return go.Figure(), []

                # Get indices of visible locations
                visibility_choices = create_location_choices(geocoded_locations)
                visible_indices = [
                    i
                    for i, choice in enumerate(visibility_choices)
                    if choice in selected_visibility
                ]

                # Update map and table
                fig, table_data = map_and_table_from_geocoded_locations(
                    geocoded_locations,
                    visible_indices=visible_indices,
                    selected_index=selected_index,
                )

                return fig, table_data
            except Exception as e:
                print(f"Error updating map visibility: {e}")
                return go.Figure(), []

        def highlight_location(
            evt: gr.SelectData, geocoded_locations, selected_visibility
        ):
            """Highlight selected location on map"""
            try:
                if evt is None or not geocoded_locations:
                    return gr.update(), None

                selected_index = (
                    evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                )

                # Get visible indices
                visibility_choices = create_location_choices(geocoded_locations)
                visible_indices = [
                    i
                    for i, choice in enumerate(visibility_choices)
                    if choice in selected_visibility
                ]

                # Create updated map with highlighted marker
                fig, _ = map_and_table_from_geocoded_locations(
                    geocoded_locations,
                    visible_indices=visible_indices,
                    selected_index=selected_index,
                )

                return fig, selected_index
            except Exception as e:
                print(f"Error highlighting location: {e}")
                return gr.update(), None

        def select_all_locations(geocoded_locations):
            """Select all locations"""
            try:
                if not geocoded_locations:
                    return gr.update()
                visibility_choices = create_location_choices(geocoded_locations)
                return gr.update(value=visibility_choices)
            except Exception as e:
                print(f"Error selecting all locations: {e}")
                return gr.update()

        def deselect_all_locations():
            """Deselect all locations"""
            return gr.update(value=[])

        def select_all_chapters(analysis_info):
            """Select all chapters"""
            try:
                if not analysis_info or not analysis_info.get("chunks"):
                    return gr.update()
                labels = get_chapter_labels(analysis_info)
                return gr.update(value=labels)
            except Exception as e:
                print(f"Error selecting all chapters: {e}")
                return gr.update()

        def deselect_all_chapters():
            """Deselect all chapters"""
            return gr.update(value=[])

        # Model management functions
        def update_selected_model(model_name):
            """Update the selected model"""
            try:
                return gr.update(value=f"**Model Status:** Using {model_name}"), model_name, gr.update(value=f"**Model:** Using {model_name}")
            except Exception as e:
                return gr.update(value=f"**Model Status:** Error updating model: {str(e)}"), gr.update(), gr.update(value="**Model:** Using gemini-2.5-flash")

        # Prompt management functions
        def save_custom_prompt(prompt_text):
            """Save the custom prompt to state"""
            try:
                if not prompt_text.strip():
                    return gr.update(value="**Status:** ❌ Error - Prompt cannot be empty"), gr.update(), gr.update(value="**Prompt:** Using default prompt")
                
                return gr.update(value="**Status:** ✅ Custom prompt saved successfully - Will be used for next extraction"), prompt_text, gr.update(value="**Prompt:** Using custom prompt")
            except Exception as e:
                return gr.update(value=f"**Status:** ❌ Error saving prompt: {str(e)}"), gr.update(), gr.update(value="**Prompt:** Using default prompt")

        def reset_to_default_prompt():
            """Reset prompt to default"""
            try:
                return gr.update(value=MASTER_PROMPT), gr.update(value="**Status:** ✅ Reset to default prompt"), gr.update(value="**Prompt:** Using default prompt")
            except Exception as e:
                return gr.update(), gr.update(value=f"**Status:** ❌ Error resetting prompt: {str(e)}"), gr.update(value="**Prompt:** Using default prompt")

        def test_custom_prompt(prompt_text, test_text, selected_model):
            """Test the custom prompt with sample text"""
            try:
                if not prompt_text.strip() or not test_text.strip():
                    return gr.update(value="**Status:** ❌ Both prompt and test text are required")
                
                # Create a GeminiExtractor with the custom prompt and model
                from gemini_extractor import GeminiExtractor
                extractor = GeminiExtractor(GEMINI_API_KEY, custom_prompt=prompt_text, model_name=selected_model)
                
                # Test the prompt
                result = extractor.try_extract_locations_from_chunk(test_text, 0)
                
                # Convert LocationMention objects to dictionaries for JSON display
                result_dicts = []
                for loc in result:
                    result_dicts.append({
                        "name": loc.name,
                        "text_reference": loc.text_reference,
                        "confidence": loc.confidence,
                        "scale": loc.scale,
                        "model_used": loc.model_used
                    })
                
                return gr.update(value="**Status:** ✅ Test completed successfully"), result_dicts
            except Exception as e:
                return gr.update(value=f"**Status:** ❌ Test failed: {str(e)}"), gr.update()

        convert_btn.click(
            fn=convert_to_text,
            inputs=[doc_input],
            outputs=[text_input],
            show_progress="minimal",
        )

        analyze_btn.click(
            fn=analyze_callback,
            inputs=[text_input],
            outputs=[
                num_chapters,
                num_chunks,
                chapter_select,
                analysis_info_state,
                tabs,
            ],
            show_progress="full",
        )

        extract_btn.click(
            fn=extract_callback,
            inputs=[text_input, chapter_select, scale_select, analysis_info_state, custom_prompt_state, selected_model_state],
            outputs=[
                map_plot,
                locations_table,
                geocoded_locations_state,
                locations_visibility,
                tabs,
                setup_accordion,
            ],
            show_progress="full",
        )

        # Visibility control
        locations_visibility.change(
            fn=update_map_visibility,
            inputs=[
                locations_visibility,
                geocoded_locations_state,
                selected_location_index,
            ],
            outputs=[map_plot, locations_table],
            show_progress="minimal",
        )

        # Row selection for highlighting
        locations_table.select(
            fn=highlight_location,
            inputs=[geocoded_locations_state, locations_visibility],
            outputs=[map_plot, selected_location_index],
            show_progress="minimal",
        )

        # Select/Deselect all buttons
        select_all_btn.click(
            fn=select_all_locations,
            inputs=[geocoded_locations_state],
            outputs=[locations_visibility],
            show_progress="minimal",
        )

        deselect_all_btn.click(
            fn=deselect_all_locations,
            outputs=[locations_visibility],
            show_progress="minimal",
        )

        # Chapter selection buttons
        select_all_chapters_btn.click(
            fn=select_all_chapters,
            inputs=[analysis_info_state],
            outputs=[chapter_select],
            show_progress="minimal",
        )

        deselect_all_chapters_btn.click(
            fn=deselect_all_chapters,
            outputs=[chapter_select],
            show_progress="minimal",
        )

        # Prompt tab event handlers
        save_prompt_btn.click(
            fn=save_custom_prompt,
            inputs=[prompt_editor],
            outputs=[prompt_status, custom_prompt_state, prompt_status_indicator],
            show_progress="minimal",
        )

        reset_prompt_btn.click(
            fn=reset_to_default_prompt,
            outputs=[prompt_editor, prompt_status, prompt_status_indicator],
            show_progress="minimal",
        )

        test_prompt_btn.click(
            fn=test_custom_prompt,
            inputs=[prompt_editor, test_text, selected_model_state],
            outputs=[prompt_status, test_result],
            show_progress="full",
        )

        # Model selection event handler
        model_select.change(
            fn=update_selected_model,
            inputs=[model_select],
            outputs=[model_status, selected_model_state, model_status_indicator],
            show_progress="minimal",
        )

    return demo3


if __name__ == "__main__":
    chapter_scale_ui().launch(debug=True, share=False)
