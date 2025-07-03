import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import os
import google.generativeai as genai
import googlemaps
import gradio
from my_keys import GEMINI_API_KEY, GOOGLE_MAPS_KEY, MASTER_PROMPT

USE_GEMINI = True  # Set to False to use OpenRouter (TODO) instead 
GEMINI_VERSION = "gemini-2.5-flash"  # Free version for debugging

@dataclass
class LocationMention:
    name: str
    text_reference: str
    confidence: float
    chunk_index: int
    model_used: str
    scale: str

class TextPreprocessor:
    """Splits input text into chunks by chapter or by max words."""
    def __init__(self, max_words: int = 4000, chapter_keywords: Optional[List[str]] = None):
        self.max_words = max_words
        self.chapter_keywords = chapter_keywords or ["chapter ", "prologue", "epilogue"]

    def split_by_chapter(self, text: str) -> List[str]:
        import re
        # Split on chapter keywords (case-insensitive)
        pattern = r'(?i)(' + '|'.join(map(re.escape, self.chapter_keywords)) + r')'
        splits = re.split(pattern, text)
        # Recombine so each chunk starts with the chapter keyword
        chunks = []
        i = 1
        while i < len(splits):
            chunk = splits[i] + splits[i+1] if i+1 < len(splits) else splits[i]
            chunks.append(chunk.strip())
            i += 2
        return [c for c in chunks if c.strip()]

    def split_by_tokens(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.max_words):
            chunk = ' '.join(words[i:i + self.max_words])
            chunks.append(chunk)
        return chunks

    def chunk(self, text: str, method: str = "chapter") -> List[str]:
        if method == "chapter":
            chapters = self.split_by_chapter(text)
            if len(chapters) > 1:
                return chapters
            # fallback to tokens if no chapters found
        return self.split_by_tokens(text)

class GeminiExtractor:
    """Extracts locations from text chunks using Gemini API."""
    def __init__(self, gemini_api_key: str):
        self.api_key = gemini_api_key
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(GEMINI_VERSION)
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=4000,
        )

    def get_combined_prompt(self, chunk: str) -> str:
        return f"""{MASTER_PROMPT} {chunk}"""

    async def extract_locations_from_chunk(self, chunk: str, chunk_index: int) -> List[LocationMention]:
        try:
            prompt = self.get_combined_prompt(chunk)
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            content = response.text.strip()
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
            locations_data = json.loads(content)
            locations = []
            for loc_data in locations_data:
                locations.append(LocationMention(
                    name=loc_data['name'],
                    text_reference=loc_data['text_reference'],
                    confidence=loc_data['confidence'],
                    chunk_index=chunk_index,
                    model_used="Gemini Pro (Free)",
                    scale=loc_data['scale']
                ))
            return locations
        except Exception as e:
            print(f"Error with Gemini on chunk {chunk_index}: {e}")
            return []

    async def process_all_chunks(self, chunks: List[str]) -> List[LocationMention]:
        all_locations = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)} with Gemini...")
            locations = await self.extract_locations_from_chunk(chunk, i)
            all_locations.extend(locations)
            if i < len(chunks) - 1:
                await asyncio.sleep(1.1)
        return all_locations

class GoogleMapsExtractor:
    """Geocodes locations and creates Google Maps HTML/export."""
    def __init__(self, api_key: str):
        self.client = googlemaps.Client(key=api_key)

    def maps_geocode(self, locations: List[LocationMention]) -> List[Dict[str, Any]]:
        geocoded = []
        for loc in locations:
            geocode_result = self.client.geocode(loc.name)
            if geocode_result:
                location = geocode_result[0]['geometry']['location']
                geocoded.append({
                    "name": loc.name,
                    "confidence": loc.confidence,
                    "lat": location["lat"],
                    "lng": location["lng"],
                    "text_reference": loc.text_reference,
                    "scale": loc.scale
                })
            else:
                print(f"⚠️ No geocoding results for {loc.name}")
        return geocoded

    def export_gmaps_list(self, geocoded_locations: List[Dict[str, Any]]) -> str:
        # Export as JSON string (could be CSV/KML as needed)
        return json.dumps(geocoded_locations, indent=2)

    def create_map_html(self, geocoded_locations: List[Dict[str, Any]]) -> str:
        map_html = """
        <html>
        <head>
            <title>Location Map</title>
            <script src=\"https://maps.googleapis.com/maps/api/js?key={key}\"></script>
            <script>
                function initMap() {{
                    var map = new google.maps.Map(document.getElementById('map'), {{
                        zoom: 2,
                        center: {{lat: 20, lng: 0}}
                    }});
                    var bounds = new google.maps.LatLngBounds();
                    {markers}
                }}
            </script>
        </head>
        <body onload=\"initMap()\">
            <div id=\"map\" style=\"height: 600px; width: 100%;\"></div>
        </body>
        </html>
        """.format(
            key=GOOGLE_MAPS_KEY,
            markers=''.join(
                f"var marker{i} = new google.maps.Marker({{position: {{lat: {loc['lat']}, lng: {loc['lng']}}}, map: map, title: '{loc['name']}'}}); bounds.extend(marker{i}.getPosition());"
                for i, loc in enumerate(geocoded_locations)
            )
        )
        return map_html

class UserInterface:
    """Displays the Google Maps route and results to the user."""
    def __init__(self):
        pass

    def launch_interface(self, map_html: str, locations: List[Dict[str, Any]]):
        # Save the map HTML to a file
        with open("map.html", "w") as f:
            f.write(map_html)
        print("🌐 Map saved to map.html. Open this file in your browser to view the interactive map.")
        print("(Note: Gradio cannot display interactive Google Maps. Use your browser for full interactivity.)")

# --- MAIN PIPELINE ---
async def main_pipeline():
    # Sample book text (replace with file reading as needed)
    sample_book_text = """
    Chapter 1: The Journey Begins
    Sarah stepped off the plane at Charles de Gaulle Airport, the humid Parisian air hitting her face. 
    She had always dreamed of seeing the City of Light. After taking the RER train into the city, 
    she found herself standing before the iconic iron tower that Gustave Eiffel had built for the 1889 World's Fair.
    The next morning, she crossed the famous bridge over the Seine to visit the world's largest art museum, 
    where Leonardo's masterpiece smiled enigmatically from behind bulletproof glass.
    Chapter 2: Across the Channel
    Three days later, Sarah took the Eurostar through the tunnel beneath the English Channel. 
    London greeted her with its typical drizzle. She walked from St. Pancras to the Thames, 
    where the famous clock tower chimed noon. The Gothic revival palace nearby housed the British Parliament.
    Chapter 3: Germanic Adventures
    Her final destination was the German capital. Walking through the Brandenburg Gate, 
    she remembered the wall that once divided this city. Near Potsdamer Platz, she climbed 
    the victory column topped with the golden winged figure locals call "Goldelse."
    """
    # 1. Preprocess text
    preprocessor = TextPreprocessor()
    chunks = preprocessor.chunk(sample_book_text, method="chapter")
    print(f"📚 Split text into {len(chunks)} chunks")
    # 2. Extract locations
    gemini_extractor = GeminiExtractor(gemini_api_key=GEMINI_API_KEY)
    all_locations = await gemini_extractor.process_all_chunks(chunks)
    print(f"\n📍 Extracted {len(all_locations)} location mentions")
    # 3. Deduplicate
    def simple_deduplicate(locations: List[LocationMention]) -> List[LocationMention]:
        seen = {}
        deduped = []
        for loc in locations:
            key = loc.name.lower().strip()
            if key not in seen:
                seen[key] = loc
                deduped.append(loc)
            else:
                seen[key].confidence = max(seen[key].confidence, loc.confidence)
        return deduped
    unique_locations = simple_deduplicate(all_locations)
    print(f"📍 After deduplication: {len(unique_locations)} unique locations")
    # 4. Geocode
    gmaps_extractor = GoogleMapsExtractor(api_key=GOOGLE_MAPS_KEY)
    geocoded_locations = gmaps_extractor.maps_geocode(unique_locations)
    print(f"🗺️  Successfully geocoded {len(geocoded_locations)} locations")
    # 5. Export and display
    map_html = gmaps_extractor.create_map_html(geocoded_locations)
    ui = UserInterface()
    ui.launch_interface(map_html, geocoded_locations)
    # Optionally export list
    with open("debug_results.json", "w") as f:
        f.write(gmaps_extractor.export_gmaps_list(geocoded_locations))
    print("💾 Results saved to debug_results.json")
    print("✅ Pipeline complete!")

if __name__ == "__main__":
    asyncio.run(main_pipeline())
