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
import re
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
    DEFAULT_CHAPTER_PATTERNS = [
        r"^Chapter\s+\d+\b",
        r"^Section\s+\d+\b",
        r"^Part\s+[IVXLC]+\b",
        r"^CHAPTER\s+[A-Z]+\b",
        r"^Prologue\b",
        r"^Epilogue\b",
        r"^Introduction\b",
    ]

    def __init__(self, chunk_size=1500, overlap=300, chapter_patterns=None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chapter_patterns = chapter_patterns or self.DEFAULT_CHAPTER_PATTERNS
        self.compiled_patterns = [
            re.compile(p, re.MULTILINE) for p in self.chapter_patterns
        ]

    def find_anchors(self, text):
        """Detects chapter/section markers using compiled regex."""
        anchors = []
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                anchors.append((match.start(), match.group()))
        return sorted(anchors, key=lambda x: x[0])

    def segment_text_by_anchors(self, text, anchors):
        """Yields labeled text segments (start/end indexes only)."""
        for i, (start_idx, label) in enumerate(anchors):
            end_idx = anchors[i + 1][0] if i + 1 < len(anchors) else len(text)
            yield {"label": label.strip(), "start": start_idx, "end": end_idx}

    def chunk_section(self, text, section):
        """Yields chunks from a section using start/end indexes with overlap."""
        label = section["label"]
        section_text = text[section["start"] : section["end"]]
        start = 0
        chunk_num = 1
        section_len = len(section_text)

        while start < section_len:
            end = min(start + self.chunk_size, section_len)
            chunk = section_text[start:end]
            yield {
                "parent_label": label,
                "chunk_id": f"{label}.{chunk_num}",
                "preview": chunk[:200].strip(),
                "full_text": chunk.strip(),
            }
            start += self.chunk_size - self.overlap
            chunk_num += 1

    def process(self, text):
        """Main method to process full text."""
        anchors = self.find_anchors(text)
        chapter_titles = [label for _, label in anchors]

        if not anchors:
            sections = [{"label": "Unlabeled", "start": 0, "end": len(text)}]
        else:
            sections = list(self.segment_text_by_anchors(text, anchors))

        chunks = []
        for section in sections:
            chunks.extend(self.chunk_section(text, section))

        return {
            "num_chapters": len(anchors),
            "num_chunks": len(chunks),
            "chapter_titles": chapter_titles,
            "chunks": chunks,
        }


class GeminiExtractor:
    """Extracts locations from text chunks using Gemini API."""
    def __init__(self, gemini_api_key: str):
        self.api_key = gemini_api_key
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(GEMINI_VERSION)
        self.generation_config = genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=4000,
        )

    def get_combined_prompt(self, chunk: str) -> str:
        return f"""{MASTER_PROMPT} {chunk}"""
    
    def try_extract_locations_from_chunk(self, chunk: str, chunk_index: int, model: str = GEMINI_VERSION) -> List[LocationMention]:
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

    async def extract_locations_from_chunk(self, chunk: str, chunk_index: int) -> List[LocationMention]:
        try:
            output = self.try_extract_locations_from_chunk(chunk, chunk_index)
        except Exception as e:
            print(f"Error with Gemini on chunk {chunk_index}: {e}")
            # try again with a different model
            try:
                output = self.try_extract_locations_from_chunk(chunk, chunk_index, model="gemini-2.0-flash")
            except Exception as e:
                print(f"Error with Gemini on chunk {chunk_index} with model gemini-2.0-flash: {e}")
                return []
        return output

    async def process_all_chunks(self, chunks: List[str]) -> List[LocationMention]:
        tasks = []
        for i, chunk in enumerate(chunks):
            print(f"Creating task for chunk {i+1}/{len(chunks)}...")
            task = self.extract_locations_from_chunk(chunk, i)
            tasks.append(task)
        
        print("Processing all chunks in parallel...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # filter out exceptions, and empty lists
        results = [result for result in results if not isinstance(result, Exception) and result]
        return results

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

# TODO: KML export
    def export_gmaps_list(self, geocoded_locations: List[Dict[str, Any]]) -> str:
        # Export as JSON string (could be CSV/KML as needed)
        return json.dumps(geocoded_locations, indent=2)

# --- MAIN PIPELINE ---

def extract_and_geocode_locations(chunks: List[str], selected_scales: List[str]) -> List[Dict[str, Any]]:
    """
    Given a list of text chunks and selected scales, extract locations, deduplicate, filter by scale, geocode, and return geocoded location dicts.
    Synchronous wrapper for Gradio UI.
    Deduplication: same places (by name, case-insensitive) are merged, text references concatenated, and highest confidence kept.
    """
    async def pipeline():
        gemini_extractor = GeminiExtractor(gemini_api_key=GEMINI_API_KEY)
        all_locations = await gemini_extractor.process_all_chunks(chunks)
        # Deduplicate by name (case-insensitive), concatenate text references, keep highest confidence
        deduped = {}
        for loc in all_locations:
            key = loc.name.lower().strip()
            if key not in deduped:
                deduped[key] = loc
            else:
                # Concatenate text references
                deduped[key].text_reference += ", " + loc.text_reference
                # Keep highest confidence
                deduped[key].confidence = max(deduped[key].confidence, loc.confidence)
        unique_locations = list(deduped.values())
        # Filter by selected scales
        filtered_locations = [loc for loc in unique_locations if loc.scale in selected_scales]
        # Geocode
        gmaps_extractor = GoogleMapsExtractor(api_key=GOOGLE_MAPS_KEY)
        geocoded_locations = gmaps_extractor.maps_geocode(filtered_locations)
        return geocoded_locations
    return asyncio.run(pipeline())

if __name__ == "__main__":
    pass
