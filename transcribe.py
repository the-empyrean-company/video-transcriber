#!/usr/bin/env python3
"""
transcribe.py — Extract transcription from a video using OpenAI Whisper API.

Usage:
    python3 transcribe.py
    (you will be prompted to enter the path to your video file)

Setup:
    1. Copy .env.example to .env and fill in your API key
    2. pip3 install openai python-dotenv
    3. brew install ffmpeg
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from the same folder as this script
load_dotenv(dotenv_path=Path(__file__).parent / ".env")


def extract_audio(video_path: str, tmp_dir: str) -> str:
    """Extract audio from a video file using ffmpeg."""
    audio_path = os.path.join(tmp_dir, "audio.mp3")
    print(f"🎬 Extracting audio from: {video_path}")
    result = subprocess.run(
        [
            "ffmpeg",
            "-i", video_path,
            "-vn",                  # no video
            "-ar", "16000",         # 16kHz sample rate (optimal for Whisper)
            "-ac", "1",             # mono
            "-q:a", "0",            # best quality
            "-y",                   # overwrite if exists
            audio_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("❌ ffmpeg error:")
        print(result.stderr)
        sys.exit(1)
    print("✅ Audio extracted successfully.")
    return audio_path


def transcribe_audio(audio_path: str, language: Optional[str]) -> str:
    """Send audio to OpenAI Whisper API and return the transcription text."""
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ openai package not found. Install it with: pip3 install openai")
        sys.exit(1)

    # Try Streamlit secrets first (when running on Streamlit Cloud),
    # then fall back to .env / environment variable for local use
    api_key = os.environ.get("OPENAI_API_KEY")
    try:
        import streamlit as st
        api_key = st.secrets.get("OPENAI_API_KEY", api_key)
    except Exception:
        pass

    if not api_key:
        print("❌ OPENAI_API_KEY not found.")
        print("   Copy .env.example to .env and add your key there.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print("🤖 Sending audio to Whisper API...")
    with open(audio_path, "rb") as audio_file:
        kwargs = {
            "model": "whisper-1",
            "file": audio_file,
            "response_format": "text",
        }
        if language:
            kwargs["language"] = language

        transcript = client.audio.transcriptions.create(**kwargs)

    print("✅ Transcription complete.")
    return transcript  # response_format="text" returns a plain string


def save_transcript(text: str, output_path: str) -> None:
    """Save transcription text to a .txt file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"💾 Transcript saved to: {output_path}")


def main():
    print("🎙️  Video Transcriber (OpenAI Whisper)")
    print("--------------------------------------")

    # Ask for the video file path interactively
    video_path = input("📁 Drag & drop your video file here (or type the path): ").strip()

    # Strip surrounding quotes (macOS drag & drop sometimes adds them)
    video_path = video_path.strip("'\"")
    # Unescape shell-escaped characters (e.g. \, \& \  from drag & drop)
    video_path = re.sub(r'\\(.)', r'\1', video_path)

    if not os.path.isfile(video_path):
        print(f"❌ File not found: {video_path}")
        sys.exit(1)

    # Ask for optional language
    language = input("🌐 Language code (e.g. en, fr, es) — press Enter to auto-detect: ").strip() or None

    # Determine output path
    output_path = str(Path(video_path).with_suffix(".txt"))

    # Extract audio and transcribe
    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path = extract_audio(video_path, tmp_dir)
        transcript = transcribe_audio(audio_path, language)

    save_transcript(transcript, output_path)

    print("\n--- Preview (first 500 chars) ---")
    print(transcript[:500])
    print("..." if len(transcript) > 500 else "")


if __name__ == "__main__":
    main()
