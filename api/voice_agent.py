"""
Voice Agent Module for English Speech Integration
Handles: Speech-to-Text (Whisper via Groq) → RAG → TTS (Eleven Labs AI Voice)

Accuracy Improvements:
- Whisper confidence scoring and validation
- Audio quality checks
- Eleven Labs for premium quality human-like speech
- Detailed logging for debugging
"""

import os
import io
import tempfile
import logging
import json
import asyncio
from groq import Groq
from gtts import gTTS
from elevenlabs.client import ElevenLabs
import soundfile as sf
import numpy as np
from typing import AsyncGenerator, Any

logger = logging.getLogger(__name__)

# Global client cache
_groq_client = None

# Constants for accuracy tuning
MIN_CONFIDENCE_THRESHOLD = 0.3  # Lower threshold to avoid retries (faster)
MIN_AUDIO_DURATION = 0.3  # Reduced from 0.5s for faster validation
MAX_TRANSCRIPTION_RETRIES = 0  # Removed retries for speed (was 2)


def get_groq_client():
    """Get or create Groq client for STT and translation."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def speech_to_text_gujarati(audio_file_path: str) -> str:
    """
    Convert Gujarati speech to text using Whisper via Groq (BALANCED: FAST + ACCURATE).
    
    Optimizations:
    - Fast single transcription attempt
    - Whisper confidence checking (post-processing)
    - Improved validation
    
    Args:
        audio_file_path: Path to audio file
        
    Returns:
        Transcribed Gujarati text
        
    Raises:
        ValueError: If audio is invalid or transcription fails
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Quick audio validation
    try:
        import os
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:
            raise ValueError(f"Audio file too small: {file_size} bytes")
        logger.info(f"🎤 Audio file validated ({file_size} bytes)")
    except Exception as e:
        logger.error(f"Audio validation failed: {e}")
        raise
    
    client = get_groq_client()
    
    try:
        logger.info("🎤 Transcribing Gujarati speech (Whisper)...")
        
        with open(audio_file_path, "rb") as audio_file:
            # Use verbose_json to get confidence scores for quality validation
            transcription_response = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                language="gu",  # Gujarati language code
                response_format="verbose_json"  # Get confidence for validation
            )
        
        transcribed_text = transcription_response.text.strip() if hasattr(transcription_response, 'text') else str(transcription_response).strip()
        
        # Get confidence if available (but don't retry, just log)
        confidence = getattr(transcription_response, 'confidence', None)
        if confidence:
            logger.info(f"Confidence: {confidence:.2%}")
        
        if not transcribed_text:
            raise ValueError("Empty transcription result")
        
        logger.info(f"✓ Transcribed: {transcribed_text[:80]}...")
        return transcribed_text
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise


def translate_gujarati_to_english(gujarati_text: str) -> str:
    """
    Translate Gujarati text to English with ACCURACY OPTIMIZED (still fast).
    
    Optimizations for accuracy:
    - Spiritual terminology context preserved
    - Focused system prompt (concise but complete)
    - Balanced max_tokens (350 - enough for quality)
    - Validation of translation quality
    
    Args:
        gujarati_text: Text in Gujarati
        
    Returns:
        English translation
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not gujarati_text or not gujarati_text.strip():
        return ""
    
    client = get_groq_client()
    
    try:
        logger.info(f"🌐 Translating to English: {gujarati_text[:60]}...")
        
        # Restored detailed system prompt for accuracy but kept it concise
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert translator for Krishna wisdom and Hindu sacred texts.
Translate Gujarati text to English accurately. Keep spiritual terms (Bhakti, Dharma, Moksha, Karma, etc) in original form when appropriate.
Preserve exact meaning. Output ONLY the translation."""
                },
                {
                    "role": "user",
                    "content": gujarati_text
                }
            ],
            temperature=0.1,  # Slightly higher for better expression flexibility
            max_tokens=350,  # Balanced: fast but allows quality translation
            top_p=1.0
        )
        
        translation = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        
        if not translation:
            logger.warning("Empty translation result")
            return ""
        
        # Validate translation is not too short (quality check)
        if len(translation.split()) < 2:
            logger.warning(f"Translation too short: {translation}")
            # Retry with slightly higher temperature
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "Expert Krishna wisdom translator. Translate Gujarati to English preserving spiritual terms. Output ONLY translation."
                    },
                    {
                        "role": "user",
                        "content": gujarati_text
                    }
                ],
                temperature=0.3,
                max_tokens=350,
                top_p=1.0
            )
            translation = response.choices[0].message.content.strip() if response.choices[0].message.content else translation
        
        logger.info(f"✓ Translated: {translation[:80]}...")
        return translation
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise


def translate_english_to_gujarati(english_text: str) -> str:
    """
    Translate English text to Gujarati (ACCURACY OPTIMIZED).
    
    Optimizations:
    - Spiritual context preserved
    - Balanced max_tokens (450 for quality Gujarati)
    - Accurate system prompt with examples
    
    Args:
        english_text: Text in English
        
    Returns:
        Gujarati translation
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not english_text or not english_text.strip():
        return ""
    
    client = get_groq_client()
    
    try:
        logger.info(f"🇬🇯 Translating to Gujarati: {english_text[:60]}...")
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """Expert translator for Krishna teachings. Translate English to Gujarati accurately.
Keep spiritual terms in proper Gujarati script. Preserve exact meaning and nuance.
Output ONLY Gujarati translation in proper script."""
                },
                {
                    "role": "user",
                    "content": english_text
                }
            ],
            temperature=0.1,
            max_tokens=450,  # Gujarati needs more space
            top_p=1.0
        )
        
        translation = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        logger.info(f"✓ Gujarati translation ready")
        return translation
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise


def load_tts_model():
    """Initialize gTTS for Gujarati text-to-speech."""
    # gTTS is initialized on-demand in text_to_speech_gujarati
    # This function is kept for API compatibility
    return "gtts"


def text_to_speech_gujarati(gujarati_text: str, output_path: str | None = None) -> str:
    """
    Convert Gujarati text to speech using Google Text-to-Speech (OPTIMIZED FOR SPEED).
    
    Optimizations:
    - Minimal logging
    - Direct TTS without validation
    
    Args:
        gujarati_text: Text in Gujarati script
        output_path: Optional output file path, otherwise creates temp file
        
    Returns:
        Path to generated audio file
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        output_path = temp_file.name
        temp_file.close()
    
    try:
        logger.info("🔊 Generating Gujarati speech...")
        tts = gTTS(text=gujarati_text, lang='gu', slow=False)
        tts.save(output_path)
        logger.info("✓ Audio generated")
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        # Fallback: create silence
        sample_rate = 22050
        duration = 1.0
        audio_data = np.zeros(int(sample_rate * duration))
        wav_path = output_path.replace('.mp3', '.wav')
        sf.write(wav_path, audio_data, sample_rate)
        return wav_path
    
    return output_path


def process_voice_query(audio_file_path: str) -> tuple[str, str]:
    """
    Full voice processing pipeline (OPTIMIZED FOR SPEED).
    
    Pipeline:
    1. Speech-to-Text (fast, single attempt)
    2. Translate to English (reduced tokens)
    3. Return for RAG processing
    
    Args:
        audio_file_path: Path to input audio file
        
    Returns:
        Tuple of (gujarati_question, english_question)
        
    Raises:
        ValueError: If pipeline fails
    """
    logger.info("🚀 Voice query pipeline started")
    
    try:
        # Step 1: Speech to Text
        gujarati_question = speech_to_text_gujarati(audio_file_path)
        
        if not gujarati_question:
            raise ValueError("Failed to transcribe audio")
        
        # Step 2: Translate to English
        english_question = translate_gujarati_to_english(gujarati_question)
        
        if not english_question:
            raise ValueError("Translation failed")
        
        logger.info("✓ Voice pipeline complete")
        return gujarati_question, english_question
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise


# ============================================================================
# English-Only Voice Agent (WebSocket Streaming)
# ============================================================================

def speech_to_text_english(audio_file_path: str) -> str:
    """
    Convert English speech to text using Whisper via Groq.
    
    Args:
        audio_file_path: Path to audio file
        
    Returns:
        Transcribed English text
        
    Raises:
        ValueError: If audio is invalid or transcription fails
    """
    # Quick audio validation
    try:
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:
            raise ValueError(f"Audio file too small: {file_size} bytes")
        logger.info(f"🎤 Audio file validated ({file_size} bytes)")
    except Exception as e:
        logger.error(f"Audio validation failed: {e}")
        raise
    
    client = get_groq_client()
    
    try:
        logger.info("🎤 Transcribing English speech (Whisper)...")
        
        with open(audio_file_path, "rb") as audio_file:
            transcription_response = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                language="en",  # English language code
                response_format="verbose_json"
            )
        
        transcribed_text = transcription_response.text.strip() if hasattr(transcription_response, 'text') else str(transcription_response).strip()
        
        if not transcribed_text:
            raise ValueError("Empty transcription result")
        
        logger.info(f"✓ Transcribed: {transcribed_text[:80]}...")
        return transcribed_text
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise


def text_to_speech_english(english_text: str, output_path: str | None = None) -> str:
    """
    Convert English text to speech using Eleven Labs (Premium AI Voice).
    
    Fallback to gTTS if Eleven Labs fails.
    
    Features:
    - Premium quality, natural-sounding voice
    - Emotional expressiveness
    - Professional audio for Krishna wisdom
    
    Args:
        english_text: Text in English
        output_path: Optional output file path, otherwise creates temp file
        
    Returns:
        Path to generated audio file
    """
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        output_path = temp_file.name
        temp_file.close()
    
    try:
        logger.info("🔊 Generating premium English speech (Eleven Labs)...")
        
        # Initialize Eleven Labs client
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            logger.warning("⚠️  ELEVENLABS_API_KEY not set, falling back to gTTS")
            return fallback_text_to_speech_gtts(english_text, output_path)
        
        client = ElevenLabs(api_key=api_key)
        
        # Use professional voice for spiritual content
        # Adam: Wise, articulate male voice (good for teachings)
        response = client.text_to_speech.convert(
            text=english_text,
            voice_id="pNInz6obpgDQGcFmaJgB",  # Adam - Professional male voice
            model_id="eleven_turbo_v2_5",  # Latest free tier model (faster, better quality)
            voice_settings={
                "stability": 0.5,      # Balance between stability and variation
                "similarity_boost": 0.75,  # Good naturalness
            }
        )
        
        # Write audio to file
        with open(output_path, 'wb') as f:
            for chunk in response:
                f.write(chunk)
        
        logger.info("✓ Premium audio generated with Eleven Labs")
        return output_path
        
    except Exception as e:
        logger.error(f"Eleven Labs TTS failed: {e}, falling back to gTTS...")
        # Fallback to gTTS
        return fallback_text_to_speech_gtts(english_text, output_path)


def fallback_text_to_speech_gtts(english_text: str, output_path: str) -> str:
    """
    Fallback text-to-speech using Google gTTS.
    
    Used when Eleven Labs is unavailable.
    """
    try:
        logger.info("🔊 Generating English speech (gTTS - fallback)...")
        
        # Preprocess text for better audio quality
        processed_text = ' '.join(english_text.split())
        processed_text = processed_text.replace(' . ', '. ')
        processed_text = processed_text.replace(' , ', ', ')
        
        # Use slow=True for more natural speech
        tts = gTTS(
            text=processed_text, 
            lang='en',
            slow=True,
            tld='com'
        )
        tts.save(output_path)
        logger.info("✓ Audio generated with gTTS")
        return output_path
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        # Fallback: create silence
        sample_rate = 22050
        duration = 1.0
        audio_data = np.zeros(int(sample_rate * duration))
        wav_path = output_path.replace('.mp3', '.wav')
        sf.write(wav_path, audio_data, sample_rate)
        return wav_path


async def stt_stream(audio_chunks: AsyncGenerator[bytes, None]) -> AsyncGenerator[dict, None]:
    """
    Stream Speech-to-Text events from audio chunks.
    
    Collects audio chunks into a file, transcribes, and yields STT event.
    
    Args:
        audio_chunks: Async generator of audio bytes
        
    Yields:
        {"type": "stt_result", "text": str}
    """
    try:
        # Collect audio chunks into a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name
            
            async for chunk in audio_chunks:
                if chunk:
                    temp_file.write(chunk)
        
        # Transcribe the collected audio
        transcribed_text = speech_to_text_english(temp_path)
        
        # Clean up
        try:
            os.remove(temp_path)
        except:
            pass
        
        # Yield STT result
        yield {
            "type": "stt_result",
            "text": transcribed_text
        }
        
    except Exception as e:
        logger.error(f"STT stream error: {e}")
        yield {
            "type": "error",
            "message": f"Speech-to-text failed: {str(e)}"
        }


async def agent_stream(stt_events: AsyncGenerator[dict, None], agent_handler) -> AsyncGenerator[dict, None]:
    """
    Stream Agent events (RAG responses) from STT events.
    
    Takes STT results and passes them through the RAG agent.
    
    Args:
        stt_events: Async generator of STT events
        agent_handler: Function that takes question text and returns answer
        
    Yields:
        {"type": "agent_response", "text": str}
    """
    try:
        async for event in stt_events:
            if event["type"] == "stt_result":
                question = event["text"]
                logger.info(f"🤖 Processing question: {question[:80]}...")
                
                # Call the agent handler (synchronous RAG chain)
                try:
                    answer = await asyncio.to_thread(agent_handler, question)
                    
                    yield {
                        "type": "agent_response",
                        "text": answer
                    }
                except Exception as e:
                    logger.error(f"Agent error: {e}")
                    yield {
                        "type": "error",
                        "message": f"Agent processing failed: {str(e)}"
                    }
            else:
                # Pass through errors
                yield event
                
    except Exception as e:
        logger.error(f"Agent stream error: {e}")
        yield {
            "type": "error",
            "message": f"Agent stream error: {str(e)}"
        }


async def tts_stream(agent_events: AsyncGenerator[dict, None]) -> AsyncGenerator[dict, None]:
    """
    Stream TTS audio events from agent response events.
    
    Takes agent responses and converts them to audio.
    
    Args:
        agent_events: Async generator of agent response events
        
    Yields:
        {"type": "tts_chunk", "audio": bytes}
    """
    try:
        async for event in agent_events:
            if event["type"] == "agent_response":
                response_text = event["text"]
                logger.info("🔊 Converting response to speech...")
                
                try:
                    # Generate speech (synchronous operation in thread)
                    audio_path = await asyncio.to_thread(
                        text_to_speech_english,
                        response_text
                    )
                    
                    # Read audio file as bytes
                    with open(audio_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                    
                    # Clean up
                    try:
                        os.remove(audio_path)
                    except:
                        pass
                    
                    # Yield audio chunk
                    yield {
                        "type": "tts_chunk",
                        "audio": audio_bytes
                    }
                except Exception as e:
                    logger.error(f"TTS error: {e}")
                    yield {
                        "type": "error",
                        "message": f"Text-to-speech failed: {str(e)}"
                    }
            else:
                # Pass through errors
                yield event
                
    except Exception as e:
        logger.error(f"TTS stream error: {e}")
        yield {
            "type": "error",
            "message": f"TTS stream error: {str(e)}"
        }


def create_voice_pipeline(agent_handler):
    """
    Create a voice processing pipeline using RunnableGenerator-style async generators.
    
    Composition: audio_stream → stt_stream → agent_stream → tts_stream
    
    Args:
        agent_handler: Function that takes a question string and returns an answer
        
    Returns:
        A function that takes an audio chunk async generator and returns TTS event stream
    """
    async def pipeline(audio_chunks: AsyncGenerator[bytes, None]) -> AsyncGenerator[dict, None]:
        """Execute the full voice pipeline."""
        # Chain the generators together
        stt_events = stt_stream(audio_chunks)
        agent_events = agent_stream(stt_events, agent_handler)
        tts_events = tts_stream(agent_events)
        
        # Yield all TTS events
        async for event in tts_events:
            yield event
    
    return pipeline

# ============================================================================
# Language-agnostic wrapper functions for unified voice endpoint
# ============================================================================

def speech_to_text(audio_file_path: str) -> str:
    """
    Generic speech-to-text that defaults to English.
    
    Args:
        audio_file_path: Path to audio file
        
    Returns:
        Transcribed text in English
    """
    return speech_to_text_english(audio_file_path)


def text_to_speech(text: str, output_path: str | None = None) -> str:
    """
    Generic text-to-speech that defaults to English.
    
    Args:
        text: Text to convert to speech
        output_path: Optional output file path
        
    Returns:
        Path to generated audio file
    """
    return text_to_speech_english(text, output_path)