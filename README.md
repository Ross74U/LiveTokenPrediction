## VoskWhisper 
A hybrid, Vosk + OpenAI whisper based transcription library with per token perctimer logging
 - Inspired by RealtimeSTT (https://github.com/KoljaB/RealtimeSTT)
 - Threaded Silero Silence Detection 
 - Vosk cpu-bound partial-result processing
 - FastWhisper cuda-bound utterance processing 
 - Detailed .JSONL Performance Logging on per-token level

Designed as a general research testbed for Token Prediction latency reduction approaches 
and "True Realtime Chatbot Functionality" applications:
- **Realtime Backbone Embedding** - continuous sub-millisecond decoding and feedback done by tiny_decoder
    - implementation of artificial patience, interruption, and other current human-bound behavior
 
- **Mixed Audio and Transcription Token** - see CSM by Sesame AI (https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice)  

To run
```
import voskwhisper.voskwhisper_audiodetection

vw = voskwhisper_audiodetection.VoskWhisper()
vw.start() 
```
