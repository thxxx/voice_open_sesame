import torch
import asyncio
import threading
import inspect
from app.session import Session
from llm.openai import chat_followup
from chatterbox_infer.mtl_tts import ChatterboxMultilingualTTS
from librosa.util import normalize
from utils.process import pcm16_b64
from utils.constants import COMMON_STARTERS, DEFAULT_VOICE_PATH, DEFAULT_KOREAN_VOICE_PATH, FOLLOWUP_SILENCE_DELAY
from concurrent.futures import ThreadPoolExecutor
import orjson as json
import time
import torchaudio
import re
import random
import opuslib
import numpy as np
import struct

MAGIC = b'\xA1\x51'

def pack_frame(seq, ts_usec, payload: bytes, is_final=False):
    flags = 1 if is_final else 0
    header = MAGIC + struct.pack('<BBIQI', flags, 0, seq, ts_usec, len(payload))
    return header + payload

class OpusEnc:
    def __init__(self, sr=24000, channels=1, bitrate=32000):
        self.sr, self.ch = sr, channels
        self.frame_ms = 20                           # 20ms 권장
        self.frame_size = sr * self.frame_ms // 1000 # samples per channel
        self.enc = opuslib.Encoder(sr, channels, opuslib.APPLICATION_AUDIO)
        try:
            self.enc.bitrate = bitrate
        except Exception:
            self.enc.set_bitrate(bitrate)

    def encode(self, pcm_f32: np.ndarray) -> list[bytes]:
        """
        pcm_f32: shape (N,), float32 [-1, 1]
        returns: list of raw opus packets (bytes), each ~= 20ms
        """
        # float32 -> int16 (mono interleaved)
        pcm_i16 = np.clip(pcm_f32, -1.0, 1.0)
        pcm_i16 = (pcm_i16 * 32767.0).astype(np.int16, copy=False)

        frames = []
        n = pcm_i16.shape[0]
        i = 0
        while i + self.frame_size <= n:
            chunk_i16 = pcm_i16[i:i+self.frame_size]
            pkt = self.enc.encode(chunk_i16.tobytes(), self.frame_size)
            frames.append(pkt)
            i += self.frame_size
        return frames


ENC_EXEC = ThreadPoolExecutor(max_workers=6)

tts_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

def jdumps(o): return json.dumps(o).decode()

@torch.inference_mode()
async def chatter_streamer(sess: Session):
    try:
        loop = asyncio.get_running_loop()
        sr = 24000
        OVERLAP = int(0.03 * sr)

        sess.tts_buffer_sr = sr
        sess.tts_pcm_buffer = np.empty(0, dtype=np.float32)

        opus_enc = OpusEnc(sr=sr, channels=1, bitrate=32000)
        seq_ref = {"seq": 0}
        session_t0 = time.time()
        
        sess.out_q.put_nowait(jdumps({
            "type": "tts_audio_meta",
            "format": "opus",
            "sample_rate": sr,
            "channels": 1,
            "frame_ms": 20,
        }))

        def put_binary(out_q, b: bytes):
            out_q.put_nowait(b)
        
        async def emit_opus_frames(
            out_q: asyncio.Queue,
            enc: OpusEnc,
            pcm: torch.Tensor,
            sr: int,
            seq_ref: dict,
            is_final: bool = False,
            t0: float | None = None,
        ):
            """
            Encode pcm to 20ms Opus frames and push to out_q.
            """
            if pcm is None or pcm.numel() == 0:
                # no payload end frame
                ts_usec = int(((time.time()) - (t0 or 0.0)) * 1e6) if t0 else 0
                put_binary(out_q, pack_frame(seq_ref["seq"], ts_usec, b"", is_final=is_final))
                seq_ref["seq"] += 1
                return
        
            if pcm.dim() == 2:
                pcm = pcm.squeeze(0)
            pcm = pcm.detach().cpu().to(torch.float32).clamp_(-1.0, 1.0)
            sess.tts_pcm_buffer = np.concatenate([sess.tts_pcm_buffer, pcm.numpy()], axis=0)
        
            pcm_f32 = pcm.numpy().astype(np.float32, copy=False)
        
            frames = enc.encode(pcm_f32)
        
            now = time.time()
            for pkt in frames:
                ts_usec = int((now - (t0 or now)) * 1e6)
                put_binary(out_q, pack_frame(seq_ref["seq"], ts_usec, pkt, is_final=False))
                seq_ref["seq"] += 1
        
            if is_final:
                ts_usec = int((time.time() - (t0 or now)) * 1e6)
                put_binary(out_q, pack_frame(seq_ref["seq"], ts_usec, b"", is_final=True))
                seq_ref["seq"] += 1

        def start_tts_producer_in_thread(text_chunk: str, ref_audio, out_q: asyncio.Queue):
            stop_evt = sess.tts_stop_event
            text_chunk = re.sub(r"\\n", " ... ", text_chunk)
            text_chunk = re.sub(r"\n", " ... ", text_chunk)

            async def produce():
                try:
                    agen = tts_model.generate_stream(
                        text_chunk,
                        audio_prompt_path=ref_audio,
                        language_id=sess.language,
                        chunk_size=28, # This should be adjusted for realtime factor
                        exaggeration=0.4,
                        cfg_weight=0.5,
                        temperature=0.75,
                        repetition_penalty=1.3,
                        min_p=0.02,
                        top_p=0.9
                    )
                    if inspect.isasyncgen(agen):
                        async for evt in agen:
                            if stop_evt.is_set():
                                break
                            audio = evt.get("audio")
                            loop.call_soon_threadsafe(out_q.put_nowait, ("chunk", audio))
                    else:
                        for evt in agen:
                            if stop_evt.is_set():
                                break
                            audio = evt.get("audio")
                            loop.call_soon_threadsafe(out_q.put_nowait, ("chunk", audio))

                    loop.call_soon_threadsafe(out_q.put_nowait, ("eos", None))
                except Exception as e:
                    loop.call_soon_threadsafe(out_q.put_nowait, ("error", str(e)))

            def thread_target():
                asyncio.run(produce())

            t = threading.Thread(target=thread_target, daemon=True)
            t.start()

        async def consume_loop():
            while sess.running:
                text_chunk = await sess.tts_in_q.get()
                if not text_chunk or sess.tts_stop_event.is_set():
                    print("[chatter_streamer] TTS stop event is set", text_chunk)
                    continue
                is_last = True
                if "<cont>" in text_chunk:
                    is_last=False
                    text_chunk = re.sub('<cont>', '', text_chunk)
                
                if isinstance(text_chunk, tuple) and len(text_chunk) == 2 and text_chunk[0] == "__silence__":
                    try:
                        silence_sec = float(text_chunk[1]) * 0.4
                        if silence_sec > 0:
                            num_samples = int(silence_sec * sr)
                            silence_wav = torch.zeros(1, num_samples, dtype=torch.float32)
                            await emit_opus_frames(sess.out_q, opus_enc, silence_wav, sr, seq_ref, is_final=False, t0=session_t0)
                            print(f"[silence] sent {silence_sec:.2f}s (opus)")
                    except Exception as e:
                        print(f"[silence] error: {e}")
                    continue


                # === Prepare reference audio ===
                if hasattr(sess, "ref_audios") and not getattr(sess, "ref_audios").empty():
                    ref_audio = sess.ref_audios.get()
                    sess.ref_audios.put(ref_audio)
                    ref_audio = normalize(ref_audio[-int(16000 * 15):])
                else:
                    ref_audio = DEFAULT_VOICE_PATH if sess.language != 'ko' else DEFAULT_KOREAN_VOICE_PATH

                cancel_silence_nudge(sess)
                
                # === Thread → Main loop chunk queue ===
                tts_chunk_q: asyncio.Queue = asyncio.Queue(maxsize=6)

                # === (Add) Detect starter → Send sample wav ===
                stripped = text_chunk.lstrip()
                matched = None
                for s in COMMON_STARTERS:
                    if stripped.startswith(s):
                        matched = s
                        break

                if matched is not None:
                    token = matched.lower().strip().replace(".", "").replace(",", "")
                    idx = random.choice([0, 1, 2])
                    sample_path = f"./audio_samples/{token}_{idx}.wav"
                    try:
                        wav, sr_file = torchaudio.load(sample_path)  # (ch, T)
                        if wav.dim() == 2 and wav.size(0) > 1:
                            wav = wav.mean(dim=0, keepdim=True)  # mono
                        if sr_file != sr:
                            wav = torchaudio.functional.resample(wav, sr_file, sr)
                        # Send to Opus
                        asyncio.create_task(emit_opus_frames(sess.out_q, opus_enc, wav, sr, seq_ref, is_final=False, t0=session_t0))
                    except Exception as e:
                        print(f"[starter] failed to load '{sample_path}': {e}")

                last_length = 0
                last_tail: torch.Tensor | None = None
                start_time = time.time()

                # === TTS producer (thread) start (send sample wav and simultaneous) ===
                if matched is not None:
                    text_chunk = re.sub(matched, '', text_chunk[:10]) + text_chunk[10:]
                
                start_tts_producer_in_thread(text_chunk, ref_audio, tts_chunk_q)
                start_sending_at = 0
                total_audio_seconds = 0

                # === Consume: Get from queue and cross-fade & send ===
                while True:
                    if sess.tts_stop_event.is_set():
                        try:
                            while True:
                                _ = tts_chunk_q.get_nowait()
                                tts_chunk_q.task_done()
                        except asyncio.QueueEmpty:
                            pass
                        break
                    evt_type, payload = await tts_chunk_q.get()

                    if evt_type == "chunk":
                        wav: torch.Tensor = payload  # (ch, T)
                        if wav is None or wav.numel() == 0:
                            await asyncio.sleep(0)
                            continue
                        new_total = wav.shape[-1]
                        delta = new_total - last_length
                        if delta <= 0:
                            await asyncio.sleep(0)
                            continue
                        new_part = wav[:, last_length:new_total]

                        # --- Cross-fade ---
                        if last_tail is None:
                            out_chunk = new_part
                        else:
                            L = min(OVERLAP, new_part.shape[-1], last_tail.shape[-1])
                            if L > 0:
                                dtype = wav.dtype
                                device = wav.device
                                fade_in  = torch.linspace(0, 1, L, device=device, dtype=dtype)
                                fade_out = 1.0 - fade_in
                                mixed = last_tail[:, -L:] * fade_out + new_part[:, :L] * fade_in
                                tail  = new_part[:, L:]
                                out_chunk = torch.cat([mixed, tail], dim=-1)
                            else:
                                out_chunk = new_part

                        new_tail_start = max(0, new_total - OVERLAP)
                        last_tail = wav[:, new_tail_start:new_total].detach()

                        print(f"[TTS {(new_total-last_length)/24000:.3f}] - takes {time.time() - start_time:.3f}")
                        total_audio_seconds += (new_total-last_length)/24000
                        
                        await emit_opus_frames(sess.out_q, opus_enc, out_chunk, sr, seq_ref, is_final=False, t0=session_t0)

                        if start_sending_at == 0:
                            start_sending_at = time.time()
                        
                        last_length = new_total
                        await asyncio.sleep(0)

                    elif evt_type == "eos":
                        if not is_last:
                            break
                            
                        if last_tail is not None and last_tail.numel() > 0:
                            await emit_opus_frames(sess.out_q, opus_enc, last_tail, sr, seq_ref, is_final=True, t0=session_t0)
                        else:
                            await emit_opus_frames(sess.out_q, opus_enc, None, sr, seq_ref, is_final=True, t0=session_t0)
                        sess.tts_pcm_buffer = np.empty(0, dtype=np.float32)

                        taken = time.time() - start_sending_at
                        remaining_until_audio_end = total_audio_seconds - taken + 1
                        print(f"Taken : {taken:.3f}, Total audio : {total_audio_seconds:.3f}, Remain : {remaining_until_audio_end:.3f}")
                        
                        schedule_silence_nudge(sess, delay=FOLLOWUP_SILENCE_DELAY, remain=remaining_until_audio_end)
                        
                        break

                    elif evt_type == "error":
                        print("🥊 [chatter_streamer] TTS error:", payload)
                        sess.out_q.put_nowait(jdumps({"type": "tts_error", "message": payload}))
                        break

        await consume_loop()

    except asyncio.CancelledError:
        raise
    except Exception as e:
        await sess.out_q.put(jdumps({"type": "tts_error", "message": str(e)}))
    finally:
        pass

async def proactive_say(sess: Session):
    loop = asyncio.get_running_loop()

    def run_blocking():
        return chat_followup(
            prev_scripts=sess.transcripts[-6:],
            prev_answers=sess.outputs[-6:],
            language=sess.language,
            name=sess.name,
            current_time=sess.current_time,
        )

    try:
        output = await loop.run_in_executor(None, run_blocking)
        tuples = (output.get("text", "") or "")
        if tuples[0] is None or tuples[0] == '':
            return
        if tuples[1] == 'wait':
            return
        text = tuples[0]
        
        sess.answer = text.strip()
        sess.outputs[-1] = sess.outputs[-1] + " (User silence for six seconds) " + text

    except Exception as e:
        print("NUDGE ERROR : ", e)
    
    loop.call_soon_threadsafe(sess.tts_in_q.put_nowait, text)
    loop.call_soon_threadsafe(
        sess.out_q.put_nowait,
        jdumps({"type": "translated", "script": "", "text": text, "is_final": True})
    )

def cancel_silence_nudge(sess: Session):
    """Cancel silence nudge task"""
    task = getattr(sess, "silence_nudge_task", None)
    if task and not task.done():
        task.cancel()
    setattr(sess, "silence_nudge_task", None)

def schedule_silence_nudge(sess: Session, delay: float = 5.0, remain: float = 1.0):
    cancel_silence_nudge(sess)

    async def waiter():
        try:
            await asyncio.sleep(remain)
            if getattr(sess, "current_audio_state", "none") != "none":
                return
            await asyncio.sleep(delay)
            if getattr(sess, "current_audio_state", "none") == "none":
                print("\nproactive say\n")
                await proactive_say(sess)
        except asyncio.CancelledError:
            pass

    # Register new timer
    if random.random() > 0.7:
        sess.silence_nudge_task = asyncio.create_task(waiter())
