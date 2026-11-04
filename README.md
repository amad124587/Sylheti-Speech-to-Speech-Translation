# Sylheti Speech-to-Speech Translation

Code for the BSc thesis **"A Low-Resource Speech-to-Speech Translation System for Sylheti to Bangla and English"**.

The system is a cascade of three modules:

| Module | Model | Notebooks |
|---|---|---|
| ASR (Sylheti speech → Sylheti text) | Whisper-large-v3-turbo (fine-tuned), Whisper-small as baseline | Step2d, Step2e, Step10, Step10b |
| MT (Sylheti text → Bangla / English) | NLLB-200-distilled-600M (fine-tuned) | Step3 |
| TTS (Bangla / English text → speech) | MMS-TTS (`facebook/mms-tts-ben`, `facebook/mms-tts-eng`) | Step4, Step4b, Step6, Step6c |

The Sylheti sentences and their Bangla and English translations come from the publicly available Kothon text corpus; the speech recordings were collected for this thesis.

## Repository contents

```
notebooks/   Colab notebooks (outputs and run metadata removed)
data/        round2_clean.csv - Round 2 recording metadata (speaker codes only)
scripts/     convert_to_wav.py - convert recordings to 16 kHz mono WAV
thesis/      Thesis draft (PDF)
```

### Notebooks

| Notebook | Purpose |
|---|---|
| Sylheti_Data_Cleaning | Build metadata and apply corrections (Round 1) |
| Step1b_Recording_Check, Step1c_Recheck | Automatic recording check (Whisper transcription + fuzzy matching) |
| Step2d_ASR_Turbo | Fine-tune and test Whisper-large-v3-turbo (Round 1) |
| Step2e_ASR_Small | Fine-tune and test Whisper-small (Round 1) |
| Step3_MT_Training | Fine-tune and test NLLB-200 for Sylheti → Bangla / English |
| Step3b_Cascade | Cascade ASR → MT evaluation (Round 1 ASR models) |
| Step4_TTS, Step4_TTS_normalized, Step4b_TTS_Compare | TTS generation and intelligibility evaluation |
| Step6_TTS_VoiceClone, Step6b_TTS_VoiceClone_v2, Step6c_TTS_VoiceClone_v3 | Voice-cloning experiments (MMS-TTS fine-tuning) |
| Step7_MOS_prepare | Audio samples for the listening test (MOS) |
| Step7b_MOS_analysis | MOS results and significance tests |
| Step9_Round2_Recheck | Round 2 recording check |
| Step10_ASR_Round2, Step10b_ASR_Round2_Small | Round 2 training, test and paired bootstrap |
| Step11_Cascade_Round2 | Round 2 cascade ASR → MT evaluation and bootstrap, all four ASR models |
| Step12_Demo_Round2, Step12_Demo_Round2_V1voice | Interactive end-to-end demo (Gradio) |

The notebooks expect the project data in Google Drive (`/content/drive/MyDrive/Sylheti_Thesis/...`) and were run on Google Colab GPUs (T4 / L4 / A100).

## Main results (from the thesis)

| | test_seen | test_unseen (A2) |
|---|---|---|
| ASR WER / CER, Whisper-large-v3-turbo, Round 2 | 36.1 / 12.3 | 31.5 / 8.3 |
| ASR WER / CER, Whisper-small, Round 2 | 40.7 / 14.7 | 37.6 / 10.8 |
| Cascade Sylheti → Bangla BLEU / chrF (turbo, Round 2) | 35.83 / 66.06 | 39.46 / 69.80 |
| Cascade Sylheti → English BLEU / chrF (turbo, Round 2) | 30.26 / 51.38 | 32.81 / 54.01 |
| MT on gold Sylheti text, Sylheti → Bangla / English BLEU | 51.32 / 38.25 | |

## Privacy

Speakers are identified only by codes (A1, A2, A4, A6, A7, B01–B12). No audio recordings, speaker names or name-to-code lists are included in this repository. Recordings from speakers who withdrew consent were removed from the dataset.
