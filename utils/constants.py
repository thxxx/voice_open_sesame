COMPANION_NAME = "Maya"
LLM_MODEL = "local" # "api" or "local"
OPENAI_KEY = ""

DEFAULT_VOICE_PATH = "./audio_samples/output_full.wav"
DEFAULT_KOREAN_VOICE_PATH = "./audio_samples/shogun.wav"

FOLLOWUP_SILENCE_DELAY = 6.0

LANGUAGE_CODE = {
    "Arabic": "ar",
    "Danish": "da",
    "German": "de",
    "Greek": "el",
    "English": "en",
    "Spanish": "es",
    "Finnish": "fi",
    "French": "fr",
    "Hebrew": "he",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Malay": "ms",
    "Dutch": "nl",
    "Norwegian": "no",
    "Polish": "pl",
    "Portuguese": "pt",
    "Russian": "ru",
    "Swedish": "sv",
    "Swahili": "sw",
    "Turkish": "tr",
    "Chinese": "zh",
}

LANGUAGE_CODE_REVERSED = {v: k for k, v in LANGUAGE_CODE.items()}


COMMON_STARTERS = [
    "Yeah.. ",
    "Yep.. ",
    "Nah.. ",
    "Right.. ",
    "Okay.. ",
    "Alright.. ",
    "Well.. ",
    "So, ",
    "Anyway, ",
    "By the way, ",
    "Actually, ",
    "Honestly, ",
    "Seriously, ",
    "Basically, ",
    "Like",
    "You know, ",
    "I mean, ",
    "I guess, ",
    "I think, ",
    "Apparently, ",
    "Obviously, ",
    "Literally, ",
    "Maybe, ",
    "Probably, ",
    "Exactly, ",
    "Sure, ",
    "Uh...",
    "Uhm...",
    "Ah...",
    "Oh!",
    "Cool.. ",
    "Nice.. "
]