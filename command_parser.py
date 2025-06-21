# command_parser.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load FLAN-T5-small (only ~80MB model weights)
# google/flan-t5-small

# flan-t5-large: 800MB
# google/flan-t5-large
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

ALLOWED_COMMANDS = ["spawn", "reclassify","delete","move_up","move_down","move_left","move_right", "noop"]

def interpret_command(text):
    prompt = (
        f"You are a voice assistant in a drawing app. "
        f"Recognize and extract only the commands or similar from this user's speech: '{text}'. "
        f"Use only the words from this list: {', '.join(ALLOWED_COMMANDS)}. "
        f"Respond with a comma-separated list of only matching command(s). If none match, return 'noop'.\n\n"
        
        f"Good examples:\n"
        f"- 'I want a drawing to the left of the canvas' → spawn, move_left, move_left\n"
        f"- 'move greatly to the right' → move_right, move_right\n"
        f"- 'can you delete that and reclassify it?' → delete, reclassify\n"
        f"- 'bring it up a bit' → move_up\n"
        f"- 'place a shape and shift it down' → spawn, move_down\n\n"

        f"Bad examples:\n"
        f"- '1111111000000000' → spawn, reclassify (nonsensical input)\n"
        f"- 'aamief hheeeee' → move_right, delete (gibberish)\n"
        f"- 'I had a sandwich yesterday' → move_left (irrelevant input)\n"
        f"- 'just go boom boom lefty woo' → move_left, spawn (unclear/made-up language)\n\n"

        f"Now extract the command(s) from: '{text}'"
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_new_tokens=40)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Normalize and return as a list
    commands = [cmd.strip().lower() for cmd in output_text.split(",")]
    valid_commands = [cmd for cmd in commands if cmd in ALLOWED_COMMANDS]
    return valid_commands
