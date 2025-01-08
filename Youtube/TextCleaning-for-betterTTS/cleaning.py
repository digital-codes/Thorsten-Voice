from nemo_text_processing.text_normalization.normalize import Normalizer
import sys

def normalize_text(input_text, language='de'):
    normalizer = Normalizer(
        lang=language,
        input_case='cased',
    )

    normalized = normalizer.normalize(
        text=input_text,
        punct_post_process=True,
        verbose=False
    )

    return normalized

for line in sys.stdin:
    input_text_stdin = str(line)

output_text = normalize_text(input_text=input_text_stdin,language='en')
#print("INPUT: " + input_text_stdin)
#print("OUTPUT: " + output_text)
sys.stdout.write(output_text)