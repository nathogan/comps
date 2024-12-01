
from pydub import AudioSegment

def cut_wav_file(input_file, output_file, start_time_ms, end_time_ms):
    """Cuts a WAV file from start_time_ms to end_time_ms and saves the result to output_file."""

    audio = AudioSegment.from_wav(input_file)
    audio_segment = audio[start_time_ms:end_time_ms]
    audio_segment.export(output_file, format="wav")



# Read line by line
with open('/Users/nhogan/Desktop/languages/en/info.txt', 'r') as file:
    for line in file:
        words = line.split()
        cut_wav_file("/Users/nhogan/Desktop/languages/en/grid_and_wav/" + words[0] + ".wav", "temp_wavs/" + words[0] + ".wav", float(words[1])*1000, float(words[2])*1000)
        
        print(words)
        
        print(line, end='')  # end='' removes extra newline
