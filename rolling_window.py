from pydub import AudioSegment

def cut_wav_file(input_file, output_file, start_time_ms, end_time_ms):
    """Cuts a WAV file from start_time_ms to end_time_ms and saves the result to output_file."""

    audio = AudioSegment.from_wav(input_file)
    audio_segment = audio[start_time_ms:end_time_ms]
    audio_segment.export(output_file, format="wav")



for i in range(30):
    cut_wav_file("vat_of_shade.wav", "vat_of_shade_test/" + str(i) + ".wav", i * 100, (i + 2) * 100)
