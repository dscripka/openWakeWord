import librosa
import soundfile as sf
from pathlib import Path
import subprocess
from pathlib import Path

def convert_and_resample(input_dir, output_dir, target_sr=16000):
    """
    Convert M4A files to WAV format and resample them to a specified sampling rate.

    Args:
        input_dir (str): Path to the directory containing the original M4A files.
        output_dir (str): Path where the converted and resampled WAV files will be saved.
        target_sr (int, optional): Target sampling rate for the WAV files. Default is 16000 Hz.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for m4a_file in input_dir.glob('*.m4a'):
        wav_file = output_dir / (m4a_file.stem + '.wav')
        # Build ffmpeg command
        command = [
            'ffmpeg',
            '-i', str(m4a_file),  # Input file
            '-ar', str(target_sr),  # Set audio sampling rate
            '-ac', '1',  # Set audio channels to 1
            '-y',  # Overwrite output files without asking
            str(wav_file)  # Output file
        ]

        # Execute the command
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Print status
        print(f'Converted and resampled {m4a_file} to {wav_file}')

# Example usage
input_directory = '/Users/SAI/Documents/Code/wakeWord/wakeWordForked/Untitled/verifier_data/epp_hey_stryker/mp4'
output_directory = '/Users/SAI/Documents/Code/wakeWord/wakeWordForked/Untitled/verifier_data/resamples_stryker_epp'
convert_and_resample(input_directory, output_directory)
