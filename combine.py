from pathlib import Path
import wave
import logging
import os
import glob

import scipy.io.wavfile as wavfile
from scipy import signal
from scipy.signal import butter, sosfilt
import numpy as np
import noisereduce as nr

from typing import List, Optional, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table




class Audio_processing:
    def __init__(self, root_directory: Optional[str] = None):
        if root_directory is None:
            raise ValueError("root_directory cannot be None")
        self.root_directory = root_directory
        self.root_path = Path(root_directory)
        self.out_path = self.root_path/Path('combined/')
        self.final_path = self.root_path/Path('process/')
        self.noise_file = Path(root_directory)
        self.npz_file = self.root_path/Path('noise/noise.npz')
        self.console = Console()
        self.logger = self._setup_logging()
        
        #create path output directory
        Path(self.out_path).mkdir(parents=True, exist_ok=True)

        """
        #Load Noise profile
        self.noise_spectrum= self.load_noise_profile()

        self.sample_rate = 256000
    
        self.noise_spectrum = np.array([])
        self.frequencies = np.array([])
        """

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger(__name__)

    def find_wave_files(self, root_directory: str) -> List[str]:
        """
        Recursively find all WAV files in directory and subdirectories.

        Args:
            root_directory: Path to the root directory to search

        Returns:
            List of WAV file paths
        """
        wav_files = []
        for root, dirs, files in os.walk(root_directory):
            wav_pattern = os.path.join(root, "*.wav")
            wav_files.extend(glob.glob(wav_pattern))

        self.logger.info(f"Found {len(wav_files)} WAV files in {root_directory}")
        return sorted(wav_files)
    
    def load_audio_file(self, filepath: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Load a WAV file and return audio data and sample rate.

        Args:
            filepath: Path to the WAV file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            sample_rate, audio_data = wavfile.read(filepath)

            # Convert to float32 and normalize
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.uint8:
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
            elif audio_data.dtype == np.float32:
            # Already float32, but ensure it's in [-1, 1] range
                if np.max(np.abs(audio_data)) > 1.0:
                    audio_data = np.clip(audio_data, -1.0, 1.0)
            elif audio_data.dtype == np.float64:
            # Convert float64 to float32
                audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = np.clip(audio_data, -1.0, 1.0)
            else:
                self.logger.warning(f"Unsupported audio format {audio_data.dtype} in {filepath}")
            # Try to convert to float32 anyway
            audio_data = audio_data.astype(np.float32)

        # Ensure audio_data is contiguous in memory for better performance
            audio_data = np.ascontiguousarray(audio_data)

            return audio_data, sample_rate

        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")
            return None, None
        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {e}")
            return None, None

    def consolidate_audio_files(self, wave_file: list, out_file: Path) -> bool:
        """
        Recursively find all WAV files in directory and subdirectories. 
        Args:
            root_directory: Path to the root directory to search
        Returns:
            List of WAV file paths
        """
        try:
            data = []
            for wave_file in wave_file:
                with wave.open(str(wave_file), 'rb') as w:
                    data.append([w.getparams(), w.readframes(w.getnframes())])
        
            with wave.open(str(out_file), 'wb') as output:
                if data:  # Check if we have data
                    output.setparams(data[0][0])
                    for params, frames in data:
                        output.writeframes(frames)
        
            return True
        except Exception as e:
            print(f"Error consolidating audio files: {e}")
            return False
                    
    
    def load_noise_profile(self):
        # Load and compute noise spectrum from noise file
        try:
            noise_data = wavfile.read(self.noise_file)
            #convert to float and normalize
            noise_data_array = np.array(noise_data)
            converted_noise_data = noise_data_array.astype(float)
            if converted_noise_data.dtype == np.int16:
                converted_noise_data = converted_noise_data.astype(np.float32) / 32768.0
            elif converted_noise_data == np.int32:
                converted_noise_data = converted_noise_data.astype(np.float32) / 2147483648.0
        except Exception as e:
            raise Exception(f"Error loading audio file: {e}")
             
    def compute_noise_spectrum(self, noise_data, nperseg=8192, overlap=0.75, window='hann', detrend='constant'):
        #using Welch's method
        noverlap = int(nperseg * overlap)

        frequencies, psd = signal.welch(
            noise_data,
            fs = self.sample_rate,
            window = window,
            nperseg = nperseg,
            noverlap = noverlap,
            detrend = detrend, 
            scaling = 'density'
        )

        #store file for background subtraction
        self.frequencies = frequencies
        self.noise_spectrum = psd

        return frequencies, psd
    
    def get_noise_floor_db(self):
        if self.noise_spectrum is None:
            raise ValueError("compute noise spectrum first")
        noise_db = 10 * np.log10(self.noise_spectrum + 1e-12)
        return noise_db

    def get_US_band_noise(self, freq_min=15000, freq_max=120000):
        if self.noise_spectrum is None:
            raise ValueError("Compute noise spectrum first")

        freq_mask = (self.frequencies >= freq_min) & (self.frequencies <= freq_max)   
        band_freqs = self.frequencies[freq_mask]
        band_noise = self.noise_spectrum[freq_mask]

        stats = {
            'freq_range' : (freq_min, freq_max),
            'mean_noise_linear' : np.mean(band_noise),
            'mean_noise_db': 10 * np.log10(np.mean(band_noise) + 1e-12),
            'median_noise_db' : 10 * np.log10(np.median(band_noise) + 1e-12),
            'std_noise_db' : np.std(10 * np.log10(band_noise + 1e-12)),
            'frequencies' : band_freqs,
            'noise_psd' : band_noise
        }
        return stats

    def save_noise_spectrum(self):
        """
        Save computed noise spectrum to file
        """
        if self.noise_spectrum is None:
            raise ValueError("Compute noise spectrum first")
        
        np.savez(str(self.noise_file),
                 frequencies=self.frequencies,
                 noise_psd=self.noise_spectrum,
                 sampling_rate=self.sample_rate)
        print(f"Noise spectrum saved to {self.noise_file}")
    
    def load_noise_spectrum(self, filepath):
        """
        Load previously computed noise spectrum
        
        Parameters:
        filepath (str): Input file path (NPZ format)
        """
        data = np.load(filepath)
        self.frequencies = data['frequencies']
        self.noise_spectrum = data['noise_psd']
        self.sample_rate = int(data['sampling_rate'])
        print(f"Noise spectrum loaded from {filepath}")
    
    def subtract_noise_from_signal(self, signal_psd, method='simple', 
                                 alpha=2.0, beta=0.01):
        """
        Perform spectral subtraction for noise reduction
        
        Parameters:
        signal_psd (array): Power spectral density of signal + noise
        method (str): 'simple' or 'wiener'
        alpha (float): Over-subtraction factor
        beta (float): Spectral floor factor
        
        Returns:
        array: Noise-reduced PSD
        """
        if self.noise_spectrum is None:
            raise ValueError("Compute noise spectrum first")
        
        if method == 'simple':
            # Simple spectral subtraction
            clean_psd = signal_psd - alpha * self.noise_spectrum
            # Apply spectral floor
            clean_psd = np.maximum(clean_psd, beta * signal_psd)
            
        elif method == 'wiener':
            # Wiener filter approach
            snr_est = signal_psd / (self.noise_spectrum + 1e-12)
            wiener_gain = snr_est / (1 + snr_est)
            clean_psd = wiener_gain * signal_psd
            
        else:
            raise ValueError("Method must be 'simple' or 'wiener'")
        
        return clean_psd
        
           
    
    def bandpass_filter(self, signal, sample_rate, low_freq=18000, high_freq=100000, order=5):
        """
        Apply bandpass filter to signal
        
        Args:
            signal: Input signal
            sample_rate: Sample rate
            low_freq: Low cutoff frequency (Hz)
            high_freq: High cutoff frequency (Hz)
            order: Filter order
        
        Returns:
            Filtered signal
        """
        # Check if frequencies are within Nyquist limit
        nyquist = sample_rate / 2
        if high_freq >= nyquist:
            high_freq = nyquist * 0.95  # Set to 95% of Nyquist
            print(f"Warning: High frequency adjusted to {high_freq:.0f} Hz (Nyquist limit)")
        
        if low_freq >= nyquist:
            print(f"Error: Low frequency {low_freq} Hz exceeds Nyquist frequency {nyquist} Hz")
            return signal
        
        # Normalize frequencies
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Design Butterworth bandpass filter
        sos = butter(order, [low_norm, high_norm], btype='band', output='sos')
        
        # Apply filter
        filtered_signal = sosfilt(sos, signal)
        return filtered_signal
    
       
        ##set up for current code above.
    def process_directory(self, 
        input_dir: str,
        output_dir: str,
        noise_profile_path: str
        ):

        """
        Complete processing pipeline for a directory.

        Args:
        input_dir: Directory containing WAV files
        output_dir: Directory for output files
        preset: Filter preset to apply
        gap_seconds: Gap between consolidated files
        noise_profile_path: Path to noise profile file for noise reduction

        Returns:
            True if successful, False otherwise
        """
    
        # Create output directory if it doesn't exist

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Find all WAV files
        wav_files = self.find_wave_files(input_dir)

        if not wav_files:
            self.console.print("[red]No WAV files found in directory[/red]")
            return False

        # Display found files
        self.display_found_files(wav_files)

        # Consolidate files
        consolidated_path = Path (output_dir) / "consolidated.wav"
        if not self.consolidate_audio_files(wav_files, consolidated_path):
            return False

        # Apply filters
        filtered_path = Path(output_dir, f"filtered_.wav")
        if noise_profile_path:
            if not self.bandpass_filter(consolidated_path, noise_profile_path):
                return False
            
    def display_found_files(self, wav_files: List[str]):
        """Display a table of found WAV files."""
        table = Table(title="Found WAV Files")
        table.add_column("Index", style="cyan", no_wrap=True)
        table.add_column("File Name", style="magenta")
        table.add_column("Directory", style="green")

        for i, filepath in enumerate(wav_files, 1):
            filename = os.path.basename(filepath)
            directory = os.path.dirname(filepath)
            table.add_row(str(i), filename, directory)

        self.console.print(table)


        self.console.print("[green]Processing completed successfully![/green]")
        return True




      