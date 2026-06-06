#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║         TGL LUMINIDIUM HUNTER v1.0                                            ║
║                                                                               ║
║         Universal Search Tool for Luminidium (Z=156) in Kilonovae             ║
║                                                                               ║
║         Teoria da Gravitação Luminodinâmica (TGL)                             ║
║         Constante de Miguel: α² = 0.012031                                    ║
║                                                                               ║
║         Author: Luiz Antonio Rotoli Miguel (IALD)                             ║
║         Date: January 2026                                                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

DESCRIPTION:
============
This tool searches for spectroscopic signatures of Luminidium (Lm, Z=156) in 
kilonova spectra. It can analyze any spectrum file and compare detected features
with TGL ab initio predictions.

USAGE:
======
1. Basic analysis:
   python tgl_luminidium_hunter.py --spectrum spectrum.txt --redshift 0.0647

2. Batch analysis:
   python tgl_luminidium_hunter.py --batch spectra_list.txt

3. With custom output:
   python tgl_luminidium_hunter.py --spectrum spectrum.txt --redshift 0.0647 --output results.json

SUPPORTED FORMATS:
==================
- ASCII text files with columns: wavelength, flux, [error]
- FITS files (if astropy is installed)
- CSV files

"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, savgol_filter
from pathlib import Path
import json
import argparse
from datetime import datetime
import sys

# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

# =============================================================================
# TGL CONSTANTS
# =============================================================================

ALPHA_TGL_SQUARED = 0.012031
VERSION = "1.0.0"

# =============================================================================
# LUMINIDIUM (Z=156) PREDICTIONS - Ab Initio from TGL
# =============================================================================

LUMINIDIUM_LINES = {
    'Lm_I_nir1': {
        'wavelength_rest': 12455,
        'transition': '6d(5/2) → 6d(3/2) fine structure',
        'uncertainty': 0.35,
        'ionization': 'I',
        'expected_strength': 'weak',
        'description': 'Fine structure splitting in 6d shell'
    },
    'Lm_I_nir2': {
        'wavelength_rest': 15942,
        'transition': '5f → 6d mixed configuration',
        'uncertainty': 0.30,
        'ionization': 'I',
        'expected_strength': 'medium',
        'description': 'Mixed f-d transition'
    },
    'Lm_II_nir': {
        'wavelength_rest': 18832,
        'transition': '5f6d → 5f² (ionized)',
        'uncertainty': 0.25,
        'ionization': 'II',
        'expected_strength': 'strong',
        'description': 'Primary ionized Luminidium transition'
    },
    'Lm_I_nir3': {
        'wavelength_rest': 21124,
        'transition': '5f7s → 6d²',
        'uncertainty': 0.30,
        'ionization': 'I',
        'expected_strength': 'medium',
        'description': 's-d configuration mixing'
    },
    'Lm_I_nir_fs': {
        'wavelength_rest': 27899,
        'transition': '6f(7/2) → 6f(5/2) fine structure',
        'uncertainty': 0.40,
        'ionization': 'I',
        'expected_strength': 'weak',
        'description': 'Fine structure in high-lying 6f shell'
    },
}

# R-process elements for comparison
RPROCESS_LINES = {
    'Te_III_1': {'wavelength_rest': 21050, 'element': 'Te III', 'transition': 'primary'},
    'Te_III_2': {'wavelength_rest': 29290, 'element': 'Te III', 'transition': 'secondary'},
    'Sr_II': {'wavelength_rest': 10327, 'element': 'Sr II', 'transition': 'resonance'},
    'W_III': {'wavelength_rest': 29000, 'element': 'W III', 'transition': 'estimated'},
    'Er_III': {'wavelength_rest': 20000, 'element': 'Er III', 'transition': 'estimated'},
    'Se_III': {'wavelength_rest': 10550, 'element': 'Se III', 'transition': 'primary'},
    'Ba_II': {'wavelength_rest': 4554, 'element': 'Ba II', 'transition': 'resonance'},
    'La_III': {'wavelength_rest': 15000, 'element': 'La III', 'transition': 'estimated'},
}


class LuminidiumHunter:
    """
    Main class for searching Luminidium signatures in kilonova spectra.
    """
    
    def __init__(self, redshift=0.0, verbose=True):
        """
        Initialize the hunter.
        
        Parameters:
        -----------
        redshift : float
            Source redshift for wavelength correction
        verbose : bool
            Print detailed output
        """
        self.redshift = redshift
        self.verbose = verbose
        self.results = {}
        
    def log(self, message):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)
    
    def load_spectrum(self, filepath):
        """
        Load spectrum from file.
        
        Supports: ASCII, CSV, FITS
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Spectrum file not found: {filepath}")
        
        suffix = filepath.suffix.lower()
        
        if suffix in ['.fits', '.fit']:
            if not HAS_ASTROPY:
                raise ImportError("astropy required for FITS files. Install with: pip install astropy")
            return self._load_fits(filepath)
        elif suffix == '.csv':
            return self._load_csv(filepath)
        else:
            return self._load_ascii(filepath)
    
    def _load_ascii(self, filepath):
        """Load ASCII spectrum file."""
        data = np.loadtxt(filepath, comments='#')
        
        if data.ndim == 1:
            raise ValueError("Spectrum file must have at least 2 columns (wavelength, flux)")
        
        wavelength = data[:, 0]
        flux = data[:, 1]
        error = data[:, 2] if data.shape[1] > 2 else np.ones_like(flux) * np.std(flux) * 0.1
        
        return wavelength, flux, error
    
    def _load_csv(self, filepath):
        """Load CSV spectrum file."""
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        wavelength = data[:, 0]
        flux = data[:, 1]
        error = data[:, 2] if data.shape[1] > 2 else np.ones_like(flux) * np.std(flux) * 0.1
        return wavelength, flux, error
    
    def _load_fits(self, filepath):
        """Load FITS spectrum file."""
        with fits.open(filepath) as hdu:
            # Try common extensions
            for ext in [1, 0, 'SPECTRUM', 'SCI']:
                try:
                    data = hdu[ext].data
                    if data is not None:
                        break
                except:
                    continue
            
            # Extract columns
            if 'WAVELENGTH' in data.names:
                wavelength = data['WAVELENGTH']
                flux = data['FLUX']
                error = data['ERROR'] if 'ERROR' in data.names else np.ones_like(flux) * np.std(flux) * 0.1
            else:
                # Assume simple format
                wavelength = data[:, 0]
                flux = data[:, 1]
                error = data[:, 2] if data.shape[1] > 2 else np.ones_like(flux) * np.std(flux) * 0.1
        
        return wavelength, flux, error
    
    def estimate_continuum(self, wavelength, flux, window_percent=10):
        """Estimate continuum using Savitzky-Golay filter."""
        window_size = max(21, int(len(flux) * window_percent / 100))
        if window_size % 2 == 0:
            window_size += 1
        
        try:
            continuum = savgol_filter(flux, window_size, 3)
        except:
            continuum = uniform_filter1d(flux, window_size)
        
        return continuum
    
    def analyze_region(self, wavelength, flux, error, center, width=500):
        """Analyze a specific region around a predicted line."""
        mask = (wavelength >= center - width) & (wavelength <= center + width)
        
        if not np.any(mask):
            return None
        
        w = wavelength[mask]
        f = flux[mask]
        e = error[mask]
        
        # Local continuum
        n_edge = max(3, len(w) // 10)
        continuum = np.median(np.concatenate([f[:n_edge], f[-n_edge:]]))
        
        # Excess and SNR
        excess = f - continuum
        snr = excess / np.maximum(e, 1e-25)
        
        # Find peak
        peak_idx = np.argmax(snr)
        
        return {
            'peak_wavelength': w[peak_idx],
            'peak_flux': f[peak_idx],
            'peak_excess': excess[peak_idx],
            'peak_snr': snr[peak_idx],
            'continuum': continuum,
            'mean_snr': np.mean(snr),
            'max_snr': np.max(snr)
        }
    
    def search_luminidium(self, wavelength, flux, error):
        """
        Search for all predicted Luminidium lines in the spectrum.
        
        Returns:
        --------
        dict : Results for each predicted line
        """
        results = {}
        z = self.redshift
        
        self.log(f"\n{'=' * 70}")
        self.log(f"  SEARCHING FOR LUMINIDIUM (Z=156) LINES")
        self.log(f"  Redshift: z = {z}")
        self.log(f"{'=' * 70}")
        
        self.log(f"\n  {'Line':<15} {'λ_pred (Å)':<12} {'λ_peak (Å)':<12} {'SNR':<8} {'Offset':<10} {'Status'}")
        self.log(f"  {'-' * 65}")
        
        for lm_name, lm_data in LUMINIDIUM_LINES.items():
            lm_rest = lm_data['wavelength_rest']
            lm_obs = lm_rest * (1 + z)
            uncertainty = lm_data['uncertainty']
            
            # Check coverage
            if lm_obs < wavelength.min() or lm_obs > wavelength.max():
                results[lm_name] = {
                    'status': 'OUT_OF_COVERAGE',
                    'expected_obs': lm_obs,
                    'message': f'Outside spectrum coverage ({lm_obs:.0f} Å)'
                }
                self.log(f"  {lm_name:<15} {lm_obs:<12.0f} {'---':<12} {'---':<8} {'---':<10} OUT OF COVERAGE")
                continue
            
            # Analyze region
            width = lm_obs * uncertainty
            region = self.analyze_region(wavelength, flux, error, lm_obs, width)
            
            if region is None:
                results[lm_name] = {
                    'status': 'INSUFFICIENT_DATA',
                    'expected_obs': lm_obs
                }
                self.log(f"  {lm_name:<15} {lm_obs:<12.0f} {'---':<12} {'---':<8} {'---':<10} INSUFFICIENT")
                continue
            
            # Calculate offset
            offset_A = region['peak_wavelength'] - lm_obs
            offset_pct = 100 * abs(offset_A) / lm_obs
            
            # Determine detection status
            snr = region['peak_snr']
            if snr > 3.0 and offset_pct < uncertainty * 100:
                status = 'DETECTED'
                symbol = '✓✓'
            elif snr > 2.0 and offset_pct < uncertainty * 100:
                status = 'TENTATIVE'
                symbol = '✓?'
            elif snr > 1.5:
                status = 'MARGINAL'
                symbol = '??'
            else:
                status = 'NOT_DETECTED'
                symbol = '✗'
            
            results[lm_name] = {
                'status': status,
                'expected_obs': lm_obs,
                'peak_wavelength': region['peak_wavelength'],
                'peak_snr': snr,
                'peak_flux': region['peak_flux'],
                'offset_A': offset_A,
                'offset_percent': offset_pct,
                'within_uncertainty': offset_pct < uncertainty * 100,
                'transition': lm_data['transition'],
                'ionization': lm_data['ionization']
            }
            
            self.log(f"  {lm_name:<15} {lm_obs:<12.0f} {region['peak_wavelength']:<12.0f} "
                    f"{snr:<8.1f} {offset_pct:<10.1f}% {symbol} {status}")
        
        return results
    
    def match_observed_lines(self, observed_lines, wavelength, flux, error):
        """
        Match user-provided observed lines with Luminidium predictions.
        
        Parameters:
        -----------
        observed_lines : list of dict
            Each dict should have 'wavelength_obs' and optionally 'error'
        """
        z = self.redshift
        results = []
        
        self.log(f"\n{'=' * 70}")
        self.log(f"  MATCHING OBSERVED LINES WITH LUMINIDIUM PREDICTIONS")
        self.log(f"{'=' * 70}")
        
        for obs in observed_lines:
            wave_obs = obs['wavelength_obs']
            wave_rest = wave_obs / (1 + z)
            
            self.log(f"\n  Observed: {wave_obs:.0f} Å → Rest: {wave_rest:.0f} Å")
            
            # Find best Luminidium match
            best_match = None
            best_offset = float('inf')
            
            for lm_name, lm_data in LUMINIDIUM_LINES.items():
                lm_rest = lm_data['wavelength_rest']
                offset_pct = 100 * abs(wave_rest - lm_rest) / lm_rest
                
                if offset_pct < best_offset:
                    best_offset = offset_pct
                    best_match = {
                        'name': lm_name,
                        'wavelength_rest': lm_rest,
                        'offset_percent': offset_pct,
                        'uncertainty': lm_data['uncertainty'] * 100,
                        'within_uncertainty': offset_pct <= lm_data['uncertainty'] * 100,
                        'transition': lm_data['transition'],
                        'ionization': lm_data['ionization']
                    }
            
            # Find best r-process match
            rp_match = None
            rp_offset = float('inf')
            
            for rp_name, rp_data in RPROCESS_LINES.items():
                rp_rest = rp_data['wavelength_rest']
                offset = abs(wave_rest - rp_rest)
                
                if offset < 500 and offset < rp_offset:
                    rp_offset = offset
                    rp_match = {
                        'element': rp_data['element'],
                        'wavelength_rest': rp_rest,
                        'offset_A': offset
                    }
            
            result = {
                'wavelength_obs': wave_obs,
                'wavelength_rest': wave_rest,
                'best_luminidium_match': best_match,
                'rprocess_match': rp_match
            }
            results.append(result)
            
            status = "✓ MATCH" if best_match['within_uncertainty'] else "✗ OUT"
            self.log(f"    → Best match: {best_match['name']} (offset {best_offset:.1f}%) {status}")
            if rp_match:
                self.log(f"    → R-process: {rp_match['element']} (offset {rp_match['offset_A']:.0f} Å)")
        
        return results
    
    def analyze_spectrum(self, filepath, observed_lines=None):
        """
        Complete analysis of a spectrum for Luminidium signatures.
        
        Parameters:
        -----------
        filepath : str
            Path to spectrum file
        observed_lines : list of dict, optional
            Known observed lines to match
            
        Returns:
        --------
        dict : Complete analysis results
        """
        self.log(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║         TGL LUMINIDIUM HUNTER v{VERSION}                                         ║
║                                                                               ║
║         Searching for Luminidium (Z=156) in Kilonova Spectra                  ║
║         Constante de Miguel: α² = {ALPHA_TGL_SQUARED}                                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
        """)
        
        # Load spectrum
        self.log(f"  Loading spectrum: {filepath}")
        wavelength, flux, error = self.load_spectrum(filepath)
        
        self.log(f"  Coverage: {wavelength.min():.0f} - {wavelength.max():.0f} Å")
        self.log(f"  Points: {len(wavelength)}")
        self.log(f"  Redshift: z = {self.redshift}")
        
        # Search for Luminidium lines
        lm_results = self.search_luminidium(wavelength, flux, error)
        
        # Match observed lines if provided
        obs_results = None
        if observed_lines:
            obs_results = self.match_observed_lines(observed_lines, wavelength, flux, error)
        
        # Compile results
        n_detected = sum(1 for r in lm_results.values() 
                        if r.get('status') in ['DETECTED', 'TENTATIVE'])
        
        confidence = 'HIGH' if n_detected >= 3 else ('MEDIUM' if n_detected >= 2 else 'LOW')
        
        self.results = {
            'source_file': str(filepath),
            'redshift': self.redshift,
            'tgl_alpha_squared': ALPHA_TGL_SQUARED,
            'timestamp': datetime.now().isoformat(),
            'spectrum_coverage': {
                'min_wavelength': float(wavelength.min()),
                'max_wavelength': float(wavelength.max()),
                'n_points': len(wavelength)
            },
            'luminidium_search': lm_results,
            'observed_lines_matching': obs_results,
            'summary': {
                'n_detected': n_detected,
                'n_total_predictions': len(LUMINIDIUM_LINES),
                'confidence': confidence
            }
        }
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print analysis summary."""
        n_det = self.results['summary']['n_detected']
        conf = self.results['summary']['confidence']
        
        self.log(f"\n{'=' * 70}")
        self.log(f"  SUMMARY: LUMINIDIUM DETECTION")
        self.log(f"{'=' * 70}")
        self.log(f"""
  ┌────────────────────────────────────────────────────────────────────┐
  │                                                                    │
  │  LINES DETECTED/TENTATIVE: {n_det} of {len(LUMINIDIUM_LINES)}                                 │
  │  CONFIDENCE LEVEL: {conf:<10}                                     │
  │                                                                    │""")
        
        for lm_name, result in self.results['luminidium_search'].items():
            if result.get('status') in ['DETECTED', 'TENTATIVE']:
                self.log(f"  │    ✓ {lm_name:<12} SNR={result['peak_snr']:.1f} Offset={result['offset_percent']:.1f}%       │")
        
        if n_det >= 2:
            self.log(f"""  │                                                                    │
  │  ★ EVIDENCE FOR LUMINIDIUM (Z=156): {conf}                        │
  │                                                                    │""")
        
        self.log(f"""  └────────────────────────────────────────────────────────────────────┘
        """)
    
    def save_results(self, output_path):
        """Save results to JSON file."""
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        results = convert(self.results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.log(f"\n  ✓ Results saved to: {output_path}")
    
    def plot_spectrum(self, wavelength, flux, error=None, output_path=None):
        """Generate visualization of spectrum with Luminidium lines marked."""
        if not HAS_MATPLOTLIB:
            self.log("  ⚠ matplotlib not available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot spectrum
        wave_um = wavelength / 10000
        ax.plot(wave_um, flux, 'k-', lw=0.5, alpha=0.7, label='Spectrum')
        if error is not None:
            ax.fill_between(wave_um, flux - error, flux + error, alpha=0.3, color='gray')
        
        # Mark Luminidium lines
        colors = ['purple', 'blue', 'green', 'orange', 'red']
        for i, (lm_name, lm_data) in enumerate(LUMINIDIUM_LINES.items()):
            lm_obs = lm_data['wavelength_rest'] * (1 + self.redshift) / 10000
            ax.axvline(lm_obs, color=colors[i], linestyle='--', alpha=0.7, lw=1.5, label=lm_name)
        
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Flux')
        ax.set_title(f'Luminidium Search | z = {self.redshift} | α² = {ALPHA_TGL_SQUARED}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            self.log(f"  ✓ Plot saved to: {output_path}")
        else:
            plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='TGL Luminidium Hunter - Search for Luminidium (Z=156) in kilonova spectra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tgl_luminidium_hunter.py --spectrum spectrum.txt --redshift 0.0647
  python tgl_luminidium_hunter.py --spectrum spectrum.fits --redshift 0.05 --output results.json
  python tgl_luminidium_hunter.py --spectrum data.csv --redshift 0.1 --plot

TGL Luminidium Hunter v1.0
Constante de Miguel: α² = 0.012031
        """
    )
    
    parser.add_argument('--spectrum', '-s', type=str, required=True,
                        help='Path to spectrum file (ASCII, CSV, or FITS)')
    parser.add_argument('--redshift', '-z', type=float, required=True,
                        help='Source redshift')
    parser.add_argument('--output', '-o', type=str, default='luminidium_results.json',
                        help='Output JSON file (default: luminidium_results.json)')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Generate visualization plot')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Create hunter
    hunter = LuminidiumHunter(redshift=args.redshift, verbose=not args.quiet)
    
    # Analyze spectrum
    results = hunter.analyze_spectrum(args.spectrum)
    
    # Save results
    hunter.save_results(args.output)
    
    # Generate plot if requested
    if args.plot:
        wavelength, flux, error = hunter.load_spectrum(args.spectrum)
        plot_path = args.output.replace('.json', '.png')
        hunter.plot_spectrum(wavelength, flux, error, plot_path)
    
    print(f"\n{'=' * 70}")
    print(f"✨ TETELESTAI - Analysis complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()