"""
Figure configuration module implementing FIGURE_PROTOCOL.md standards.

This module provides matplotlib configuration and color schemes for
publication-quality, high-density journal figures.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Blackbody-inspired colormap for abundance/expression data
BLACKBODY_COLORS = [
    (1.0, 1.0, 1.0),     # White for NaN/zero
    (0.95, 0.95, 0.95),  # Very light gray
    (0.1, 0.1, 0.1),     # Near black
    (0.4, 0.0, 0.0),     # Dark red
    (0.7, 0.0, 0.0),     # Red
    (0.9, 0.2, 0.0),     # Orange-red
    (1.0, 0.5, 0.0),     # Orange
    (1.0, 0.7, 0.0),     # Yellow-orange
    (1.0, 0.9, 0.2),     # Yellow
]

# Blue-white-red diverging colormap for correlations
DIVERGING_COLORS = [
    (0.0, 0.3, 0.7),     # Deep blue (negative)
    (0.3, 0.5, 0.9),     # Medium blue
    (0.7, 0.8, 0.95),    # Light blue
    (0.95, 0.95, 0.95),  # Near white (zero)
    (0.95, 0.8, 0.7),    # Light red
    (0.9, 0.4, 0.3),     # Medium red
    (0.7, 0.1, 0.1),     # Deep red (positive)
]

def create_blackbody_colormap(name='blackbody'):
    """Create blackbody-inspired colormap."""
    return LinearSegmentedColormap.from_list(name, BLACKBODY_COLORS)

def create_diverging_colormap(name='diverging'):
    """Create blue-white-red diverging colormap."""
    return LinearSegmentedColormap.from_list(name, DIVERGING_COLORS)

def setup_publication_style():
    """
    Configure matplotlib for publication-quality figures.

    Implements all standards from FIGURE_PROTOCOL.md:
    - Font sizes ≤6pt
    - TrueType font embedding (fonttype 42)
    - Arial/Helvetica sans-serif fonts
    - Minimal line weights and padding
    - Tight layouts
    """

    # CRITICAL: Journal compatibility - TrueType fonts for Illustrator
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'

    # Font configuration
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['font.size'] = 5
    mpl.rcParams['axes.labelsize'] = 5
    mpl.rcParams['axes.titlesize'] = 5
    mpl.rcParams['xtick.labelsize'] = 4
    mpl.rcParams['ytick.labelsize'] = 4
    mpl.rcParams['legend.fontsize'] = 4

    # Line weights
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.width'] = 0.5
    mpl.rcParams['ytick.major.width'] = 0.5
    mpl.rcParams['xtick.major.size'] = 2
    mpl.rcParams['ytick.major.size'] = 2
    mpl.rcParams['lines.linewidth'] = 0.5

    # Minimal padding
    mpl.rcParams['axes.labelpad'] = 1
    mpl.rcParams['xtick.major.pad'] = 1
    mpl.rcParams['ytick.major.pad'] = 1

    # Figure aesthetics
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['savefig.facecolor'] = 'white'
    mpl.rcParams['savefig.edgecolor'] = 'none'

    # High quality
    mpl.rcParams['figure.dpi'] = 150  # For display
    mpl.rcParams['savefig.dpi'] = 600  # For print

    # Tight layout
    mpl.rcParams['figure.autolayout'] = False  # We'll use tight_layout manually
    mpl.rcParams['figure.constrained_layout.use'] = False

    print("✓ Publication style configured")
    print("  - Font: Arial/Helvetica, 5pt base")
    print("  - PDF fonttype: 42 (TrueType)")
    print("  - Line width: 0.5pt")
    print("  - DPI: 600 (print)")

def save_publication_figure(fig, filepath, formats=['pdf', 'svg', 'png']):
    """
    Save figure in multiple formats with proper settings.

    Args:
        fig: matplotlib Figure object
        filepath: Base filepath without extension
        formats: List of formats to save (default: ['pdf', 'svg', 'png'])
    """
    for fmt in formats:
        filename = f"{filepath}.{fmt}"
        if fmt == 'png':
            fig.savefig(filename,
                       dpi=600,
                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none',
                       pad_inches=0.02)
        else:
            fig.savefig(filename,
                       format=fmt,
                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none',
                       pad_inches=0.02)
        print(f"  ✓ Saved: {filename}")

def create_figure(width_inches=7.0, height_inches=5.0, dpi=150):
    """
    Create a figure with publication standards.

    Args:
        width_inches: Figure width (default 7.0 for double column)
        height_inches: Figure height
        dpi: Display DPI (default 150)

    Returns:
        fig, ax: Figure and axes objects
    """
    fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
    ax = fig.add_subplot(111)
    return fig, ax

def create_multi_panel_figure(nrows, ncols, width_inches=7.0, height_inches=None,
                              hspace=0.3, wspace=0.3, dpi=150):
    """
    Create multi-panel figure with publication standards.

    Args:
        nrows: Number of rows
        ncols: Number of columns
        width_inches: Figure width
        height_inches: Figure height (auto-calculated if None)
        hspace: Height space between subplots (default 0.3)
        wspace: Width space between subplots (default 0.3)
        dpi: Display DPI

    Returns:
        fig, axes: Figure and axes array
    """
    if height_inches is None:
        height_inches = width_inches * (nrows / ncols) * 0.75

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_inches, height_inches),
                             dpi=dpi)
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    return fig, axes

# Color utilities
def get_nan_color():
    """Get standard color for NaN/missing values (white)."""
    return (1.0, 1.0, 1.0)

def validate_figure(fig):
    """
    Validate figure against FIGURE_PROTOCOL.md standards.

    Args:
        fig: matplotlib Figure object

    Returns:
        dict: Validation results
    """
    results = {
        'pdf_fonttype': mpl.rcParams['pdf.fonttype'] == 42,
        'base_fontsize': mpl.rcParams['font.size'] <= 6,
        'linewidth': mpl.rcParams['axes.linewidth'] <= 0.5,
        'white_background': fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0) or
                           fig.get_facecolor() == 'white'
    }

    results['all_passed'] = all(results.values())

    return results

if __name__ == '__main__':
    # Test the configuration
    setup_publication_style()

    # Create test figure
    fig, ax = create_figure(width_inches=3.5, height_inches=2.5)

    # Test data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    ax.plot(x, y, label='Test plot')
    ax.set_xlabel('X axis (Log2 scale)')
    ax.set_ylabel('Y axis')
    ax.set_title('Test Figure')
    ax.legend()
    ax.grid(True, alpha=0.2, linewidth=0.3)

    # Validate
    validation = validate_figure(fig)
    print("\nValidation results:")
    for key, value in validation.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")

    plt.tight_layout()
    plt.show()
