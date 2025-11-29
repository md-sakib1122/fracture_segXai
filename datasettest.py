"""
FracAtlas Fracture Pixel Distribution Analysis
Check how many fractures fall into each pixel range
"""

import os
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def analyze_fracture_distribution(data_root):
    """
    Analyze fracture pixel distribution across all splits
    Categorize by pixel ranges: <500, 500-1500, 1500-2000, 2000+
    """

    splits = ['train', 'val', 'test']

    # Define ranges
    ranges = [
        (0, 500, "< 500px (Tiny)"),
        (500, 1500, "500-1500px (Small)"),
        (1500, 2000, "1500-2000px (Medium)"),
        (2000, float('inf'), "> 2000px (Large)")
    ]

    print("=" * 80)
    print("FRACTURE PIXEL DISTRIBUTION ANALYSIS")
    print("=" * 80)

    all_results = {}

    for split in splits:
        mask_dir = os.path.join(data_root, 'FracAtlas', split, 'masks')

        if not os.path.exists(mask_dir):
            print(f"⚠️  {split.upper()} mask directory not found: {mask_dir}")
            continue

        print(f"\n{'=' * 80}")
        print(f"{split.upper()} SET ANALYSIS")
        print(f"{'=' * 80}")

        # Initialize counters
        range_counts = defaultdict(int)
        pixel_values = []
        fracture_samples = []
        non_fracture_samples = 0

        # Scan all masks
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('_mask.png')])

        print(f"Scanning {len(mask_files)} masks...")

        for mask_file in mask_files:
            mask_path = os.path.join(mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"  ⚠️  Could not read: {mask_file}")
                continue

            # Binarize and count fracture pixels
            binary_mask = (mask > 127).astype(np.uint8)
            fracture_pixels = binary_mask.sum()

            pixel_values.append(fracture_pixels)

            if fracture_pixels == 0:
                non_fracture_samples += 1
            else:
                fracture_samples.append((mask_file, fracture_pixels))

                # Categorize into ranges
                for min_px, max_px, label in ranges:
                    if min_px <= fracture_pixels < max_px:
                        range_counts[label] += 1
                        break

        # Print results
        print(f"\n{'CATEGORY':<30} {'COUNT':<10} {'PERCENTAGE':<12} {'VISUALIZATION'}")
        print("-" * 80)

        # Non-fractured
        pct = (non_fracture_samples / len(mask_files) * 100) if len(mask_files) > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"{'No Fracture':<30} {non_fracture_samples:<10} {pct:>6.2f}%      {bar}")

        # Fractured by range
        total_fractured = sum(range_counts.values())
        for min_px, max_px, label in ranges:
            count = range_counts[label]
            pct = (count / len(mask_files) * 100) if len(mask_files) > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"{label:<30} {count:<10} {pct:>6.2f}%      {bar}")

        # Statistics
        print(f"\n{'STATISTICS':<30}")
        print("-" * 80)
        print(f"{'Total masks':<30} {len(mask_files)}")
        print(f"{'With fractures':<30} {total_fractured} ({total_fractured / len(mask_files) * 100:.2f}%)")
        print(f"{'Without fractures':<30} {non_fracture_samples}")

        pixel_values_array = np.array(pixel_values)
        print(f"\n{'Pixel count statistics':<30}")
        print("-" * 80)
        print(f"{'Minimum':<30} {pixel_values_array.min()}")
        print(f"{'Maximum':<30} {pixel_values_array.max()}")
        print(f"{'Mean':<30} {pixel_values_array.mean():.0f}")
        print(f"{'Median':<30} {np.median(pixel_values_array):.0f}")
        print(f"{'Std Dev':<30} {pixel_values_array.std():.0f}")

        # Percentiles
        print(f"\n{'PERCENTILES':<30}")
        print("-" * 80)
        for p in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(pixel_values_array, p)
            print(f"{p}th percentile:<30 {val:.0f}")

        # Examples from each range
        print(f"\n{'EXAMPLE FRACTURES FROM EACH RANGE':<30}")
        print("-" * 80)
        for min_px, max_px, label in ranges:
            examples = [s for s in fracture_samples if min_px <= s[1] < max_px]
            if examples:
                print(f"\n{label}:")
                # Show first 3 examples
                for fname, pixels in sorted(examples, key=lambda x: x[1])[:3]:
                    print(f"  - {fname:<40} {pixels:>6} pixels")

        all_results[split] = {
            'range_counts': dict(range_counts),
            'pixel_values': pixel_values_array,
            'total': len(mask_files),
            'with_fractures': total_fractured,
            'without_fractures': non_fracture_samples
        }

    return all_results


def create_visualization(results):
    """Create visualizations of the distribution"""

    if not results:
        print("No results to visualize")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fracture Pixel Distribution Analysis', fontsize=16, fontweight='bold')

    splits = list(results.keys())

    # 1. Bar chart: Distribution across ranges
    ax = axes[0, 0]
    ranges_labels = ["< 500px", "500-1500px", "1500-2000px", "> 2000px"]

    for split in splits:
        counts = results[split]['range_counts']
        values = [counts.get(f"{min_px}-{max_px}px", 0)
                  for min_px, max_px, _ in [(0, 500), (500, 1500), (1500, 2000), (2000, float('inf'))]]
        ax.plot(ranges_labels, values, marker='o', label=split.upper(), linewidth=2)

    ax.set_xlabel('Fracture Size Range', fontsize=11)
    ax.set_ylabel('Number of Samples', fontsize=11)
    ax.set_title('Fracture Distribution by Range', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Histogram: Pixel value distribution (all splits combined)
    ax = axes[0, 1]
    all_pixels = np.concatenate([results[split]['pixel_values'] for split in splits])
    ax.hist(all_pixels, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Fracture Pixels', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Histogram: Fracture Pixel Distribution (All Sets)', fontsize=12, fontweight='bold')
    ax.set_xlim(left=0)

    # Add range markers
    for val in [500, 1500, 2000]:
        ax.axvline(val, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # 3. Pie chart: With vs Without fractures (train only)
    ax = axes[1, 0]
    if 'train' in results:
        train_data = results['train']
        sizes = [train_data['without_fractures'], train_data['with_fractures']]
        labels = [f"No Fracture\n({sizes[0]})", f"With Fracture\n({sizes[1]})"]
        colors = ['#ff9999', '#66b3ff']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Train Set: Fracture Distribution', fontsize=12, fontweight='bold')

    # 4. Box plot: Pixel distribution by split
    ax = axes[1, 1]
    data_to_plot = [results[split]['pixel_values'] for split in splits]
    bp = ax.boxplot(data_to_plot, labels=[s.upper() for s in splits], patch_artist=True)

    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Fracture Pixels', fontsize=11)
    ax.set_title('Fracture Pixel Distribution by Split', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add range markers
    for val in [500, 1500, 2000]:
        ax.axhline(val, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig('fracture_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'fracture_distribution.png'")
    plt.show()


def print_summary_table(results):
    """Print a comprehensive summary table"""

    print("\n" + "=" * 100)
    print("SUMMARY TABLE: ALL SPLITS")
    print("=" * 100)

    print(
        f"\n{'Split':<10} {'Total':<10} {'No Frac':<12} {'<500px':<12} {'500-1500':<12} {'1500-2000':<12} {'>2000px':<12}")
    print("-" * 100)

    for split in ['train', 'val', 'test']:
        if split not in results:
            continue

        data = results[split]
        total = data['total']
        no_frac = data['without_fractures']

        # Get range counts
        range_counts = data['range_counts']
        r1 = range_counts.get('< 500px (Tiny)', 0)
        r2 = range_counts.get('500-1500px (Small)', 0)
        r3 = range_counts.get('1500-2000px (Medium)', 0)
        r4 = range_counts.get('> 2000px (Large)', 0)

        print(f"{split.upper():<10} {total:<10} {no_frac:<12} {r1:<12} {r2:<12} {r3:<12} {r4:<12}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    # CHANGE THIS TO YOUR DATA ROOT
    data_root = "./"  # Adjust this path

    print(f"\nData root: {data_root}\n")

    # Run analysis
    results = analyze_fracture_distribution(data_root)

    # Print summary table
    if results:
        print_summary_table(results)

    # Create visualizations
    if results:
        create_visualization(results)