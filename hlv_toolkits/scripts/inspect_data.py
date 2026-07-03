#!/usr/bin/env python3
"""
Data inspection script for NLI datasets.

This script provides detailed statistics and examples for NLI datasets,
including label distributions, text length statistics, and sample previews.

Example usage:
    # Inspect SNLI test set
    python -m hlv_toolkits.scripts.inspect_data \
        --source snli \
        --split test \
        --num_examples 5

    # Inspect ChaosNLI dataset
    python -m hlv_toolkits.scripts.inspect_data \
        --source chaosnli \
        --split test \
        --chaosnli_path data/chaosnli.jsonl \
        --num_examples 10
"""

import argparse
from collections import Counter
from typing import List, Union

from hlv_toolkits.data import (
    ChaosNLIReader,
    NLIDistributionSample,
    NLISample,
    NLI_ID2LABEL,
    NLI_LABELS,
    SNLIReader,
)


def compute_text_statistics(samples: List[Union[NLISample, NLIDistributionSample]]) -> dict:
    """
    Compute text length statistics for premises and hypotheses.
    
    Args:
        samples: List of NLI samples
        
    Returns:
        Dictionary with statistics including mean, min, max, and percentiles
    """
    premise_lengths = [len(s.premise.split()) for s in samples if s.premise]
    hypothesis_lengths = [len(s.hypothesis.split()) for s in samples if s.hypothesis]
    
    stats = {}
    
    if premise_lengths:
        stats["premise"] = {
            "mean": sum(premise_lengths) / len(premise_lengths),
            "min": min(premise_lengths),
            "max": max(premise_lengths),
            "median": sorted(premise_lengths)[len(premise_lengths) // 2],
        }
    
    if hypothesis_lengths:
        stats["hypothesis"] = {
            "mean": sum(hypothesis_lengths) / len(hypothesis_lengths),
            "min": min(hypothesis_lengths),
            "max": max(hypothesis_lengths),
            "median": sorted(hypothesis_lengths)[len(hypothesis_lengths) // 2],
        }
    
    return stats


def compute_label_distribution(samples: List[Union[NLISample, NLIDistributionSample]]) -> dict:
    """
    Compute label distribution statistics.
    
    Args:
        samples: List of NLI samples
        
    Returns:
        Dictionary with label counts and percentages
    """
    label_counts = Counter()
    total_samples = len(samples)
    
    for sample in samples:
        if isinstance(sample, NLIDistributionSample) and sample.human_dist:
            # For distribution samples, count based on majority label
            label_counts[sample.label] += 1
        elif isinstance(sample, NLISample):
            # For single-label samples
            if sample.label >= 0:
                label_counts[sample.label] += 1
    
    distribution = {}
    for label_id, label_name in enumerate(NLI_LABELS):
        count = label_counts[label_id]
        percentage = (count / total_samples * 100) if total_samples > 0 else 0.0
        distribution[label_name] = {
            "count": count,
            "percentage": percentage,
        }
    
    return distribution


def compute_distribution_statistics(samples: List[NLIDistributionSample]) -> dict:
    """
    Compute statistics about human label distributions for ChaosNLI.
    
    Args:
        samples: List of NLIDistributionSample objects
        
    Returns:
        Dictionary with distribution statistics
    """
    if not samples or not isinstance(samples[0], NLIDistributionSample):
        return {}
    
    dist_samples = [s for s in samples if s.human_dist]
    if not dist_samples:
        return {}
    
    # Compute entropy for each distribution
    import math
    entropies = []
    for sample in dist_samples:
        entropy = -sum(p * math.log(p + 1e-10) for p in sample.human_dist if p > 0)
        entropies.append(entropy)
    
    # Compute average distribution
    avg_dist = [0.0, 0.0, 0.0]
    for sample in dist_samples:
        for i, p in enumerate(sample.human_dist):
            avg_dist[i] += p
    avg_dist = [p / len(dist_samples) for p in avg_dist]
    
    return {
        "num_samples_with_dist": len(dist_samples),
        "avg_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
        "avg_distribution": {
            label: avg_dist[i] for i, label in enumerate(NLI_LABELS)
        },
    }


def check_data_quality(samples: List[Union[NLISample, NLIDistributionSample]]) -> dict:
    """
    Check data quality issues.
    
    Args:
        samples: List of NLI samples
        
    Returns:
        Dictionary with quality check results
    """
    issues = {
        "empty_premise": 0,
        "empty_hypothesis": 0,
        "invalid_label": 0,
        "missing_id": 0,
    }
    
    for sample in samples:
        if not sample.premise or not sample.premise.strip():
            issues["empty_premise"] += 1
        if not sample.hypothesis or not sample.hypothesis.strip():
            issues["empty_hypothesis"] += 1
        if sample.label < 0 or sample.label >= len(NLI_LABELS):
            issues["invalid_label"] += 1
        if not sample.id or not sample.id.strip():
            issues["missing_id"] += 1
    
    return issues


def print_statistics(
    samples: List[Union[NLISample, NLIDistributionSample]],
    source: str,
    split: str,
    num_examples: int = 5,
) -> None:
    """
    Print comprehensive statistics about the dataset.
    
    Args:
        samples: List of NLI samples
        source: Data source name
        split: Data split name
        num_examples: Number of example samples to display
    """
    print("=" * 80)
    print(f"Dataset Inspection: {source.upper()} - {split.upper()} Split")
    print("=" * 80)
    print()
    
    # Basic statistics
    print("Basic Statistics:")
    print(f"  Total samples: {len(samples)}")
    if samples:
        print(f"  Data source: {samples[0].source}")
        print(f"  Task: {samples[0].task}")
    print()
    
    # Label distribution
    print("Label Distribution:")
    label_dist = compute_label_distribution(samples)
    for label_name, stats in label_dist.items():
        print(f"  {label_name:15s}: {stats['count']:6d} ({stats['percentage']:5.2f}%)")
    print()
    
    # Text length statistics
    print("Text Length Statistics (word count):")
    text_stats = compute_text_statistics(samples)
    for text_type in ["premise", "hypothesis"]:
        if text_type in text_stats:
            stats = text_stats[text_type]
            print(f"  {text_type.capitalize()}:")
            print(f"    Mean:   {stats['mean']:.2f}")
            print(f"    Median: {stats['median']:.2f}")
            print(f"    Min:    {stats['min']}")
            print(f"    Max:    {stats['max']}")
    print()
    
    # Distribution statistics (for ChaosNLI)
    if samples and isinstance(samples[0], NLIDistributionSample):
        print("Human Label Distribution Statistics:")
        dist_stats = compute_distribution_statistics(samples)
        if dist_stats:
            print(f"  Samples with distribution: {dist_stats['num_samples_with_dist']}")
            print(f"  Average entropy: {dist_stats['avg_entropy']:.4f}")
            print("  Average distribution:")
            for label, prob in dist_stats["avg_distribution"].items():
                print(f"    {label:15s}: {prob:.4f}")
        print()
    
    # Data quality checks
    print("Data Quality Checks:")
    quality_issues = check_data_quality(samples)
    has_issues = any(count > 0 for count in quality_issues.values())
    if has_issues:
        for issue, count in quality_issues.items():
            if count > 0:
                print(f"  ⚠️  {issue.replace('_', ' ').title()}: {count}")
    else:
        print("  ✓ No quality issues detected")
    print()
    
    # Example samples
    print(f"Example Samples (showing {min(num_examples, len(samples))}):")
    print("-" * 80)
    for i, sample in enumerate(samples[:num_examples], 1):
        print(f"\nExample {i}:")
        print(f"  ID: {sample.id}")
        print(f"  Label: {NLI_ID2LABEL.get(sample.label, 'unknown')} ({sample.label})")
        if isinstance(sample, NLIDistributionSample) and sample.human_dist:
            print(f"  Human Distribution: {sample.human_dist}")
        print(f"  Premise:    {sample.premise[:200]}{'...' if len(sample.premise) > 200 else ''}")
        print(f"  Hypothesis: {sample.hypothesis[:200]}{'...' if len(sample.hypothesis) > 200 else ''}")
    
    print()
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and analyze NLI dataset statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["snli", "chaosnli"],
        help="Data source: 'snli' or 'chaosnli'",
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "valid", "validation", "test"],
        help="Data split to inspect",
    )
    
    parser.add_argument(
        "--chaosnli_path",
        type=str,
        default="data/external/chaosnli/chaosNLI_snli.jsonl",
        help="Path to ChaosNLI JSONL file (required if source=chaosnli)",
    )
    
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of example samples to display (default: 5)",
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.source.upper()} {args.split} split...")
    try:
        if args.source == "snli":
            reader = SNLIReader()
            samples = reader.load_split(args.split)
        elif args.source == "chaosnli":
            if args.chaosnli_path is None:
                raise ValueError("--chaosnli_path is required when source=chaosnli")
            reader = ChaosNLIReader(data_path=args.chaosnli_path)
            samples = reader.load_split(args.split)
        else:
            raise ValueError(f"Unknown source: {args.source}")
        
        print(f"Loaded {len(samples)} samples\n")
        
        # Print statistics
        print_statistics(samples, args.source, args.split, args.num_examples)
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
