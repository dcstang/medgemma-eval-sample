#!/usr/bin/env python3
"""
Analysis script for MedGemma evaluation results by specialty.
Creates visualizations and summary statistics comparing different medical specialties.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_and_analyze_data(json_file_path):
    """Load JSON data and analyze by specialty."""
    
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Extract the three main score columns
    score_columns = [
        'free_of_medical_jargon_(geval)_score',
        'medical_accuracy_(geval)_score', 
        'medical_safety_(geval)_score'
    ]
    
    # Calculate average scores by specialty
    specialty_scores = df.groupby('specialty')[score_columns].mean()
    
    # Add count of questions per specialty
    specialty_counts = df['specialty'].value_counts()
    specialty_scores['question_count'] = specialty_counts
    
    # Rename columns for better readability
    specialty_scores.columns = ['Jargon-Free Score', 'Medical Accuracy Score', 'Medical Safety Score', 'Question Count']
    
    return df, specialty_scores

def create_visualizations(specialty_scores, output_dir='.'):
    """Create various visualizations of the specialty comparison data."""
    
    # Set up the plotting style
    sns.set_theme(style="white", palette="husl")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. Bar chart comparing all three scores by specialty
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort specialties by safety score in descending order
    specialty_scores_sorted = specialty_scores.sort_values('Medical Safety Score', ascending=False)
    
    y = np.arange(len(specialty_scores_sorted)) * 1.2  # Add spacing between specialties
    height = 0.33
    
    bars1 = ax.barh(y - height, specialty_scores_sorted['Jargon-Free Score'], height, 
                   label='Jargon-free', alpha=0.4, edgecolor='none')
    bars2 = ax.barh(y, specialty_scores_sorted['Medical Accuracy Score'], height, 
                   label='Medical Accuracy', alpha=0.4, edgecolor='none')
    bars3 = ax.barh(y + height, specialty_scores_sorted['Medical Safety Score'], height, 
                   label='Medical Safety', alpha=0.99, edgecolor='none')
    
    # Get the color of the safety bar from the current palette
    safety_color = bars3.patches[0].get_facecolor()
    
    # Only add value labels for the last occurrence of each unique safety score
    # Find the last index for each unique score
    safety_scores = list(specialty_scores_sorted['Medical Safety Score'])
    last_indices = {score: idx for idx, score in enumerate(safety_scores)}
    for idx, bar in enumerate(bars3):
        width = bar.get_width()
        if last_indices[width] == idx:
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{width * 100:.1f}', ha='left', va='center', fontsize=6, color=safety_color)
    
    ax.set_xlabel('Quality measures (LLM-as-judge G-evals by Gemini2.5-flash)')
    ax.set_title('MedGemma Performance by Medical Specialty')
    ax.set_yticks(y)
    ax.set_yticklabels(specialty_scores_sorted.index)
    ax.set_xlim(0, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    # Invert legend order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='lower center', bbox_to_anchor=(0.73, 0.5), frameon=True, facecolor='white', edgecolor='black')
    
    # Add dashed vertical lines at 1.0 and 0.5
    ax.axvline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
    
    # Reduce white space at top and bottom
    ax.set_ylim(-0.7, len(specialty_scores_sorted) * 1.2 - 0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'specialty_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Heatmap of scores by specialty
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for heatmap (exclude question count)
    heatmap_data = specialty_scores[['Jargon-Free Score', 'Medical Accuracy Score', 'Medical Safety Score']]
    
    sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='RdYlGn', 
                cbar_kws={'label': 'Average Score'}, ax=ax)
    ax.set_title('MedGemma Performance Heatmap by Specialty')
    ax.set_xlabel('Medical Specialty')
    ax.set_ylabel('Evaluation Metric')
    
    plt.tight_layout()
    plt.savefig(output_path / 'specialty_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Radar chart for top specialties
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Select top 6 specialties by overall performance
    overall_scores = specialty_scores[['Jargon-Free Score', 'Medical Accuracy Score', 'Medical Safety Score']].mean(axis=1)
    top_specialties = overall_scores.nlargest(6).index
    
    # Prepare data for radar chart
    categories = ['Jargon-Free', 'Accuracy', 'Safety']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Plot each specialty
    for i, specialty in enumerate(top_specialties):
        values = specialty_scores.loc[specialty, ['Jargon-Free Score', 'Medical Accuracy Score', 'Medical Safety Score']].tolist()
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=specialty)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Top 6 Specialties - Performance Radar Chart', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path / 'top_specialties_radar.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Question count vs performance scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate overall performance score
    specialty_scores['Overall Score'] = specialty_scores[['Jargon-Free Score', 'Medical Accuracy Score', 'Medical Safety Score']].mean(axis=1)
    
    scatter = ax.scatter(specialty_scores['Question Count'], specialty_scores['Overall Score'], 
                        s=100, alpha=0.7, c=specialty_scores['Overall Score'], cmap='viridis')
    
    # Add specialty labels
    for idx, row in specialty_scores.iterrows():
        ax.annotate(idx, (row['Question Count'], row['Overall Score']), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Number of Questions')
    ax.set_ylabel('Overall Average Score')
    ax.set_title('Specialty Performance vs Question Count')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Overall Score')
    
    plt.tight_layout()
    plt.savefig(output_path / 'performance_vs_questions.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_statistics(specialty_scores):
    """Print detailed summary statistics."""
    
    print("=" * 80)
    print("MEDGEMMA EVALUATION RESULTS BY SPECIALTY")
    print("=" * 80)
    
    # Overall statistics
    print(f"\nTotal number of specialties evaluated: {len(specialty_scores)}")
    print(f"Total number of questions: {specialty_scores['Question Count'].sum()}")
    
    # Best performing specialties
    print("\n" + "=" * 50)
    print("TOP PERFORMING SPECIALTIES BY METRIC")
    print("=" * 50)
    
    for metric in ['Jargon-Free Score', 'Medical Accuracy Score', 'Medical Safety Score']:
        print(f"\n{metric}:")
        top_3 = specialty_scores.nlargest(3, metric)[[metric, 'Question Count']]
        for specialty, row in top_3.iterrows():
            print(f"  {specialty}: {row[metric]:.3f} ({row['Question Count']} questions)")
    
    # Overall performance ranking
    print("\n" + "=" * 50)
    print("OVERALL PERFORMANCE RANKING")
    print("=" * 50)
    
    overall_scores = specialty_scores[['Jargon-Free Score', 'Medical Accuracy Score', 'Medical Safety Score']].mean(axis=1)
    overall_ranking = overall_scores.sort_values(ascending=False)
    
    for i, (specialty, score) in enumerate(overall_ranking.items(), 1):
        question_count = specialty_scores.loc[specialty, 'Question Count']
        print(f"{i:2d}. {specialty:<20} {score:.3f} ({question_count} questions)")
    
    # Detailed table
    print("\n" + "=" * 80)
    print("DETAILED RESULTS TABLE")
    print("=" * 80)
    
    # Format the table nicely
    detailed_df = specialty_scores.copy()
    detailed_df['Overall Score'] = overall_scores
    detailed_df = detailed_df.sort_values('Overall Score', ascending=False)
    
    # Round scores for display
    display_df = detailed_df.round(3)
    
    print(display_df.to_string())
    
    # Statistical summary
    print("\n" + "=" * 50)
    print("STATISTICAL SUMMARY")
    print("=" * 50)
    
    for metric in ['Jargon-Free Score', 'Medical Accuracy Score', 'Medical Safety Score']:
        print(f"\n{metric}:")
        print(f"  Mean: {specialty_scores[metric].mean():.3f}")
        print(f"  Std:  {specialty_scores[metric].std():.3f}")
        print(f"  Min:  {specialty_scores[metric].min():.3f}")
        print(f"  Max:  {specialty_scores[metric].max():.3f}")

def main():
    """Main function to run the analysis."""
    
    # File path
    json_file = "medgemma_eval_results.json"
    
    try:
        # Load and analyze data
        print("Loading and analyzing data...")
        df, specialty_scores = load_and_analyze_data(json_file)
        
        # Print summary statistics
        print_summary_statistics(specialty_scores)
        
        # Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(specialty_scores)
        
        print(f"\nAnalysis complete! Visualizations saved to current directory.")
        print("Files created:")
        print("  - specialty_comparison_bars.png")
        print("  - specialty_heatmap.png") 
        print("  - top_specialties_radar.png")
        print("  - performance_vs_questions.png")
        
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}")
        print("Please make sure the JSON file is in the same directory as this script.")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main() 