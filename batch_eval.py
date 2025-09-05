
# Command-line tool for processing lots of reviews at once
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm  # Progress bars for long operations

from sentiment_llm import analyze_sentiment

def load_reviews_from_file(file_path):
    """Read reviews from a CSV file and make sure it has the right format."""
    try:
        dataframe = pd.read_csv(file_path)
        
        # Check that the CSV has the expected structure
        if 'review' not in dataframe.columns:
            raise ValueError("The CSV file must have a 'review' column containing the movie reviews.")
        
        return dataframe
        
    except Exception as error:
        print(f"Error reading CSV file: {error}")
        sys.exit(1)

def calculate_performance_metrics(dataframe):
    """Calculate how accurate our predictions were (if we have ground truth labels)."""
    # If there's no ground truth data, we can't measure accuracy
    if 'true_sentiment' not in dataframe.columns:
        return {"message": "No ground truth labels found - skipping accuracy calculation"}
    
    # Clean up the labels to make sure they match properly
    dataframe['true_clean'] = dataframe['true_sentiment'].str.strip().str.title()
    dataframe['predicted_clean'] = dataframe['predicted_sentiment'].str.strip().str.title()
    
    # Only look at rows with valid sentiment labels
    valid_labels = {'Positive', 'Negative', 'Neutral'}
    valid_mask = (
        dataframe['true_clean'].isin(valid_labels) & 
        dataframe['predicted_clean'].isin(valid_labels)
    )
    valid_data = dataframe[valid_mask].copy()
    
    if len(valid_data) == 0:
        return {"error": "No valid label pairs found for evaluation"}
    
    # Calculate overall accuracy - how many did we get right?
    correct_predictions = (valid_data['true_clean'] == valid_data['predicted_clean']).sum()
    total_predictions = len(valid_data)
    overall_accuracy = correct_predictions / total_predictions
    
    # Calculate detailed metrics for each sentiment class
    class_metrics = {}
    for label in valid_labels:
        # Count true positives, false positives, false negatives
        true_positives = ((valid_data['true_clean'] == label) & 
                         (valid_data['predicted_clean'] == label)).sum()
        false_positives = ((valid_data['true_clean'] != label) & 
                          (valid_data['predicted_clean'] == label)).sum()
        false_negatives = ((valid_data['true_clean'] == label) & 
                          (valid_data['predicted_clean'] != label)).sum()
        
        # Calculate precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'support': (valid_data['true_clean'] == label).sum()
        }
    
    # Build confusion matrix - shows which labels got confused with which
    confusion_matrix = {}
    for true_label in valid_labels:
        confusion_matrix[true_label] = {}
        for predicted_label in valid_labels:
            count = ((valid_data['true_clean'] == true_label) & 
                    (valid_data['predicted_clean'] == predicted_label)).sum()
            confusion_matrix[true_label][predicted_label] = int(count)
    
    # Check if confidence scores correlate with accuracy
    valid_data['is_correct'] = valid_data['true_clean'] == valid_data['predicted_clean']
    avg_confidence_correct = valid_data[valid_data['is_correct']]['confidence'].mean()
    avg_confidence_wrong = valid_data[~valid_data['is_correct']]['confidence'].mean()
    
    return {
        'accuracy': overall_accuracy,
        'total_samples': total_predictions,
        'correct_predictions': int(correct_predictions),
        'class_metrics': class_metrics,
        'confusion_matrix': confusion_matrix,
        'avg_confidence_correct': avg_confidence_correct,
        'avg_confidence_wrong': avg_confidence_wrong,
        'evaluated_samples': len(valid_data),
        'skipped_samples': len(dataframe) - len(valid_data)
    }

def save_results_to_files(dataframe, output_path, metrics):
    """Save the analysis results and performance stats to files."""
    # Save the main results as CSV
    dataframe.to_csv(output_path, index=False)
    print(f"✅ Analysis results saved to: {output_path}")
    
    # If we have performance metrics, save those too
    if metrics and 'error' not in metrics and 'message' not in metrics:
        metrics_path = output_path.replace('.csv', '_metrics.json')
        
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file, indent=2, default=str)
        print(f"📊 Performance metrics saved to: {metrics_path}")

def print_summary_report(metrics):
    """Display a nice summary of how well the analysis performed."""
    # Handle cases where we couldn't calculate metrics
    if not metrics or 'error' in metrics:
        print("\n⚠️  No performance metrics available")
        if 'error' in metrics:
            print(f"Issue: {metrics['error']}")
        return
    
    # Show informational messages if needed
    if 'message' in metrics:
        print(f"\n📝 {metrics['message']}")
        return
    
    # Print a nice formatted report header
    print("\n" + "="*65)
    print("📊 ANALYSIS RESULTS SUMMARY")
    print("="*65)
    
    # Show the key performance numbers
    print(f"🎯 Overall Accuracy: {metrics['accuracy']:.1%}")
    print(f"📝 Total Reviews Analyzed: {metrics['total_samples']}")
    print(f"✅ Correct Predictions: {metrics['correct_predictions']}")
    
    # Note if any reviews were skipped
    if metrics.get('skipped_samples', 0) > 0:
        print(f"⚠️  Reviews Skipped: {metrics['skipped_samples']} (invalid labels)")
    
    # Show confidence score patterns
    print(f"\n📈 Average Confidence Scores:")
    print(f"   Correct predictions: {metrics.get('avg_confidence_correct', 0):.1%}")
    print(f"   Incorrect predictions: {metrics.get('avg_confidence_wrong', 0):.1%}")
    
    # Break down performance by each sentiment class
    print(f"\n📋 Performance by Sentiment:")
    for sentiment, stats in metrics['class_metrics'].items():
        print(f"\n   {sentiment}:")
        print(f"      Precision: {stats['precision']:.3f}")
        print(f"      Recall: {stats['recall']:.3f}")
        print(f"      F1-Score: {stats['f1_score']:.3f}")
        print(f"      Number of samples: {stats['support']}")
    
    # Display confusion matrix in a readable format
    print(f"\n🔄 Confusion Matrix:")
    print("       ", end="")
    labels = list(metrics['confusion_matrix'].keys())
    for label in labels:
        print(f"{label[:6]:>8}", end="")
    print()
    
    for true_label in labels:
        print(f"{true_label[:6]:>8} ", end="")
        for predicted_label in labels:
            count = metrics['confusion_matrix'][true_label][predicted_label]
            print(f"{count:>8}", end="")
        print()

def main():
    """The main command-line interface for batch processing reviews."""
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Batch sentiment analysis for movie reviews",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_eval.py reviews.csv
  python batch_eval.py reviews.csv --output results.csv
  python batch_eval.py reviews.csv --sample 100 --verbose
        """
    )
    
    # Define what command-line arguments we accept
    parser.add_argument("input_file", help="CSV file containing movie reviews (must have 'review' column)")
    parser.add_argument("--output", "-o", help="Output CSV file path (default: adds '_results' to input name)")
    parser.add_argument("--sample", "-s", type=int, help="Only process first N reviews (useful for testing)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed progress information")
    
    args = parser.parse_args()
    
    # Check that the API key is available before starting
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable is required")
        print("Obtain your free API key from: https://makersuite.google.com/app/apikey")
        print("Configure it with: export GEMINI_API_KEY='your_key_here'")
        sys.exit(1)
    
    # Figure out where to save the results
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input_file)
        output_file = input_path.parent / f"{input_path.stem}_results.csv"
    
    # Show what we're about to do
    print("🎬 Movie Review Sentiment Analysis - Batch Processing")
    print(f"📁 Input file: {args.input_file}")
    print(f"💾 Output file: {output_file}")
    
    # Load the input file and check its format
    print("\n📖 Loading reviews from file...")
    reviews_df = load_reviews_from_file(args.input_file)
    
    # If user requested a sample, take just the first N reviews
    if args.sample:
        reviews_df = reviews_df.head(args.sample)
        print(f"📝 Processing first {len(reviews_df)} reviews (sample mode)")
    else:
        print(f"📝 Processing {len(reviews_df)} reviews")
    
    # Start the main analysis process
    print(f"\n🤖 Analyzing sentiment...")
    start_time = time.time()
    
    # Track results and failures
    analysis_results = []
    failed_analyses = 0
    
    # Process each review with a progress bar
    with tqdm(total=len(reviews_df), desc="Processing reviews") as progress_bar:
        for index, row in reviews_df.iterrows():
            review_text = str(row['review']).strip()
            
            # Handle empty or invalid reviews
            if not review_text or review_text.lower() in ['nan', 'none', '']:
                result = {
                    'label': 'Neutral',
                    'confidence': 0.0,
                    'explanation': 'Empty or missing review text',
                    'evidence_phrases': []
                }
            # Try to analyze the review
            else:
                try:
                    result = analyze_sentiment(review_text)
                    
                    # Show detailed progress if requested
                    if args.verbose:
                        sentiment = result['label']
                        confidence = result['confidence']
                        tqdm.write(f"Review {index+1}: {sentiment} ({confidence:.2f})")
                        
                # Handle analysis failures gracefully
                except Exception as error:
                    if args.verbose:
                        tqdm.write(f"Analysis failed for review {index+1}: {error}")
                    
                    failed_analyses += 1
                    result = {
                        'label': 'Neutral',
                        'confidence': 0.0,
                        'explanation': f'Analysis failed: {str(error)}',
                        'evidence_phrases': []
                    }
            
            analysis_results.append(result)
            progress_bar.update(1)
    
    # Calculate how long the whole process took
    total_time = time.time() - start_time
    
    # Add the analysis results to our dataframe
    reviews_df['predicted_sentiment'] = [r['label'] for r in analysis_results]
    reviews_df['confidence'] = [r['confidence'] for r in analysis_results]
    reviews_df['explanation'] = [r['explanation'] for r in analysis_results]
    reviews_df['evidence_phrases'] = [', '.join(r['evidence_phrases']) for r in analysis_results]
    
    # If we have ground truth labels, mark which predictions were correct
    if 'true_sentiment' in reviews_df.columns:
        reviews_df['correct'] = (
            reviews_df['true_sentiment'].str.strip().str.title() == 
            reviews_df['predicted_sentiment']
        )
    
    # Report completion and timing
    print(f"\n✅ Analysis completed successfully!")
    print(f"⏱️  Total processing time: {total_time:.1f} seconds ({total_time/len(reviews_df):.1f}s per review)")
    
    # Report any failures
    if failed_analyses > 0:
        print(f"⚠️  {failed_analyses} reviews failed analysis and were assigned default values")
    
    # Calculate performance metrics and save everything
    print("\n📊 Calculating performance metrics...")
    performance_metrics = calculate_performance_metrics(reviews_df)
    
    save_results_to_files(reviews_df, output_file, performance_metrics)
    
    print_summary_report(performance_metrics)
    
    # Show the overall sentiment breakdown
    sentiment_distribution = reviews_df['predicted_sentiment'].value_counts()
    print(f"\n📈 Sentiment Distribution:")
    for sentiment, count in sentiment_distribution.items():
        percentage = count / len(reviews_df) * 100
        print(f"   {sentiment}: {count} reviews ({percentage:.1f}%)")
    
    # Show average confidence and wrap up
    avg_confidence = reviews_df['confidence'].mean()
    print(f"\n🎯 Average Confidence Score: {avg_confidence:.1%}")
    
    print(f"\n📋 Results saved successfully!")

# Run the main function when this script is executed directly
if __name__ == "__main__":
    main()