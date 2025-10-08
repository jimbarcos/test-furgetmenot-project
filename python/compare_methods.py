#!/usr/bin/env python3
"""
Compare ImageNet vs Siamese Model Performance
This script tests both approaches on the same query and shows the difference.
"""

import sys
import os
import subprocess
import json

def run_comparison(query_image, pet_types, gallery_dir, top_k=10):
    """Run both methods and compare results."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    compute_script = os.path.join(script_dir, 'python', 'compute_matches.py')
    
    print("=" * 80)
    print("COMPARISON: ImageNet vs Siamese Model")
    print("=" * 80)
    print(f"Query Image: {query_image}")
    print(f"Pet Types: {pet_types}")
    print(f"Gallery: {gallery_dir}")
    print(f"Top K: {top_k}")
    print()
    
    # Run ImageNet-only method
    print("-" * 80)
    print("METHOD 1: ImageNet Embeddings + Cosine Similarity")
    print("-" * 80)
    cmd_imagenet = [
        'python', compute_script,
        query_image, pet_types, gallery_dir, str(top_k),
        '--debug', '--imagenet-only'
    ]
    
    result_imagenet = subprocess.run(cmd_imagenet, capture_output=True, text=True)
    
    if result_imagenet.returncode == 0:
        data_imagenet = json.loads(result_imagenet.stdout)
        if data_imagenet['ok']:
            print(f"✅ Success! Found {len(data_imagenet['matches'])} matches")
            print("\nTop 5 Matches:")
            for i, match in enumerate(data_imagenet['matches'][:5], 1):
                print(f"  {i}. {match['filename']}: {match['similarity']:.4f} ({match['confidence']:.2f}%)")
        else:
            print(f"❌ Error: {data_imagenet.get('error', 'Unknown')}")
    else:
        print(f"❌ Failed to run: {result_imagenet.stderr}")
    
    print()
    
    # Run full Siamese model
    print("-" * 80)
    print("METHOD 2: Full Siamese Model (with training)")
    print("-" * 80)
    cmd_siamese = [
        'python', compute_script,
        query_image, pet_types, gallery_dir, str(top_k),
        '--debug'
    ]
    
    result_siamese = subprocess.run(cmd_siamese, capture_output=True, text=True)
    
    if result_siamese.returncode == 0:
        data_siamese = json.loads(result_siamese.stdout)
        if data_siamese['ok']:
            print(f"✅ Success! Found {len(data_siamese['matches'])} matches")
            print("\nTop 5 Matches:")
            for i, match in enumerate(data_siamese['matches'][:5], 1):
                print(f"  {i}. {match['filename']}: {match['similarity']:.4f} ({match['confidence']:.2f}%)")
        else:
            print(f"❌ Error: {data_siamese.get('error', 'Unknown')}")
    else:
        print(f"❌ Failed to run: {result_siamese.stderr}")
    
    print()
    
    # Compare results
    if result_imagenet.returncode == 0 and result_siamese.returncode == 0:
        data_imagenet = json.loads(result_imagenet.stdout)
        data_siamese = json.loads(result_siamese.stdout)
        
        if data_imagenet['ok'] and data_siamese['ok']:
            print("-" * 80)
            print("COMPARISON SUMMARY")
            print("-" * 80)
            
            # Compare top match
            top_img = data_imagenet['matches'][0]
            top_sia = data_siamese['matches'][0]
            
            print(f"\nTop Match Comparison:")
            print(f"  ImageNet:  {top_img['filename']} ({top_img['similarity']:.4f})")
            print(f"  Siamese:   {top_sia['filename']} ({top_sia['similarity']:.4f})")
            
            if top_img['filename'] == top_sia['filename']:
                print(f"  Result: ✅ Both methods agree on top match")
                diff = abs(top_img['similarity'] - top_sia['similarity'])
                print(f"  Confidence difference: {diff:.4f}")
                if top_img['similarity'] > top_sia['similarity']:
                    print(f"  → ImageNet has HIGHER confidence (+{diff:.4f})")
                else:
                    print(f"  → Siamese has HIGHER confidence (+{diff:.4f})")
            else:
                print(f"  Result: ⚠️  Methods DISAGREE on top match!")
            
            # Compare average scores
            avg_img = sum(m['similarity'] for m in data_imagenet['matches'][:5]) / 5
            avg_sia = sum(m['similarity'] for m in data_siamese['matches'][:5]) / 5
            
            print(f"\nAverage Top-5 Similarity:")
            print(f"  ImageNet:  {avg_img:.4f}")
            print(f"  Siamese:   {avg_sia:.4f}")
            
            if avg_img > avg_sia:
                diff = avg_img - avg_sia
                print(f"  → ImageNet produces HIGHER similarities (+{diff:.4f})")
            else:
                diff = avg_sia - avg_img
                print(f"  → Siamese produces HIGHER similarities (+{diff:.4f})")
            
            # Overlap analysis
            img_files = set(m['filename'] for m in data_imagenet['matches'][:5])
            sia_files = set(m['filename'] for m in data_siamese['matches'][:5])
            overlap = img_files & sia_files
            
            print(f"\nTop-5 Overlap: {len(overlap)}/5 matches in common")
            if len(overlap) < 5:
                print(f"  ImageNet only: {img_files - sia_files}")
                print(f"  Siamese only:  {sia_files - img_files}")
            
            # Recommendation
            print("\n" + "=" * 80)
            if avg_img > avg_sia + 0.05:
                print("RECOMMENDATION: Use ImageNet embeddings (--imagenet-only)")
                print("Reason: Higher confidence scores and likely better generalization")
            elif avg_sia > avg_img + 0.05:
                print("RECOMMENDATION: Use Full Siamese model")
                print("Reason: Higher confidence scores, training is beneficial")
            else:
                print("RECOMMENDATION: Results are similar, either method works")
                print("Consider ImageNet for simplicity, or Siamese if training data is good")
            print("=" * 80)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python compare_methods.py <query_image> <pet_types> <gallery_dir> [top_k]")
        print("\nExample:")
        print("  python compare_methods.py uploads/pet.jpg cat,dog ./Preprocessed 10")
        sys.exit(1)
    
    query_image = sys.argv[1]
    pet_types = sys.argv[2]
    gallery_dir = sys.argv[3]
    top_k = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    
    run_comparison(query_image, pet_types, gallery_dir, top_k)
