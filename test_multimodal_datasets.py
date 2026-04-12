#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for multimodal KG datasets (DB15K-mm, FB15K-mm, YAGO15K-mm).
Shows dataset statistics, how to load image embeddings, and numerical attributes.
"""

import os
import sys
import torch
import h5py
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pfn.datasets import DB15KMM, FB15KMM, YAGO15KMM


def print_separator(title=""):
    """Print a separator line."""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")
    else:
        print(f"{'='*80}")


def clean_uri(uri, dataset_name=None, fb_dataset=None):
    """Remove common URI prefixes to make output more readable."""
    # Remove angle brackets if present
    uri = uri.strip('<>')

    # For FB15K, try to get human-readable name
    if dataset_name == 'FB15K-mm' and fb_dataset is not None:
        if uri.startswith('/m/'):
            name = fb_dataset.get_entity_name(uri)
            if name != uri:
                return f"{name} ({uri})"

    prefixes = [
        'http://dbpedia.org/resource/',
        'http://dbpedia.org/ontology/',
        'http://www.w3.org/2000/01/rdf-schema#',
        'http://www.w3.org/2003/01/geo/wgs84_pos#',
        'http://yago-knowledge.org/resource/',
        'http://rdf.freebase.com/ns/',
        'http://www.w3.org/2001/XMLSchema#',
    ]

    for prefix in prefixes:
        if uri.startswith(prefix):
            return uri[len(prefix):]

    # If no prefix matched, return as is
    return uri


def test_dataset(dataset_class, dataset_name, root="./datasets"):
    """Test a multimodal dataset and print statistics."""
    print_separator(f"Testing {dataset_name}")

    try:
        # Load dataset
        print(f"\n[1] Loading {dataset_name}...")
        dataset = dataset_class(root=root)

        # Get train/valid/test splits
        train_data = dataset[0][0]
        valid_data = dataset[1][0]
        test_data = dataset[2][0]

        # Get metadata
        metadata = dataset.get_metadata()
        inv_entity_vocab = metadata['inv_entity_vocab']
        inv_rel_vocab = metadata['inv_rel_vocab']
        entity_to_image = metadata['entity_to_image']
        numerical_data = metadata['numerical_data']

        # Print basic statistics
        print(f"\n[2] Basic Statistics:")
        print(f"  - Total entities: {len(inv_entity_vocab)}")
        print(f"  - Total relations: {len(inv_rel_vocab)}")
        print(f"  - Entities with images: {len(entity_to_image)}")
        print(f"  - Numerical attribute triples: {len(numerical_data)}")

        print(f"\n[3] Split Statistics:")
        print(f"  - Train triples: {train_data.target_edge_index.shape[1]}")
        print(f"  - Valid triples: {valid_data.target_edge_index.shape[1]}")
        print(f"  - Test triples: {test_data.target_edge_index.shape[1]}")
        print(f"  - Total triples: {train_data.target_edge_index.shape[1] + valid_data.target_edge_index.shape[1] + test_data.target_edge_index.shape[1]}")

        # Show sample entities and relations
        print(f"\n[4] Sample Entities (first 5):")
        entity_vocab = {v: k for k, v in inv_entity_vocab.items()}
        for i in range(min(5, len(entity_vocab))):
            entity_uri = entity_vocab[i]
            entity_clean = clean_uri(entity_uri, dataset_name, dataset)
            has_image = "✓" if i in entity_to_image else "✗"
            # Truncate long URIs
            if len(entity_clean) > 70:
                entity_clean = entity_clean[:67] + "..."
            print(f"  [{i}] {entity_clean} (image: {has_image})")

        print(f"\n[5] Sample Relations (first 5):")
        rel_vocab = {v: k for k, v in inv_rel_vocab.items()}
        for i in range(min(5, len(rel_vocab))):
            rel_uri = rel_vocab[i]
            rel_clean = clean_uri(rel_uri, dataset_name, dataset)
            if len(rel_clean) > 70:
                rel_clean = rel_clean[:67] + "..."
            print(f"  [{i}] {rel_clean}")

        # Test loading image embeddings
        print(f"\n[6] Testing Image Embedding Loading:")
        print(f"  - Image HDF5 path: {metadata['image_h5_path']}")

        # Load embeddings for first 3 entities with images
        entities_with_images = list(entity_to_image.keys())[:3]
        if entities_with_images:
            embeddings = dataset.load_image_embeddings(entities_with_images)
            print(f"  - Loaded embeddings for {len(embeddings)} entities")
            for ent_id, emb in embeddings.items():
                img_id = entity_to_image[ent_id]
                print(f"    Entity {ent_id} (image: {img_id}): shape={emb.shape}, dtype={emb.dtype}")
                print(f"      First 5 values: {emb.flatten()[:5].tolist()}")
        else:
            print(f"  - No entities with images found")

        # Show sample numerical attributes
        print(f"\n[7] Sample Numerical Attributes (first 5):")
        for i, num_attr in enumerate(numerical_data[:5]):
            entity_id = num_attr['entity']
            relation_id = num_attr['relation']
            value = num_attr['value']
            entity_uri = entity_vocab.get(entity_id, f"Entity_{entity_id}")
            rel_uri = rel_vocab.get(relation_id, f"Relation_{relation_id}")

            # Clean URIs
            entity_clean = clean_uri(entity_uri, dataset_name, dataset)
            rel_clean = clean_uri(rel_uri, dataset_name, dataset)

            # Truncate URIs
            if len(entity_clean) > 50:
                entity_clean = entity_clean[:47] + "..."
            if len(rel_clean) > 50:
                rel_clean = rel_clean[:47] + "..."

            print(f"  [{i}] Entity: {entity_clean}")
            print(f"      Relation: {rel_clean}")
            print(f"      Value: {value}")

        # Direct HDF5 inspection
        print(f"\n[8] Direct HDF5 File Inspection:")
        h5_path = metadata['image_h5_path']
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as f:
                num_images = len(f.keys())
                sample_keys = list(f.keys())[:5]
                print(f"  - Total images in HDF5: {num_images}")
                print(f"  - Sample image IDs: {sample_keys}")
                if sample_keys:
                    first_key = sample_keys[0]
                    first_shape = f[first_key].shape
                    first_dtype = f[first_key].dtype
                    print(f"  - Image embedding shape: {first_shape}")
                    print(f"  - Image embedding dtype: {first_dtype}")
        else:
            print(f"  - HDF5 file not found: {h5_path}")

        print(f"\n✓ {dataset_name} test completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Error testing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print_separator("Multimodal KG Dataset Test Suite")
    print("\nThis script tests the three multimodal KG datasets:")
    print("  - DB15K-mm")
    print("  - FB15K-mm")
    print("  - YAGO15K-mm")
    print("\nData source: /data/gaoyisen/mmkb/")

    # Set dataset root
    root = "./datasets"
    os.makedirs(root, exist_ok=True)

    # Test each dataset
    results = {}

    # Test DB15K-mm
    results['DB15K-mm'] = test_dataset(DB15KMM, 'DB15K-mm', root)

    # Test FB15K-mm
    results['FB15K-mm'] = test_dataset(FB15KMM, 'FB15K-mm', root)

    # Test YAGO15K-mm
    results['YAGO15K-mm'] = test_dataset(YAGO15KMM, 'YAGO15K-mm', root)

    # Summary
    print_separator("Test Summary")
    for dataset_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {dataset_name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
