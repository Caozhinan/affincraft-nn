#!/usr/bin/env python3  
"""  
AffinCraft Embedding Test Script  
Tests the complete embedding pipeline from PKL files to embedded representations  
"""  
  
import torch  
import pickle  
import numpy as np  
from pathlib import Path  
import sys  
import argparse  
from torch.utils.data import DataLoader  
  
# Add the Dynaformer path to sys.path if needed  
sys.path.append('/xcfhome/zncao02/affincraft/Dynaformer')  
  
from dynaformer.data.affincraft_dataset import (  
    AffinCraftDataset,   
    affincraft_collator,   
    create_affincraft_dataloader  
)  
from dynaformer.modules.affincraft_graph_encoder import AffinCraftGraphEncoder  
  
def test_embedding_pipeline(pkl_file_path, batch_size=4, max_complexes=10):  
    """  
    Test the complete embedding pipeline  
      
    Args:  
        pkl_file_path: Path to PKL file containing multiple complexes  
        batch_size: Batch size for processing  
        max_complexes: Maximum number of complexes to test (for debugging)  
    """  
      
    print(f"Loading PKL file: {pkl_file_path}")  
      
    # Load PKL file  
    with open(pkl_file_path, 'rb') as f:  
        complexes = pickle.load(f)  
      
    print(f"Loaded {len(complexes)} complexes")  
      
    # Limit complexes for testing  
    if max_complexes and len(complexes) > max_complexes:  
        complexes = complexes[:max_complexes]  
        print(f"Limited to {len(complexes)} complexes for testing")  
      
    # Create temporary PKL files for each complex (since your dataset expects file paths)  
    temp_pkl_files = []  
    temp_dir = Path("./temp_test_pkls")  
    temp_dir.mkdir(exist_ok=True)  
      
    try:  
        for i, complex_data in enumerate(complexes):  
            temp_pkl_path = temp_dir / f"complex_{i}.pkl"  
            with open(temp_pkl_path, 'wb') as f:  
                pickle.dump([complex_data], f)  # Wrap in list as expected  
            temp_pkl_files.append(str(temp_pkl_path))  
          
        print(f"Created {len(temp_pkl_files)} temporary PKL files")  
          
        # Create dataset and dataloader  
        dataset = AffinCraftDataset(temp_pkl_files)  
        dataloader = DataLoader(  
            dataset,  
            batch_size=batch_size,  
            shuffle=False,  
            collate_fn=affincraft_collator,  
            num_workers=0  # Set to 0 for debugging  
        )  
          
        print(f"Created dataset with {len(dataset)} items")  
          
        # Initialize the AffinCraft encoder  
        encoder = AffinCraftGraphEncoder(  
            num_encoder_layers=6,  # Smaller for testing  
            embedding_dim=256,     # Smaller for testing  
            ffn_embedding_dim=256,  
            num_attention_heads=8,  
            dropout=0.1,  
            node_feat_dim=9,  
            use_masif=True,  
            use_gbscore=True  
        )  
          
        # Set to evaluation mode  
        encoder.eval()  
          
        print("Testing embedding pipeline...")  
          
        total_processed = 0  
          
        with torch.no_grad():  
            for batch_idx, batch_data in enumerate(dataloader):  
                if batch_data is None:  
                    print(f"Batch {batch_idx}: Skipped (None)")  
                    continue  
                  
                print(f"\nBatch {batch_idx}:")  
                print(f"  Batch size: {len(batch_data['pdbid'])}")  
                print(f"  Node features shape: {batch_data['node_feat'].shape}")  
                print(f"  Edge features: {len(batch_data['edge_feat'])} items")  
                print(f"  GB-Score shape: {batch_data['gbscore'].shape}")  
                print(f"  MaSIF desc shape: {batch_data['masif_desc_straight'].shape}")  
                  
                try:  
                    # Run through encoder  
                    inner_states, graph_rep = encoder(batch_data)  
                      
                    print(f"  ✓ Embedding successful!")  
                    print(f"  Graph representation shape: {graph_rep.shape}")  
                    print(f"  Number of transformer states: {len(inner_states)}")  
                    print(f"  Final state shape: {inner_states[-1].shape}")  
                      
                    # Print some statistics  
                    print(f"  Graph rep mean: {graph_rep.mean().item():.4f}")  
                    print(f"  Graph rep std: {graph_rep.std().item():.4f}")  
                    print(f"  Graph rep range: [{graph_rep.min().item():.4f}, {graph_rep.max().item():.4f}]")  
                    print(f"Input node features contain NaN: {torch.isnan(batch_data['node_feat']).any()}")  
                    print(f"Input edge features contain NaN: {torch.isnan(batch_data['edge_feat']).any()}")  
                    print(f"Input GB-Score contain NaN: {torch.isnan(batch_data['gbscore']).any()}")  
                    print(f"Input MaSIF contain NaN: {torch.isnan(batch_data['masif_desc_straight']).any()}")
                    total_processed += len(batch_data['pdbid'])  
                      
                except Exception as e:  
                    print(f"  ✗ Embedding failed: {e}")  
                    import traceback  
                    traceback.print_exc()  
                    break  
          
        print(f"\nTest completed! Processed {total_processed} complexes successfully.")  
          
        # Test individual complex processing  
        print("\nTesting individual complex processing...")  
        sample_data = dataset[0]  
        print(f"Sample complex: {sample_data.get('pdbid', 'Unknown')}")  
        print(f"Node features: {sample_data['node_feat'].shape}")  
        print(f"Edge features: {sample_data['edge_feat'].shape}")  
          
    finally:  
        # Clean up temporary files  
        for temp_file in temp_pkl_files:  
            Path(temp_file).unlink(missing_ok=True)  
        temp_dir.rmdir()  
        print("Cleaned up temporary files")  
  
def main():  
    parser = argparse.ArgumentParser(description='Test AffinCraft embedding pipeline')  
    parser.add_argument('pkl_file', help='Path to PKL file containing complexes')  
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')  
    parser.add_argument('--max_complexes', type=int, default=10, help='Max complexes to test')  
      
    args = parser.parse_args()  
      
    if not Path(args.pkl_file).exists():  
        print(f"Error: PKL file not found: {args.pkl_file}")  
        sys.exit(1)  
      
    test_embedding_pipeline(args.pkl_file, args.batch_size, args.max_complexes)  
  
if __name__ == "__main__":  
    main()