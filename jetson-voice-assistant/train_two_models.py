"""
Train two FastText models for Jetson comparison:
- Small: Lightweight, fast inference, lower memory
- Medium: Better accuracy, slower inference, higher memory
"""

import fasttext
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_training_data(csv_path: str):
    """Load training data from CSV"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} training examples")
    print(f"Intents: {df['intent'].nunique()}")
    return df


def prepare_fasttext_data(df, train_file, valid_file, train_split=0.8):
    """Prepare data in FastText format: __label__<intent> <text>"""
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split
    split_idx = int(len(df) * train_split)
    train_df = df[:split_idx]
    valid_df = df[split_idx:]
    
    # Write train file
    with open(train_file, 'w', encoding='utf-8') as f:
        for _, row in train_df.iterrows():
            f.write(f"__label__{row['intent']} {row['text']}\n")
    
    # Write validation file
    with open(valid_file, 'w', encoding='utf-8') as f:
        for _, row in valid_df.iterrows():
            f.write(f"__label__{row['intent']} {row['text']}\n")
    
    print(f"\nTrain: {len(train_df)} examples")
    print(f"Valid: {len(valid_df)} examples")
    
    return train_file, valid_file


def train_small_model(train_file, output_path):
    """
    Train SMALL model for Jetson
    - Low dimensions (dim=25)
    - Small bucket size
    - Fewer epochs
    - Target: <10 MB, <50ms inference
    """
    print("\n" + "="*60)
    print("🔹 Training SMALL model (Jetson-optimized)")
    print("="*60)
    
    model = fasttext.train_supervised(
        input=train_file,
        lr=0.5,              # Learning rate
        epoch=50,            # Fewer epochs
        wordNgrams=2,        # Bigrams
        dim=25,              # Low dimensions (was 50)
        bucket=50000,        # Small bucket (was 100000)
        loss='softmax',
        verbose=2
    )
    
    model.save_model(str(output_path))
    
    # Get model size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\n✅ Small model saved: {output_path}")
    print(f"   Size: {size_mb:.2f} MB")
    
    return model


def train_medium_model(train_file, output_path):
    """
    Train MEDIUM model for Jetson
    - Medium dimensions (dim=50)
    - Medium bucket size
    - More epochs
    - Target: <20 MB, <100ms inference
    """
    print("\n" + "="*60)
    print("🔸 Training MEDIUM model (balanced)")
    print("="*60)
    
    model = fasttext.train_supervised(
        input=train_file,
        lr=0.5,
        epoch=100,           # More epochs
        wordNgrams=2,
        dim=50,              # Medium dimensions
        bucket=100000,       # Medium bucket
        loss='softmax',
        verbose=2
    )
    
    model.save_model(str(output_path))
    
    # Get model size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\n✅ Medium model saved: {output_path}")
    print(f"   Size: {size_mb:.2f} MB")
    
    return model


def validate_model(model, valid_file, model_name):
    """Validate model accuracy"""
    print(f"\n📊 Validating {model_name}...")
    
    # Test on validation set
    result = model.test(valid_file)
    n_samples = result[0]
    precision = result[1]
    recall = result[2]
    
    print(f"   Samples: {n_samples}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   Accuracy: {precision:.2%}")
    
    return precision


def test_inference_speed(model, test_texts, model_name):
    """Test inference speed"""
    import time
    
    print(f"\n⏱️  Testing {model_name} inference speed...")
    
    # Warmup
    for text in test_texts[:5]:
        model.predict(text)
    
    # Measure
    start = time.time()
    for text in test_texts:
        model.predict(text)
    elapsed = time.time() - start
    
    avg_time_ms = (elapsed / len(test_texts)) * 1000
    print(f"   Avg inference time: {avg_time_ms:.2f} ms")
    
    return avg_time_ms


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train FastText intent models")
    parser.add_argument("--data", "-d", default="data/training_data_menu.csv",
                        help="Path to training data CSV")
    parser.add_argument("--models-dir", "-m", default="models",
                        help="Directory to save models")
    args = parser.parse_args()
    
    # Paths
    csv_path = args.data
    models_dir = Path(args.models_dir)
    models_dir.mkdir(exist_ok=True)
    
    train_file = "training_data_fasttext.txt"
    valid_file = "validation_data_fasttext.txt"
    
    small_model_path = models_dir / "restaurant_intent_small.bin"
    medium_model_path = models_dir / "restaurant_intent_medium.bin"
    
    # Load data
    print(f"📂 Loading training data from: {csv_path}")
    df = load_training_data(csv_path)
    
    # Prepare FastText format
    prepare_fasttext_data(df, train_file, valid_file, train_split=0.85)
    
    # Train both models
    small_model = train_small_model(train_file, small_model_path)
    medium_model = train_medium_model(train_file, medium_model_path)
    
    # Validate both
    small_acc = validate_model(small_model, valid_file, "SMALL")
    medium_acc = validate_model(medium_model, valid_file, "MEDIUM")
    
    # Test inference speed
    test_texts = [
        "i want a large pizza with pepperoni",
        "show me the menu",
        "add two burgers to my order",
        "what's my total",
        "cancel that",
        "can i get a coke",
        "remove the fries",
        "checkout please",
        "do you have desserts",
        "gimme something spicy"
    ]
    
    small_time = test_inference_speed(small_model, test_texts, "SMALL")
    medium_time = test_inference_speed(medium_model, test_texts, "MEDIUM")
    
    # Summary
    print("\n" + "="*60)
    print("📊 COMPARISON SUMMARY")
    print("="*60)
    
    small_size = Path(small_model_path).stat().st_size / (1024 * 1024)
    medium_size = Path(medium_model_path).stat().st_size / (1024 * 1024)
    
    print(f"\n{'Model':<15} {'Size':<12} {'Accuracy':<12} {'Inference'}")
    print("-" * 60)
    print(f"{'SMALL':<15} {small_size:>6.2f} MB   {small_acc:>7.2%}     {small_time:>6.2f} ms")
    print(f"{'MEDIUM':<15} {medium_size:>6.2f} MB   {medium_acc:>7.2%}     {medium_time:>6.2f} ms")
    
    print("\n💡 Recommendation:")
    if small_acc >= 0.94 and small_time < 50:
        print("   ✅ Use SMALL model - Good accuracy, fast inference, low memory")
    elif medium_acc - small_acc > 0.02:
        print("   ✅ Use MEDIUM model - Better accuracy worth the extra resources")
    else:
        print("   ⚖️  Both models viable - test on actual Jetson hardware")
    
    print("\n📁 Models saved:")
    print(f"   - {small_model_path}")
    print(f"   - {medium_model_path}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
