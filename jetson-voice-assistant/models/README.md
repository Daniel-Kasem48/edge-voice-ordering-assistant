# Models Directory

This directory contains FastText intent classification models. Models are not included in git due to their size.

## Required Models

### Option 1: Train from Menu (Recommended)
```bash
# Generate training data from your menu
python3 generate_training_from_menu.py --menu data/menu.json

# Train models
python3 train_two_models.py --data data/training_data_menu.csv
```

This will create:
- `restaurant_intent_small.bin` (5MB) - Fast, lower accuracy
- `restaurant_intent_medium.bin` (19MB) - Better accuracy, recommended

### Option 2: Download Pre-trained
Contact the development team for pre-trained models if available.

## Model Specifications

| Model | Size | Accuracy | Use Case |
|-------|------|----------|----------|
| Small | 5MB | ~77% | Low-memory devices |
| Medium | 19MB | ~80% | Production (recommended) |

## Training Your Own

The models are trained on your specific menu items, so they understand:
- Your item names (e.g., "Margherita Pizza", "Latte")
- Your categories (e.g., "pizza", "drink", "dessert")
- Common variations and typos

Training takes ~10 seconds on a laptop.
