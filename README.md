# MNIST Classification with GitHub Actions

![Build Status](https://github.com/[YOUR_USERNAME]/GitHubActions_MNIST/actions/workflows/model_tests.yml/badge.svg)

This repository contains a lightweight CNN model for MNIST digit classification that achieves >95% accuracy in a single epoch while maintaining fewer than 25,000 parameters.

## Model Architecture

The model uses a carefully designed CNN architecture with the following key features:
- Multiple convolutional layers with batch normalization
- Max pooling to reduce spatial dimensions
- Efficient channel progression (8 → 16 → 32 → 10)
- Global average pooling instead of dense layers
- Total parameters: < 25,000

## Requirements

- Python 3.8+
- PyTorch 1.7.0+
- torchvision 0.8.1+
- pytest 6.0.0+
- tqdm 4.50.0+

Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

The model includes:
- Image augmentation (rotation and translation)
- Batch normalization for faster convergence
- SGD optimizer with momentum
- One epoch training to achieve >95% accuracy

To train the model:
```bash
python train.py
```

## Tests

The repository includes comprehensive tests that verify:
1. Model parameter count (< 25,000)
2. Output shape correctness
3. Forward pass stability
4. Training accuracy (>95% in one epoch)
5. Image augmentation effectiveness

Run tests:
```bash
pytest test_model.py -v
```

## GitHub Actions

The repository uses GitHub Actions to automatically:
- Run all tests on every push and pull request
- Verify model meets parameter constraints
- Confirm accuracy requirements
- Validate image augmentation
- Display build status badge

## Image Augmentation

The training pipeline includes robust image augmentation:
- Random rotation (-7° to 7°)
- Random translation (±10% in both dimensions)
- Verified on 100+ images during testing

## License

MIT
