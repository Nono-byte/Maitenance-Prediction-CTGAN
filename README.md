# 🏭 Predictive Maintenance — Synthetic Data Generation & Failure Classification

*Building a custom Conditional Tabular GAN (CTGAN) from scratch to solve extreme class imbalance in CNC machine failure detection — and quantifying the business value of getting it right.*

---

## Project Overview

Real industrial sensor datasets are imbalanced by design. Machines are built not to fail. When they do, failure events are rare — sometimes as few as 12 records across an entire dataset of 10,000 observations.

A standard classifier trained on this raw data learns one thing: predict "No Failure" for everything. It achieves 97% accuracy and catches zero actual failures. This is not a model. It is a liability.

This project builds a custom Conditional Tabular GAN (CTGAN) from scratch to synthesise realistic failure records, augments the training set, and quantifies the exact lift in multi-class failure classification — translated directly into annual business value.

---

## Business Problem

A CNC manufacturing facility runs 100 machines continuously. Each machine fails approximately 3 times per year. An unplanned failure costs $50,000 in downtime. A planned maintenance intervention — triggered by an accurate prediction — costs $5,000. Every failure caught early saves $45,000.

The problem is not the model. The problem is that the model cannot learn to predict failures when it has seen only 12–70 examples of each failure type. Synthetic data is the solution.

---

## Results

The baseline Gradient Boosting classifier, trained on raw imbalanced data, achieved a Macro F1 of 0.1649. Every single minority failure class scored F1 = 0.00. The model predicted "No Failure" for everything.

After augmenting the training set with 2,796 CTGAN-generated failure records, the same classifier improved to a Macro F1 of 0.2319 — a lift of 40.7%. Macro Recall improved from 16.65% to 24.91%. Failure classes that were previously invisible to the model became detectable for the first time.

For a 100-machine fleet, this recall improvement translates to 24.78 additional failures caught per year. At $45,000 savings per catch, the incremental annual value of CTGAN augmentation is approximately $1,115,100.

---

## Dataset

The dataset mirrors the NASA AI4I 2020 Predictive Maintenance Dataset, simulating a CNC milling machine with five sensor streams: air temperature, process temperature, rotational speed, torque, and tool wear. Two derived features are added — power (torque × speed / 9550) and temperature differential (process minus air). Sensor correlations are physics-informed: process temperature tracks air temperature, and torque has an inverse relationship with rotational speed.

The dataset contains 10,000 records across six classes. No Failure accounts for 9,796 records. The five failure classes range from 70 samples for Tool Wear down to just 12 for Random Failure. Tool Wear failures are triggered by high tool wear combined with high torque. Heat Failures arise from high temperature differential at low rotational speed. Power Failures follow excessive power draw. Overstrain occurs when the product of tool wear and torque exceeds a threshold. Random Failures have no sensor signal — they are unpredictable by definition.

The rarest class has 12 examples competing against 9,796 counter-examples. No standard classifier survives that ratio intact.

---

## CTGAN Architecture

This is a custom implementation built from scratch in pure NumPy — not a library import. The decision to build rather than import is deliberate: understanding every component is what allows you to diagnose failures in production, not just run a notebook.

The generator uses a two-phase strategy. The first phase fits a Gaussian Mixture Model independently for each failure class, using only the real samples from that class. A GMM captures the full covariance structure between sensors — not just how torque behaves individually, but how torque and tool wear move together specifically during Tool Wear failures. This distinction matters. GMMs work reliably with as few as 12 training samples, which makes them the right tool for this problem.

The second phase trains a three-layer neural network to learn the cross-feature interactions the GMM misses. The architecture takes 32-dimensional Gaussian noise as input, passes it through two hidden layers of 64 units with LeakyReLU activations, and outputs synthetic sensor values through a tanh layer. Instead of the standard adversarial loss — which is unstable when training samples number in the single digits — the generator is trained with a distribution-matching loss. This loss has three terms: mean matching, standard deviation matching, and correlation matrix matching. The correlation term is the critical one for physics-constrained industrial data. A synthetic Tool Wear failure where tool wear and torque are uncorrelated is physically impossible and will corrupt the model if it enters the training set.

Each synthetic record is a blend of 70% GMM sample and 30% GAN-generated sample. The GMM provides statistical stability. The GAN provides realistic variation. Both are conditioned on a specific failure class label — Tool Wear generators only produce Tool Wear records, not generic failures.

---

## Synthetic Data Quality Validation

Before any synthetic record enters the training set it is validated against two standards. Jensen-Shannon Divergence is computed per feature per class, measuring how similar the synthetic distribution is to the real distribution on a scale from 0 (identical) to 1 (completely different). Any feature-class combination scoring above 0.20 triggers retraining of the generator. The target is below 0.10.

Inter-sensor correlation matrices are computed separately for real and synthetic data and plotted side by side. The mean absolute difference between the two matrices is reported, with a target below 0.05. A model trained on synthetic data that violates known physical relationships between sensors will learn the wrong patterns — this validation step catches that before it reaches training.

---

## How to Run

```bash
git clone https://github.com/yourusername/predictive-maintenance-ctgan.git
cd predictive-maintenance-ctgan
pip install pandas numpy scikit-learn scipy matplotlib
jupyter notebook notebooks/predictive_maintenance_ctgan.ipynb
```

The dataset is generated programmatically inside the notebook. No external data file is required.

---

## Key Design Decisions

The SDV/CTGAN library is excellent for production use. Building from scratch forces explicit understanding of every component — why GMM before GAN, why distribution-matching loss instead of adversarial loss, why the 70/30 blend ratio. That understanding is what allows you to diagnose when something goes wrong in production.

The generator is trained only on failure records, not the full dataset. Training it alongside 9,796 "No Failure" records would force it to spend most of its capacity learning the majority class — the one class we already have plenty of. Separate generators are trained per failure class, each conditioned on that class's sensor signature.

Macro F1 is used as the primary metric rather than accuracy or weighted F1. A model predicting "No Failure" 100% of the time achieves 97.9% accuracy while catching zero failures. Macro F1 weights every class equally and penalises the model for ignoring minority classes. It is the only metric that tells the truth on severely imbalanced data.

The baseline and augmented models use identical hyperparameters and random seeds. If you change both the data and the model simultaneously, you cannot isolate what caused the improvement. Any performance difference between baseline and augmented is attributable solely to data augmentation.

---

## Limitations

Minority class test set sizes are 3–14 samples. Per-class F1 carries high variance at this scale. A minimum of 50–100 real failure examples per class are needed for reliable test-set metrics in a production deployment.

Finite-difference gradients are slow and approximate. A production version of this generator would use PyTorch or JAX with automatic differentiation for faster, more accurate training.

The 70/30 GMM/GAN blend ratio was set manually through experimentation. It could be treated as a hyperparameter and optimised via cross-validation on a validation set held out from the real failure data.

JSD and correlation checks validate that synthetic distributions match real distributions. They do not validate that synthetic data generalises to truly unseen failures collected after training. Temporal holdout data — failures that occurred after the training period — is the only gold standard for that test.

Random Failure has no sensor signal by definition. No amount of synthetic data or modelling sophistication will predict it reliably. A model that returns low confidence on Random Failure is behaving correctly, not failing. This is an important distinction when presenting results to stakeholders.

---

## Related Projects

[Credit Risk Scorecard](https://github.com/yourusername/credit-risk-scorecard) — End-to-end credit scoring pipeline with FICO-style scorecard and Power BI outputs.

---

## Author

**Nonkululeko Ntuli**
Freelance Data Scientist

---

*This project is part of a portfolio demonstrating how synthetic data generation solves real industrial data problems. The business impact framing is intentional — models that cannot be translated into dollars do not get deployed.*
