# Optimizing-Duckworth-Lewis-Method
## Duckworth–Lewis Resource Model Implementation

## Overview

This project implements the **Duckworth–Lewis (DL) resource model** used in limited-overs cricket to estimate the expected runs remaining based on the number of **overs remaining** and **wickets in hand**.

The implementation estimates the model parameters directly from historical One Day International (ODI) cricket data using nonlinear optimization. The final model produces smooth run-production curves for each wicket state and evaluates the quality of fit using a normalized likelihood-based loss function.

---

## Objectives

* Preprocess historical ODI match data.
* Estimate initial resource parameters for every wicket state.
* Fit the Duckworth–Lewis exponential run production model.
* Learn a common decay parameter across all wicket states.
* Generate run production curves.
* Evaluate the model using normalized loss.

---

## Mathematical Model

Let

* **u** = overs remaining
* **w** = wickets in hand
* **Z(u,w)** = expected runs remaining

The Duckworth–Lewis model assumes an exponential run production function

```text
Z(u,w) = Z₀(w) × (1 − exp(−L × u / Z₀(w)))
```

where

* **Z₀(w)** is the total scoring potential with **w** wickets remaining.
* **L** is the common decay parameter controlling scoring rate.

As the number of remaining overs increases,

```text
lim(u → ∞) Z(u,w) = Z₀(w)
```

indicating that the expected runs asymptotically approach the maximum scoring potential.

---

## Loss Function

The model parameters are estimated by minimizing the negative log-likelihood loss

```text
Loss(ŷ,y) =
(ŷ + 1)
log((ŷ + 1)/(y + 1))
− ŷ + y
```

where

* **y** = observed runs remaining
* **ŷ** = predicted runs remaining

The total optimization objective is

```text
min Σ Loss(ŷ,y)
```

across all observations.

---

## Optimization Strategy

The model is trained in two stages.

### Stage 1: Preliminary Model

For every wicket state,

* Estimate an independent **Z₀(w)**
* Estimate an independent **L(w)**

using nonlinear optimization.

---

### Stage 2: Final Model

Compute a weighted average

```text
L = Σ Ni Li / Σ Ni
```

where

* **Ni** is the number of observations for wicket state *i*.

Keeping this common **L** fixed,

only **Z₀(w)** is re-estimated for each wicket.

This produces the final Duckworth–Lewis resource model containing

* 10 values of **Z₀(w)**
* 1 common decay parameter **L**

---

## Methodology

1. Load historical ODI match data.
2. Filter first innings only.
3. Compute overs remaining.
4. Remove incomplete observations.
5. Estimate initial scoring potentials.
6. Fit preliminary model using nonlinear optimization.
7. Estimate common decay parameter.
8. Refit final model.
9. Generate resource curves.
10. Compute normalized loss.

---

## Features

* Data preprocessing using Pandas
* Duckworth–Lewis exponential resource model
* Nonlinear parameter estimation using SciPy
* Likelihood-based loss function
* Weighted estimation of common decay parameter
* Resource curve visualization
* Normalized loss evaluation

---

## Technologies Used

* Python
* NumPy
* Pandas
* SciPy
* Matplotlib

---

## Project Structure

```text
.
├── data/
│   └── cricket_1999to2011.csv
│
├── code.py
├── README.md
│
└── results/
    ├── preliminary_fit.png
    └── best_fit.png
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/duckworth-lewis-model.git

cd duckworth-lewis-model
```

Install dependencies

```bash
pip install numpy pandas scipy matplotlib pandas
```

---

## Running the Project

Update the dataset path inside the script and execute

```bash
python code.py
```

The program will

* preprocess the cricket dataset,
* estimate the preliminary parameters,
* compute the common decay parameter,
* fit the final resource model,
* generate run production plots,
* report the normalized loss.

---

## Results

The model estimates

* Initial scoring potential **Z₀(w)** for every wicket state.
* Common decay parameter **L**.
* Resource curves for wickets 1–10.
* Normalized likelihood loss.

The generated plots illustrate how expected runs increase with overs remaining and with additional wickets in hand.

---

## Future Improvements

* Compare with the Duckworth–Lewis–Stern (DLS) model.
* Cross-validation using multiple cricket seasons.
* Bayesian parameter estimation.
* Interactive visualization of resource tables.
* Predict revised targets for rain-interrupted matches.

---

## References

1. Duckworth, F. C., & Lewis, A. J. (1998). *A Fair Method for Resetting the Target in Interrupted One-Day Cricket Matches.*
2. Stern, S. (2014). *The Duckworth–Lewis–Stern Method.*
3. SciPy Optimization Documentation.

---

## License

This project is released under the MIT License.

---

## Author

Developed as part of a sports analytics and statistical modeling project to implement the Duckworth–Lewis resource model from first principles using scientific computing in Python.
