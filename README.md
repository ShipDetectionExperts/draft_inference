# Vessel detection as an Application Package 

This repository contains the code for the vessel detection application package. It is intended  to be on the EOEPCA platform.

## What is an Application Package?

A platform independent and self-contained representation of an Application, providing executables, metadata and dependencies such that it can be deployed to and executed within an Exploitation Platform. Learn more about the Application Package [here](https://docs.ogc.org/bp/20-089r1.html#toc0).

## What is the vessel detection application package?

The vessel detection application package is an ongoing project of demonstrating the capabilities of the EOEPCA platform. The Application Package contains a machine learning model trained to detect vessels on Sentinel 2 imagery. 

## How to use the vessel detection application package?


## Contributors


# draft_inference

A sample jupyter notebook for showcasing the inference capabilities of the vessel detection model.

# Simple tutorial usage

0. Clone or fork this repository

```bash
cd draft_inference
```

1. Create a virtual environment

```bash
python3 -m venv vessel-detection
```

2. Activate the environment

```bash
source vessel-detection/bin/activate
```

3. Install the required libraries

```bash
pip install -r requirements.txt
```

4. Explore the *inference_notebook.ipynb* notebook
