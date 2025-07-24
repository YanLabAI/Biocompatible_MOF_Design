# Machine Learning-Assisted Discovery of Biocompatible Metal-Organic Frameworks

This repository contains the data and code for our manuscript: "Machine Learning-Assisted Discovery of Biocompatible Metal Organic Frameworks Using Simulation Data and Toxicological Profiles: Adsorptive Removal of Benzene Series Pollutants from Water".

This project develops and applies a machine learning framework to identify novel, high-performance, and biocompatible Metal-Organic Frameworks (MOFs) for environmental remediation. The framework focuses on two primary predictive tasks:
1.  Predicting the adsorption capacities of MOFs for benzene and toluene.
2.  Developing models to predict the aquatic toxicity of MOF organic linkers and classify their environmental risk (PMT/vPvM).

---


-   **/Data**: Contains all datasets used for model training and validation.
-   **/code**: Contains the main python file (`ML.py`) with the complete machine learning pipeline.

---

## Datasets

The following datasets are provided in the `/Data` directory:

-   `MOF adsorption dataset.csv`: Contains the adsorption values (adsorption capacities) of MOF for the Benzene and Toluene.
-   `IBC50.csv`: Ecotoxicity data for 1,188 compounds, detailing the 15-minute 50% bioluminescence inhibition concentration in *Vibrio fischeri*.
-   `IGC50.csv`: Ecotoxicity data for 1,792 compounds, detailing the 40-hour 50% growth inhibition concentration in *Tetrahymena pyriformis*.
-   `LC50DM.csv`: Ecotoxicity data for 353 compounds, detailing the 48-hour 50% lethal concentration for *Daphnia magna*.
-   `LC50.csv`: Ecotoxicity data for 823 compounds, detailing the 96-hour 50% lethal concentration for the fathead minnow (*Pimephales promelas*).
-   `PMT.csv`: Dataset of 3,111 chemicals with validated classifications for Persistent, Mobile, and Toxic (PMT) or very Persistent and very Mobile (vPvM) characteristics.

---


## Getting Started

Follow these steps to set up the environment and run the analysis.

### Running the code

This project uses Python 3.9. You will need to install the required libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap

git clone https://github.com/YanLabAI/Biocompatible_MOF_Design.git
cd Biocompatible_MOF_Design

python ML.py
