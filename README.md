# Stress Detection via Wrist-Based EDA and Machine Learning

## Project Description

The goal of this project is to replicate and evaluate the findings of the 2023 paper:

> **L. Zhu et al.**, *Stress Detection Through Wrist-Based Electrodermal Activity Monitoring and Machine Learning*,  
> IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 5, pp. 2155–2165, May 2023.  
> DOI: 10.1109/JBHI.2023.3239305

In addition to replication, the project evaluates:
- Different EDA component separation methods provided by NeuroKit
- Various NaN handling strategies

---

## Setup Instructions

1. Ensure your Python installation is **Python 3.12.x**
2. Run the setup script:
   ```bash
   python setup.py
   ```
3. Run the main pipeline:
   ```bash
   python main.py
   ```
4. Or look at the results in `results/`
   - Newly generated results will replace the old results 

---

## Implemented Functionalities

- Signal segmentation
- EDA component separation
- Feature extraction
- Database creation
- CSV export of the database
- Data splitting
- Machine learning–based classification
- Testing and result persistence
- Accuracy evaluation
- Comparison of NeuroKit EDA component separation methods
- Visualization utilities, including:
  - Segment plots
  - Radviz plots
  - Boxplots
  - Additional illustrative plots for the processing pipeline

---

## Planned Functionalities

- No further development is currently planned

---

## Optional / Experimental Functionalities

- Low-pass filter vs. no-filter comparison
- Integration of additional datasets
- Multimodal approaches (e.g., EEG, PPG)

---

## Research Papers Used

- **L. Zhu et al.**, *Stress Detection Through Wrist-Based Electrodermal Activity Monitoring and Machine Learning*,  
  IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 5, pp. 2155–2165, May 2023

- **A. Greco et al.**, *cvxEDA: A Convex Optimization Approach to Electrodermal Activity Processing*,  
  IEEE Transactions on Biomedical Engineering, vol. 63, no. 4, pp. 797–804, April 2016

- **J. Birjandtalab et al.**, *A Non-EEG Biosignals Dataset for Assessment and Visualization of Neurological Status*,  
  IEEE International Workshop on Signal Processing Systems (SiPS), Dallas, TX, USA, 2016, pp. 110–114
