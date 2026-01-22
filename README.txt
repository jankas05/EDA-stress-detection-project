Description:
This projects goal is to replicate the findings of the 2023 paper:
L. Zhu et al., "Stress Detection Through Wrist-Based Electrodermal Activity Monitoring and Machine Learning," in IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 5, pp. 2155-2165, May 2023, doi: 10.1109/JBHI.2023.3239305.
Currently the code also evaluates the component seperation methods of Neurokit and NaN handling methods. 

Setup: 
- ensure the python installation is 3.12.x
- run setup.py 
- run main.py to gather results or look at the existing results in results/

Currently following functionalities are implemented:
-Segmentation of the signal
-Component Separation
-Feature Extraction
-Database creation
-CSV export of database
-Data splitting
-Classification using machine learning techniques
-Testing and saving of results
-Accuracy review
-Comparison of Neurokit EDA component seperation methods
-Multiple functions to illustrate the process(plotting of segments, plotting of radviz, boxplot diagrams, ... )

Planned functionalities:
-currently no further development is planned

Optional functionalities:
-low pass filter vs no filter comparison
-more datasets
-multimodal approach( EEG, PPA,...)


Research papers used in this project:  
- L. Zhu et al., "Stress Detection Through Wrist-Based Electrodermal Activity Monitoring and Machine Learning," 
  in IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 5, pp. 2155-2165, May 2023
- A. Greco, G. Valenza, A. Lanata, E. P. Scilingo and L. Citi, "cvxEDA: A Convex Optimization Approach to Electrodermal Activity Processing," 
  in IEEE Transactions on Biomedical Engineering, vol. 63, no. 4, pp. 797-804, April 2016
- J. Birjandtalab, D. Cogan, M. B. Pouyan and M. Nourani, "A Non-EEG Biosignals Dataset for Assessment and Visualization of Neurological Status," 
  2016 IEEE International Workshop on Signal Processing Systems (SiPS), Dallas, TX, USA, 2016, pp. 110-114
