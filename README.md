### Repo for Paper ["Only a Little to the Left: A Theory-grounded Measure of Political Bias in Large Language Models"](https://arxiv.org/abs/2503.16148) (Accepted at ACL 2025, Main conference)

## Directory Structure

**analysis/** - Contains all plots, data analysis notebooks, and robustness checks for bias analysis  
**stance_detector/** - Contains all files related to the stance detection model implementation  
**dataset/** - Contains scripts for dataset preparation and synthetic label generation  

## Directories that need to be downloaded from GDrive
**data/** - Contains processed bias dictionaries, statistics, and comparison data files  
**model/** - Contains stance detector model files 


### Analysis Directory
- `figure_1.ipynb` - Generates Figure 1: Political bias for instruction vs. base models over the two dimensions of political ideology, disaggregated by the measurement instruments 
- `figure 5.ipynb` - Creates Figure 5 : Political bias disaggregated by prefix and data source. Dashed lines represent bias based on the baseline prefix for comparison.
- `figure_4.ipynb` - Produces Figure 4: Difference in political bias between data sources.
- `figure_6.ipynb` - Generates Figure 6: Difference in political bias induced by likert and baseline prefixes to mean bias of other responses.
- `helper_functions.py` - Contains utility functions used across analysis notebooks for data processing and visualization
- `robustness_check.ipynb` - Performs robustness check analysis with original and Llama 405B 3.1 generated variations. 
- `summary_table.ipynb` - Generates summary tables of bias analysis results

### Stance Detector Directory
- `eval.py` - Evaluates stance detection model performance on test datasets
- `figure_3.py` - Generates Figure 3: Contrasting the performance of the zero- shot and fine-tuned stance classifier. 
- `finetune-bart.py` - Fine-tunes BART model for stance detection task
- `sampling.py` - Handles data sampling strategies for stance detection training/evaluation

### Dataset Directory
- `get_synth_labels.py` - Generates synthetic labels for training data augmentation

## External Data Links
data: https://drive.google.com/drive/folders/1WXNrLzSo7RqZtGmrVpmqSQGVjfL8YgM9?usp=sharing  
stance detector model files: https://drive.google.com/drive/folders/1XdXieCpV1YxomS7h7jFcZDs3dCCkTtWD?usp=sharing

If you use our code and datasets, please consider citing our paper:

- Faulborn, M., Sen, I., Pellert, M., Spitz, A., & Garcia, D. (2025). Only a Little to the Left: A Theory-grounded Measure of Political Bias in Large Language Models. arXiv preprint arXiv:2503.16148. To appear at ACL 2025
