# DECAF: An Interpretable Deep Cascading Framework for ICU Mortality Prediction

./data is used in the experiment, including the data of 6 wards, and each ward has the following data:
（1）exam_disease: Ward network structure
（2）hadm_death: Whether the patient died in hospital, 1 is death, 0 is survival
（3）hadm_record: Abnormal examination and disease of all patients during hospitalization
（4）Initial_character: Initial characteristics of network nodes
（5）Initial_failure_probability: Initial failure probability of patients in disease network
（6）weight: Weights of directed edges in networks

Program start
First，run ./get_CF_curve_node_eff_pro.py 或 get_CF_curve_node_isfail.py
Second，run ./predict_GRU_CUDA.py ./predict_lstm_CUDA.py ./predict_Transformer_CUDA.py
