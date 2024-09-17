# Simulation with clustered data

This directory contains the analysis related to the simulation studies in the paper, including the results of "output dimension collapse (Figure 8)" and "Simulation with clustered features (Figure 9)". You can follow the instructions below and reproduce the figures and results.

---

### Figure 8
To reproduce the result and plot Figure 8, run the following command from the `Spurious_reconstruction` directory:

```
python ./analysis/2_formal-analysis/cluster_data_simulation/Figure_output_dimension_collapse.py
```

### Figures 9B, 9D, and S3 FigB
To reproduce the simulation analysis related to Figures 9B, 9D, and S3 FigB, use this command:
```
python ./analysis/2_formal-analysis/cluster_data_simulation/simulation_cluster_identification.py
```

If you prefer to skip this resource-heavy and time-consuming analysis, you can download the pre-computed results:
```
python ./download.py "simulation results"
```
After downloading the results, reproduce the figures by running:
```
python ./analysis/2_formal-analysis/cluster_data_simulation/Figure_simulation_cluster_identification.py
```
### Figure 9E
To reproduce the results for Figure 9E, use the following command:
```
python ./analysis/2_formal-analysis/cluster_data_simulation/Figure_simulation_cluster_identification_simple_case.py
```
### S3 FigA
To reproduce the simulation analysis related to S3 FigA, use this command:
```
python ./analysis/2_formal-analysis/cluster_data_simulation/simulation_cluster_identification_change_ratio.py
```

Once the analysis is complete, you can reproduce the corresponding figure by running:
```
python ./analysis/2_formal-analysis/cluster_data_simulation/Figure_simulation_cluster_identification_change_ratio.py
```

---

Ensure that the necessary data files are prepared before running these commands.

