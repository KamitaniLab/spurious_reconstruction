To reproduce and plot the Figure 8, use the following command at the Spurious_reconstruction directory:

```
python ./analysis/2_formal-analysis/cluster_data_simulation/Figure_output_dimension_collapse.py
```

To reproduce the simulation analysis realted to Figure 9 B, D and S3 FigB, use the following command at the Spurious_reconstruction directory:


```
python ./analysis/2_formal-analysis/cluster_data_simulation/simulation_cluster_identificaiton.py
```

Alternatively, you can skip this heavy and time-consuming analysis by downloading the resutls:

```
python ./download.py  "simulation results"
```

Then you can reproduce these figures by running:
```
python ./analysis/2_formal-analysis/cluster_data_simulation/Figure_simulation_cluster_identification.py

```

To reproduce the results of Figure 9E, use the following command:
```
python ./analysis/2_formal-analysis/cluster_data_simulation/Figure_simulation_cluster_identification_simple_case.py
```



To reproduce the simulation analysis realted to S3 FigA, use the following command:
```
python ./analysis/2_formal-analysis/cluster_data_simulation/simulation_cluster_identificaiton_change_ratio.py
```


Then you can reproduce this figure by running:
```
python ./analysis/2_formal-analysis/cluster_data_simulation/Figure_simulation_cluster_identification_change_ratio.py

```