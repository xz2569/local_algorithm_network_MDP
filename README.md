
**Preparation step** 

1. make sure all required python packages and license are installed
* networkx
* pandas
* gurobipy (license required) 
* matplotlib

2. make sure the following folders are created
* graphs
* instances
* payoffs
* results
* solns

**Experiment**

1. set up graph size
$ s=500

2. generate instance: underlying graph saved to the graphs/ folder; instances saved to the instances/ folder.
$ python3 ./generate_instance.py $s

3. find solutions: solutions saved to solns/ folder
$ for i in {0..9}; do for c in 0.1 0.2 0.3 0.4 0.5; do for L in 0 2 4 6 -1; do python3 ./find_solution.py $s $c $L $i; done; done; done

4. compute payoff (need to edit file `compute_payoff.py`: `cs` and `Ls`): payoffs saved to payoffs/ folder
$ for i in {0..9}; python3 ./compute_payoff.py $s $i; done

5. visualize result: plot saved to results/ folders
$ python3 ./visualize_results.py $s



