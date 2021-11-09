# FEBR: Expert-Based Recommendation Framework

## Prerequisite
###  Running environment

- python 3.7 (recommended version)
- Gym 0.17.2 (recommended version)
- Dopamine 0.7 (recommended version)
- dopamine-rl 3.0.1 (or later)
- matplotlib 3.0.3 (or later)
- numpy 1.16.3 (or later)
- pandas 0.24.2 (or later)
- scipy 1.4.1 (or later)
- scikit-learn 0.21.3 (or later)
- tensorflow 1.15.0 (used for recFSQ experiments, it requires this version to correctly work with Dopamine)
- tensorflow-estimator 1.15.1 (used for recFSQ experiments, it requires this version to correctly work with Dopamine)
- 'recsim 0.2.4' framework is cloned to our work under the Apache License 2.0 , and exploited to serve our development and research purposes. 

### Installation as a project repository
```
git clone https://github.com/FEBR-rec/ExpertDrivenRec.git
```
In this case, you need to manually install the dependencies.

## Presentation of the structure of the code source
We propose **FEBR**, a content-driven framework for video recommendations based on expert demonstrations. 
FEBR is contsructed based on RECSIM [Ie et al., 2019b] simulation plateforme (see 'recsim' folder and you can refer to its Github repository for more information).
The structure of the code source is as follows:
### AL/IRL component
1. file **ExpertRecEval.py**: we developed an expert evaluation environment that represents a dynamic recommendation environment 
   to simulate expert behavior, while watching and rating videos according to her domaine of expertise.
   We provide within this environment a simple evaluation model based on some video features (accuracy, importance, entertainment and pedagogy).

2. file **maxEnt_irl.py**: we implemented the Maximum entropy inverse reinforcement learning (MaxEnt-IRL) [Ziebart et al., 2008] for our 
   recommendation problem. 

3. file **rl.py**: implements some RL algorithm (value_iteration and policy_iteration) that can be used by the MaxEnt-IRL algorithm to optimize 
   the rewards and its derived policy. 

4. file **febr_al_irl.py** enables running AL/IRL model to learn the expert policy using a given configuration.
   You can run it by this commande: 
   > pyhton febr_al_irl.py num_topics slate_size corpus_size steps num_trajs num_experts 
   
   The outputs after execution are 3 files located in 'datasets_states' folder: policyTIMESTAMP.npy, rewardsTIMESTAMP.npy and statesTIMESTAMP.npy
   
***Note 1***: The execution can take a long time depending on the size of the simulation and the technical infrastructure provided.
 
***Note 2***: We only need 'policyTIMESTAMP.npy' and 'statesTIMESTAMP.npy' for the recommendation component.


### Recommendation component
This is the exploitation component of the expert policy learned by AL/IRL system. We used for
our simulation the 'interest evolution environment' from recsim framework.

6. file **irl_agent.py**: contains the class 'InverseRLAgent' that exploits the expert policy learned by the MaxEnt-IRL algorithm to build an 
   agent, which recommends slates to users using a classification algorithm based on useful comparison margins. 

7. file **eval_baseline.py**: enables to run the baseline methodes with our approach
   to build the comparison result figures:
   - w_t_comparTIMESTAMP.png: total watching time of all episodes for recNaive, recFEBR and recFSQ.
   - q_t_comparTIMESTAMP.png: average total quality of episodes for recNaive, recFEBR and recFSQ.
   These figures are stored  in 'eval_results' folder. The command to launch this experiment is:
   > python eval_baseline.py number_episodes slate_size corpus_size name_states_file name_policy_file 

8. file **models_eval.ipynb** (in folder 'notebooks'): is a jupyter notebook where you can produce the same experiments explained in point 7. 
   In addition, it provides additional visualisation graphs and statistical information. 


***Note 3***: Basically, both expert and end-user recommendations are supposed to happen in the same recommendation system. Then, while configuring
 experiments (AL/IRL and final recommendations that exploit AL/IRL results), the system should deal with same number of video topics for both expert
 and user environments, which can be fixed by initializing the variable NUM_FEATURES in classes 'IEvUserState' and 'IEvVideo' 
 (resp. 'ExpertUserState' and 'EEVideo') in the file 'interest_evolution' (resp. ExpertRecEval.py).
