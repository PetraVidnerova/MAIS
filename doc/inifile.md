# INI FILE

All the user adjustable configuration is defined by an INI file. The INI file is parsed by python's
module [configparser](https://docs.python.org/3/library/configparser.html) and follows the common INI file syntax.

This document lists all the sections and their keys.

# TASK

|key|type|default|meaning|
|---|---|---|---|
|duration|number|60|duration of the simulation in days|
|print_interval|number|1|period of printing the information of numbers of nodes in individual states, in days, -1 = never print| 
|verbose|string|Yes|Use no for no output during simulation| 
|model|string|SimulationDrivenModel|type of model (select from InfoSIRModel, InfoTippingModel, SimulationDrivenModel; other Models are no more supported, use the at your own risk)|
|save_node_states|string|No|if "Yes", states of all nodes are saved| 
|monitor_node|number|None|If you provide a number of a node, this node is tracked and information about state changes, testing, quarantine, etc. are printed. Hint: Gabriela is 29681|
|output_dir|string|.|path to directory which is used for saving output files|

# GRAPH

|key|type|default|meaning|
|---|---|---|---|
|type|string|light|only "light" is supported by SimulationDrivenModel; for backward compatibility with older models| 
|nodes|string|p.csv|filename for nodes|
|edges|string|edges.csv|filename for edges|
|layers|string|etypes.csv|filename for etypes| 
|externals|string|e.csv|filename for export nodes|
|nums|string|nums.csv|not used; for backward compatibility| 
|objects|string|objects.csv|filename for objects| 
|quarantine|string|--|filename for quarantine specification, optional, needed only by graphs with custom layers|
|layer_groups|string|--|layer groups definition for contact tracing, optional, needed only by graphs with custom layers|
|file|string|town.pickle|filename to save the pickled graph|

# MODEL and POLICY Sections 

MODEL and POLICY sections depend on the model type used. 

## Information Spread Models Configuration 

<details>
<summary> click to expand </summary>
<p>
  
For **InfoSirModel** use: 

# MODEL

|key|type|default|meaning|
|---|---|---|---|
|beta|number|0|transmission rate|
|I_duration|number|1|time the I state|
|init_X|number|0|initial number of nodes in state X; the rest of nodes is asigned to S| 

<p>
  
For **InfoTippingModel** use:

# MODEL

|key|type|default|meaning|
|---|---|---|---|
|theta|number|0|transmission threashold|
|init_Active|number|0|initial number of nodes in the Active state; the rest of nodes is asigned to S| 

<p>
Both InfoSIRModel and InfoTipping model can be used together with Spreader policy. The corresponding config section is as follows:

# POLICY

|key|type|default|meaning|
|---|---|---|---|
|filename|string|info_spreader|the filename with the policy code|
|name|string|Spreader|the name of the policy class|

# POLICY_SETUP

|key|type|default|meaning|
|---|---|---|---|
|quantile|string|1.0|centrality quantile to choose node for seeding|
</details>

## Infection (covid) Spread Model Configuration 

<details>
<summary> click to expand </summary>
<p>

  For **SimulationDrivenModel** (infection spread) use:

# MODEL

|key|type|default|meaning|
|---|---|---|---|
|start_day|number|0|day to start the simulation|
|durations_file|string|../config/duration_probs.json|file with probs for calculation durations of RNA positivity, infectious time and incubation perios|
|prob_death_file|string|../data/prob_death.csv|file with probabilities of death by age|
|mu|number|0|multiplies the probability of death|
|ext_epi|number|0|probability of beeing infectious for external nodes|
|beta|number|0|rate of transmission (upon exposure) (note that transmission depands also on intensity of edge of contact)|
|beta_reduction|number|0|multiplication coefficient for beta of asymptomatic nodes|
|theta_Is|number|0|prob of being tested after the decision of going to the test is made|      
|test_rate|number|0|prob of deciding to go for test if symptomatic|
|asymptomatic_rate|number|0|rate of asymtomatic flow after being exposed|
|init_X|number|0|initial number of nodes in state X; the rest of nodes is asigned to S| 

# POLICY

|key|type|default|meaning|
|---|---|---|---|
|filename|string|None|filename for policy code|
|name|string|None|name of the policy object| 

# POLICY SETUP

This depends on your policy. The following parameters are used for `customised_policy`. This policy enables you to
control various parameters of the model and also to run other policies.

|key|type|default|meaning|
|---|---|---|---|
|layer_changes_filename|string|None|layer weights calendar, csv file|
|policy_calendar_filename|string|None|json file with policy calendar|
|beta_factor_filename|string|None|csv file, values between 0.0 and 1.0, compliance with protective meassures|
|face_masks_filename|string|None|csv file, values between 0.0 and 1.0, compliance with wearing masks|
|theta_filename|string|None|csv file, multiplication of theta_Is calendar| 
|test_rate_filename|string|None|csv file, multiplication of test rate calender|
|init_filename|string|None|json, additional seeding with E nodes|
|reduction_coef1|number|1.0|controls reduction by wearing masks|
|reduction_coef2|number|1.0|controls reduction by protective meassures|
|new_beta|string|None|must be 'Yes', for backward compability| 
|sub_policies|comma separated|None|list of aliases for other policies to run|
|<POLICY_ALIAS>_filename|string|None|name of file with policy code|
|<POLICY_ALIAS>_name|string|None|name of policy object|
|<POLICY_ALIAS>_config|string|None|file with policy config, see [POLICY](policy.md)|

</details>
