import numpy as np
import pandas as pd

import logging
import time
from models.engine import BaseEngine
from utils.history_utils import TimeSeries, TransitionHistory
import utils.global_configs as global_configs
from utils.global_configs import monitor



EXPECTED_NUM_DAYS = 300 

class SimulationEngine(BaseEngine):

    states = []
    num_states = len(states)
    state_str_dict = {}

    ext_code = 0

    transitions = []
    num_transitions =  len(transitions)

    final_states = []     # no transition from them 
    invisible_states = [] # does not count into population 
    unstable_states = []  # can change 


    fixed_model_parameters = {}
    model_parameters = {} 

    common_arguments = {
        "random_seed": (None, "random seed value"),
        "start_day": (1, "day to start")
    }
    

    def __init__(self, G, **kwargs):

        self.G = G  # backward compatibility
        self.graph = G

        self.init_kwargs = kwargs

        # 2. model initialization
        self.inicialization()

        # 3. time and history setup
        self.setup_series_and_time_keeping()

        # 4. init states and their counts
        self.states_and_counts_init(ext_nodes=self.num_ext_nodes,
                                    ext_code=self.ext_code)


        # 5. set callback to None
        self.periodic_update_callback = None

        self.T = self.start_day - 1

    def update_graph(self, new_G):
        if new_G is not None:
            self.G = new_G  # just for backward compability
            self.graph = new_G
            self.num_nodes = self.graph.num_nodes
            try:
                self.num_ext_nodes = self.graph.num_nodes - self.graph.num_base_nodes
            except AttributeError:
                #  for saved old graph
                self.num_ext_nodes = 0
            self.nodes = np.arange(self.graph.number_of_nodes).reshape(-1, 1)



    def inicialization(self):

        super().inicialization()

        # node indexes
        self.nodes = np.arange(self.graph.num_nodes).reshape(-1, 1)
        self.num_nodes = self.graph.num_nodes

        

    def setup_series_and_time_keeping(self):

        super().setup_series_and_time_keeping()
        

        tseries_len = self.num_transitions * self.num_nodes

        self.tseries = TimeSeries(tseries_len, dtype=float)
        self.history = TransitionHistory(tseries_len)

        # state history
        if global_configs.SAVE_NODES:
            history_len = EXPECTED_NUM_DAYS
        else:
            history_len = 1
        self.states_history = TransitionHistory(
            history_len, width=self.num_nodes)

        if global_configs.SAVE_DURATIONS:
            self.states_durations = {
                s: []
                for s in self.states
            }

        self.durations = np.zeros(self.num_nodes, dtype=int)

        # state_counts ... numbers of inidividuals in given states
        self.state_counts = {
            state: TimeSeries(EXPECTED_NUM_DAYS, dtype=int)
            for state in self.states
        }

        self.state_increments = {
            state: TimeSeries(EXPECTED_NUM_DAYS, dtype=int)
            for state in self.states
        }

        # N ... actual number of individuals in population
        self.N = TimeSeries(EXPECTED_NUM_DAYS, dtype=float)


    def states_and_counts_init(self, ext_nodes=None, ext_code=None):
        super().states_and_counts_init(ext_nodes, ext_code)

        # time to go until I move to the state state_to_go
        self.time_to_go = np.full(
            self.num_nodes, fill_value=-1, dtype="int32").reshape(-1, 1)
        self.state_to_go = np.full(
            self.num_nodes, fill_value=-1, dtype="int32").reshape(-1, 1)

        self.current_state = self.states_history[0].copy().reshape(-1, 1)


        # need update = need to recalculate time to go and state_to_go
        self.need_update = np.ones(self.num_nodes, dtype=bool)

        
    def daily_update(self, nodes):
        """
        Everyday checkup
        """
        pass 


    def change_states(self, nodes, target_state=None):
        """
        nodes that just entered a new state, update plan
        """
        # discard current state
        self.memberships[:, nodes == True] = 0


        for node in nodes.nonzero()[0]:
            if target_state is None:
                new_state = self.state_to_go[node][0]
            else:
                new_state = target_state
            old_state = self.current_state[node, 0]

            self.memberships[new_state, node] = 1
            self.state_counts[new_state][self.t] += 1
            self.state_counts[old_state][self.t] -= 1
            self.state_increments[new_state][self.t] += 1
            if global_configs.SAVE_NODES:
                self.states_history[self.t][node] = new_state

        if target_state is None:
            self.current_state[nodes] = self.state_to_go[nodes]
        else:
            self.current_state[nodes] = target_state
        self.update_plan(nodes)

    def update_plan(self, nodes):
        """ This is done for nodes that  just changed their states.
        New plans are generated according the state."""
        pass

    def _get_target_nodes(self, nodes, state):
        ret = nodes.copy().ravel()
        is_target_state = self.memberships[state, ret, 0]
        ret[nodes.flatten()] = is_target_state
        # ret = np.logical_and(
        #     self.memberships[state].flatten(),
        #     nodes.flatten()
        # )
        return ret
    
    def print(self, verbose=False):
        if verbose:
            print(f"T = {self.T} ({self.t})")
            for state in self.states:
                    print(f"\t {self.state_str_dict[state]} = {self.state_counts[state][self.t]}")

    def save_durations(self, f):
        for s in self.states:
            line = ",".join([str(x) for x in self.states_durations[s]])
            print(f"{self.state_str_dict[s]},{line}", file=f)

    def save_node_states(self, filename):
        if global_configs.SAVE_NODES is False:
            logging.warning(
                "Nodes states were not saved, returning empty data frame.")
            return pd.DataFrame()
        index = range(0, self.t+1)
        columns = self.states_history.values
        df = pd.DataFrame(columns, index=index)
        df.to_csv(filename)
        # df = df.replace(self.state_str_dict)
        # df.to_csv(filename)
        # print(df)

    def to_df(self):

        df = super().to_df()
        if self.start_day != 1:
            df["day"] = self.start_day + df["day"] - 1
            df.index = self.start_day + df.index - 1
        return df

    def run(self, T, print_interval=10, verbose=False):

        if global_configs.MONITOR_NODE is not None:
            monitor(0, f" being monitored, now in {self.state_str_dict[self.current_state[global_configs.MONITOR_NODE,0]]}")

        running = True
        self.tidx = 0
        self.T = self.start_day - 1
        if print_interval >= 0:
            self.print(verbose)

        for self.t in range(1, T+1):

            self.T = self.start_day + self.t - 1

            if __debug__ and print_interval >= 0 and verbose:
                print(flush=True)

            if (self.t >= len(self.state_counts[0])):
                # room has run out in the timeseries storage arrays; double the size of these arrays
                self.increase_data_series_length()

            if print_interval > 0 and verbose:
                start = time.time()
            running = self.run_iteration()

            # run periodical update
            if self.periodic_update_callback is not None:
                self.periodic_update_callback.run()

            if print_interval > 0 and (self.t % print_interval == 0):
                self.print(verbose)
                if verbose:
                    end = time.time()
                    print(f"Last day took: {end - start} seconds")

        if self.t < T:
            for t in range(self.t+1, T+1):
                if (t >= len(self.state_counts[0])):
                    self.increase_data_series_length()
                for state in self.states:
                    self.state_counts[state][t] = self.state_counts[state][t-1]
                    self.state_increments[state][t] = 0

        # finalize durations
        if global_configs.SAVE_DURATIONS:
            for s in self.states:
                durations = self.durations[self.memberships[s].flatten() == 1]
                durations = durations[durations != 0]
                self.states_durations[s].extend(list(durations))

        if print_interval >= 0:
            self.print(verbose)
        self.finalize_data_series()
        return True
 
    
    def run_iteration(self):

        logging.debug("DBG run iteration")

        # prepare
        # add timeseries members
        for state in self.states:
            self.state_counts[state][self.t] = self.state_counts[state][self.t-1]
            self.state_increments[state][self.t] = 0
        self.N[self.t] = self.N[self.t-1]

        self.durations += 1
        if global_configs.SAVE_NODES:
                self.states_history[self.t] = self.states_history[self.t-1]

        #print("DBG Time to go", self.time_to_go)
        #print("DBG State to go", self.state_to_go)

        # update times_to_go and states_to_go and
        # do daily_checkup
        self.daily_update(self.need_check)

        self.time_to_go -= 1
        #print("DBG Time to go", self.time_to_go)
        nodes_to_move = self.time_to_go == 0

        if global_configs.MONITOR_NODE and nodes_to_move[global_configs.MONITOR_NODE]:
            node = global_configs.MONITOR_NODE
            monitor(self.t,
                    f"changing state from {self.state_str_dict[self.current_state[node,0]]} to {self.state_str_dict[self.state_to_go[node,0]]}")

        orig_states = self.current_state[nodes_to_move]
        durs = self.durations[nodes_to_move.flatten()]
        self.change_states(nodes_to_move)
        self.durations[nodes_to_move.flatten()] = 0

        if global_configs.SAVE_DURATIONS:
            for s, d in zip(orig_states, durs):
                assert(d > 0)
                self.states_durations[s].append(d)



    
