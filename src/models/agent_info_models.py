import json
import numpy as np
import pandas as pd

import time
import logging

from models.simulation_engine import SimulationEngine
from utils.random_utils import RandomDuration
from utils.random_utils import gen_tuple
from utils.history_utils import TimeSeries, TransitionHistory, ShortListSeries

from utils.global_configs import monitor
import utils.global_configs as global_configs



class STATES():
    S = 0
    I = 1 
    R = 2
    EXT = 10

class Tipping():
    S = 0
    ACTIVE = 1
    EXT = 10


class RumourModel(SimulationEngine):
    states = [
        STATES.S,
        STATES.I,
        STATES.R,
        STATES.EXT
    ]

    num_states = len(states)
    state_str_dict = {
        STATES.S: "S",
        STATES.I: "I",
        STATES.R: "R",
        STATES.EXT: "EXT"
    }
    ext_code = STATES.EXT

    
    transitions = [
        (STATES.S, STATES.I),
        (STATES.I, STATES.R)
    ]

    num_transitions = len(transitions)

    final_states = [
        STATES.R
    ]

    invisible_states = [
        STATES.EXT
    ]

    unstable_states = [
        STATES.I
    ]

    fixed_model_parameters = {
        "I_duration": (1, "time in the I state"),
        "beta": (0,  "rate of transmission (exposure)")
    }

    def inicialization(self):
        super().inicialization()
        

    def setup_series_and_time_keeping(self):

        super().setup_series_and_time_keeping()


    def states_and_counts_init(self, ext_nodes=None, ext_code=None):
        super().states_and_counts_init(ext_nodes, ext_code)

        # need_check - state that needs regular checkup
        self.need_check = self.memberships[STATES.S]

        self.update_plan(np.ones(self.num_nodes, dtype=bool))

    
    def prob_of_contact(self, source_state, dest_state, beta):
        """
        Evaluates if transition happens.
        Edge goes from source to dest, dest is the infected node, source is the one that can become exposed. 
        :returns list of nodes (possibly with duplicates) that are exposed
        """ 

        # source_states - states that can be infected
        # dest_states - states that are infectious

        main_s = time.time()
        
        # is source in feasible state?
        is_relevant_source = self.memberships[source_state, self.graph.e_source, 0]
        
        # is dest in feasible state?
        is_relevant_dest = self.memberships[dest_state, self.graph.e_dest, 0]
        
        is_relevant_edge = np.logical_and(
            is_relevant_source,
            is_relevant_dest
        )

        assert type(beta) == float

        relevant_sources = self.graph.e_source[is_relevant_edge]
        relevant_dests = self.graph.e_dest[is_relevant_edge]

        # len(relevant_sources) == len(relevant_dests) --> for each edge that must be considered take source and dest
        # for each such edge draw a random number a test if < beta ==> get exposed nodes (the ones that are exposed at least once)

        num_relevant_edges = is_relevant_edge.sum() 
        r = np.random.rand(num_relevant_edges) 
        is_exposed = r < beta # for each relevant edge if transmission happens 

        is_exposed = is_exposed.ravel()
        
        exposed_nodes = relevant_sources[is_exposed]
        
        main_e = time.time()
        logging.info(f"PROBS OF CONTACT {main_e - main_s}")
        return exposed_nodes


    def daily_update(self, nodes):
        """
        Everyday checkup
        """

        # S
        target_nodes = self._get_target_nodes(nodes, STATES.S)      

        # try infection 
        exposed_nodes = self.prob_of_contact(
            STATES.S,
            STATES.I,  
            self.beta,
        ).flatten()

            
        self.time_to_go[exposed_nodes] = 1
        self.state_to_go[exposed_nodes] = STATES.I


    def update_plan(self, nodes):
        """ This is done for nodes that  just changed their states.
        New plans are generated according the state."""

        # STATES.S:     "S",
        target_nodes = self._get_target_nodes(nodes, STATES.S)
        
        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = STATES.S
        self.need_check[target_nodes] = True

        # STATES.I:   "I"
        target_nodes = self._get_target_nodes(nodes, STATES.I)
        self.time_to_go[target_nodes] = self.I_duration 
        self.state_to_go[target_nodes] = STATES.R
        self.need_check[target_nodes] = False

        # STATES.R:   "R",
        target_nodes = self._get_target_nodes(nodes, STATES.R)
        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = -1
        self.need_check[target_nodes] = False



    def run_iteration(self):
        super().run_iteration()

    
class RumourModelInfo(RumourModel):

    fixed_model_parameters = {
        "I_duration": (1, "time in the I state"),
        "beta": (0,  "rate of transmission (exposure)"),
        "scale": (1, "scaling factor for the exposure probability")
    }
    
    def prob_of_contact(self, source_state, dest_state, beta):
        """
        Evaluates if transition happens.
        Edge goes from source to dest, dest is the infected node, source is the one that can become exposed. 
        :returns list of nodes (possibly with duplicates) that are exposed
        """ 

        # source_states - states that can be infected
        # dest_states - states that are infectious

        main_s = time.time()
        
        # is source in feasible state?
        is_relevant_source = self.memberships[source_state, self.graph.e_source, 0]
        
        # is dest in feasible state?
        is_relevant_dest = self.memberships[dest_state, self.graph.e_dest, 0]
        
        is_relevant_edge = np.logical_and(
            is_relevant_source,
            is_relevant_dest
        )

        assert type(beta) == float

        relevant_sources = self.graph.e_source[is_relevant_edge]

        N = self.graph.number_of_nodes 
        counts = np.bincount(relevant_sources, minlength=N)

        #print(len(counts), self.memberships.shape[1], N)
        assert len(counts) == self.memberships.shape[1] == N 
        
        r = np.random.rand(N)
        # for all nodes, even those who are not relevant! for now  
        is_exposed = r < beta * np.tanh(counts/self.scale) #np.log(counts+1)/beta 
        
        exposed_nodes = np.arange(N)[is_exposed]

        main_e = time.time()
        logging.info(f"PROBS OF CONTACT {main_e - main_s}")
        return exposed_nodes

class InfoSIRModel(SimulationEngine):
    states = [
        STATES.S,
        STATES.I,
        STATES.R,
        STATES.EXT
    ]

    num_states = len(states)
    state_str_dict = {
        STATES.S: "S",
        STATES.I: "I",
        STATES.R: "R",
        STATES.EXT: "EXT"
    }
    ext_code = STATES.EXT

    
    transitions = [
        (STATES.S, STATES.I),
        (STATES.I, STATES.R)
    ]

    num_transitions = len(transitions)

    final_states = [
        STATES.R
    ]

    invisible_states = [
        STATES.EXT
    ]

    unstable_states = [
        STATES.I
    ]

    fixed_model_parameters = {
        "I_duration": (1, "time in the I state"),
    }

    model_parameters = {
                "beta": (0,  "rate of transmission (exposure)"),    
    }

    def inicialization(self):
        super().inicialization()
        

    def setup_series_and_time_keeping(self):

        super().setup_series_and_time_keeping()


    def states_and_counts_init(self, ext_nodes=None, ext_code=None):
        super().states_and_counts_init(ext_nodes, ext_code)

        # need_check - state that needs regular checkup
        self.need_check = self.memberships[STATES.S]

        self.update_plan(np.ones(self.num_nodes, dtype=bool))

    def prob_of_contact(self, source_state, dest_state, beta):
        """
        Evaluates if transition happens.
        Edge goes from source to dest, dest is the infected node, source is the one that can become exposed. 
        """ 

        # source_states - states that can be infected
        # dest_states - states that are infectious

        main_s = time.time()

        edges_probs = self.graph.get_all_edges_probs()
        num_edges = len(edges_probs)

        r = np.random.rand(num_edges)
        active_edges = (r < edges_probs).nonzero()[0]
        logging.info(f"active_edges {len(active_edges)}")

        source_nodes = self.graph.e_source[active_edges]
        dest_nodes = self.graph.e_dest[active_edges]
        types = self.graph.e_types[active_edges]
        # contact_info = (
        #     np.concatenate([source_nodes, dest_nodes]),
        #     np.concatenate([dest_nodes, source_nodes]),
        #     np.concatenate([types, types])
        # )
        # self.contact_history.append(contact_info)

        # take them in both directions
        n = len(active_edges)
        active_edges = np.concatenate([active_edges, active_edges])
        active_edges_dirs = np.ones(2*n, dtype=bool)
        active_edges_dirs[n:] = False

        source_nodes, dest_nodes = self.graph.get_edges_nodes(
            active_edges,
            active_edges_dirs
        )

        # is source in feasible state?
        is_relevant_source = self.memberships[source_state, source_nodes, 0]
        
        # is dest in feasible state?
        is_relevant_dest = self.memberships[dest_state, dest_nodes, 0]
        
        is_relevant_edge = np.logical_and(
            is_relevant_source,
            is_relevant_dest
        )

        ##########################
        relevant_edges = active_edges[is_relevant_edge]

        intensities = self.graph.get_edges_intensities(
            relevant_edges).reshape(-1, 1)
        relevant_sources, relevant_dests = self.graph.get_edges_nodes(
            relevant_edges, active_edges_dirs[is_relevant_edge])

        b_intensities = beta[relevant_sources]

        r = np.random.rand(intensities.ravel().shape[0]).reshape(-1, 1)
        is_exposed = r < (b_intensities * intensities)

        
        if np.all(is_exposed == False):
            return np.zeros((self.num_nodes, 1))

        is_exposed = is_exposed.ravel()
        
        exposed_nodes = relevant_sources[is_exposed]
        
        #print(exposed_nodes)
        
        ret = np.zeros((self.num_nodes, 1))
        ret[exposed_nodes] = 1

        main_e = time.time()
        logging.info(f"PROBS OF CONTACT {main_e - main_s}")
        return ret


    def daily_update(self, nodes):
        """
        Everyday checkup
        """

        # S
        target_nodes = self._get_target_nodes(nodes, STATES.S)      

        # if we have external nodes
        if self.num_ext_nodes > 0:
            raise NotImplementedError("External nodes not implemented yet.")    

        # try infection 
        P_infection = self.prob_of_contact(
                                      STATES.S,
                                      STATES.I,  
                                      self.beta
                                      ).flatten()

        exposed = P_infection[target_nodes]
        
        exposed_mask = np.zeros(self.num_nodes, dtype=bool)
        exposed_mask[target_nodes] = (exposed == 1)

        print("Number of infected",  exposed_mask.sum())
        if exposed_mask.sum() > 0:
            print(exposed_mask)
            
        self.time_to_go[exposed_mask] = 1
        self.state_to_go[exposed_mask] = STATES.I


    def update_plan(self, nodes):
        """ This is done for nodes that  just changed their states.
        New plans are generated according the state."""

        # STATES.S:     "S",
        target_nodes = self._get_target_nodes(nodes, STATES.S)
        
        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = STATES.S
        self.need_check[target_nodes] = True

        # STATES.I:   "I"
        target_nodes = self._get_target_nodes(nodes, STATES.I)
        self.time_to_go[target_nodes] = self.I_duration 
        self.state_to_go[target_nodes] = STATES.R
        self.need_check[target_nodes] = False

        # STATES.R:   "R",
        target_nodes = self._get_target_nodes(nodes, STATES.R)
        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = -1
        self.need_check[target_nodes] = False



    def run_iteration(self):
        super().run_iteration()

 
class InfoTippingModel(SimulationEngine):

    states = [
        Tipping.S,
        Tipping.ACTIVE,
        Tipping.EXT
    ]

    num_states = len(states)

    state_str_dict = {
        Tipping.S : "S",
        Tipping.ACTIVE: "Active",
        Tipping.EXT: "Ext"
    }
    ext_code = STATES.EXT

    
    transitions = [
        (Tipping.S, Tipping.ACTIVE)
    ]

    num_transitions = len(transitions)

    model_parameters = {
                "theta": (0,  "threshold"),    
    }


    def states_and_counts_init(self, ext_nodes=None, ext_code=None):
        super().states_and_counts_init(ext_nodes, ext_code)

        # need_check - state that needs regular checkup
        self.need_check = self.memberships[Tipping.S]

        self.update_plan(np.ones(self.num_nodes, dtype=bool))

    def _transmission(self):
        """ Returns boolean vector, for each node whether it is newly activated. """ 
        ret = np.zeros(self.num_nodes, dtype=bool)
        active_nodes = self.memberships[Tipping.S]

        edges_probs = self.graph.get_all_edges_probs()
        num_edges = len(edges_probs)
        print("num edges", num_edges)
        r = np.random.rand(num_edges)
        active_edges = (r < edges_probs) #.nonzero()[0] #bitmap
        
        for i, node in enumerate(self.graph.nodes):
            # bitmap of node's edges
            my_edges = self.graph.e_source == node
            print("my_edges", my_edges.shape)
            # keep only those that are active 
            my_edges = np.logical_and(
                my_edges,
                active_edges
            ).nonzero()[0]
            print("my_edges (after logical and)", my_edges.shape, my_edges.dtype)
            
            if len(my_edges) == 0:
                continue

            # take destination nodes
            my_neighbours = self.graph.e_dest[my_edges]
            print("my neighbours", my_neighbours.shape)
        
            active_neighbours = my_neighbours[(self.memberships[Tipping.ACTIVE] == 1)[my_neighbours].ravel()]
            print("active neighbours", active_neighbours.shape)

            my_edges_to_active = np.isin(self.graph.e_dest[my_edges], active_neighbours)

            print("my_edges_to_active", my_edges_to_active.shape)

            sum_all = self.graph.e_intensities[my_edges].sum()
            sum_active = self.graph.e_intensities[my_edges][my_edges_to_active].sum()

            if sum_active / sum_all > self.theta[node]:
                ret[i] = 1.0
        
        return ret 

    def daily_update(self, nodes):
        """
        Everyday checkup
        """

        # S
        target_nodes = self._get_target_nodes(nodes, Tipping.S)      

        # if we have external nodes
        if self.num_ext_nodes > 0:
            raise NotImplementedError("External nodes not present in Tipping Model.")    

        # try infection 
        transmission = self._transmission().flatten()

        self.time_to_go[transmission] = 1
        self.state_to_go[transmission] = Tipping.ACTIVE


    def update_plan(self, nodes):
        """ This is done for nodes that  just changed their states.
        New plans are generated according the state."""

        # STATES.S:     "S",
        target_nodes = self._get_target_nodes(nodes, Tipping.S)
        
        self.time_to_go[target_nodes] = -1
        self.state_to_go[target_nodes] = Tipping.S
        self.need_check[target_nodes] = True

        # STATES.Active:   "Active"
        target_nodes = self._get_target_nodes(nodes, Tipping.ACTIVE)
        self.time_to_go[target_nodes] = -1  
        self.state_to_go[target_nodes] = Tipping.ACTIVE
        self.need_check[target_nodes] = False



            
  
