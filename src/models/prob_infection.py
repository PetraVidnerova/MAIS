"""Network contact and infection probability utilities.

This module provides standalone functions that compute which edges become
"active" on a given day and whether susceptible nodes are exposed via those
active edges.  Two main entry points are provided:

* :func:`prob_of_contact` – the current production implementation.
* :func:`prob_of_contact_old` – a legacy implementation retained for
  reference.

Helper functions :func:`select_active_edges`, :func:`archive_active_edges`,
and :func:`get_relevant_edges` are used internally by both entry points.
"""

import time
import numpy as np
import numpy_indexed as npi

import logging

from models.states import STATES


def select_active_edges(model, source_states, source_candidate_states, dest_states, dest_candidate_states):
    """Stochastically select active (contact) edges between two node sets.

    For each edge connecting a node in *source_candidate_states* to a node in
    *dest_candidate_states*, flips a biased coin with the edge's daily-contact
    probability.  Returns only edges where contact actually occurred.

    Args:
        model: The active simulation model instance (must expose
            ``memberships``, ``graph``, ``num_nodes``).
        source_states (list of int): States whose nodes can become exposed
            (used for final filtering, not edge selection).
        source_candidate_states (list of int): Superset of *source_states*
            used for initial candidate-edge selection.
        dest_states (list of int): Infectious states (used for filtering).
        dest_candidate_states (list of int): Superset of *dest_states* used
            for initial candidate-edge selection.

    Returns:
        tuple: ``(active_edges, active_edges_dirs)`` where both are
        ``numpy.ndarray`` of shape ``(num_active_edges,)``, or
        ``(None, None)`` if no edges are active.
    """
    assert type(dest_states) == list and type(source_states) == list

    # 1. select active edges
    # candidates for active edges are edges between source_candidate_states and dest_candidate_states
    # source (the one who is going to be infected)
    # dest (the one who can offer infection)

    s = time.time()
    source_candidate_flags = model.memberships[source_candidate_states, :, :].reshape(
        len(source_candidate_states), model.num_nodes).sum(axis=0)
    # source_candidate_indices = source_candidate_flags.nonzero()[0]

    dest_candidate_flags = model.memberships[dest_candidate_states, :, :].reshape(
        len(dest_candidate_states), model.num_nodes).sum(axis=0)
    # dest_candidate_indices = dest_candidate_flags.nonzero()[0]
    e = time.time()
    logging.info(f"Create flags {e-s}")

    s = time.time()
    possibly_active_edges, possibly_active_edges_dirs = model.graph.get_edges(
        source_candidate_flags,
        dest_candidate_flags
    )
    num_possibly_active_edges = len(possibly_active_edges)
    e = time.time()
    logging.info(f"Select possibly active {e-s}")

    s = time.time()

    if num_possibly_active_edges == 0:
        return None, None

    # for each possibly active edge flip coin
    r = np.random.rand(num_possibly_active_edges)
    e = time.time()
    logging.info(f"Random numbers: {e-s}")
    s = time.time()
    # edges probs
    p = model.graph.get_edges_probs(possibly_active_edges)
    e = time.time()
    logging.info(f"Get intensities: {e-s}")

    s = time.time()
    active_indices = (r < p).nonzero()[0]
    num_active_edges = len(active_indices)
    if num_active_edges == 0:
        return None, None

    active_edges = possibly_active_edges[active_indices]
    active_edges_dirs = possibly_active_edges_dirs[active_indices]
    e = time.time()
    logging.info(f"Select active edges: {e-s}")

    return active_edges, active_edges_dirs


def archive_active_edges(model, active_edges, active_edges_dirs):
    """Record today's active edges in the model's contact-history buffer.

    Retrieves source and destination node indices for each active edge and
    appends them (along with edge type) to ``model.contact_history``.

    Args:
        model: The active simulation model instance.
        active_edges (numpy.ndarray): Indices of today's active edges.
        active_edges_dirs (numpy.ndarray): Direction flags for each edge
            (``True`` = canonical direction, ``False`` = reversed).
    """
    s = time.time()
    # get source and dest nodes for active edges
    # source and dest met today, dest is possibly infectious, source was possibly infected
    source_nodes, dest_nodes = model.graph.get_edges_nodes(
        active_edges, active_edges_dirs)
    # add to contact_history (infectious node goes first!!!)
    contact_indices = list(zip(dest_nodes, source_nodes, active_edges))
    model.contact_history.append(contact_indices)

    # uncoment for statistics
    # for dest_node, source_node, _ in contact_indices:
    #     model.nodes_inf_contacts[source_node] += 1

    # if model.t == 6:
    #     contact_numbers = {}
    #     for e in active_edges:
    #         layer_number = model.graph.get_layer_for_edge(e)
    #         contact_numbers[layer_number] = contact_numbers.get(
    #             layer_number, 0) + 1
    #     contact_num_str = ', '.join([f'{layer_type}:{number}' for layer_type, number in contact_numbers.items()])
    #     logging.debug(f"DBG contact numbers {contact_num_str}")

    # print("Potkali se u kolina:", contact_indices)
    logging.info(f"Todays contact num:  {len(contact_indices)}")
    e = time.time()
    logging.info(f"Archive active edges: {e-s}")


def get_relevant_edges(model, active_edges, active_edges_dirs, source_states, dest_states):
    """Filter active edges to those connecting relevant source/destination states.

    From the set of *active_edges* retains only those where the source node is
    in one of *source_states* and the destination node is in one of
    *dest_states*.

    Args:
        model: The active simulation model instance.
        active_edges (numpy.ndarray): Indices of today's active edges.
        active_edges_dirs (numpy.ndarray): Direction flags for *active_edges*.
        source_states (list of int): States eligible to be exposed.
        dest_states (list of int): Infectious states.

    Returns:
        tuple: ``(active_relevant_edges, active_relevant_edges_dirs)`` or
        ``(None, None)`` if no relevant edges remain after filtering.
    """
    # restrict the selection to only relevant states
    # (ie. candidates can be both E and I, relevant are only I)
    # candidates are those who will be possibly relevant in future
    s = time.time()
    dest_flags = model.memberships[dest_states, :, :].reshape(
        len(dest_states), model.num_nodes).sum(axis=0)
    source_flags = model.memberships[source_states, :, :].reshape(
        len(source_states), model.num_nodes).sum(axis=0)

    relevant_edges, relevant_edges_dirs = model.graph.get_edges(
        source_flags, dest_flags)
    if len(relevant_edges) == 0:
        return None, None

    # get intersection
    active_relevant_edges = np.intersect1d(active_edges, relevant_edges)

    if len(active_relevant_edges) == 0:
        return None, None
    # print(active_relevant_edges.shape)
    # print(relevant_edges.shape)
    # print((active_relevant_edges[:, np.newaxis] == relevant_edges).shape)
    try:
        # this causes exceptin, but only sometimes ???
        # where_indices = (
        #    active_relevant_edges[:, np.newaxis] == relevant_edges).nonzero()[1]
        # lets try npi instead
        where_indices = npi.indices(relevant_edges, active_relevant_edges)
    except AttributeError as e:
        print(e)
        print("Lucky we are ...")
        print(active_relevant_edges.shape)
        np.save("active_relevant_edges", active_relevant_edges)
        print(relevant_edges.shape)
        np.save("relevant_edges", relevant_edges)
        print(active_relevant_edges[:, np.newaxis] == relevant_edges)
        exit()
    #        print(where_indices, len(where_indices))
    # always one index! (sources and dest must be disjunct)
    active_relevant_edges_dirs = relevant_edges_dirs[where_indices]
    e = time.time()
    logging.info(f"Get relevant active edges: {e-s}")
    return active_relevant_edges, active_relevant_edges_dirs


def prob_of_contact_old(model, source_states, source_candidate_states, dest_states, dest_candidate_states, beta, beta_in_family):
    """Legacy implementation of the contact-probability computation (abandoned).

    Retained for historical reference only.  Prefer :func:`prob_of_contact`.

    Args:
        model: The active simulation model instance.
        source_states (list of int): States that can be exposed.
        source_candidate_states (list of int): Candidate source states for
            edge selection.
        dest_states (list of int): Infectious states.
        dest_candidate_states (list of int): Candidate destination states for
            edge selection.
        beta (numpy.ndarray): Per-node non-family transmission rate,
            shape ``(num_nodes, 1)``.
        beta_in_family (numpy.ndarray): Per-node family transmission rate,
            shape ``(num_nodes, 1)``.

    Returns:
        numpy.ndarray: Per-node binary exposure indicator, shape
        ``(num_nodes, 1)``.
    """

    main_s = time.time()
    active_edges, active_edges_dirs = select_active_edges(model,
                                                          source_states, source_candidate_states, dest_states, dest_candidate_states)
    if active_edges is None:  # we have no active edges today
        return np.zeros((model.num_nodes, 1))

    #archive_active_edges(model, active_edges, active_edges_dirs)

    active_relevant_edges, active_relevant_edges_dirs = get_relevant_edges(model,
                                                                           active_edges,
                                                                           active_edges_dirs,
                                                                           source_states,
                                                                           dest_states)
    if active_relevant_edges is None:
        return np.zeros((model.num_nodes, 1))

    s = time.time()
    intensities = model.graph.get_edges_intensities(
        active_relevant_edges).reshape(-1, 1)
    relevant_sources, relevant_dests = model.graph.get_edges_nodes(
        active_relevant_edges, active_relevant_edges_dirs)

    is_family_edge = model.graph.is_family_edge(
        active_relevant_edges).reshape(-1, 1)

    if False:
        is_class_edge = model.graph.is_class_edge(
            active_relevant_edges).reshape(-1, 1)
        is_pub_edge = model.graph.is_pub_edge(
            active_relevant_edges).reshape(-1, 1)
        is_super_edge = model.graph.is_super_edge(
            active_relevant_edges).reshape(-1, 1)
        is_family_edge = np.logical_or.reduce((is_family_edge,
                                               is_class_edge,
                                               is_super_edge,
                                               is_pub_edge))

    #        assert len(relevant_sources) == len(set(relevant_sources))
    # TOD beta - b_ij
    # new beta depands on the one who is going to be infected
    #        b_intensities = beta[relevant_sources]
    #        b_f_intensities = beta_in_family[relevant_sources]

    # reduce asymptomatic
    is_A = model.memberships[STATES.I_n][relevant_dests]

    b_original_intensities = (
        beta_in_family[relevant_sources] * (1 - is_A) +
        model.beta_A_in_family[relevant_sources] * is_A
    )
    b_reduced_intensities = (
        beta[relevant_sources] * (1 - is_A) +
        model.beta_A[relevant_sources] * is_A
    )

    b_intensities = (
        b_original_intensities * is_family_edge +
        b_reduced_intensities * (1 - is_family_edge)
    )

    assert b_intensities.shape == intensities.shape

    # relevant_sources_unique, unique_indices = np.unique(
    #     relevant_sources, return_inverse=True)

    # print(len(relevant_sources_unique))
    # print(b_intensities, b_intensities.shape)
    # print(active_relevant_edges, active_relevant_edges.shape)
    # exit()

    r = np.random.rand(b_intensities.ravel().shape[0]).reshape(-1, 1)
    # print(b_intensities.shape)
    # print(intensities.shape)
    # print((b_intensities*intensities).shape)
    is_exposed = r < (b_intensities * intensities)

    #        print(is_exposed, is_exposed.shape)
    if np.all(is_exposed == False):
        return np.zeros((model.num_nodes, 1))

    is_exposed = is_exposed.ravel()
    exposed_nodes = relevant_sources[is_exposed]
    ret = np.zeros((model.num_nodes, 1))
    ret[exposed_nodes] = 1

    sourced_nodes = relevant_dests[is_exposed]
    model.successfull_source_of_infection[sourced_nodes] += 1

    succesfull_edges = active_relevant_edges[is_exposed]
    successfull_layers = model.graph.get_layer_for_edge(succesfull_edges)
    for e in successfull_layers:
        model.stat_successfull_layers[e][model.t] += 1

    main_e = time.time()
    logging.info(f"PROBS OF CONTACT {main_e - main_s}")
    return ret
    # no_infection = (1 - b_intensities * intensities).ravel()

    # res = np.ones(len(relevant_sources_unique), dtype='float32')
    # for i in range(len(unique_indices)):
    #     res[unique_indices[i]] = res[unique_indices[i]]*no_infection[i]
    # prob_of_no_infection = res
    # # prob_of_no_infection2 = np.fromiter((np.prod(no_infection, where=(relevant_sources==v).T)
    # #                         for v in relevant_sources_unique), dtype='float32')

    # result = np.zeros(model.num_nodes)
    # result[relevant_sources_unique] = 1 - prob_of_no_infection
    # e = time.time()
    # print("Comp probability", e-s)
    #        return result.reshape(model.num_nodes, 1)


def prob_of_contact(model, source_states, source_candidate_states, dest_states, dest_candidate_states, beta, beta_in_family):
    """Evaluate per-node exposure via stochastic edge activation (production).

    For each graph edge, independently activates it with the edge's daily
    contact probability.  For active edges connecting a susceptible source to
    an infectious destination, draws a Bernoulli trial with the effective
    transmission rate (accounting for family vs. non-family edges and
    asymptomatic reduction).  Returns a binary indicator vector: 1 for each
    node that was exposed on this day.

    *source_candidate_states* and *dest_candidate_states* are accepted for
    backward compatibility but are no longer used; the function queries all
    edges.

    Updates the following model statistics:

    * ``model.contact_history`` – active edge contacts.
    * ``model.successfull_source_of_infection`` – per-node successful
      infection counts.
    * ``model.stat_successfull_layers`` – per-layer successful transmission
      counts.

    Args:
        model: The active simulation model instance.
        source_states (list of int): States that can become exposed.
        source_candidate_states (list of int): Unused (backward compat.).
        dest_states (list of int): Infectious states.
        dest_candidate_states (list of int): Unused (backward compat.).
        beta (numpy.ndarray): Per-node non-family transmission rate,
            shape ``(num_nodes, 1)``.
        beta_in_family (numpy.ndarray): Per-node family transmission rate,
            shape ``(num_nodes, 1)``.

    Returns:
        numpy.ndarray: Per-node binary exposure indicator, shape
        ``(num_nodes, 1)``.  Value is 1 if the node was newly exposed, 0
        otherwise.
    """

    # source_states - states that can be infected
    # dest_states - states that are infectious

    main_s = time.time()

    edges_probs = model.graph.get_all_edges_probs()
    num_edges = len(edges_probs)

    r = np.random.rand(num_edges)
    active_edges = (r < edges_probs).nonzero()[0]
    logging.info(f"active_edges {len(active_edges)}")

    source_nodes = model.graph.e_source[active_edges]
    dest_nodes = model.graph.e_dest[active_edges]
    types = model.graph.e_types[active_edges]
    contact_info = (
        np.concatenate([source_nodes, dest_nodes]),
        np.concatenate([dest_nodes, source_nodes]),
        np.concatenate([types, types])
    )
    model.contact_history.append(contact_info)

    # take them in both directions
    n = len(active_edges)
    active_edges = np.concatenate([active_edges, active_edges])
    active_edges_dirs = np.ones(2*n, dtype=bool)
    active_edges_dirs[n:] = False

    source_nodes, dest_nodes = model.graph.get_edges_nodes(
        active_edges,
        active_edges_dirs
    )

    # is source in feasible state?
    is_relevant_source = model.memberships[source_states[0], source_nodes, 0]
    for state in source_states[1:]:
        is_relevant_source += model.memberships[state, source_nodes, 0]

    # is dest in feasible state?
    is_relevant_dest = model.memberships[dest_states[0], dest_nodes, 0]
    for state in dest_states[1:]:
        is_relevant_dest += model.memberships[state, dest_nodes, 0]

    is_relevant_edge = np.logical_and(
        is_relevant_source,
        is_relevant_dest
    )

    ##########################
    relevant_edges = active_edges[is_relevant_edge]

    intensities = model.graph.get_edges_intensities(
        relevant_edges).reshape(-1, 1)
    relevant_sources, relevant_dests = model.graph.get_edges_nodes(
        relevant_edges, active_edges_dirs[is_relevant_edge])

    is_family_edge = model.graph.is_family_edge(
        relevant_edges).reshape(-1, 1)

    # reduce asymptomatic
    is_A = model.memberships[STATES.I_n][relevant_dests]

    b_original_intensities = (
        beta_in_family[relevant_sources] * (1 - is_A) +
        model.beta_A_in_family[relevant_sources] * is_A
    )
    b_reduced_intensities = (
        beta[relevant_sources] * (1 - is_A) +
        model.beta_A[relevant_sources] * is_A
    )

    b_intensities = (
        b_original_intensities * is_family_edge +
        b_reduced_intensities * (1 - is_family_edge)
    )

    assert b_intensities.shape == intensities.shape

    r = np.random.rand(b_intensities.ravel().shape[0]).reshape(-1, 1)
    is_exposed = r < (b_intensities * intensities)

    if np.all(is_exposed == False):
        return np.zeros((model.num_nodes, 1))

    is_exposed = is_exposed.ravel()
    exposed_nodes = relevant_sources[is_exposed]
    ret = np.zeros((model.num_nodes, 1))
    ret[exposed_nodes] = 1

    # save stats
    sourced_nodes = relevant_dests[is_exposed]
    model.successfull_source_of_infection[sourced_nodes] += 1

    succesfull_edges = relevant_edges[is_exposed]
    successfull_layers = model.graph.get_layer_for_edge(succesfull_edges)
    for e in successfull_layers:
        model.stat_successfull_layers[e][model.t] += 1

    main_e = time.time()
    logging.info(f"PROBS OF CONTACT {main_e - main_s}")
    return ret
