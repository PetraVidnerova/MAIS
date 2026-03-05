"""Epidemic state definitions for the agent-based network model.

This module defines the integer codes used to identify each epidemiological
compartment and provides a human-readable mapping via ``state_codes``.
"""

# states for agent based model


class STATES():
    """Integer constants representing each epidemiological compartment.

    Attributes:
        S (int): Susceptible.
        S_s (int): Susceptible with false symptoms.
        E (int): Exposed (latent infection).
        I_n (int): Infectious, asymptomatic (non-symptomatic track).
        I_a (int): Infectious, pre-symptomatic (asymptomatic phase before
            symptoms develop).
        I_s (int): Infectious, symptomatic.
        J_s (int): Post-infectious, symptomatic (RNA-positive, symptomatic).
        J_n (int): Post-infectious, asymptomatic (RNA-positive, no symptoms).
        R (int): Recovered.
        D (int): Dead.
        EXT (int): External node (not counted in the population).
    """

    S = 0
    S_s = 1
    E = 2
    I_n = 3
    I_a = 4
    I_s = 5
    J_s = 6
    J_n = 7
    R = 8
    D = 9
    EXT = 10

    pass


# Mapping from STATES integer code to its short string label.
state_codes = {
    STATES.S:     "S",
    STATES.S_s:   "S_s",
    STATES.E:     "E",
    STATES.I_n:   "I_n",
    STATES.I_a:   "I_a",
    STATES.I_s:   "I_s",
    STATES.J_s:   "J_s",
    STATES.J_n:   "J_n",
    STATES.R:   "R",
    STATES.D:   "D",
    STATES.EXT: "EXT"
}
