# SurveyPropagation

Implementation of the majority of the algorithms developed for Survey Propogation as described in [1]. It aims at solving the K-SAT problem using heuristics inspired from statistical physics. This project was implemented for a course called "Mathematics for Neurosciences" during the French master "Mathématiques, Vision, Apprentissage".

Let's mention some details on the implementation:
------

- I use the python package networkX which scales poorly because it does not rely on a C++ code. It should be modified using the package igraph to improve speed
- The algorithm is not stable and is frequently UNSAT when approaching the critical ratio described in [1]. One cause might be that the normalization of probabilities omits a term which should be zero (namely a "contradiction probability") but which isn't during numerical tests. The code needs further corrections to solve this issue

Source:
------

[1] Braunstein, A., Mézard, M., & Zecchina, R. (2005). Survey propagation: An algorithm for satisfiability. Random Structures & Algorithms, 27(2), 201-226. (https://arxiv.org/abs/cs/0212002) 
