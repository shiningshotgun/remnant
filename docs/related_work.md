# Contribution and related work

Remnant is an original project and controlled study. It is not a reproduction of a single
paper, and it does not propose a new episodic-control algorithm.

## What Remnant contributes

- **The Vesper time-loop setting and its implementation:** a deterministic 72-node tidal
  island where the world resets every 54 strides, a rewarding pearl detour makes escape
  impossible within the same tide, and only the agent's stores persist across dawns
  ([environment code](../remnant_core/worlds/reach.py)).
- **A persistent lived-memory formulation:** everything Wick carries across dawns is built
  from strides it actually lived (the Return Tally and the experienced map). Returns are
  bounded by the tide: nothing bootstraps across the dawn reset.
- **A controlled demonstration that separates discovery from learned escape:** curiosity
  tides (supplied) find the escape; the frozen learned policy then escapes on its own, and
  matched controls (legacy learner, amnesic, no curiosity) show which parts are needed
  ([results](results.md)).

## Model-free episodic control

Blundell, Uria, Pritzel, Li, Ruderman, Leibo, Rae, Wierstra and Hassabis, *Model-Free
Episodic Control* (2016): [arXiv abstract](https://arxiv.org/abs/1606.04460),
[HTML version](https://arxiv.org/html/1606.04460).

Their method keeps a table of the highest return experienced for each state and action and
acts greedily on it, estimating unseen states from nearby entries. The Return Tally follows
that idea directly. Remnant's differences are implementation choices, not a new method:

- exact-match tabular keys over a small discrete state `(position, items, pearl marker)`,
  with no nearest-neighbour estimate (unvisited actions get a fixed value of 0.0);
- returns end at the tide boundary (escape or the 54-stride toll), so no return crosses
  the dawn reset;
- discovery comes from a supplied mechanism (curiosity tides on a fixed schedule), not from
  the episodic table's own exploration.

## Remember, return, then explore

Ecoffet, Huizinga, Lehman, Stanley and Clune, *First return, then explore* (Nature, 2021):
[publication page](https://adrien.ecoffet.com/publication/2021-02-24-first-return-then-explore).

Go-Explore remembers promising states, returns to them, then explores from them. Remnant's
curiosity tides share the broad principle of walking back through remembered territory to
its edge before trying something new. This is a related principle only: Remnant does not
implement Go-Explore, and there is no benchmark or equivalence claim.

## Episodic curiosity

Savinov, Raichuk, Marinier, Vincent, Pollefeys, Lillicrap and Gelly, *Episodic Curiosity
through Reachability* (ICLR 2019):
[publication page](https://research.google/pubs/episodic-curiosity-through-reachability/).

That method learns a reachability network and gives the agent an intrinsic novelty reward
for observations that are not reachable from its episodic memory. Remnant uses no intrinsic
reward and no learned reachability: a curiosity tide is a breadth-first walk over the
agent's experienced map to the nearest untried action, run on a fixed seeded schedule.

## Software

Remnant's study code uses only the Python standard library. Figures use
[Matplotlib](https://matplotlib.org/) (optional), with its dependencies, under their own
licences. No third-party code is vendored in this tree.
