# What this taught me

Remnant went through a carousel of learning rules before it worked. Bandit, stateful, traces, TD, Monte Carlo. Each one was defensible and the agent still never escaped. At one point I had to ask if we had moved away from the actual idea. We had. The point of the project was always Wick getting out because of what it remembered.

What worked: Wick keeps a Return Tally, the best discounted return it has actually lived from each state and action, and that memory survives the daily reset. No planner, no replay, no dreaming. After 100 lived tides, the frozen policy escaped on every evaluation tide in 30 of 30 seeds. A legacy Q-learner with the same exploration escaped in 1 of 30. Wiping the memory at dawn gives 0 of 30.

Why the Q-learner fails in this world: our diagnosis is dawn bootstrapping, value re-derived from a dawn that looks like the old one. We did not isolate that cause separately, so it stays a diagnosis. The measured gap stands either way.

Finding the escape and learning from it are separate problems here. Without the curiosity tides, no seed found the escape in 100 tides. The curiosity mechanism is supplied, not learned, and the README says so. Hiding that would make the result look bigger than it is.

Keeping the evidence hash-locked and the claims pre-registered paid off: when a figure annotation overstated the discovery timing, the records settled it and the figure was corrected.

These results cover one world, one start and one escape. I haven't tested whether the same learning rule works outside that setup.
