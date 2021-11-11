# mdp-extras

Flexible methods for specifying MDPs, particularly in the context of Inverse
Reinforcement Learning.

Inverse Reinforcement Learning (IRL) algorithms have different requirements to
Reinforcement Learning (RL) algorithms.
Many existing APIs for specifying Markov Decision Processes (MDPs) are oriented toward
RL applications, and lack many critical features that would allow them to be used for
IRL.

For example, are the MDP transition dynamics explicit (a full transition matrix is
available to the algorithm), implicit (the algorithm can request a transition sample
from any state-action tuple), or available by simulation only (the algorithm can only
observe sequential transitions)?

The same concerns also apply to state, action and observation spaces, feature functions,
reward functions, discount factors, policy classes, etc.

This utility library aims to provide a flexible way to specify MDP components such that
they can be used in IRL applications.

## Installation

This package is not distributed on PyPI - you'll have to install from source.

```bash
git clone https://github.com/aaronsnoswell/mdp-extras.git
cd mdp-extras
pip install -e .
```

## Usage

### [extras.py](mdp_extras/extras.py)

Provides the 'extra' pieces that OpenAI Gym Environment definitions are missing.

 * `DiscreteExplicitExtras` - A MDP with Discrete States and Actions, and Explicit
   dynamics (i.e. a full transition matrix). Sometimes known as a tabular MDP.

### [features.py](mdp_extras/features.py)

Provides a general feature function interface.

 * `FeatureFunction` - abstract feature vector function base class.
   Feature functions can consume observations, observation-action tuples, or full
   observation-action-observation triples.
 * `Indicator` - An indicator feature function - sometimes known as a one-hot feature
   function.
 * `Disjoint` - A disjoint feature function.
   I.e. every point in the input space is mapped to one and only one point in the output
   space.
   The output space can be smaller than the input space.

### [rewards.py](mdp_extras/rewards.py)

Structured reward function representations.

 * `RewardFunction` - abstract reward function base class.
 * `Linear` - A reward function that is linear with some pre-defined feature function.

### [soln.py](mdp_extras/soln.py)

Value/Policy iteration methods for solving tabular MDPs exactly.
Also includes various policy definitions, a policy evaluation method, and estimators of
the Q-function gradient.

### [utils.py](mdp_extras/utils.py)

Various miscellaneous utilities.

### [/envs](mdp_extras/envs)

Utilities for converting OpenAI Gym `Environment` objects to the equivalent
`mdp-extras` representations.
