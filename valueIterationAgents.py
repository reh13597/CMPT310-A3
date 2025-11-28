# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        # run value iteration for # of iterations
        for i in range(self.iterations):
            # create a new counter to store the new values
            newValues = util.Counter()

            # update the values for each state
            for state in self.mdp.getStates():
                # check if the current state is a terminal state, if so,
                # set the value to 0 since terminal states have no values, and
                # move onto the next staet
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                    continue

                # get the possible actions based on the current sate
                actions = self.mdp.getPossibleActions(state)

                # if there are no legal actions, set the value to 0 and move
                # onto the next state
                if not actions:
                    newValues[state] = 0
                    continue

                # init the max Q value
                maxQValue = float('-inf')

                # check all the actions to get the max Q value
                for action in actions:
                    qValue = self.computeQValueFromValues(state, action)

                    if qValue > maxQValue:
                        maxQValue = qValue

                # set the current value to this max Q value
                newValues[state] = maxQValue

            # replace the old values with the new ones for the next iteration
            # (batch update, Vk is calculated from Vk-1)
            self.values = newValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # init the Q value
        qValue = 0.0

        # get the next states and their probabilities
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)

        # sum over all the transitions
        for nextState, prob in transitions:
            # get the reward based off the current state, action, and next state
            reward = self.mdp.getReward(state, action, nextState)
            # Q(s, a) = sum over s' T(s, a, s')[R(s, a, s') + gamma * V(s')]
            qValue += prob * (reward + self.discount * self.values[nextState])

        return qValue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # check if the terminal state was reached, if so, return None since
        # terminal states don't have actions
        if self.mdp.isTerminal(state):
            return None

        # get the possible actions based on the current state
        actions = self.mdp.getPossibleActions(state)

        # if there are no legal actions, return None
        if not actions:
            return None

        # init the best action and Q value
        bestAction = None
        bestQValue = float('-inf')

        # check each action to see if its Q value is better than the current best one
        for action in actions:
            # get the Q value of the current state and action
            qValue = self.computeQValueFromValues(state, action)

            # if the Q value is better than than the current best one, replace it
            if qValue > bestQValue:
                bestQValue = qValue
                bestAction = action

        return bestAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


