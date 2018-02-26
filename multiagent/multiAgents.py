# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        rows = newFood.height
        collums = newFood.width
        foodlocations = list()
        for y in range(0,collums):
            currentrow =  newFood.__getitem__(y)
            for x in range(0,rows - 1):
                if currentrow[x] :
                    foodlocations.append((y,x))   
        totalmindistance = 0
        currentlocation = newPos

        for count in range(0,len(foodlocations)):
            nearestfooddistance = 999999999
            nearestfood = None
            for foodlocation in foodlocations :
                newdistance = abs(foodlocation[0] - currentlocation[0]) + abs(foodlocation[1] - currentlocation[1])
                if newdistance < nearestfooddistance :
                    nearestfooddistance = newdistance
                    nearestfood = foodlocation
            totalmindistance = totalmindistance + nearestfooddistance
            currentlocation = nearestfood
            foodlocations.remove(nearestfood)

        for ghostState in newGhostStates :
           
            if abs(ghostState.getPosition()[0] - newPos[0]) + abs(ghostState.getPosition()[1] - newPos[1]) < 2:
                totalmindistance  = totalmindistance + 999999
        return -totalmindistance 

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        "Node structure is (agentindex, depth, parent, action to get here, gamestate)"
        from collections import deque
        from operator import itemgetter
        numagents = gameState.getNumAgents()

        firstnode = (self.index, self.depth, None, None, gameState)
        "keeps track of possible values put by children and the action to get there"
        possiblevalues = {firstnode : []}
        nodes = deque()
        leafnodes = list();
        for i in range(0, self.depth + 1):
            leafnodes.append(list());
        leafnodes[self.depth].append(firstnode);

        nodes.append(firstnode)
        while(len(nodes) > 0) :
            currentnode = nodes.pop()
            for legalaction in currentnode[4].getLegalActions(currentnode[0]):
                newindex = (currentnode[0] + 1) % numagents 
                newdepth = currentnode[1]
                
                if(newindex == 0) :
                    newdepth = newdepth - 1
                newgamestate = currentnode[4].generateSuccessor(currentnode[0], legalaction)
                newnode = (newindex, newdepth, currentnode, legalaction, newgamestate)
                leafnodes[newdepth].append(newnode);
                if newdepth == 0 or newgamestate.isWin() or newgamestate.isLose() :
                    possiblevalues[newnode] = [(self.evaluationFunction(newnode[4]), legalaction)]
                else:
                    nodes.append(newnode)
                    possiblevalues[newnode] = [] 
        for i in range(0, self.depth + 1) :
            while(len(leafnodes[i]) > 0) :
                node = leafnodes[i].pop()
                if(len( possiblevalues[node]) != 0 and node[2] != None) :
                    if(node[0] == 0 ) :
                        possiblevalues[node[2]].append((max(possiblevalues[node],key=itemgetter(0))[0], node[3]))
                    else :
                        possiblevalues[node[2]].append((min(possiblevalues[node],key=itemgetter(0))[0], node[3]))
        return (max(possiblevalues[firstnode],key=itemgetter(0)))[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        posinf = 9999999
        neginf = -9999999
        numagents = gameState.getNumAgents()
        """ returns (max action) """
        def getMax(alpha, beta, gamestate, currentid, currentdepth):
            value = (neginf, None);
            newindex = (currentid + 1) % numagents 
            newdepth = currentdepth - 1;

            for legalaction in gamestate.getLegalActions(currentid) :

                newgamestate = gamestate.generateSuccessor(currentid, legalaction)
                if (newgamestate.isWin() or newgamestate.isLose() or newdepth == 0 or len(newgamestate.getLegalActions(newindex)) == 0) :
                    newvalue = (self.evaluationFunction(newgamestate), legalaction)
                    if newvalue[0] > value[0] :
                        value = newvalue
                else :
                    if (newindex == 0) :
                        newmax = (getMax(alpha, beta, newgamestate, newindex, newdepth)[0], legalaction)
                        if(newmax[0] > value[0]) :
                            value = newmax
                    else :
                        newmax = (getMin(alpha, beta, newgamestate, newindex, newdepth)[0], legalaction)
                        if(newmax[0] > value[0]) :
                            value = newmax
                if(value[0] > beta) :
                    return value
                if(value[0] > alpha) :
                   alpha = value[0]
            return value

        def getMin(alpha, beta, gamestate, currentid, currentdepth):
            value = (posinf, None);
            newindex = (currentid + 1) % numagents
            newdepth = currentdepth - 1
 
            for legalaction in gamestate.getLegalActions(currentid) :

                newgamestate = gamestate.generateSuccessor(currentid, legalaction)
                if (newgamestate.isWin() or newgamestate.isLose() or  newdepth == 0 or len(newgamestate.getLegalActions(newindex)) == 0) :
                    newvalue = (self.evaluationFunction(newgamestate), legalaction)
             
                    if newvalue[0] < value[0] :
                        value = newvalue
                else :
                    if (newindex == 0) :
                        newmin = (getMax(alpha, beta, newgamestate, newindex, newdepth)[0] , legalaction)
                        if(newmin[0] < value[0]) :
                            value = newmin
                    else :
                        newmin = (getMin(alpha, beta, newgamestate, newindex, newdepth)[0], legalaction)
                        if(newmin[0] < value[0]) :
                            value = newmin
                if(value[0] < alpha) :
                    return value
                if(value[0] < beta) :
                   beta = value[0]
            
            return value
        return getMax(neginf, posinf, gameState, 0, self.depth * numagents)[1]
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        from collections import deque
        from operator import itemgetter
        numagents = gameState.getNumAgents()

        firstnode = (self.index, self.depth, None, None, gameState)
        "keeps track of possible values put by children and the action to get there"
        possiblevalues = {firstnode : []}
        nodes = deque()
        leafnodes = list();
        for i in range(0, self.depth + 1):
            leafnodes.append(list());
        leafnodes[self.depth].append(firstnode);

        nodes.append(firstnode)
        while(len(nodes) > 0) :
            currentnode = nodes.pop()
            for legalaction in currentnode[4].getLegalActions(currentnode[0]):
                newindex = (currentnode[0] + 1) % numagents 
                newdepth = currentnode[1]
                
                if(newindex == 0) :
                    newdepth = newdepth - 1
                newgamestate = currentnode[4].generateSuccessor(currentnode[0], legalaction)
                newnode = (newindex, newdepth, currentnode, legalaction, newgamestate)
                leafnodes[newdepth].append(newnode);
                if newdepth == 0 or newgamestate.isWin() or newgamestate.isLose() :
                    possiblevalues[newnode] = [(self.evaluationFunction(newnode[4]), legalaction)]
                else:
                    nodes.append(newnode)
                    possiblevalues[newnode] = [] 
        for i in range(0, self.depth + 1) :
            while(len(leafnodes[i]) > 0) :
                node = leafnodes[i].pop()
                if(len( possiblevalues[node]) != 0 and node[2] != None) :
                    if(node[0] == 0 ) :
                        possiblevalues[node[2]].append((max(possiblevalues[node],key=itemgetter(0))[0], node[3]))
                    else :
                        possiblevalues[node[2]].append((sum([pair[0] for pair in possiblevalues[node]]) / len(possiblevalues[node]) , node[3]))
        return (max(possiblevalues[firstnode],key=itemgetter(0)))[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did -- lose = the worst outcome otherwise the negtive length of the shortest path 
      through all food the antighost capsuls count as food and a scared ghost counts as food plus a bit of randomness to break ties>
    """
    "*** YOUR CODE HERE ***"
    newGhostStates = currentGameState.getGhostStates()
    newPos = currentGameState.getPacmanPosition()
    if currentGameState.isLose():
        return -99999999
    nearestghostkiller = None



    if  nearestghostkiller == None:
        newFood = currentGameState.getFood()
        rows = newFood.height
        collums = newFood.width
        foodlocations = list()
        rows = newFood.height
        collums = newFood.width
        foodlocations = list()
        for y in range(0,collums):
            currentrow =  newFood.__getitem__(y)
            for x in range(0,rows - 1):
                if currentrow[x] :
                    foodlocations.append((y,x)) 
        for ghostkiller in  currentGameState.getCapsules() :
            foodlocations.append(ghostkiller)
        for ghostState in newGhostStates :
            if abs(ghostState.getPosition()[0] - newPos[0]) + abs(ghostState.getPosition()[1] - newPos[1]) < 3:
                if(ghostState.scaredTimer > 2) :
                    foodlocations.append(ghostState.getPosition())
        totalmindistance = 0
        currentlocation = newPos
        onfood =  newPos in foodlocations 
        for count in range(0,len(foodlocations)):
            nearestfooddistance = 999999999
            nearestfood = None
            for foodlocation in foodlocations :
                newdistance = abs(foodlocation[0] - currentlocation[0]) + abs(foodlocation[1] - currentlocation[1])
                if newdistance < nearestfooddistance :
                    nearestfooddistance = newdistance
                    nearestfood = foodlocation
            totalmindistance = totalmindistance + nearestfooddistance
            currentlocation = nearestfood
            foodlocations.remove(nearestfood)
        if currentGameState.getNumFood() == 0 and totalmindistance < 2 :
            return  10 * random.random()
        return -totalmindistance + random.random() * 2
# Abbreviation
better = betterEvaluationFunction

