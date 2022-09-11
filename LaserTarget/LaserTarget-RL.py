# Import libraries
import numpy as np
import RPi.GPIO as GPIO
from time import sleep

# Set Global Servo Check
CheckServo = True

# Setup Pins
GPIO.setmode(GPIO.BOARD)
GPIO.setup(16, GPIO.OUT)

ServoA = GPIO.PWM(16, 50)
ServoA.start(0)


# Define the shape of the environment (i.e., its states)
environment_columns = 8

# Create a 3D numpy array to hold the current Q-values for each state and action pair: Q(s, a) 
# The array contains 8 columns (to match the shape of the environment), as well as a third "action" dimension.
# The "action" dimension consists of 5 layers that will allow us to keep track of the Q-values for each possible action in
# Each state (see next cell for a description of possible actions). 
# The value of each (state, action) pair is initialized to 0.
q_values = np.zeros((environment_columns, 2))

# Define actions
actions = ['right', 'left']

# Create a 2D numpy array to hold the rewards for each state. 
# The array contains 8 columns (to match the shape of the environment), and each value is initialized to -100.
rewards = np.full((environment_columns), -1.) #set the reward for the packaging area to 100

# Define aisle locations (i.e., white squares) for rows 1 through 9
aisles = {} # Store locations in a dictionary
aisles[1] = [i for i in range(1, 10)]
aisles[2] = [i for i in range(1, 10)]
aisles[3] = [i for i in range(1, 10)]
aisles[4] = [i for i in range(1, 10)]
aisles[5] = [i for i in range(1, 10)]
aisles[6] = [i for i in range(1, 10)]
aisles[7] = [i for i in range(1, 10)]
aisles[8] = [i for i in range(1, 10)]
aisles[9] = [i for i in range(1, 10)]
  
rewards[4] = 100.

# Print rewards matrix
for row in rewards:
  print(row)

# Define a function that determines if the specified location is a terminal state
def is_terminal_state(current_column_index):
  # If the reward for this location is -1, then it is not a terminal state (i.e., it is a 'white square')
  if rewards[current_column_index] == -1.  :
    return False
  else:
    return True

# Define a function that will choose a random, non-terminal starting location
def get_starting_location():
  # Get a random column index
  current_column_index = 4

  while current_column_index == 4:
    current_column_index = np.random.randint(environment_columns)

  return current_column_index

# Define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
def get_next_action(current_column_index, epsilon):
  # If a randomly chosen value between 0 and 1 is less than epsilon, 
  # Then choose the most promising value from the Q-table for this state.
  if np.random.random() < epsilon:
    return np.argmax(q_values[current_column_index])
  else:
    return np.random.randint(2)

# Define a function that will get the next location based on the chosen action
def get_next_location(current_column_index, action_index):
  new_column_index = current_column_index

  if actions[action_index] == 'right' and current_column_index < environment_columns - 1:
    new_column_index += 1

    # Check Servo && Run It
    if CheckServo:
      ServoA.ChangeDutyCycle(7.5) # 0 deg position
      sleep(0.05)

  elif actions[action_index] == 'left' and current_column_index > 0:
    new_column_index -= 1

    # Check Servo && Run It
    if CheckServo:
      ServoA.ChangeDutyCycle(10) # 90 deg position
      sleep(0.05)


  return new_column_index

# Define a function that will get the shortest path between any location within the warehouse that 
# The robot is allowed to travel and the item packaging location.
def get_shortest_path(start_column_index):
    current_column_index = start_column_index
    shortest_path = []
    shortest_path.append(current_column_index)

    # Continue moving along the path until we reach the goal (i.e., the item packaging location)
    while not is_terminal_state(current_column_index):
      # Change CheckServo
      CheckServo = True
      # Get the best action to take
      action_index = get_next_action(current_column_index, 1.)

      # Move to the next location on the path, and add the new location to the list
      current_column_index = get_next_location(current_column_index, action_index)
      shortest_path.append(current_column_index)
    return shortest_path

def main():
  # Define training parameters
  epsilon = 0.9 # The percentage of time when we should take the best action (instead of a random action)
  discount_factor = 0.9 # Discount factor for future rewards
  learning_rate = 0.9 # The rate at which the AI agent should learn

  # Run through 1000 training episodes
  for episode in range(250):
    # Get the starting location for this episode
    column_index = get_starting_location()

    # Continue taking actions (i.e., moving) until we reach a terminal state
    while not is_terminal_state(column_index):
      # Choose which action to take (i.e., where to move next)
      action_index = get_next_action(column_index, epsilon)

      # Perform the chosen action, and transition to the next state (i.e., move to the next location)
      old_column_index = column_index #store the old column indexes
      column_index = get_next_location(column_index, action_index)

      # Receive the reward for moving to the new state, and calculate the temporal difference
      reward = rewards[column_index]
      old_q_value = q_values[old_column_index, action_index]
      temporal_difference = reward + (discount_factor * np.max(q_values[column_index])) - old_q_value

      # Update the Q-value for the previous state and action pair
      new_q_value = old_q_value + (learning_rate * temporal_difference)
      q_values[old_column_index, action_index] = new_q_value
main()

# Display a few shortest paths
shortest_path = get_shortest_path(1)

def read_moves(shortest_path):
    # Create Varibles
    current_node = 0
    move_array = []

    while current_node + 1 <= len(shortest_path):
        if (shortest_path[current_node] - (shortest_path[current_node] + 1)) < 0:
            move_array.append("Right")

        else:
            move_array.append("Left")

        current_node += 1

    print(move_array)

read_moves(shortest_path)

# Exit Pins
ServoA.stop()
GPIO.cleanup()