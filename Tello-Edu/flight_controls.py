# Import the necessary libraries
import time
from djitellopy import Tello

# Create a Tello object
tello = Tello()

# Connect to the drone
tello.connect()

# Take off
tello.takeoff()

# Wait for a few seconds
time.sleep(5)

# Move forward for 50cm
tello.move_forward(20)

# Wait for a few seconds
time.sleep(5)

# Move backward for 50cm
tello.move_back(20)

# Wait for a few seconds
time.sleep(5)

# Move left for 50cm
tello.move_left(20)

# Wait for a few seconds
time.sleep(5)

# Move right for 50cm
tello.move_right(20)

# Wait for a few seconds
time.sleep(5)

# Move up for 50cm
tello.move_up(20)

# Wait for a few seconds
time.sleep(5)

# Move down for 50cm
tello.move_down(20)

# Wait for a few seconds
time.sleep(5)

# Rotate clockwise by 90 degrees
tello.rotate_clockwise(90)

# Wait for a few seconds
time.sleep(5)

# Rotate counterclockwise by 90 degrees
tello.rotate_counter_clockwise(90)

# Wait for a few seconds
time.sleep(5)

# Land the drone
tello.land()

# Close the connection
tello.end()
