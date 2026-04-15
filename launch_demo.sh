#!/bin/bash
# Launch 3 tmux panes with all commands pre-typed.
# Run each pane with Enter when ready (top to bottom).
#
# Usage: ./launch_demo.sh

SESSION="trajectory_demo"

tmux kill-session -t $SESSION 2>/dev/null

tmux new-session -d -s $SESSION -x 200 -y 50

# Pane 0 (top): Gazebo
tmux send-keys -t $SESSION "source /opt/ros/humble/setup.bash && export TURTLEBOT3_MODEL=burger && export GAZEBO_PLUGIN_PATH=/opt/ros/humble/lib:\$GAZEBO_PLUGIN_PATH && ros2 launch turtlebot3_gazebo empty_world.launch.py" ""

# Pane 1 (middle): Our pipeline
tmux split-window -v -t $SESSION
tmux send-keys -t $SESSION "source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch trajectory_smoother trajectory_tracking.launch.py" ""

# Pane 2 (bottom): RViz
tmux split-window -v -t $SESSION
tmux send-keys -t $SESSION "source /opt/ros/humble/setup.bash && source install/setup.bash && rviz2 -d install/trajectory_smoother/share/trajectory_smoother/config/trajectory_tracking.rviz" ""

# Even out pane sizes
tmux select-layout -t $SESSION even-vertical

# Focus on top pane (start Gazebo first)
tmux select-pane -t $SESSION:0.0

tmux attach -t $SESSION

echo ""
echo "Press Enter in each pane (top to bottom) to start:"
echo "  Pane 0: Gazebo + TurtleBot3"
echo "  Pane 1: Trajectory tracking pipeline"
echo "  Pane 2: RViz visualization"