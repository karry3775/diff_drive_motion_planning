#!/bin/bash
# Launch 3 tmux panes with all commands pre-typed.
# Press Enter in each pane top to bottom.

SESSION="motion_planner_demo"
tmux kill-session -t $SESSION 2>/dev/null

tmux new-session -d -s $SESSION -x 200 -y 50

# Pane 0: Gazebo
tmux send-keys -t $SESSION "source /opt/ros/humble/setup.bash && export TURTLEBOT3_MODEL=burger && export GAZEBO_PLUGIN_PATH=/opt/ros/humble/lib:\$GAZEBO_PLUGIN_PATH && ros2 launch turtlebot3_gazebo empty_world.launch.py" ""

# Pane 1: Motion planner pipeline
tmux split-window -v -t $SESSION
tmux send-keys -t $SESSION "source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch motion_planner_service motion_planner.launch.py" ""

# Pane 2: RViz
tmux split-window -v -t $SESSION
tmux send-keys -t $SESSION "source /opt/ros/humble/setup.bash && source install/setup.bash && rviz2 -d install/motion_planner_service/share/motion_planner_service/config/trajectory_tracking.rviz" ""

tmux select-layout -t $SESSION even-vertical
tmux select-pane -t $SESSION:0.0
tmux attach -t $SESSION
