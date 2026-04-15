"""Controller interface and factory."""

from abc import ABC, abstractmethod
from motion_planner_core.types import RobotState, VelocityCommand, Trajectory


class Controller(ABC):

    @abstractmethod
    def compute_command(self, state: RobotState, trajectory: Trajectory,
                        elapsed_time: float) -> VelocityCommand:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @property
    @abstractmethod
    def goal_reached(self) -> bool:
        ...

    @property
    def faulted(self) -> bool:
        return False


def create_controller(config: dict) -> Controller:
    """Factory: create controller from config dict."""
    ctrl_type = config.get('controller', {}).get('type', 'pure_pursuit')
    ctrl_cfg = config.get('controller', {})
    traj_cfg = config.get('trajectory', {})

    if ctrl_type == 'pid':
        from motion_planner_core.pid_controller import PIDController
        return PIDController(
            kp_lateral=ctrl_cfg.get('kp_lateral', 2.0),
            kp_longitudinal=ctrl_cfg.get('kp_longitudinal', 1.0),
            kd_lateral=ctrl_cfg.get('kd_lateral', 0.1),
            max_linear_vel=traj_cfg.get('max_velocity', 0.22),
            max_angular_vel=ctrl_cfg.get('max_angular_velocity', 2.84),
            max_cross_track_error=ctrl_cfg.get('max_cross_track_error', 0.5),
            goal_tolerance=ctrl_cfg.get('goal_tolerance', 0.1),
        )
    elif ctrl_type == 'feedforward':
        from motion_planner_core.feedforward_controller import FeedforwardController
        return FeedforwardController(
            goal_tolerance=ctrl_cfg.get('goal_tolerance', 0.1),
        )
    else:
        from motion_planner_core.pure_pursuit import PurePursuitController
        return PurePursuitController(
            lookahead_distance=ctrl_cfg.get('lookahead_distance', 0.3),
            goal_tolerance=ctrl_cfg.get('goal_tolerance', 0.1),
            max_linear_vel=traj_cfg.get('max_velocity', 0.22),
            max_angular_vel=ctrl_cfg.get('max_angular_velocity', 2.84),
        )
