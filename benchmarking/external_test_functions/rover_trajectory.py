import torch
from botorch.test_functions.synthetic import (
    SyntheticTestFunction,
    ConstrainedBaseTestProblem,
)
from gpytorch.kernels.kernel import Distance
import numpy as np
import itertools
import scipy.interpolate as si


# Rover utility classes and functions from the original code
class Trajectory:
    def __init__(self):
        pass

    def set_params(self, start, goal, params):
        raise NotImplementedError

    def get_points(self, t):
        raise NotImplementedError

    @property
    def param_size(self):
        raise NotImplementedError


class PointBSpline(Trajectory):
    """
    dim : number of dimensions of the state space
    num_points : number of internal points used to represent the trajectory.
                    Note, internal points are not necessarily on the trajectory.
    """

    def __init__(self, dim, num_points):
        self.tck = None
        self.d = dim
        self.npoints = num_points
        self.control_points = None  # Store control points for plotting

    def set_params(self, params, start, goal=None):
        assert start is not None

        points = np.hstack((start[:, None], params.reshape((-1, self.d)).T)).cumsum(
            axis=1
        )

        # Store the control points before spline fitting
        self.control_points = points.T.copy()

        # Bug Fix: Remove all occurances of the same (x,y) point repeated twice from path
        xp = points[0]
        yp = points[1]
        okay = np.where(np.abs(np.diff(xp)) + np.abs(np.diff(yp)) > 0)
        xp = np.r_[xp[okay], xp[-1]]
        yp = np.r_[yp[okay], yp[-1]]
        self.tck, u = si.splprep([xp, yp], k=3, s=0)

    def get_points(self, t):
        assert self.tck is not None, (
            "Parameters have to be set with set_params() before points can be queried."
        )
        return np.vstack(si.splev(t, self.tck)).T

    def get_control_points(self):
        """Return the control points used to define the spline"""
        return self.control_points

    @property
    def param_size(self):
        return self.d * self.npoints


class AABoxes:
    def __init__(self, lows, highs):
        self.l = lows
        self.h = highs

    def contains(self, X):
        if X.ndim == 1:
            X = X[None, :]

        lX = self.l.T[None, :, :] <= X[:, :, None]
        hX = self.h.T[None, :, :] > X[:, :, None]

        return lX.all(axis=1) & hX.all(axis=1)


class NegGeom:
    def __init__(self, geometry):
        self.geom = geometry

    def contains(self, X):
        return ~self.geom.contains(X)


class UnionGeom:
    def __init__(self, geometries):
        self.geoms = geometries

    def contains(self, X):
        return np.any(
            np.hstack([g.contains(X) for g in self.geoms]), axis=1, keepdims=True
        )


class ConstObstacleCost:
    def __init__(self, geometry, cost):
        self.geom = geometry
        self.c = cost

    def __call__(self, X):
        return self.c * self.geom.contains(X)


class ConstCost:
    def __init__(self, cost):
        self.c = cost

    def __call__(self, X):
        if X.ndim == 1:
            X = X[None, :]
        return np.ones((X.shape[0], 1)) * self.c


class AdditiveCosts:
    def __init__(self, fns):
        self.fns = fns

    def __call__(self, X):
        return np.sum(np.hstack([f(X) for f in self.fns]), axis=1)


class RoverDomain:
    """
    Rover domain defined on R^d
    cost_fn : vectorized function giving a scalar cost to states
    start : a start state for the rover
    goal : a goal state
    traj : a parameterized trajectory object offering an interface
            to interpolate point on the trajectory
    s_range : the min and max of the state with s_range[0] in R^d are
                the mins and s_range[1] in R^d are the maxs
    """

    def __init__(
        self,
        cost_fn,
        start,
        goal,
        traj,
        force_start=True,
        force_goal=False,
        rnd_stream=None,
    ):
        self.cost_fn = cost_fn
        self.start = start
        self.goal = goal
        self.traj = traj
        self.force_start = force_start
        self.force_goal = force_goal
        self.rnd_stream = rnd_stream

        if self.rnd_stream is None:
            self.rnd_stream = np.random.RandomState(np.random.randint(0, 2**32 - 1))

    # return the negative cost which need to be optimized
    def __call__(self, params, n_samples=1000):
        self.set_params(params)
        return -1 * self.estimate_cost(n_samples=n_samples)

    def set_params(self, params):
        self.traj.set_params(
            params,
            self.start if self.force_start else None,
            self.goal if self.force_goal else None,
        )

    def estimate_cost(self, n_samples=1000):
        # get points on the trajectory
        points = self.traj.get_points(np.linspace(0, 1.0, n_samples, endpoint=True))

        # compute cost at each point
        costs = self.cost_fn(points)

        # estimate (trapezoidal) the integral of the cost along traj
        avg_cost = 0.5 * (costs[:-1] + costs[1:])
        l = np.linalg.norm(points[1:] - points[:-1], axis=1)
        total_cost = np.sum(l * avg_cost)

        assert self.force_start
        if not self.force_goal:
            total_cost += 100 * np.linalg.norm(points[-1] - self.goal, 1)
        return total_cost

    def trajectory(self, params, n_samples=1000):
        self.set_params(params)
        return self.traj.get_points(np.linspace(0, 1.0, n_samples, endpoint=True))

    def get_control_points(self, params):
        """Get the control points for the trajectory"""
        self.set_params(params)
        return self.traj.get_control_points()

    def trajectory_length(self, params, n_samples=1000):
        # Compute the length of the trajectory
        self.set_params(params)
        points = self.traj.get_points(np.linspace(0, 1.0, n_samples, endpoint=True))
        dists = np.sqrt(((points[1:, :] - points[:-1, :]) ** 2).sum(-1))
        trajectory_length = dists.sum()
        return trajectory_length

    def distance_from_goal(self, params, n_samples=1000):
        self.set_params(params)
        points = self.traj.get_points(np.linspace(0, 1.0, n_samples, endpoint=True))
        return np.linalg.norm(points[-1] - self.goal, 1)

    @property
    def input_size(self):
        return self.traj.param_size


class ConstantOffsetFn:
    def __init__(self, fn_instance, offset):
        self.fn_instance = fn_instance
        self.offset = offset

    def __call__(self, x):
        return self.fn_instance(x) + self.offset

    def get_range(self):
        return self.fn_instance.get_range()


def create_cost_large():
    # a = [0.0, 0.4, 0.8, 1.0]
    a = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    c = np.array(list(itertools.product(a, a)))
    c = c[1:-1]
    obstacle_delta = 0.1

    l = c - obstacle_delta / 2
    h = c + obstacle_delta / 2

    r_box = np.array([[0.5, 0.5]])
    r_l = r_box - 0.5
    r_h = r_box + 0.5

    trees = AABoxes(l, h)
    r_box = NegGeom(AABoxes(r_l, r_h))
    obstacles = UnionGeom([trees, r_box])

    start = np.zeros(2) + 0.05
    goal = np.array([0.95, 0.95])

    costs = [ConstObstacleCost(obstacles, cost=20.0), ConstCost(0.05)]
    cost_fn = AdditiveCosts(costs)
    return cost_fn, start, goal, l, obstacle_delta


def create_large_domain(n_points=30, force_start=True, force_goal=False):
    cost_fn, start, goal, obstacle_l, obstacle_delta = create_cost_large()
    traj = PointBSpline(dim=2, num_points=n_points)
    domain = RoverDomain(
        cost_fn,
        start=start,
        goal=goal,
        traj=traj,
        force_start=force_start,
        force_goal=force_goal,
    )
    domain.obstacle_l = obstacle_l
    domain.obstacle_delta = obstacle_delta
    return domain


class RoverObjective(SyntheticTestFunction):
    """
    Rover optimization task implemented as a BoTorch SyntheticTestFunction.

    Goal is to find a policy for the Rover which results in a trajectory
    that moves the rover from start point to end point while avoiding
    obstacles, thereby maximizing reward.
    """

    def __init__(
        self,
        dim=60,
        seed=None,
        noise_std=1e-5,
        negate=False,
    ):
        assert dim % 2 == 0, "Dimension must be even"

        self.dim = dim
        self.n_points = dim // 2

        # Setting bounds to match the original Rover class
        lb = -0.5 * 4 / dim
        ub = 4 / dim
        bounds = [(lb, ub) for _ in range(self.dim)]

        super().__init__(
            noise_std=noise_std,
            negate=negate,
            bounds=bounds,
        )

        self.lb = lb
        self.ub = ub
        

        # Use the rover utilities defined above

        # Create rover domain
        self.domain = create_large_domain(n_points=self.n_points)

        # Create rover oracle
        f_max = 5.0  # default maximum value
        self.oracle = ConstantOffsetFn(self.domain, f_max)

        # Create distance module for trajectory comparison
        self.dist_module = Distance()

        # Dictionary to cache trajectories for efficiency
        self.xs_to_trajectories_dict = {}

        # Rover oracle needs torch.double datatype
        self.tkwargs = {"dtype": torch.double}

        # For consistency with random seed initialization
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def evaluate_true(self, X_in):
        """
        Evaluate the rover objective function.
        X: tensor of shape [batch_size, dim] containing trajectory parameters
        """
        X = torch.Tensor(X_in)
        # Handle single vs. batch inputs
        if X.dim() == 1:
            X = X.unsqueeze(0)

        batch_size = X.shape[0]
        result = torch.zeros(batch_size, dtype=X.dtype, device=X.device)

        for i in range(batch_size):
            x = X[i].to(**self.tkwargs)
            # Query the oracle for reward
            reward = torch.tensor(self.oracle(x.cpu().numpy())).to(**self.tkwargs)
            result[i] = reward

        return result.numpy()  # .squeeze() if batch_size == 1 else result

    def get_trajectory(self, x):
        """
        Get trajectory for a given parameter vector.
        Uses caching to avoid recomputing trajectories.
        """
        # Convert to hashable key for caching
        x_key = tuple(x.cpu().numpy().tolist())

        if x_key not in self.xs_to_trajectories_dict:
            trajectory = torch.from_numpy(self.domain.trajectory(x.cpu().numpy())).to(
                **self.tkwargs
            )
            self.xs_to_trajectories_dict[x_key] = trajectory

        return self.xs_to_trajectories_dict[x_key]

    def get_control_points(self, x):
        """Get control points for a given parameter vector."""
        return self.domain.get_control_points(x.cpu().numpy())

    def divf(self, x1, x2):
        """
        Compute trajectory diversity function between two parameter vectors.
        """
        traj1 = self.get_trajectory(x1)
        traj2 = self.get_trajectory(x2)
        return self.get_one_way_distance(traj1, traj2)

    def get_one_way_distance(self, trajA, trajB):
        """
        Returns one way distance (OWD) between two trajectories
        (https://zheng-kai.com/paper/vldbj_2019.pdf)

        d_proj(A,B) = (1/N_points) * Sum of Euclidean dist from each point
                      in A to NEAREST point in B
        d(A, B) = mean(d_proj(A, B), d_proj(B, A)) --> symmetric distance metric

        Also supports batches of pairwise trajectories.
        """
        N = trajA.shape[-2]

        # Move to GPU if available
        device = trajA.device
        trajA, trajB = trajA.to(device), trajB.to(device)

        # Compute distance matrix between all points in both trajectories
        dist_matrix = self.dist_module._dist(trajA, trajB, postprocess=False)

        # Find minimum distances
        dists_AB, _ = torch.min(dist_matrix, dim=-1)
        dists_BA, _ = torch.min(dist_matrix, dim=-2)

        # Compute symmetric one-way distance
        return (
            0.5
            * ((dists_AB.sum(dim=-1) / N) + (dists_BA.sum(dim=-1) / N)).detach().cpu()
        )

    def get_rover_trajectory_info(self, x):
        """
        Additional utility function to get detailed trajectory information.
        """
        x = x.to(**self.tkwargs)

        # Get trajectory points
        trajectory = self.get_trajectory(x)

        # Get control points
        control_points = self.get_control_points(x)

        # Get additional metrics
        trajectory_length = self.domain.trajectory_length(x.cpu().numpy())
        distance_from_goal = self.domain.distance_from_goal(x.cpu().numpy())

        return {
            "trajectory": trajectory,
            "control_points": control_points,
            "length": trajectory_length,
            "goal_distance": distance_from_goal,
            "reward": self.oracle(x.cpu().numpy()),
        }



# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create rover objective
    rover = RoverObjective(dim=60)

    data = [
        -0.00506287,
        0.02060261,
        0.00730977,
        0.0016916,
        0.01141071,
        0.02159766,
        0.0492761,
        0.06291116,
        -0.03199102,
        0.02348931,
        0.02233347,
        0.02839941,
        0.06485065,
        0.05000235,
        0.05023854,
        0.03068282,
        -0.0220901,
        0.0289274,
        0.06695845,
        0.02539399,
        0.00969425,
        0.05434358,
        0.01565729,
        0.0167469,
        0.01752062,
        0.0180939,
        0.06431996,
        0.02859457,
        0.01903339,
        0.01130354,
        0.02219424,
        0.01133928,
        0.00718087,
        0.0188761,
        0.04755801,
        0.05707142,
        0.04676004,
        0.02120233,
        0.01717024,
        0.0533459,
        0.05599315,
        0.03648545,
        0.02536945,
        0.02635085,
        0.05733743,
        0.04463304,
        0.02018814,
        0.01227643,
        0.04309239,
        0.04966166,
        0.0460362,
        0.03778013,
        0.03390507,
        0.0331698,
        0.04890742,
        0.05239467,
        0.03219414,
        0.03429793,
        0.05689583,
        -0.01187175,
    ]

    data_turbo = [
        -0.03011088,
        0.03862359,
        0.02111968,
        0.05352444,
        0.04358243,
        0.05432966,
        -0.00984167,
        0.00503739,
        0.02663427,
        0.01677385,
        0.05939965,
        0.06429878,
        0.05656003,
        0.01846387,
        0.06213769,
        0.06561479,
        -0.02048193,
        0.05034107,
        0.04905614,
        -0.00364762,
        0.05205394,
        0.06331383,
        0.04530138,
        0.00797831,
        0.01672425,
        0.01224421,
        0.06587455,
        0.02451582,
        -0.00287866,
        -0.00856491,
        0.03529423,
        -0.0331671,
        0.02429823,
        -0.00279762,
        0.05858013,
        0.04688421,
        0.04291008,
        0.0193781,
        0.01279675,
        0.02405777,
        0.05269766,
        0.04347129,
        0.04756951,
        0.06503204,
        0.02711504,
        0.05819203,
        0.02462525,
        -0.02586376,
        0.03599225,
        0.02715711,
        0.01974291,
        0.03586825,
        0.02920148,
        0.06198646,
        -0.00024728,
        0.04537533,
        0.03589285,
        0.0482811,
        0.01847443,
        0.02357918,
    ]

    x_test = data

    # # Test with the provided data
    # x_test = torch.tensor(
    #     [
    #         0.04180854,
    #         0.01349452,
    #         0.03545127,
    #         0.04143764,
    #         0.06261411,
    #         0.01600458,
    #         0.05684595,
    #         0.02679812,
    #         0.02456654,
    #         0.01221882,
    #         0.04009978,
    #         0.02052418,
    #         0.03019811,
    #         0.05808693,
    #         0.04044085,
    #         0.04624122,
    #         0.02477061,
    #         0.01580308,
    #         0.01339954,
    #         0.01267901,
    #         0.01207312,
    #         0.01240416,
    #         0.05002585,
    #         0.01180359,
    #         0.04829229,
    #         0.04438353,
    #         0.02450956,
    #         0.05724017,
    #         0.02363357,
    #         0.01382697,
    #         0.02033668,
    #         0.02668583,
    #         0.06165221,
    #         0.01580004,
    #         0.03389649,
    #         0.03818482,
    #         0.01239468,
    #         0.04392796,
    #         -0.0079251,
    #         0.05126693,
    #         0.00967846,
    #         0.02522195,
    #         0.03708684,
    #         0.02202524,
    #         0.04338896,
    #         0.01903247,
    #         0.0335147,
    #         0.02094862,
    #         0.04944709,
    #         0.05959421,
    #         0.00808542,
    #         0.0350554,
    #         0.01701962,
    #         0.01136674,
    #         -0.00322388,
    #         0.0310858,
    #         0.02617229,
    #         0.04197587,
    #         0.0282273,
    #         0.05501545,
    #     ]
    # ).reshape(-1)

    # Evaluate objective
    reward = rover.evaluate_true(x_test)
    print(f"Reward: {reward.item():.4f}")

    # Get trajectory information
    traj_info = rover.get_rover_trajectory_info(torch.tensor(x_test))
    print(f"Trajectory length: {traj_info['length']:.4f}")
    print(f"Distance from goal: {traj_info['goal_distance']:.4f}")

    # Plot trajectory if 2D
    trajectory = traj_info["trajectory"]
    control_points = traj_info["control_points"]

    if trajectory.shape[1] == 2:
        plt.figure(figsize=(10, 8))

        # Plot the smooth trajectory
        plt.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            "b-",
            linewidth=2,
            label="Smooth Trajectory",
        )

        # Plot the control points
        plt.plot(
            control_points[:, 0],
            control_points[:, 1],
            "o-",
            color="orange",
            markersize=6,
            linewidth=1,
            label="Control Points",
            alpha=0.7,
        )

        # Highlight start and goal
        plt.plot(
            rover.domain.start[0],
            rover.domain.start[1],
            "go",
            markersize=12,
            label="Start",
        )
        plt.plot(
            rover.domain.goal[0],
            rover.domain.goal[1],
            "ro",
            markersize=12,
            label="Goal",
        )

        # Plot obstacles if available
        if hasattr(rover.domain, "obstacle_l"):
            import matplotlib.patches as patches

            delta = rover.domain.obstacle_delta
            for i in range(len(rover.domain.obstacle_l)):
                rect = patches.Rectangle(
                    (rover.domain.obstacle_l[i, 0], rover.domain.obstacle_l[i, 1]),
                    delta,
                    delta,
                    linewidth=1,
                    edgecolor="darkred",
                    facecolor="darkred",
                    alpha=0.7,
                )
                plt.gca().add_patch(rect)

        plt.title(f"Rover Trajectory (Reward: {reward.item():.3f})")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis("equal")

        # Add some padding to the plot
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)

        plt.tight_layout()
        plt.show(block=False)
        plt.savefig("rover_sample.pdf", format="pdf", bbox_inches="tight", pad_inches=0.0)
