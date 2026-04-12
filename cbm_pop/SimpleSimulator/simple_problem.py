import math
from enum import Enum

import numpy as np


class ProblemClass(Enum):
    SimpleGrid = "Simple_Grid"
    RandomSpread = "Random_Spread"
    LinearRows = "Linear_Rows"
    DensityGradient = "Density_Gradient"
    WindFarmGrid = "Wind_Farm_Grid"
    DisjointRegions = "Disjoint_Regions"
    BottleneckCorridor = "Bottleneck_Corridor"
    ConcentricRings = "Concentric_Rings"
    HierarchicalClusters = "Hierarchical_Clusters"


class SimpleProblem:
    def __init__(self, problem_class, grid_size=15, problem_seed=1, **kwargs):
        """
        Parameters
        ----------
        problem_class : ProblemClass
        grid_size : int
            Controls the environment scale. Total tasks = grid_size^2 for all
            environments, preserving comparability across layouts.
        problem_seed : int
            RNG seed for reproducibility.
        **kwargs
            Environment-specific parameters (see per-class notes below).

            padding : float
                Minimum distance from each task to the grid boundary.
                Tasks are constrained to [padding, grid_size - padding] in both
                axes. Default 0.5. Applies to all environment types.

            LinearRows
            ----------
            num_rows : int
                Number of parallel rows of tasks. Setting num_rows < num_agents
                forces agents to share rows; num_rows > num_agents forces splits.
                Default grid_size // 3, roughly one row per 3 agents.
            row_imbalance : float in [0, 1]
                0 produces equal task counts per row (uniform workload).
                1 concentrates all tasks in the first row (maximum imbalance).
                Intermediate values interpolate linearly. Default 0 (balanced).
            jitter : float
                Positional noise added to each task, expressed as a fraction of
                inter-task spacing. 0 = perfectly regular rows. Default 0.1.

            DensityGradient
            ---------------
            num_components : int
                Number of Gaussian mixture components. One component is dominant
                (weight controlled by gradient_strength); the rest share the
                remainder equally. Default 3.
            gradient_strength : float in (0, 1)
                Fraction of tasks assigned to the dominant component.
                0.5 = uniform mixture; approaching 1.0 = all tasks in one region.
                Default 0.7.
            dominant_region : str
                Where the dominant cluster is placed: 'corner', 'centre', 'edge'.
                Default 'corner', which creates the strongest agent asymmetry.

            WindFarmGrid
            ------------
            num_rows : int
                Number of turbine rows. Default max(1, size // 4), giving a
                row spacing typical of real offshore wind farms.
            row_stagger : float in [0, 1]
                Fraction of inter-turbine spacing by which alternating rows are
                laterally offset. 0.0 = aligned columns, 0.5 = classic half-pitch
                stagger (standard layout to reduce wake interaction). Default 0.5.
            points_per_blade : int
                Number of inspection waypoints distributed along each blade,
                not counting the hub. Total task poses per turbine =
                1 (hub) + 3 * points_per_blade. Default 3.
            blade_length_fraction : float in (0, 1)
                Blade length expressed as a fraction of the inter-turbine
                spacing within a row. Values above ~0.45 risk blades from
                adjacent turbines overlapping. Default 0.35.
            blade_angle_offset : float
                Rotation of the entire rotor in degrees, applied uniformly to
                all turbines. 0 = one blade pointing straight up (+y). Default 0.
            jitter : float
                Gaussian noise added to each turbine hub position as a fraction
                of inter-turbine spacing, simulating installation tolerances.
                Blade geometry is computed relative to the jittered hub so the
                rotor shape is preserved. Default 0.05.

            DisjointRegions
            ---------------
            num_regions : int
                Number of completely separated dense zones. Each region is a
                tight Gaussian cluster. Minimum separation between region
                boundaries is enforced so no two regions ever overlap, unlike
                RandomClusters where separation is only probabilistic.
                Default 3.
            region_radius : float
                Std dev of the Gaussian spread within each region, expressed
                as a fraction of the placement span. Smaller values produce
                tighter, more isolated regions with wider sparse gaps between
                them. Default 0.06.
            balance_regions : bool
                If True, tasks are distributed evenly across regions.
                If False, region sizes are drawn from a Dirichlet distribution,
                creating asymmetric workloads. Default False.

            BottleneckCorridor
            ------------------
            corridor_width : float
                Half-width of the connecting corridor expressed as a fraction
                of the grid span. Narrower values (e.g. 0.04) produce a tighter
                bottleneck; wider values (e.g. 0.15) reduce the routing
                constraint. Default 0.06.
            corridor_task_fraction : float in (0, 1)
                Fraction of total tasks placed along the corridor bridge.
                The remaining tasks are split evenly between the two endpoint
                clusters. Default 0.15.
            end_cluster_radius : float
                Std dev of the Gaussian spread within each endpoint cluster,
                expressed as a fraction of the grid span. Default 0.08.

            ConcentricRings
            ---------------
            num_rings : int
                Number of concentric circular bands. Each ring receives an
                equal share of tasks, producing uniform angular density.
                Default 4.
            ring_width : float
                Radial thickness of each ring expressed as a fraction of the
                ring spacing. Values near 0 produce thin arcs; values near 1
                produce rings that nearly touch. Default 0.3.
            centre : tuple of float or None
                (x, y) position of the common centre in grid coordinates.
                None places the centre at the grid midpoint. Default None.

            HierarchicalClusters
            --------------------
            num_macro : int
                Number of macro-regions. These are placed across the grid with
                the same minimum-separation guarantee used in DisjointRegions.
                Default 3.
            num_micro : int
                Number of micro-clusters within each macro-region. Default 4.
            macro_radius : float
                Radius of each macro-region (used as the placement margin and
                the spread for sampling micro-cluster centres). Default 2.5.
            micro_radius : float
                Std dev of the Gaussian spread within each micro-cluster.
                Should be substantially smaller than macro_radius. Default 0.5.
            balance : bool
                If True, tasks are distributed evenly across all
                macro * micro leaf clusters. If False, leaf sizes are drawn
                from a Dirichlet distribution. Default False.

        """
        self.task_poses = None
        self.initial_robot_cost_matrix = None
        self.current_robot_cost_matrix = None
        self.problem_class = problem_class

        try:
            size = int(grid_size)
        except (TypeError, ValueError):
            raise ValueError("grid_size must be an integer") from None

        if size < 1:
            raise ValueError("grid_size must be >= 1")

        self.grid_size = size

        padding = kwargs.get("padding", 0.5)
        if padding < 0:
            raise ValueError("padding must be >= 0")
        if 2 * padding >= size:
            raise ValueError("padding too large: 2 * padding must be < grid_size")
        self.padding = padding

        if problem_class == ProblemClass.SimpleGrid:
            self.task_poses = [
                (i + 0.5, j + 0.5)
                for i in range(size)
                for j in range(size)
                if padding <= i + 0.5 <= size - padding
                and padding <= j + 0.5 <= size - padding
            ]

        elif problem_class == ProblemClass.RandomSpread:
            self.task_poses = self.generate_random_spread(size, problem_seed, padding)

        elif problem_class == ProblemClass.LinearRows:
            num_rows = kwargs.get("num_rows", max(1, size // 3))
            row_imbalance = kwargs.get("row_imbalance", 0.0)
            jitter = kwargs.get("jitter", 0.1)
            self.task_poses = self.generate_linear_rows(
                size, problem_seed, num_rows, row_imbalance, jitter, padding
            )

        elif problem_class == ProblemClass.DensityGradient:
            num_components = kwargs.get("num_components", 3)
            gradient_strength = kwargs.get("gradient_strength", 0.7)
            dominant_region = kwargs.get("dominant_region", "corner")
            self.task_poses = self.generate_density_gradient(
                size, problem_seed, num_components, gradient_strength, dominant_region,
                padding,
            )

        elif problem_class == ProblemClass.WindFarmGrid:
            num_rows = kwargs.get("num_rows", max(1, size // 4))
            row_stagger = kwargs.get("row_stagger", 0.5)
            points_per_blade = kwargs.get("points_per_blade", 3)
            blade_length_fraction = kwargs.get("blade_length_fraction", 0.35)
            blade_angle_offset = kwargs.get("blade_angle_offset", 0.0)
            jitter = kwargs.get("jitter", 0.05)
            self.task_poses = self.generate_wind_farm_grid(
                size, problem_seed, num_rows, row_stagger, points_per_blade,
                blade_length_fraction, blade_angle_offset, jitter, padding,
            )

        elif problem_class == ProblemClass.DisjointRegions:
            num_regions = kwargs.get("num_regions", 3)
            region_radius = kwargs.get("region_radius", 0.06)
            balance_regions = kwargs.get("balance_regions", False)
            self.task_poses = self.generate_disjoint_regions(
                size, problem_seed, num_regions, region_radius, balance_regions,
                padding,
            )

        elif problem_class == ProblemClass.BottleneckCorridor:
            corridor_width = kwargs.get("corridor_width", 0.06)
            corridor_task_fraction = kwargs.get("corridor_task_fraction", 0.15)
            end_cluster_radius = kwargs.get("end_cluster_radius", 0.08)
            self.task_poses = self.generate_bottleneck_corridor(
                size, problem_seed, corridor_width, corridor_task_fraction,
                end_cluster_radius, padding,
            )

        elif problem_class == ProblemClass.ConcentricRings:
            num_rings = kwargs.get("num_rings", 4)
            ring_width = kwargs.get("ring_width", 0.3)
            centre = kwargs.get("centre", None)
            self.task_poses = self.generate_concentric_rings(
                size, problem_seed, num_rings, ring_width, centre, padding,
            )

        elif problem_class == ProblemClass.HierarchicalClusters:
            num_macro = kwargs.get("num_macro", 3)
            num_micro = kwargs.get("num_micro", 4)
            macro_radius = kwargs.get("macro_radius", 2.5)
            micro_radius = kwargs.get("micro_radius", 0.5)
            balance = kwargs.get("balance", False)
            self.task_poses = self.generate_hierarchical_clusters(
                size, problem_seed, num_macro, num_micro, macro_radius,
                micro_radius, balance, padding,
            )

        else:
            raise NotImplementedError(f"Unsupported problem class: {problem_class}")

        self.cost_matrix = self.calculate_cost_matrix()
        self.num_tasks = len(self.task_poses)

    # ------------------------------------------------------------------
    # Generators
    # ------------------------------------------------------------------

    def generate_random_spread(self, size, seed, padding):
        """Random spread generator. Tasks uniformly sampled within padded bounds."""
        np.random.seed(seed)
        task_poses = []
        total_tasks = size * size
        lo, hi = padding, size - padding
        for _ in range(total_tasks):
            x_task = np.random.uniform(lo, hi)
            y_task = np.random.uniform(lo, hi)
            task_poses.append((x_task, y_task))
        return task_poses

    def generate_linear_rows(self, size, seed, num_rows, row_imbalance, jitter, padding):
        """
        Linear row environment.

        Tasks are arranged in parallel horizontal rows, directly mirroring
        the physical layout of a wind farm where turbines are arranged in
        rows across the site.

        row_imbalance controls workload asymmetry:
          - 0.0: tasks distributed uniformly across rows
          - 1.0: all tasks concentrated in the first row
          Intermediate values use a linear interpolation of a geometric
          distribution, so earlier rows are progressively denser.

        jitter adds Gaussian noise to each task position (as a fraction of
        inter-task spacing), preventing unrealistically regular layouts while
        preserving row structure.

        Rows and tasks within rows are spaced evenly within the padded region.
        """
        rng = np.random.default_rng(seed)
        total_tasks = size * size

        if row_imbalance == 0.0 or num_rows == 1:
            base = total_tasks // num_rows
            tasks_per_row = [base] * num_rows
            for i in range(total_tasks % num_rows):
                tasks_per_row[i] += 1
        else:
            raw_weights = np.array(
                [(1.0 - row_imbalance) ** i for i in range(num_rows)]
            )
            raw_weights /= raw_weights.sum()
            tasks_per_row = [max(1, round(w * total_tasks)) for w in raw_weights]
            diff = sum(tasks_per_row) - total_tasks
            tasks_per_row[-1] = max(1, tasks_per_row[-1] - diff)

        lo, hi = padding, size - padding
        row_ys = np.linspace(lo, hi, num_rows + 2)[1:-1]

        task_poses = []
        for row_y, n_tasks in zip(row_ys, tasks_per_row):
            if n_tasks == 1:
                xs = np.array([(lo + hi) / 2.0])
            else:
                xs = np.linspace(lo, hi, n_tasks + 2)[1:-1]

            spacing = (hi - lo) / max(n_tasks, 1)
            for x in xs:
                noise_x = rng.normal(0, jitter * spacing)
                noise_y = rng.normal(0, jitter * spacing)
                px = float(np.clip(x + noise_x, lo, hi))
                py = float(np.clip(row_y + noise_y, lo, hi))
                task_poses.append((px, py))

        return task_poses

    def generate_density_gradient(
        self, size, seed, num_components, gradient_strength, dominant_region, padding
    ):
        """
        Density gradient environment.

        Tasks are drawn from a Gaussian mixture model in which one component
        is dominant (controlled by gradient_strength). The remaining components
        share the remainder of the probability mass equally.

        This creates an asymmetric workload problem where agents near the dense
        region have far more tasks in their natural neighbourhood than agents
        near the sparse region. It specifically stresses whether the mimetism
        mechanism transfers useful operator policies between agents facing
        structurally different local subproblems.

        dominant_region controls where the high-density centre is placed:
          - 'corner': bottom-left corner — maximises asymmetry relative to
            uniformly distributed agent start positions
          - 'centre': centre of the grid — moderate asymmetry
          - 'edge': midpoint of the left edge — asymmetric but not extreme

        All dominant region anchors and task clipping respect the padding
        boundary, so no tasks accumulate at grid edges.
        """
        rng = np.random.default_rng(seed)
        total_tasks = size * size

        lo, hi = padding, size - padding

        dominant_positions = {
            "corner": (lo + (hi - lo) * 0.15, lo + (hi - lo) * 0.15),
            "centre": ((lo + hi) / 2.0, (lo + hi) / 2.0),
            "edge":   (lo + (hi - lo) * 0.05, (lo + hi) / 2.0),
        }
        if dominant_region not in dominant_positions:
            raise ValueError(
                f"dominant_region must be one of {list(dominant_positions.keys())}"
            )
        dominant_centre = dominant_positions[dominant_region]

        component_centres = [dominant_centre]
        for _ in range(num_components - 1):
            component_centres.append(
                (rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo)),
                 rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo)))
            )

        remaining = (1.0 - gradient_strength) / max(num_components - 1, 1)
        weights = [gradient_strength] + [remaining] * (num_components - 1)

        tasks_per_component = [max(1, round(w * total_tasks)) for w in weights]
        diff = sum(tasks_per_component) - total_tasks
        tasks_per_component[-1] = max(1, tasks_per_component[-1] - diff)

        spread = (hi - lo) / (num_components * 2)
        task_poses = []
        for centre, n_tasks in zip(component_centres, tasks_per_component):
            samples = rng.normal(loc=centre, scale=spread, size=(n_tasks, 2))
            for x, y in samples:
                task_poses.append(
                    (float(np.clip(x, lo, hi)), float(np.clip(y, lo, hi)))
                )

        return task_poses

    def generate_wind_farm_grid(
        self, size, seed, num_rows, row_stagger, points_per_blade,
        blade_length_fraction, blade_angle_offset, jitter, padding,
    ):
        """
        Wind farm stencil environment.

        Each turbine is represented by a hub waypoint plus inspection waypoints
        distributed along three blades radiating at 120-degree intervals. This
        directly models the real UAV inspection task: a drone must visit points
        along each blade to capture surface imagery for defect detection.

        Layout
        ------
        Turbines are arranged in staggered horizontal rows. Alternating rows
        are laterally offset by row_stagger * inter-turbine spacing, mimicking
        the half-pitch stagger used in real offshore farms to reduce wake
        interaction between rows.

        Task poses per turbine
        ----------------------
        points_per_turbine = 1 (hub) + 3 * points_per_blade

        Blade waypoints are placed at evenly spaced intervals from just inside
        the hub out to the blade tip. The three blades are separated by 120
        degrees, with the first blade oriented at blade_angle_offset degrees
        from the positive y-axis (pointing up). All blade geometry is computed
        relative to the jittered hub position, so the rotor shape is preserved
        regardless of hub placement noise.

        Total task count
        ----------------
        num_turbines is derived so that num_turbines * points_per_turbine is
        as close to size^2 as possible. Turbines are distributed evenly across
        rows with any remainder assigned to the earliest rows. The actual task
        count may differ slightly from size^2 depending on divisibility; use
        self.num_tasks rather than size^2 when iterating over tasks.

        Blade length
        ------------
        blade_length = blade_length_fraction * inter-turbine spacing within a
        row. At the default of 0.35, blades occupy 70% of the gap between
        adjacent turbines (35% on each side), leaving clear separation.
        Values above ~0.45 risk blade tips from adjacent turbines overlapping.
        """
        rng = np.random.default_rng(seed)
        total_tasks = size * size

        lo, hi = padding, size - padding

        if lo >= hi:
            raise ValueError(
                f"No valid placement region after padding ({padding}). "
                "Reduce padding or increase grid_size."
            )

        # Points per turbine: hub + three blades.
        points_per_turbine = 1 + 3 * points_per_blade

        # Back-calculate number of turbines to keep total close to size^2.
        num_turbines = max(num_rows, total_tasks // points_per_turbine)

        # Distribute turbines evenly across rows; earlier rows absorb remainder.
        base_per_row = num_turbines // num_rows
        remainder = num_turbines % num_rows
        turbines_per_row = [
            base_per_row + (1 if i < remainder else 0)
            for i in range(num_rows)
        ]

        # Row y-positions spaced evenly within padded vertical extent.
        row_ys = np.linspace(lo, hi, num_rows + 2)[1:-1]

        # Blade angles: three blades at 120-degree intervals, offset by
        # blade_angle_offset degrees. Angles are measured from the +y axis
        # (north-up convention matching wind farm layout diagrams).
        angle_offset_rad = math.radians(blade_angle_offset)
        blade_angles = [
            angle_offset_rad + i * (2 * math.pi / 3)
            for i in range(3)
        ]

        # Blade waypoint fractions along the blade: evenly spaced between
        # the hub (exclusive) and the tip (inclusive).
        blade_ts = np.linspace(0, 1, points_per_blade + 2)[1:-1]

        task_poses = []
        for row_idx, (row_y, n_turbines) in enumerate(zip(row_ys, turbines_per_row)):

            if n_turbines == 1:
                hub_xs = np.array([(lo + hi) / 2.0])
                inter_turbine_spacing = hi - lo
            else:
                hub_xs = np.linspace(lo, hi, n_turbines + 2)[1:-1]
                inter_turbine_spacing = (hi - lo) / (n_turbines - 1)

            blade_length = blade_length_fraction * inter_turbine_spacing

            # Alternating rows offset by row_stagger * inter-turbine spacing.
            lateral_shift = (row_idx % 2) * row_stagger * inter_turbine_spacing

            for hub_x in hub_xs:
                # Jitter applied to the hub; blade geometry follows the hub.
                noise_x = rng.normal(0, jitter * inter_turbine_spacing)
                noise_y = rng.normal(0, jitter * inter_turbine_spacing)
                hx = float(np.clip(hub_x + lateral_shift + noise_x, lo, hi))
                hy = float(np.clip(row_y + noise_y, lo, hi))

                # Hub waypoint.
                task_poses.append((hx, hy))

                # Waypoints along each of the three blades.
                for angle in blade_angles:
                    # +y is north: x component uses sin, y component uses cos.
                    dx = math.sin(angle)
                    dy = math.cos(angle)
                    for t in blade_ts:
                        wx = float(np.clip(hx + t * blade_length * dx, lo, hi))
                        wy = float(np.clip(hy + t * blade_length * dy, lo, hi))
                        task_poses.append((wx, wy))

        return task_poses

    def generate_disjoint_regions(
        self, size, seed, num_regions, region_radius, balance_regions, padding,
    ):
        """
        Disjoint regions environment.

        Creates num_regions completely non-overlapping dense zones separated
        by enforced empty gaps. Unlike RandomClusters, separation is
        guaranteed rather than probabilistic: the minimum distance between
        any two region boundaries (centre distance minus two radii) is
        constrained to be at least one absolute radius, so a genuine sparse
        corridor always exists between every pair of regions.

        This directly stress-tests coalition-level mimetism: agents assigned
        to different regions face structurally unrelated local subproblems,
        and learning transfer across the coalition is the only mechanism by
        which experience from one region can benefit agents in another.

        Region centres are placed using rejection sampling with the enforced
        boundary-separation constraint. The absolute radius used for placement
        is region_radius * span, where span is the usable grid extent after
        padding. Tasks within each region are drawn from a 2D Gaussian with
        std dev = absolute_radius / 2.5, ensuring ~99% of tasks fall inside
        the nominal region boundary.

        balance_regions controls workload distribution:
          - True:  equal task counts across all regions (clean baseline)
          - False: Dirichlet-sampled sizes (asymmetric workload stress test)
        """
        rng = np.random.default_rng(seed)
        total_tasks = size * size
        lo, hi = padding, size - padding
        span = hi - lo

        abs_radius = region_radius * span

        # Minimum centre-to-centre distance: two radii + one radius gap.
        min_centre_sep = 3.0 * abs_radius
        centre_margin = abs_radius + padding
        lo_c, hi_c = centre_margin, size - centre_margin

        if lo_c >= hi_c:
            raise ValueError(
                f"DisjointRegions: region_radius ({region_radius:.3f}) too large "
                f"for grid_size={size} with padding={padding}. "
                "Reduce region_radius or padding."
            )

        centres = []
        attempts = 0
        while len(centres) < num_regions:
            candidate = rng.uniform(lo_c, hi_c, size=2)
            if all(
                math.hypot(candidate[0] - c[0], candidate[1] - c[1]) >= min_centre_sep
                for c in centres
            ):
                centres.append(tuple(float(v) for v in candidate))
            attempts += 1
            if attempts > 50_000:
                raise ValueError(
                    f"DisjointRegions: cannot place {num_regions} non-overlapping "
                    f"regions with min separation {min_centre_sep:.2f} in region "
                    f"{hi_c - lo_c:.2f}x{hi_c - lo_c:.2f}. "
                    "Reduce num_regions or region_radius."
                )

        if balance_regions:
            base = total_tasks // num_regions
            tasks_per_region = [base] * num_regions
            for i in range(total_tasks % num_regions):
                tasks_per_region[i] += 1
        else:
            proportions = rng.dirichlet(np.ones(num_regions))
            tasks_per_region = [max(1, round(p * total_tasks)) for p in proportions]
            diff = sum(tasks_per_region) - total_tasks
            tasks_per_region[-1] = max(1, tasks_per_region[-1] - diff)

        task_poses = []
        sigma = abs_radius / 2.5
        for centre, n_tasks in zip(centres, tasks_per_region):
            samples = rng.normal(loc=centre, scale=sigma, size=(n_tasks, 2))
            for x, y in samples:
                task_poses.append(
                    (float(np.clip(x, lo, hi)), float(np.clip(y, lo, hi)))
                )

        return task_poses

    def generate_bottleneck_corridor(
        self, size, seed, corridor_width, corridor_task_fraction,
        end_cluster_radius, padding,
    ):
        """
        Bottleneck corridor environment.

        Two dense endpoint clusters are connected by a narrow linear bridge
        of tasks running horizontally through the centre of the grid. The
        corridor imposes a hard topological constraint on routing: any path
        that visits both endpoint clusters must pass through the bridge,
        forcing agents to reason about sequencing across the bottleneck.

        This tests whether the D-I cycle discovers the regime change inherent
        in transitioning between dense local optimisation within a cluster and
        long-distance travel through the sparse corridor. No existing
        environment creates this kind of topological chokepoint.

        Layout
        ------
        - Left cluster:  centred at (lo + 0.15*span, mid_y)
        - Right cluster: centred at (hi - 0.15*span, mid_y)
        - Corridor:      tasks uniformly sampled along the horizontal midline
                         within a vertical band of half-width corridor_width*span,
                         between the inner edges of the two clusters.

        Parameters
        ----------
        corridor_width : float
            Half-width of the corridor as a fraction of the grid span.
        corridor_task_fraction : float
            Fraction of total tasks placed in the corridor bridge.
            The remaining tasks are split evenly between the two endpoint
            clusters.
        end_cluster_radius : float
            Std dev of the Gaussian spread within each endpoint cluster,
            expressed as a fraction of the grid span.
        """
        rng = np.random.default_rng(seed)
        total_tasks = size * size
        lo, hi = padding, size - padding
        span = hi - lo
        mid_y = (lo + hi) / 2.0

        abs_corridor_width = corridor_width * span
        abs_cluster_radius = end_cluster_radius * span

        left_centre = (lo + 0.15 * span, mid_y)
        right_centre = (hi - 0.15 * span, mid_y)

        n_corridor = max(1, round(corridor_task_fraction * total_tasks))
        n_remaining = total_tasks - n_corridor
        n_left = n_remaining // 2
        n_right = n_remaining - n_left

        # Corridor tasks: uniform x along the bridge, Gaussian y within the
        # corridor half-width. The bridge spans between the inner cluster edges.
        corridor_x_lo = left_centre[0] + abs_cluster_radius
        corridor_x_hi = right_centre[0] - abs_cluster_radius
        if corridor_x_lo >= corridor_x_hi:
            # Clusters too wide relative to grid; fall back to full midline.
            corridor_x_lo = left_centre[0]
            corridor_x_hi = right_centre[0]

        task_poses = []
        for _ in range(n_corridor):
            x = float(rng.uniform(corridor_x_lo, corridor_x_hi))
            y = float(np.clip(
                rng.normal(mid_y, abs_corridor_width / 3.0), lo, hi
            ))
            task_poses.append((x, y))

        # Endpoint cluster tasks drawn from a 2D Gaussian.
        for centre, n_tasks in [(left_centre, n_left), (right_centre, n_right)]:
            samples = rng.normal(loc=centre, scale=abs_cluster_radius, size=(n_tasks, 2))
            for x, y in samples:
                task_poses.append(
                    (float(np.clip(x, lo, hi)), float(np.clip(y, lo, hi)))
                )

        return task_poses

    def generate_concentric_rings(
        self, size, seed, num_rings, ring_width, centre, padding,
    ):
        """
        Concentric rings environment.

        Tasks are placed on concentric circular bands radiating from a shared
        centre. Each ring receives an equal share of the total task budget.
        Tasks within a ring are distributed uniformly in angle and sampled
        uniformly in radius within the ring's radial extent.

        This environment models inspection of circular assets (offshore
        platform decks, storage tank perimeters, radar arrays) and is
        specifically designed to expose the weakness of greedy nearest-
        neighbour: a greedy agent near one ring will repeatedly cross ring
        boundaries rather than sweeping a full arc, incurring large traversal
        costs. The K-NN relocation operator, which pulls spatially proximate
        tasks into the same route segment, should have a structural advantage
        here.

        Ring geometry
        -------------
        The outermost ring radius is set to fit within the padded grid:
            r_max = min(span_x, span_y) / 2 - ring_gap

        where ring_gap = r_max / (2 * num_rings) provides clearance between
        the outer ring boundary and the padding boundary.

        Ring spacing is uniform: rings are indexed 1..num_rings from the
        innermost to the outermost. The radial midpoint of ring i is:
            r_mid_i = (i / num_rings) * r_max

        Each ring spans [r_mid - half_width, r_mid + half_width] where
            half_width = (r_max / num_rings) * ring_width / 2

        Parameters
        ----------
        centre : tuple of float or None
            (x, y) of the common centre. None uses the grid midpoint.
        """
        rng = np.random.default_rng(seed)
        total_tasks = size * size
        lo, hi = padding, size - padding
        span = hi - lo

        if centre is None:
            cx, cy = (lo + hi) / 2.0, (lo + hi) / 2.0
        else:
            cx, cy = float(centre[0]), float(centre[1])

        # Maximum usable radius: half the span, minus a small clearance gap.
        r_max_raw = span / 2.0
        ring_gap = r_max_raw / (2.0 * num_rings)
        r_max = r_max_raw - ring_gap

        ring_spacing = r_max / num_rings
        half_width = ring_spacing * ring_width / 2.0

        # Equal task counts per ring; remainder goes to outermost rings.
        base = total_tasks // num_rings
        tasks_per_ring = [base] * num_rings
        for i in range(total_tasks % num_rings):
            tasks_per_ring[num_rings - 1 - i] += 1

        task_poses = []
        for ring_idx in range(num_rings):
            r_mid = (ring_idx + 1) * ring_spacing
            r_inner = max(0.0, r_mid - half_width)
            r_outer = r_mid + half_width
            n_tasks = tasks_per_ring[ring_idx]

            # Sample angle uniformly; sample radius uniformly in [r_inner, r_outer].
            # Uniform-in-area sampling would use sqrt(U)*r, but uniform-in-radius
            # is intentional here to keep ring density independent of radius,
            # which creates equal arc-length task density per unit radius.
            angles = rng.uniform(0, 2 * math.pi, size=n_tasks)
            radii = rng.uniform(r_inner, r_outer, size=n_tasks)

            for angle, r in zip(angles, radii):
                x = float(np.clip(cx + r * math.cos(angle), lo, hi))
                y = float(np.clip(cy + r * math.sin(angle), lo, hi))
                task_poses.append((x, y))

        return task_poses

    def generate_hierarchical_clusters(
        self, size, seed, num_macro, num_micro, macro_radius, micro_radius,
        balance, padding,
    ):
        """
        Hierarchical cluster environment.

        Tasks are organised at two spatial scales. At the macro scale,
        num_macro region centres are distributed across the grid with a
        guaranteed minimum separation (same algorithm as ControlledClusters).
        Within each macro-region, num_micro micro-cluster centres are placed
        at a smaller scale, and tasks are finally drawn from a Gaussian
        around each micro-cluster centre.

        This two-level structure cannot be reproduced by DisjointRegions
        with a single radius, because the intra-macro gaps and inter-macro
        gaps are qualitatively different in scale. It tests whether the
        Q-learning mechanism in Q-CBM can adapt operator selection to
        multi-scale spatial structure — switching between fine-grained
        intensification within a micro-cluster and larger diversification
        jumps between macro-regions.

        Geometry
        --------
        - Macro-region centres: placed with min separation = span / sqrt(num_macro),
          with a margin of macro_radius from the padded boundary.
        - Micro-cluster centres: drawn from a 2D Gaussian with std dev =
          macro_radius / 2 around the macro centre, then clipped to the
          padded region.
        - Task positions: drawn from a 2D Gaussian with std dev = micro_radius
          around the micro-cluster centre, clipped to the padded region.

        balance controls whether tasks are split evenly across all
        num_macro * num_micro leaf clusters (True) or sized by a single
        Dirichlet draw over all leaves (False).
        """
        rng = np.random.default_rng(seed)
        total_tasks = size * size
        lo, hi = padding, size - padding

        # --- Place macro centres ---
        margin_macro = max(padding, macro_radius)
        lo_m, hi_m = margin_macro, size - margin_macro

        if lo_m >= hi_m:
            raise ValueError(
                f"HierarchicalClusters: macro_radius ({macro_radius}) too large "
                f"for grid_size={size} with padding={padding}."
            )

        span_m = hi_m - lo_m
        min_sep_macro = span_m / math.sqrt(num_macro)

        macro_centres = []
        attempts = 0
        while len(macro_centres) < num_macro:
            candidate = rng.uniform(lo_m, hi_m, size=2)
            if all(
                math.hypot(candidate[0] - c[0], candidate[1] - c[1]) >= min_sep_macro
                for c in macro_centres
            ):
                macro_centres.append(tuple(float(v) for v in candidate))
            attempts += 1
            if attempts > 50_000:
                raise ValueError(
                    f"HierarchicalClusters: cannot place {num_macro} macro centres "
                    f"with min separation {min_sep_macro:.2f}. "
                    "Reduce num_macro or macro_radius."
                )

        # --- Place micro centres within each macro region ---
        num_leaves = num_macro * num_micro
        micro_centres = []
        for mc in macro_centres:
            for _ in range(num_micro):
                mx = float(np.clip(
                    rng.normal(mc[0], macro_radius / 2.0), lo, hi
                ))
                my = float(np.clip(
                    rng.normal(mc[1], macro_radius / 2.0), lo, hi
                ))
                micro_centres.append((mx, my))

        # --- Assign task counts to leaves ---
        if balance:
            base = total_tasks // num_leaves
            tasks_per_leaf = [base] * num_leaves
            for i in range(total_tasks % num_leaves):
                tasks_per_leaf[i] += 1
        else:
            proportions = rng.dirichlet(np.ones(num_leaves))
            tasks_per_leaf = [max(1, round(p * total_tasks)) for p in proportions]
            diff = sum(tasks_per_leaf) - total_tasks
            tasks_per_leaf[-1] = max(1, tasks_per_leaf[-1] - diff)

        # --- Sample tasks around each micro centre ---
        task_poses = []
        for (mx, my), n_tasks in zip(micro_centres, tasks_per_leaf):
            samples = rng.normal(loc=(mx, my), scale=micro_radius, size=(n_tasks, 2))
            for x, y in samples:
                task_poses.append(
                    (float(np.clip(x, lo, hi)), float(np.clip(y, lo, hi)))
                )

        return task_poses

    # ------------------------------------------------------------------
    # Cost matrix methods
    # ------------------------------------------------------------------

    def calculate_cost_matrix(self):
        """
        Returns a cost matrix representing the traversal cost from each
        task_pose to each other task_pose, calculated as Euclidean distance.
        """
        tasks    = np.array(self.task_poses, dtype=float)  # (n, 2)
        diff     = tasks[:, np.newaxis, :] - tasks[np.newaxis, :, :]  # (n, n, 2)
        cost_map = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(cost_map, 0.0)
        return cost_map

    def update_robot_cost_matrix(self, robot_poses):
        """
        Updates the cost map representing traversal cost from each current
        robot position to each task_pose.
        """
        valid_robot_poses = [pose for pose in robot_poses if pose is not None]
        robots = np.array(valid_robot_poses, dtype=float)   # (n_robots, 2)
        tasks  = np.array(self.task_poses,   dtype=float)   # (n_tasks,  2)
        diff   = robots[:, np.newaxis, :] - tasks[np.newaxis, :, :]  # (n_robots, n_tasks, 2)
        self.current_robot_cost_matrix = np.sqrt((diff ** 2).sum(axis=2))

    def initialize_robot_initial_pose_cost_matrix(self, initial_robot_poses):
        """
        Initialises the cost map representing traversal cost from each agent's
        starting position to each task_pose.
        """
        robots = np.array(initial_robot_poses, dtype=float)  # (n_robots, 2)
        tasks  = np.array(self.task_poses,     dtype=float)  # (n_tasks,  2)
        diff   = robots[:, np.newaxis, :] - tasks[np.newaxis, :, :]
        self.initial_robot_cost_matrix = np.sqrt((diff ** 2).sum(axis=2))