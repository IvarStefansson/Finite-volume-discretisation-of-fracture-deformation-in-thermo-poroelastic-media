"""
This is a setup class for solving the THM equations with contact between the fractures. 
Fluid and temperature flow in the fractures is included in the model.

The simulation is set up for four phases:
    I the system is allowed to reach equilibrium under the influence of the mechanical BCs.
    II pressure gradient from left to right is added.
    III temperature reduced at left boundary.
    IV temperature increased at left boundray.


Compare with model formulation in the paper.

The 2d geometry is borrowed from Berge et al (2019):
The domain is (0, 2) x (0, 1) and contains 7 fractures, two of which form an L intersection.
Most of theproperties are consistent with Berge et al., see also the table in XXX.

Setup at a glance
    One horizontal fracture
    Displacement condition on the north (y = y_max) boundary
    Dirichlet values for p and T at left and right boundaries
 
          \
           \ u = [0.005, -0.002]
            V
        ______________
        |            |
 p=1    |            | p=0
 T=-100 |            | T=0
    100 ______________
        \\\\\\\\\\\\\\
"""


import numpy as np
import scipy.sparse as sps
from scipy.spatial.distance import cdist
import porepy as pp
from porepy.models.thm_model import THM

import logging

logging.basicConfig(level=logging.INFO)


class ExampleModel(THM):
    """
    This class provides the parameter specification of the example, including grid/geometry,
    BCs, rock and fluid parameters and time parameters. Also provides some common modelling
    functions, such as the aperture computation from the displacement jumps, and data storage
    and export functions.
    """

    def __init__(self, params, mesh_args):
        super().__init__(params)
        # Set additional case specific fields
        self.set_fields(mesh_args)

    def create_grid(self):
        """
        Method that creates and returns the GridBucket of a 2D domain with six
        fractures. The two sides of the fractures are coupled together with a
        mortar grid.
        """
        self.frac_pts = np.array(
            [
                [0.2, 0.7],
                [0.5, 0.7],
                [0.8, 0.65],
                [0.2, 0.3],
                [0.6, 0.25],
                [1.0, 0.4],
                [1.7, 0.85],
                [1.5, 0.65],
                [2.0, 0.55],
                [1, 0.3],
                [1.8, 0.4],
                [1.5, 0.05],
                [1.4, 0.25],
            ]
        ).T
        frac_edges = np.array(
            [[0, 1], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
        ).T
        self.n_frac = frac_edges.shape[1]

        self.box = {"xmin": 0, "ymin": 0, "xmax": 2, "ymax": 1}

        network = pp.FractureNetwork2d(self.frac_pts, frac_edges, domain=self.box)
        # Generate the mixed-dimensional mesh
        gb = network.mesh(self.mesh_args)
        pp.contact_conditions.set_projections(gb)
        self.gb = gb
        self.Nd = gb.dim_max()
        self.update_all_apertures(to_iterate=False)
        self.update_all_apertures()

    def porosity(self, g):
        if g.dim == self.Nd:
            return 0.01
        else:
            return 1

    def bc_type_mechanics(self, g):
        """
        Dirichlet values at top and bottom.
        """
        all_bf, east, west, north, south, _, _ = self.domain_boundary_sides(g)
        bc = pp.BoundaryConditionVectorial(g, south + north, "dir")
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True

        return bc

    def bc_values_mechanics(self, g):
        # Retrieve the boundaries where values are assigned
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        bc_values = np.zeros((g.dim, g.num_faces))
        if self.time > -1e4:
            x = 0.005
            y = -0.002
            bc_values[0, north] = x
            bc_values[1, north] = y
        return bc_values.ravel("F")

    def bc_type_scalar(self, g):
        """
        We prescribe the pressure value at the east and west boundary
        """
        # Define boundary regions
        all_bf, east, west, *_ = self.domain_boundary_sides(g)
        return pp.BoundaryCondition(g, east + west, "dir")

    def bc_values_scalar(self, g):
        # Retrieve the boundaries where values are assigned
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        bc_values = np.zeros(g.num_faces)
        if self.time > self.phase_end_times[0]:
            bc_values[west] = 2e8 / self.scalar_scale
        return bc_values

    def bc_type_temperature(self, g):
        """
        We prescribe the temperature value at the east and west boundary
        """
        # Define boundary regions
        all_bf, east, west, *_ = self.domain_boundary_sides(g)
        return pp.BoundaryCondition(g, east + west, "dir")

    def bc_values_temperature(self, g):
        # Retrieve the boundaries where values are assigned
        all_bf, east, west, *_ = self.domain_boundary_sides(g)
        bc_values = np.zeros(g.num_faces)
        if self.time > self.phase_end_times[1]:
            bc_values[west] = -1e2 / self.temperature_scale
        if self.time > self.phase_end_times[2]:
            bc_values[west] = 1e2 / self.temperature_scale

        return bc_values

    def biot_alpha(self, g):
        if g.dim == self.Nd:
            return 1.0
        else:
            return 0.0

    def biot_beta(self, g):
        """
        For TM, the coefficient is the product of the bulk modulus (=inverse of
        the compressibility) and the volumetric thermal expansion coefficient.
        """
        if g.dim == self.Nd:
            # Factor 3 for volumetric/linear, since the pp.Granite
            # thermal expansion expansion coefficient is the linear one at 20 degrees C.
            return self.rock.BULK_MODULUS * 3 * self.rock.THERMAL_EXPANSION
        else:
            return 0

    def scalar_temperature_coupling_coefficient(self, g):
        """
        The temperature-pressure coupling coefficient is porosity times thermal
        expansion. In the energy conservation equation, the dp/dt term is also
        multiplied by T_0, but the equation is scaled by this factor. The pressure and
        scalar scale must be accounted for wherever this coefficient is used.
        """
        thermal_expansion = self.fluid.thermal_expansion(self.background_temp_C)
        coeff = -self.porosity(g) * thermal_expansion
        return coeff

    def compute_aperture(self, g, from_iterate=True):
        """
        Obtain the aperture of a subdomain. See update_all_apertures.
        """
        if from_iterate:
            return self.gb.node_props(g)[pp.STATE]["iterate_aperture"]
        else:
            return self.gb.node_props(g)[pp.STATE]["aperture"]

    def specific_volumes(self, g, from_iterate=True):
        """
        Obtain the specific volume of a subdomain. See update_all_apertures.
        """
        if from_iterate:
            return self.gb.node_props(g)[pp.STATE]["iterate_specific_volume"]
        else:
            return self.gb.node_props(g)[pp.STATE]["specific_volume"]

    def update_all_apertures(self, to_iterate=True):
        """
        To better control the aperture computation, it is done for the entire gb by a
        single function call. This also allows us to ensure the fracture apertures
        are updated before the intersection apertures are inherited.
        The aperture of a fracture is the initial aperture + normal displacement jump.
        """
        gb = self.gb
        for g, d in gb:

            apertures = np.ones(g.num_cells)
            if g.dim == (self.Nd - 1):
                # Initial aperture

                apertures *= self.initial_aperture
                # Reconstruct the displacement solution on the fracture
                g_h = gb.node_neighbors(g)[0]
                data_edge = gb.edge_props((g, g_h))
                if pp.STATE in data_edge:
                    u_mortar_local = self.reconstruct_local_displacement_jump(
                        data_edge, from_iterate=to_iterate
                    )
                    # Magnitudes of normal and tangential components
                    norm_u_n = np.absolute(u_mortar_local[-1])
                    norm_u_tau = np.linalg.norm(u_mortar_local[:-1], axis=0)
                    # Add contributions
                    slip_angle = np.pi / 2
                    apertures += norm_u_n + np.cos(slip_angle) * norm_u_tau
            if to_iterate:
                state = {
                    "iterate_aperture": apertures,
                    "iterate_specific_volume": apertures,
                }
            else:
                state = {"aperture": apertures, "specific_volume": apertures}
            pp.set_state(d, state)
        for g, d in gb:
            parent_apertures = []
            if g.dim < (self.Nd - 1):
                for edges in gb.edges_of_node(g):
                    e = edges[0]
                    g_h = e[0]

                    if g_h == g:
                        g_h = e[1]

                    if g_h.dim == (self.Nd - 1):
                        d_h = gb.node_props(g_h)
                        if to_iterate:
                            a_h = d_h[pp.STATE]["iterate_aperture"]
                        else:
                            a_h = d_h[pp.STATE]["aperture"]
                        a_h_face = np.abs(g_h.cell_faces) * a_h
                        mg = gb.edge_props(e)["mortar_grid"]
                        # Assumes g_h is master
                        a_l = (
                            mg.mortar_to_slave_int()
                            * mg.master_to_mortar_int()
                            * a_h_face
                        )
                        parent_apertures.append(a_l)
                    else:
                        raise ValueError("Intersection points not implemented in 3d")
                parent_apertures = np.array(parent_apertures)
                apertures = np.mean(parent_apertures, axis=0)
                specific_volumes = np.product(parent_apertures, axis=0)
                if to_iterate:
                    state = {
                        "iterate_aperture": apertures,
                        "iterate_specific_volume": specific_volumes,
                    }
                else:
                    state = {"aperture": apertures, "specific_volume": specific_volumes}
                pp.set_state(d, state)

        return apertures

    def set_permeability_from_aperture(self):
        """
        Cubic law in fractures, rock permeability in the matrix.
        """
        # Viscosity has units of Pa s, and is consequently divided by the scalar scale.
        viscosity = self.fluid.dynamic_viscosity() / self.scalar_scale
        gb = self.gb
        key = self.scalar_parameter_key
        for g, d in gb:
            if g.dim < self.Nd:
                # Use cubic law in fractures. First compute the unscaled
                # permeability
                apertures = self.compute_aperture(g, from_iterate=True)
                apertures_unscaled = apertures * self.length_scale
                k = np.power(apertures_unscaled, 2) / 12
                # Multiply with the cross-sectional area, which equals the apertures
                # for 2d fractures in 3d
                specific_volumes = self.specific_volumes(g)
                k *= specific_volumes
                # Divide by fluid viscosity and scale back
                kxx = k / viscosity / self.length_scale ** 2
            else:
                # Use the rock permeability in the matrix
                kxx = (
                    self.rock.PERMEABILITY
                    / viscosity
                    * np.ones(g.num_cells)
                    / self.length_scale ** 2
                )
            K = pp.SecondOrderTensor(kxx)
            d[pp.PARAMETERS][key]["second_order_tensor"] = K

        # Normal permeability inherited from the neighboring fracture g_l
        for e, d in gb.edges():
            mg = d["mortar_grid"]
            g_l, _ = gb.nodes_of_edge(e)
            data_l = gb.node_props(g_l)
            a = self.compute_aperture(g_l)
            # We assume isotropic permeability in the fracture, i.e. the normal
            # permeability equals the tangential one
            k_s = data_l[pp.PARAMETERS][key]["second_order_tensor"].values[0, 0]
            # Division through half the aperture represents taking the (normal) gradient
            kn = mg.slave_to_mortar_int() * np.divide(k_s, a ** 2 / 2)
            pp.initialize_data(mg, d, key, {"normal_diffusivity": kn})

    def set_rock_and_fluid(self):
        """
        Set rock and fluid properties to those of granite and water.
        We ignore all temperature dependencies of the parameters.
        """
        self.rock = Granite()
        self.fluid = Water()

    def set_mechanics_parameters(self):
        """ Mechanical parameters.
        Note that we divide the momentum balance equation by self.scalar_scale. 
        A homogeneous initial temperature is assumed.
        """
        gb = self.gb
        for g, d in gb:
            if g.dim == self.Nd:
                # Rock parameters
                rock = self.rock
                lam = rock.LAMBDA * np.ones(g.num_cells) / self.scalar_scale
                mu = rock.MU * np.ones(g.num_cells) / self.scalar_scale
                C = pp.FourthOrderTensor(mu, lam)

                bc = self.bc_type_mechanics(g)
                bc_values = self.bc_values_mechanics(g)
                sources = self.source_mechanics(g)

                # In the momentum balance, the coefficient hits the scalar, and should
                # not be scaled. Same goes for the energy balance, where we divide all
                # terms by T_0, hence the term originally beta K T d(div u) / dt becomes
                # beta K d(div u) / dt = coupling_coefficient d(div u) / dt.
                coupling_coefficient = self.biot_alpha(g)

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": bc_values,
                        "source": sources,
                        "fourth_order_tensor": C,
                        "biot_alpha": coupling_coefficient,
                        "time_step": self.time_step,
                    },
                )
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_temperature_parameter_key,
                    {"biot_alpha": self.biot_beta(g), "bc_values": bc_values},
                )
            elif g.dim == self.Nd - 1:

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "friction_coefficient": self._friction_coefficient(g),
                        "contact_mechanics_numerical_parameter": 10000,
                    },
                )

        for e, d in gb.edges():
            mg = d["mortar_grid"]
            # Parameters for the surface diffusion. Not used as of now.
            pp.initialize_data(
                mg,
                d,
                self.mechanics_parameter_key,
                {"mu": self.rock.MU, "lambda": self.rock.LAMBDA},
            )

    def aperture_update(self, g, d):
        """
        Because of the nonlinearity for intersections, the d/dt specific_volume term of
        the lower-dimensional mass conservation equation is computed from the previous iteration.
        """
        if g.dim == self.Nd:
            return np.zeros(g.num_cells)

        it = d[pp.STATE]["iterate_specific_volume"]
        time = d[pp.STATE]["specific_volume"]
        return -g.cell_volumes * (it - time)

    def set_scalar_parameters(self):

        for g, d in self.gb:
            a = self.compute_aperture(g)
            specific_volumes = self.specific_volumes(g)

            # Define boundary conditions for flow
            bc = self.bc_type_scalar(g)
            # Set boundary condition values
            bc_values = self.bc_values_scalar(g)

            biot_coefficient = self.biot_alpha(g)
            compressibility = self.fluid.COMPRESSIBILITY

            mass_weight = compressibility * self.porosity(g)
            if g.dim == self.Nd:
                mass_weight += (
                    biot_coefficient - self.porosity(g)
                ) / self.rock.BULK_MODULUS
            mass_weight *= self.scalar_scale * specific_volumes

            pp.initialize_data(
                g,
                d,
                self.scalar_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": mass_weight,
                    "biot_alpha": biot_coefficient,
                    "time_step": self.time_step,
                    "source": self.aperture_update(g, d),
                },
            )

            t2s_coupling = (
                self.scalar_temperature_coupling_coefficient(g)
                * specific_volumes
                * self.temperature_scale
            )
            pp.initialize_data(
                g,
                d,
                self.t2s_parameter_key,
                {"mass_weight": t2s_coupling, "time_step": self.time_step},
            )

        self.set_permeability_from_aperture()

    def set_temperature_parameters(self):
        """ temperature parameters.
        The entire equation is divided by the initial temperature in Kelvin.
        """
        div_T_scale = self.temperature_scale / self.length_scale ** 2 / self.T_0_Kelvin
        kappa_f = self.fluid.thermal_conductivity() * div_T_scale
        kappa_s = self.rock.thermal_conductivity() * div_T_scale
        heat_capacity_f = self.fluid.density() * self.fluid.specific_heat_capacity(
            self.background_temp_C
        )
        heat_capacity_s = (
            self.rock.specific_heat_capacity(self.background_temp_C) * self.rock.DENSITY
        )
        for g, d in self.gb:
            # Aperture and cross sectional area
            specific_volumes = self.specific_volumes(g)
            porosity = self.porosity(g)
            # Define boundary conditions for flow
            bc = self.bc_type_temperature(g)
            # Set boundary condition values
            bc_values = self.bc_values_temperature(g)
            # and source values
            biot_coefficient = self.biot_beta(g)

            effective_heat_capacity = (
                porosity * heat_capacity_f + (1 - porosity) * heat_capacity_s
            )
            mass_weight = (
                effective_heat_capacity
                * specific_volumes
                * self.temperature_scale
                / self.T_0_Kelvin
            )

            effective_conductivity = porosity * kappa_f + (1 - porosity) * kappa_s
            thermal_conductivity = pp.SecondOrderTensor(
                effective_conductivity * specific_volumes
            )

            advection_weight = (
                heat_capacity_f * self.temperature_scale / self.T_0_Kelvin
            )

            pp.initialize_data(
                g,
                d,
                self.temperature_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_values,
                    "mass_weight": mass_weight,
                    "second_order_tensor": thermal_conductivity,
                    "advection_weight": advection_weight,
                    "biot_alpha": biot_coefficient,
                    "time_step": self.time_step,
                    "darcy_flux": np.zeros(g.num_faces),
                    "source": np.zeros(g.num_cells),
                },
            )

            if self._thm:
                s2t_coupling = (
                    self.scalar_temperature_coupling_coefficient(g)
                    * specific_volumes
                    * self.scalar_scale
                )
                pp.initialize_data(
                    g,
                    d,
                    self.s2t_parameter_key,
                    {"mass_weight": s2t_coupling, "time_step": self.time_step},
                )

        for e, data_edge in self.gb.edges():
            g1, g2 = self.gb.nodes_of_edge(e)
            mg = data_edge["mortar_grid"]
            a1 = self.compute_aperture(g1)
            a_mortar = mg.slave_to_mortar_avg() * a1
            kappa_n = 2 / a_mortar * kappa_f
            data_edge = pp.initialize_data(
                e,
                data_edge,
                self.temperature_parameter_key,
                {"normal_diffusivity": kappa_n},
            )

    def prepare_simulation(self):
        self._set_time_parameters()
        self.set_rock_and_fluid()
        super().prepare_simulation()
        self.export_step()

    def after_newton_iteration(self, solution):
        self.update_all_apertures(to_iterate=True)
        self.set_parameters()
        super().after_newton_iteration(solution)

    def after_newton_convergence(self, solution, errors, iteration_counter):
        self.update_all_apertures(to_iterate=False)
        super().after_newton_convergence(solution, errors, iteration_counter)
        self.export_step()
        self.adjust_time_step()
        self.save_data(errors, iteration_counter)

    def set_exporter(self):
        self.exporter = pp.Exporter(
            self.gb, self.file_name, folder=self.viz_folder_name + "_vtu"
        )
        self.export_times = []

    def export_step(self):
        """
        Export the current solution to vtu. The method sets the desired values in d[pp.STATE].
        For some fields, it provides zeros in the dimensions where the variable is not defined,
        or pads the vector values with zeros so that they have three components, as required
        by ParaView.
        We use suffix _exp on all exported variables, to separate from scaled versions also
        stored in d.
        """
        if "exporter" not in self.__dict__:
            self.set_exporter()
        for g, d in self.gb:
            if g.dim == self.Nd:
                pad_zeros = np.zeros((3 - g.dim, g.num_cells))
                u = d[pp.STATE][self.displacement_variable].reshape(
                    (self.Nd, -1), order="F"
                )
                d[pp.STATE]["u_exp"] = np.vstack((u * self.length_scale, pad_zeros))
                d[pp.STATE]["traction_exp"] = np.zeros(d[pp.STATE]["u_exp"].shape)
            elif g.dim == (self.Nd - 1):
                pad_zeros = np.zeros((2 - g.dim, g.num_cells))
                g_h = self.gb.node_neighbors(g)[0]
                data_edge = self.gb.edge_props((g, g_h))

                u_mortar_local = self.reconstruct_local_displacement_jump(
                    data_edge, from_iterate=False
                )
                d[pp.STATE]["u_exp"] = np.vstack(
                    (u_mortar_local * self.length_scale, pad_zeros)
                )
                traction = d[pp.STATE][self.contact_traction_variable].reshape(
                    (self.Nd, -1), order="F"
                )

                d[pp.STATE]["traction_exp"] = (
                    np.vstack((traction, pad_zeros))
                    * self.scalar_scale
                    * self.length_scale ** g.dim
                )  # (1d area)
            else:
                d[pp.STATE]["traction_exp"] = np.zeros((3, g.num_cells))
                d[pp.STATE]["u_exp"] = np.zeros((3, g.num_cells))

            d[pp.STATE]["aperture"] = self.compute_aperture(g)
            d[pp.STATE]["p_exp"] = d[pp.STATE][self.scalar_variable] * self.scalar_scale
            if "T_exp" in self.export_fields:
                d[pp.STATE]["T_exp"] = (
                    d[pp.STATE][self.temperature_variable] * self.temperature_scale
                )
        self.exporter.write_vtk(self.export_fields, time_step=self.time)
        self.export_times.append(self.time)

    def export_pvd(self):
        """
        At the end of the simulation, after the final vtu file has been exported, the
        pvd file for the whole simulation is written by calling this method.
        """
        self.exporter.write_pvd(np.array(self.export_times))

    def save_data(self, errors, iteration_counter):
        """
        Save displacement jumps and number of iterations for visualisation purposes. 
        These are written to file and plotted against time in Figure 4.
        """
        n = self.n_frac
        if "u_jumps_tangential" not in self.__dict__:
            self.u_jumps_tangential = np.empty((1, n))
            self.u_jumps_normal = np.empty((1, n))
            self.iterations = []
        self.iterations.append(iteration_counter)
        tangential_u_jumps = np.zeros((1, n))
        normal_u_jumps = np.zeros((1, n))

        for g, d in self.gb:
            if g.dim == self.Nd - 1:
                g_h = self.gb.node_neighbors(g)[0]
                data_edge = self.gb.edge_props((g, g_h))
                u_mortar_local = self.reconstruct_local_displacement_jump(
                    data_edge, from_iterate=False
                )
                tangential_jump = np.linalg.norm(
                    u_mortar_local[:-1] * self.length_scale, axis=0
                )
                normal_jump = u_mortar_local[-1] * self.length_scale
                vol = np.sum(g.cell_volumes)
                tangential_jump_norm = (
                    np.sqrt(np.sum(tangential_jump ** 2 * g.cell_volumes)) / vol
                )
                normal_jump_norm = (
                    np.sqrt(np.sum(normal_jump ** 2 * g.cell_volumes)) / vol
                )
                ind = g.frac_num
                tangential_u_jumps[0, ind] = tangential_jump_norm
                normal_u_jumps[0, ind] = normal_jump_norm

        self.u_jumps_tangential = np.concatenate(
            (self.u_jumps_tangential, tangential_u_jumps)
        )
        self.u_jumps_normal = np.concatenate((self.u_jumps_normal, normal_u_jumps))

    def assign_discretizations(self):
        super().assign_discretizations()

    def _set_time_parameters(self):
        """
        Specify time parameters.
        """
        # For the initialization run, we use the following
        # start time
        self.time = -1e4
        # and time step
        self.time_step = -0.5 * self.time

        # We use
        self.max_time_step = 1.5e-1
        self.end_time = 5.0
        self.phase_end_times = [0, 2e-2, 2.5, self.end_time]
        self.phase_time_steps = [2e-4, 4e-3, 5e-2, 1.0]

    def _friction_coefficient(self, g):
        #  Used for comparison with traction ratio, which always is dimensionless, i.e. no scaling.
        return 0.9

    def adjust_time_step(self):
        """
        Adjust the time step so that smaller time steps are used when the driving forces
        are changed. Also make sure to exactly reach the start and end time for
        each phase. 
        """
        # Default is to just increase the time step somewhat

        self.time_step = 1.25 * self.time_step
        # We also want to make sure that we reach the end of each simulation phase:

        for i, t in enumerate(self.phase_end_times):
            diff = self.time - t
            if diff < 0 and -diff <= self.time_step:
                self.time_step = -diff

            if np.isclose(self.time, t):
                self.time_step = self.phase_time_steps[i]

        if self.time > 0:
            self.time_step = min(self.time_step, self.max_time_step)

    def set_fields(self, mesh_args):
        """
        Set various fields to be used in the model.
        """
        # We operate on the temperature difference T-T*
        self.T_0 = 0
        self.background_temp_C = 0
        # Divide all but the divd term by T0
        self.T_0_Kelvin = self.background_temp_C + 273
        self.mesh_args = mesh_args

        # Scaling coefficients
        self.scalar_scale = 1e8
        self.length_scale = 1e0
        self.temperature_scale = 1e-2
        self.s_0 = 0

        self.file_name = self.params["file_name"]
        self.folder_name = self.params["folder_name"]
        # Keywords
        self.mechanics_parameter_key_from_t = "mechanics_from_t"
        # Temperature
        self.scalar_variable = "p"
        self.mortar_scalar_variable = "mortar_" + self.scalar_variable

        self.scalar_coupling_term = "robin_" + self.scalar_variable
        self.scalar_parameter_key = "flow"

        self.export_fields = ["u_exp", "p_exp", "T_exp", "traction_exp", "aperture"]
        # Initial hydraulic aperture
        self.initial_aperture = 1e-4 / self.length_scale
        self._thm = True


class Water(pp.Water):
    """
    Adjust according to values used in Berge (2019).
    """

    def __init__(self, theta_ref=None):
        if theta_ref is None:
            self.theta_ref = 20 * (pp.CELSIUS)
        else:
            self.theta_ref = theta_ref
        self.VISCOSITY = 1 * pp.MILLI * pp.PASCAL * pp.SECOND
        # Effective compressibility is updated from Berge (C + alpha-phi/K)
        self.COMPRESSIBILITY = 1e-10 / pp.PASCAL  # Moderate dependency on theta
        self.BULK_MODULUS = 1 / self.COMPRESSIBILITY

    def dynamic_viscosity(self, theta=None):  # theta in CELSIUS
        """Units: Pa s"""
        return 1 * pp.MILLI * pp.PASCAL * pp.SECOND


class Granite(pp.Granite):
    """
    Adjust according to values used in Berge (2019).
    """

    def __init__(self, theta_ref=None):
        super().__init__(theta_ref)
        self.BULK_MODULUS = pp.params.rock.bulk_from_lame(self.LAMBDA, self.MU)
        self.FRICTION_COEFFICIENT = 0.5
        self.PERMEABILITY = 1e-11


def write_displacement_jumps_to_file(setup):
    """
    Write data summarizing the results to csv files. Four files are written:
        time steps
        Newton iterations for all time steps    
        normal displacement jumps on all fractures for all time steps
        tangential displacement jumps on all fractures for all time steps
    """

    t = np.array(np.array(setup.export_times))
    iterations = np.array(np.array(setup.iterations))

    ind = -t.size
    fn = setup.folder_name + "/"
    np.savetxt(
        fn + "tangential_displacement_jumps_" + setup.file_name + ".txt",
        setup.u_jumps_tangential[ind:],
    )
    np.savetxt(
        fn + "normal_displacement_jumps_" + setup.file_name + ".txt",
        setup.u_jumps_normal[ind:],
    )
    np.savetxt(fn + "time_steps_" + setup.file_name + ".txt", t)
    np.savetxt(fn + "iterations_" + setup.file_name + ".txt", iterations)


if __name__ == "__main__":
    # Define mesh sizes for grid generation.
    mesh_size = 0.014
    mesh_args = {
        "mesh_size_frac": mesh_size,
        "mesh_size_min": 0.2 * mesh_size,
        "mesh_size_bound": 3 * mesh_size,
    }
    params = {
        "folder_name": "results",
        "convergence_tol": 2e-7,
        "max_iterations": 20,
        "file_name": "thm",
    }

    setup = ExampleModel(params, mesh_args)
    pp.run_time_dependent_model(setup, params)
    setup.export_pvd()
    write_displacement_jumps_to_file(setup)
